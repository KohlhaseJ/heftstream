from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees.hoeffding_adaptive_tree import HoeffdingTree
from skmultiflow.utils import check_random_state
from sklearn.model_selection import KFold
from ..feature_selection.fcbf import FCBF
import numpy as np
import copy as cp
import operator as op


class HeterogeneousEnsembleForFeatureDrifts(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Heterogeneous ensemble for feature drifts (HEFT) classifier

    Parameters
    ----------
    n_estimators: int (default=10)
        Maximum number of estimators to be kept in the ensemble
    base_estimators: numpy array of skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=NaiveBayes)
        Members of the ensemble are added as an instance of the best performing base estimator.
    window_size: int (default=200)
        The size of one chunk to be processed
        (warning: the chunk size is not always the same as the batch size)
    n_splits: int (default=5)
        Number of folds to run cross-validation for computing the weight
        of a classifier in the ensemble
    min_features: int (default=5)
        Minimum number of features before appending a feature selection method
    random_state: int, RandomState instance or None, (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    verbose: bit (default=0)
        A parameter defined to print some more output when training the classifier.

    Notes
    -----
    The heterogeneous ensemble or feature drifts in data streams [1]_ is an ensemble of
    different classification models that incorporates feature selection into a
    heterogeneous ensemble to adapt to different types of concept drifts.

    References
    ----------
    .. [1] Nguyen, H. L., Woon, Y. K., Ng, W. K., & Wan, L. (2012, May).
       Heterogeneous ensemble for feature drifts in data streams. In
       Pacific-Asia conference on knowledge discovery and data mining
       (pp. 1-12). Springer, Berlin, Heidelberg.

    """

    class HeftClassifier:
        """ A wrapper that includes a base estimator, its aggregated error and the
        the feature subspace the estimator was created with. 

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The base estimator to be wrapped up with additional information.
            This estimator must already been trained on a data chunk.
        error: float
            The error associated to this estimator
        seen_labels: array
            The array containing the unique class labels of the data chunk this estimator
            is trained on.
        selected_features: array
            The array containing the indices gathered by a feature selection algorithm
            for the features this estimator is trained on.
        """

        def __init__(self, estimator, error, seen_labels, selected_features):
            """ Creates a new HEFT classifier."""
            self.estimator = estimator
            self.error = error
            self.seen_labels = seen_labels
            self.selected_features = selected_features

        def __lt__(self, other):
            """ Compares an object of this class to the other by means of the weight.
            This method helps to sort the classifier correctly in the sorted list.

            Parameters
            ----------
            other: HeftClassifier
                The other object to be compared to

            Returns
            -------
            boolean
                true if this object's weight is less than that of the other object
            """
            return self.error > other.error

    # adjust and test n_kept estimators
    def __init__(self, n_estimators=10, n_kept_estimators=10, base_estimators=np.array([NaiveBayes(), HoeffdingTree()]),
                window_size=200, n_splits=5, min_features=5, random_state=None, verbose=0):
        """ Create a new ensemble"""

        super().__init__()

        # top K classifiers
        self.n_estimators = n_estimators

        # total number of classifiers to keep
        self.n_kept_estimators = n_kept_estimators

        # base learner
        self.base_estimators = base_estimators

        # the ensemble in which the classifiers are sorted by their weight
        self.models_pool = []

        # cross validation fold
        self.n_splits = n_splits

        # threshold for feature selection
        self.min_features = min_features

        # chunk-related information
        self.window_size = window_size  # chunk size
        self.p = -1  # chunk pointer
        self.X_chunk = None
        self.y_chunk = None

        # feature selection
        self.last_selected_features = None
        
        # set random state
        self.random_state = check_random_state(random_state)

        # verbose param
        self.verbose = verbose

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        TODO: change description to fit HEFT
        Updates the ensemble when a new data chunk arrives (Algorithm 1 in the paper).
        The update is only launched when the chunk is filled up.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.
        classes: numpy.ndarray, optional (default=None)
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.
        """
        _, D = X.shape

        # initializes everything when the ensemble is first called
        if self.p == -1:
            self.X_chunk = np.zeros((self.window_size, D))
            self.y_chunk = np.zeros(self.window_size)
            self.last_selected_features = np.array([])
            self.p = 0

        # fill up the data chunk
        for i, _ in enumerate(X):
            self.X_chunk[self.p] = X[i]
            self.y_chunk[self.p] = y[i]
            self.p += 1

            if self.p == self.window_size:
                # reset the pointer
                self.p = 0

                # feature selection
                selected_features = range(len(self.X_chunk[0]))
                if len(self.X_chunk[0]) >= self.min_features:
                    selected_features, _ = FCBF(self.X_chunk, self.y_chunk, **{'delta': 0})
                
                if self.verbose == 1:
                    print("Selected {0} feature(s) out of {1}: {2}".format(len(selected_features), len(self.X_chunk[0]), selected_features))

                # retrieve the classes
                classes = np.unique(self.y_chunk)

                # check if feature drift occured
                if len(np.setdiff1d(selected_features, self.last_selected_features, assume_unique=True)) != 0:
                    if self.verbose == 1:
                        print("Feature drift occured.")

                    add_models = []
                    if self.models_pool:
                        best_model = min(self.models_pool, key=op.attrgetter("error"))
                        for base_estimator in self.base_estimators:
                            if isinstance(base_estimator, type(best_model.estimator)):
                                add_models.append(base_estimator)
                    else:
                        add_models.extend(self.base_estimators)

                    for add_model in add_models:
                        add_classifier = self.HeftClassifier(estimator=cp.deepcopy(add_model), error=0.0,
                                                                seen_labels=classes, selected_features=selected_features)

                        # add the new model to the pool if there are slots available, else remove the worst one
                        if len(self.models_pool) >= self.n_kept_estimators:
                            worst_model = max(self.models_pool, key=op.attrgetter("error"))
                            self.models_pool.remove(worst_model)
                        
                        self.models_pool.append(add_classifier)
                else:                    
                    if self.verbose == 1:
                        print("No feature drift.")
                
                # partial fit the models created with current feature subspace
                for model in self.models_pool:
                    if len(np.setdiff1d(selected_features, model.selected_features, assume_unique=True)) == 0:                            
                        for i in range(self.X_chunk.shape[0]):
                            # online bagging poisson(1)
                            m = self.random_state.poisson()
                            if m > 0:
                                for _ in range(m):
                                    model.estimator.partial_fit(np.asarray([self.X_chunk[i, selected_features]]), np.asarray([self.y_chunk[i]]))
                    # calculate aggregated error
                    model.error += self.compute_mse(model=model, X=self.X_chunk, y=self.y_chunk)
                
                # safe latest feature selection
                self.last_selected_features = selected_features
                # print types of all models
                if self.verbose == 1:
                    model_types = {}
                    for model in self.models_pool:
                        if type(model.estimator) in model_types:
                            model_types[type(model.estimator)] += 1
                        else:
                            model_types[type(model.estimator)] = 1
                    print(model_types)
                
                # instance-based pruning only happens with Cost Sensitive extension
                self.do_instance_pruning()

        return self

    def do_instance_pruning(self):
        # Only has effect if the ensemble is applied in cost-sensitive applications.
        # Not used in the current implementation.
        pass

    def predict(self, X):
        """ Predicts the labels of X in a general classification setting.

        The prediction is done via normalized weighted voting (choosing the maximum).

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.
        """
        N = len(X)

        if len(self.models_pool) == 0:
            return np.zeros(N, dtype=int)
        
        weighted_votes = [dict()] * N
        for model in self.models_pool:
            X_hat = X[:, model.selected_features]
            probabs = model.estimator.predict_proba(X_hat)
            w = 1 / (model.error + 0.001)

            for i, label in enumerate(model.seen_labels):
                for j, p in enumerate(probabs):
                    if label in weighted_votes[j]:
                        weighted_votes[j][label] += w*p[i]
                    else:
                        weighted_votes[j][label] = w*p[i]

        predict_weighted_voting = np.zeros(N, dtype=int)
        for i, dic in enumerate(weighted_votes):
            predict_weighted_voting[i] = int(max(dic.items(), key=op.itemgetter(1))[0])

        return predict_weighted_voting

    def predict_proba(self, X):
        raise NotImplementedError

    def reset(self):
        """ Resets all parameters to its default value"""
        self.models_pool = []
        self.p = -1
        self.X_chunk = None
        self.y_chunk = None

    @staticmethod
    def compute_mse(model, X, y):
        """ Computes the mean square error of a classifier, via the predicted probabilities.

        This code needs to take into account the fact that a classifier C trained on a
        previous data chunk may not have seen all the labels that appear in a new chunk
        (e.g. C is trained with only labels [1, 2] but the new chunk contains labels [1, 2, 3, 4, 5]

        Parameters
        ----------
        model: StreamModel or sklearn.BaseEstimator
            The estimator in the ensemble to compute the score on
        X: numpy.ndarray of shape (window_size, n_features)
            The data from the new chunk
        y: numpy.array
            The labels from the new chunk

        Returns
        -------
        float
            The mean square error of the model (MSE_i)
        """
        N = len(y)
        labels = model.seen_labels
        probabs = model.estimator.predict_proba(X[:, model.selected_features])

        sum_error = 0
        for i, c in enumerate(y):
            if c in labels:
                index_label_c = np.where(labels == c)[0][0]  # find the index of this label c in probabs[i]
                probab_ic = probabs[i][index_label_c]
                sum_error += (1.0 - probab_ic) ** 2
            else:
                sum_error += 1.0

        return sum_error / N
