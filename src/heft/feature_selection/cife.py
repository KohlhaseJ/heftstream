from .helper import entropy_estimators as ee
from .base import BaseSelector
import numpy as np

class CIFE(BaseSelector):
    @staticmethod
    def select(X, y):
        """
        This function implements the basic scoring criteria for linear combination of shannon information term.
        The scoring criteria is calculated based on the formula j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be a discrete data matrix
        y: {numpy array}, shape (n_samples,)
            input class labels

        Output
        ------
        F: {numpy array}, shape: (n_features,)
            index of selected features, F[0] is the most important feature

        Reference
        ---------
        Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
        """

        _, n_features = X.shape
        # index of selected features, initialized to be empty
        F = []
        # Objective function value for selected features
        J_CMI = []
        # Mutual information between feature and response
        MIfy = []
        # indicate whether the user specifies the number of features
        is_n_selected_features_specified = False
        # initialize the parameters
        beta = 1
        gamma = 1

        # select the feature whose j_cmi is the largest
        # t1 stores I(f;y) for each feature f
        t1 = np.zeros(n_features)
        # t2 stores sum_j(I(fj;f)) for each feature f
        t2 = np.zeros(n_features)
        # t3 stores sum_j(I(fj;f|y)) for each feature f
        t3 = np.zeros(n_features)
        for i in range(n_features):
            f = X[:, i]
            t1[i] = ee.midd(f, y)

        # make sure that j_cmi is positive at the very beginning
        j_cmi = 1

        while True:
            if len(F) == 0:
                # select the feature whose mutual information is the largest
                idx = np.argmax(t1)
                F.append(idx)
                J_CMI.append(t1[idx])
                MIfy.append(t1[idx])
                f_select = X[:, idx]

            if j_cmi < 0:
                break

            # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
            j_cmi = -1E30
            for i in range(n_features):
                if i not in F:
                    f = X[:, i]
                    t2[i] += ee.midd(f_select, f)
                    t3[i] += ee.cmidd(f_select, f, y)
                    # calculate j_cmi for feature i (not in F)
                    t = t1[i] - beta*t2[i] + gamma*t3[i]
                    # record the largest j_cmi and the corresponding feature index
                    if t > j_cmi:
                        j_cmi = t
                        idx = i
            F.append(idx)
            J_CMI.append(j_cmi)
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        return np.array(F)