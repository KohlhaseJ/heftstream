from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import LEDGenerator, DataStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees.hoeffding_anytime_tree import HATT
from skmultiflow.meta import AccuracyWeightedEnsemble
from src.heft.meta.heterogeneous_ensemble_for_featuredrifts import HeterogeneousEnsembleForFeatureDrifts
from src.heft.feature_selection.fcbf import FCBF
from skmultiflow.meta.accuracy_weighted_ensemble import AccuracyWeightedEnsemble
from sklearn.datasets import fetch_openml
from skmultiflow.evaluation import EvaluatePrequential

# 1. Create a stream
stream = FileStream("data/MNIST_normalized.csv")

# mnist
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# stream = DataStream(X, y=y)

# stream.prepare_for_use()
# stream = SEAGenerator(noise_percentage=0.1)
# stream = HyperplaneGenerator(noise_percentage=0.05, mag_change=0.001, n_drift_features=10) # TODO: n_drift_features = changing 10 attributes at speed ...?
# stream = LEDGenerator(noise_percentage=0.1, has_noise=True)
stream.prepare_for_use()
print(stream.target_names)

# 2. Instantiate the Ensembles
heft = HeterogeneousEnsembleForFeatureDrifts(window_size=1000, n_kept_estimators=10,
                                                min_features=5, base_estimators=[NaiveBayes(), HATT()], feature_selector=FCBF,
                                                random_ensemble=0, verbose=1)
#aweNB = AccuracyWeightedEnsemble(window_size=1000, n_kept_estimators=10, base_estimator=NaiveBayes())
#aweHAT = AccuracyWeightedEnsemble(window_size=1000, n_kept_estimators=10, base_estimator=HAT())


# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=10000)

# 4. Run evaluation
print('Evaluating HEFT Stream')
evaluator.evaluate(stream=stream, model=heft)
#print('Evaluating AWE NB')
#evaluator.evaluate(stream=stream, model=aweNB)
#print('Evaluating AWE HAT')
#evaluator.evaluate(stream=stream, model=aweHAT)
