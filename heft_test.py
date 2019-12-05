from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import LEDGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.meta import AccuracyWeightedEnsemble
from src.heft.meta.heterogeneous_ensemble_for_featuredrifts import HeterogenousEnsembleForFeatureDrifts

# 1. Create a stream
#stream = FileStream("data/unpacked/kdd.csv")
#stream.prepare_for_use()
stream = SEAGenerator(noise_percentage=0.1)
#stream = HyperplaneGenerator(noise_percentage=0.05, mag_change=0.001, n_drift_features=10) # TODO: n_drift_features = changing 10 attributes at speed ...?
#stream = LEDGenerator(noise_percentage=0.1, has_noise=True)
stream.prepare_for_use()

# 2. Instantiate the Ensemble
ensemble = HeterogenousEnsembleForFeatureDrifts(verbose=1, window_size=1000, n_kept_estimators=10)

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=1000,
                                max_samples=20000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ensemble)
