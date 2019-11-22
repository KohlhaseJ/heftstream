from skmultiflow.data import WaveformGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.meta.accuracy_weighted_ensemble import AccuracyWeightedEnsemble

# 1. Create a stream
stream = WaveformGenerator()
stream.prepare_for_use()

# 2. Instantiate the AccuracyWeighted Ensemble
ensemble = AccuracyWeightedEnsemble()

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=20000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ensemble)
