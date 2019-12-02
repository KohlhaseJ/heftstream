from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import LEDGenerator
from skmultiflow.evaluation import EvaluatePrequential
from src.heft.meta.heterogeneous_ensemble_for_featuredrifts import HeterogenousEnsembleForFeatureDrifts

# 1. Create a stream
#stream = FileStream("data/unpacked/kdd.csv")
#stream.prepare_for_use()
stream = LEDGenerator()
stream.prepare_for_use()

# 2. Instantiate the AccuracyWeighted Ensemble
ensemble = HeterogenousEnsembleForFeatureDrifts(verbose=1)

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=20000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ensemble)
