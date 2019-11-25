from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential
from src.heft.meta.heterogenous_ensemble_for_featuredrifts import HeterogenousEnsembleForFeatureDrifts

# 1. Create a stream
stream = FileStream("data/oil.csv")
stream.prepare_for_use()

# 2. Instantiate the AccuracyWeighted Ensemble
ensemble = HeterogenousEnsembleForFeatureDrifts(verbose=1)

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=20000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ensemble)
