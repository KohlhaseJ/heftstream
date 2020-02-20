from skmultiflow.data import FileStream, DataStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.trees.hoeffding_anytime_tree import HATT
from skmultiflow.meta import AccuracyWeightedEnsemble
from src.heft.meta.heterogeneous_ensemble_for_featuredrifts import HeterogeneousEnsembleForFeatureDrifts
from src.heft.feature_selection.fcbf import FCBF
from src.heft.feature_selection.cife import CIFE
from skmultiflow.evaluation import EvaluatePrequential
import sys
from io import StringIO


# 1. Create a data stream
#stream = FileStream("data/SEA.csv")
#stream = FileStream("data/HYP.csv")
stream = FileStream("data/LED.csv")
#stream = FileStream("data/KDD_PRE.csv")
#stream = FileStream("data/MNIST.csv")
stream.prepare_for_use()

# 2. Instantiate the Ensemble
window_size = 200
n_kept_estimators = 10
ensembles = {}
ensembles['heft'] = HeterogeneousEnsembleForFeatureDrifts(window_size=window_size, n_kept_estimators=n_kept_estimators,
                                                min_features=5, base_estimators=[NaiveBayes(), HATT()], feature_selector=FCBF,
                                                random_ensemble=0)
ensembles['awe'] = AccuracyWeightedEnsemble(window_size=window_size, n_kept_estimators=n_kept_estimators, base_estimator=NaiveBayes())

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=False,
                                pretrain_size=window_size,
                                metrics=['accuracy','running_time'],
                                max_samples=1000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=list(ensembles.values()), model_names=list(ensembles.keys()))

with open('results.txt', 'w') as f:
    stdout = sys.stdout
    s = StringIO()
    sys.stdout = s
    evaluator.evaluation_summary()
    s.seek(0)
    f.write('----------\n')
    f.write(str(s.read()))
    f.write(str(ensembles['heft'].print_statistics()))
    f.write('----------')
