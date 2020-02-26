from skmultiflow.data import FileStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.trees.hoeffding_anytime_tree import HATT
from skmultiflow.meta import AccuracyWeightedEnsemble
from src.heft.meta.heterogeneous_ensemble_for_featuredrifts import HeterogeneousEnsembleForFeatureDrifts
from src.heft.feature_selection.fcbf import FCBF
from skmultiflow.evaluation import EvaluatePrequential
from itertools import combinations
from functools import partial
from sklearn.neural_network import MLPClassifier
import sys
from io import StringIO


open('results.txt', 'w').close()

# 1. Create a list of filepaths for all datasets
files = ['data/KDD_PRE.csv', 'data/weather.csv', 'data/LED.csv', 'data/SEA.csv', 'data/HYP.csv']
MAX_SAMPLES = 1000
BASE_ESTIMATORS = [NaiveBayes, HATT, HAT, partial(MLPClassifier, learning_rate_init=1e-2, hidden_layer_sizes=1, max_iter=2000)]
HEFT_BASE_ESTIMATORS_LIST = []
for i in range(1, len(BASE_ESTIMATORS) + 1):
    HEFT_BASE_ESTIMATORS_LIST.extend(list(combinations(BASE_ESTIMATORS, i)))
print(len(HEFT_BASE_ESTIMATORS_LIST))
print(HEFT_BASE_ESTIMATORS_LIST)
FEATURE_SELECTOR_LIST = [FCBF, ]

for file in files:
    for FEATURE_SELECTOR in FEATURE_SELECTOR_LIST:
        for HEFT_BASE_ESTIMATORS in HEFT_BASE_ESTIMATORS_LIST:
            # 2. Create a data stream from the current file
            stream = FileStream(file)
            stream.prepare_for_use()

            # 3. Instantiate the Ensemble
            window_size = 200
            n_kept_estimators = 10
            ensembles = {
                'heft': HeterogeneousEnsembleForFeatureDrifts(window_size=window_size, n_kept_estimators=n_kept_estimators,
                                                              min_features=5, base_estimators=[c() for c in HEFT_BASE_ESTIMATORS],
                                                              feature_selector=FEATURE_SELECTOR,
                                                              random_ensemble=0),
                'awe': AccuracyWeightedEnsemble(window_size=window_size, n_kept_estimators=n_kept_estimators,
                                                base_estimator=NaiveBayes())}

            # 4. Setup the evaluator
            evaluator = EvaluatePrequential(show_plot=False,
                                            pretrain_size=window_size,
                                            metrics=['accuracy','running_time'],
                                            max_samples=MAX_SAMPLES)

            # 5. Run evaluation
            evaluator.evaluate(stream=stream, model=list(ensembles.values()), model_names=list(ensembles.keys()))

            # 6. Print the results of the evalutation to a resutl file
            with open('results.txt', 'a') as f:
                stdout = sys.stdout
                s = StringIO()
                sys.stdout = s
                evaluator.evaluation_summary()
                s.seek(0)
                f.write('----------\n')
                f.write(file + str(HEFT_BASE_ESTIMATORS) + '\n')
                f.write(str(s.read()))
                f.write(str(ensembles['heft'].print_statistics()))
                f.write('----------\n')
                sys.stdout = stdout
