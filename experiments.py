from skmultiflow.data import FileStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.trees.hoeffding_anytime_tree import HATT
from skmultiflow.meta import AccuracyWeightedEnsemble
from skmultiflow.meta import AdditiveExpertEnsemble
from src.heft.meta.heterogeneous_ensemble_for_featuredrifts import HeterogeneousEnsembleForFeatureDrifts
from src.heft.feature_selection.fcbf import FCBF
from src.heft.feature_selection.cmim import CMIM
from skmultiflow.evaluation import EvaluatePrequential
from itertools import combinations
from functools import partial
from sklearn.neural_network import MLPClassifier
import sys
from io import StringIO
import pandas as pd
import numpy as np
from evaluator_regex import get_proc_samples_accuracy_time

result_array = []

# 1. Create a list of filepaths for all datasets
# files = ['data/weather.csv', 'data/LED.csv', 'data/SEA.csv', 'data/HYP.csv', 'data/KDD_PRE.csv']
files = ['data/LED.csv', 'data/SEA.csv', 'data/HYP.csv']
MAX_SAMPLES = 25000
BASE_ESTIMATORS = [NaiveBayes, HATT,
                   partial(MLPClassifier, learning_rate_init=1e-2, hidden_layer_sizes=1, max_iter=2000)]
HEFT_BASE_ESTIMATORS_LIST = []
for i in range(2, len(BASE_ESTIMATORS) + 1):
    HEFT_BASE_ESTIMATORS_LIST.extend(list(combinations(BASE_ESTIMATORS, i)))
FEATURE_SELECTOR_LIST = [FCBF, CMIM]
window_size = 200
n_kept_estimators = 10

for file in files:
    dataset_name = file[5:-4] + '_result.csv'
    print(dataset_name)
    dataset_result = []
    stream = FileStream(file)
    stream.prepare_for_use()

    awe = AccuracyWeightedEnsemble(window_size=window_size, n_kept_estimators=n_kept_estimators,
                                   base_estimator=NaiveBayes())

    evaluator = EvaluatePrequential(show_plot=False,
                                    pretrain_size=window_size,
                                    metrics=['accuracy', 'running_time'],
                                    max_samples=MAX_SAMPLES)

    evaluator.evaluate(stream=stream, model=awe, model_names=['awe'])

    a = [file, 'awe', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stdout = sys.stdout
    s = StringIO()
    sys.stdout = s
    evaluator.evaluation_summary()
    s.seek(0)
    t = get_proc_samples_accuracy_time(s.read())
    sys.stdout = stdout
    a[11] = t[0]
    a[12] = t[1]
    a[13] = t[2]
    dataset_result.append(a)

    stream = FileStream(file)
    stream.prepare_for_use()

    aee = AdditiveExpertEnsemble(n_estimators=n_kept_estimators, base_estimator=NaiveBayes())

    evaluator = EvaluatePrequential(show_plot=False,
                                    pretrain_size=window_size,
                                    metrics=['accuracy', 'running_time'],
                                    max_samples=MAX_SAMPLES)

    evaluator.evaluate(stream=stream, model=aee, model_names=['aee'])

    a = [file, 'aee', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stdout = sys.stdout
    s = StringIO()
    sys.stdout = s
    evaluator.evaluation_summary()
    s.seek(0)
    t = get_proc_samples_accuracy_time(s.read())
    sys.stdout = stdout
    a[11] = t[0]
    a[12] = t[1]
    a[13] = t[2]
    dataset_result.append(a)

    for FEATURE_SELECTOR in FEATURE_SELECTOR_LIST:
        for rand in (0, 1):
            for HEFT_BASE_ESTIMATORS in HEFT_BASE_ESTIMATORS_LIST:
                # 2. Create a data stream from the current file
                stream = FileStream(file)
                stream.prepare_for_use()

                # 3. Instantiate the Ensemble
                heft = HeterogeneousEnsembleForFeatureDrifts(window_size=window_size,
                                                             n_kept_estimators=n_kept_estimators,
                                                             min_features=5,
                                                             base_estimators=[c() for c in HEFT_BASE_ESTIMATORS],
                                                             feature_selector=FEATURE_SELECTOR,
                                                             random_ensemble=rand)

                # 4. Setup the evaluator
                evaluator = EvaluatePrequential(show_plot=False,
                                                pretrain_size=window_size,
                                                metrics=['accuracy', 'running_time'],
                                                max_samples=MAX_SAMPLES)

                # 5. Run evaluation
                evaluator.evaluate(stream=stream, model=heft, model_names=['heft'])

                a = [file, 'heft', 0, 0, 0, FEATURE_SELECTOR.__name__, np.mean(heft.number_of_selected_features), 0,
                     0, 0, 0, 0, 0, 0]
                if NaiveBayes in HEFT_BASE_ESTIMATORS:
                    a[2] = 1
                if HATT in HEFT_BASE_ESTIMATORS:
                    a[3] = 1
                if MLPClassifier in HEFT_BASE_ESTIMATORS:
                    a[4] = 1
                model_types = heft.get_models()
                if NaiveBayes in model_types.keys():
                    a[7] = model_types[NaiveBayes]
                if HATT in model_types.keys():
                    a[8] = model_types[HATT]
                if MLPClassifier in model_types.keys():
                    a[9] = model_types[MLPClassifier]
                a[10] = heft.random_ensemble
                stdout = sys.stdout
                s = StringIO()
                sys.stdout = s
                evaluator.evaluation_summary()
                s.seek(0)
                t = get_proc_samples_accuracy_time(s.read())
                sys.stdout = stdout
                a[11] = t[0]
                a[12] = t[1]
                a[13] = t[2]
                dataset_result.append(a)
    df = pd.DataFrame(np.array(dataset_result),
                      columns=['dataset', 'ensemble', 'base_naive', 'base_hatt', 'base_MLP',
                               'feature_selector', 'avg_selected_features', 'final_naive', 'final_hatt',
                               'final_MLP', 'random_ensemble',
                               'total_samples', 'accuracy', 'running_time'])
    df.to_csv(dataset_name)
    result_array.extend(dataset_result)
