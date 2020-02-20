from skmultiflow.data import FileStream, DataStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.trees.hoeffding_anytime_tree import HATT
from skmultiflow.meta import AccuracyWeightedEnsemble
from src.heft.meta.heterogeneous_ensemble_for_featuredrifts import HeterogeneousEnsembleForFeatureDrifts
from src.heft.feature_selection.fcbf import FCBF
from src.heft.feature_selection.cife import CIFE
from skmultiflow.evaluation import EvaluatePrequential
from src.heft.classifier.CalibratedPerceptron import CalibratedPerceptron
import sys
from io import StringIO


open('results.txt', 'w').close()

# 1. Create a list of filepaths for all datasets
files = ['data/LED.csv', 'data/SEA.csv']
MAX_SAMPLES = 1000
# [NaiveBayes(), CalibratedPerceptron()], [HATT(), CalibratedPerceptron()], [HAT(), CalibratedPerceptron()]
HEFT_BASE_ESTIMATORS_LIST = [[NaiveBayes(), HATT()], [NaiveBayes(), HAT()], [HATT(), HAT()]]
# CIFE
FEATURE_SELECTOR_LIST = [FCBF]
# stream = FileStream("data/HYP.csv")
# stream = FileStream("data/KDD_PRE.csv")
# stream = FileStream("data/MNIST.csv")

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
                                                              min_features=5, base_estimators=HEFT_BASE_ESTIMATORS,
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
                f.write(file + '\n')
                f.write(str(s.read()))
                f.write(str(ensembles['heft'].print_statistics()))
                f.write('----------\n')
                sys.stdout = stdout
