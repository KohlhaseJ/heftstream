# heftstream

install scikit-multiflow from the folder with 
* pip install .
* or
* pip install scikit-multiflow

## Schedule

1. Implement HEFT stream ([Paper](https://link.springer.com/chapter/10.1007/978-3-642-30220-6_1))
    1. Possible base class: `accuracy_weighted_ensemble.py` (scikit-multiflow)
    1. FCBF feature selection ([Paper](https://bioconductor.org/packages/devel/bioc/vignettes/FCBF/inst/doc/FCBF-Vignette.html))
    1. Classifiers: Online Naive Bayes, CVFDT
2. Implement two feature selection algorithms and compare them w.r.t. accuracy
3. Implement two classifiers and compare them w.r.t accuracy
    1. Candidate: Hoeffding adaptive tree
    2. Candidate: Perceptron
    3. Candidate: SVM
