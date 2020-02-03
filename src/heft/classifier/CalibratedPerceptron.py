import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron


class CalibratedPerceptron(BaseSKMObject, ClassifierMixin):
    """ Calibrated Perceptron classifier

    """

    def __init__(self, nominal_attributes=None):
        super().__init__()
        self.perceptron = Perceptron()
        self.cc = None

    def fit(self, X, y, sample_weight=None):
        self.perceptron.fit(X, y)
        if self.cc is None:
            self.cc = CalibratedClassifierCV(self.perceptron, cv='prefit', method='isotonic')
        self.cc.fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.perceptron.partial_fit(X, y, classes, sample_weight)
        self.cc.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.perceptron.predict(X)

    def predict_proba(self, X):
        return self.cc.predict_proba(X)
