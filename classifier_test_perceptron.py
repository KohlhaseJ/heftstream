from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from src.heft.feature_selection.fcbf import FCBF
import numpy as np
import csv

X, Y = [],[]
X_test, Y_test = [], []

with open("data/lungcancer.csv") as f:
    readCSV = csv.reader(f, delimiter=',')
    for (i, line) in enumerate(readCSV):
        if i >= 20:
            X_test.append(line[:-1])
            Y_test.append(line[len(line) - 1])
        else:
            X.append(line[:-1])
            Y.append(line[len(line) - 1])

X = np.array(X).astype(np.float64)
Y = np.array(Y).astype(np.float64)

X_test = np.array(X_test).astype(np.float64)
Y_test = np.array(Y_test).astype(np.float64)

print(f"{len(X)}, {len(Y)}, {len(X_test)}, {len(Y_test)}")

z = FCBF(X, Y, **{"delta": 0})
print(z)

X = X[:, z[0]]
X_test = X_test[:, z[0]]

perceptron = Perceptron()
perceptron = CalibratedClassifierCV(perceptron, cv=2, method='isotonic')
perceptron = perceptron.fit(X, Y)

print(f"Accuracy: {perceptron.score(X_test, Y_test)}")