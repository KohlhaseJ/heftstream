import csv
import numpy as np
from src.heft.feature_selection.fcbf import FCBF

X, Y = [],[]

with open("data/lungcancer.csv") as f:
    readCSV = csv.reader(f, delimiter=',')
    for (i, line) in enumerate(readCSV):
        X.append(line[:-1])
        Y.append(line[len(line) - 1])
        if i == 199:
            break

X = np.array(X)
Y = np.array(Y)

z = FCBF(X, Y, **{"delta": 0})

print(np.array_equal([1,2],[1,2]))

print("Selected {0} feature(s) out of {1}: {2}".format(len(z[0]), len(X[0]), z[0]))