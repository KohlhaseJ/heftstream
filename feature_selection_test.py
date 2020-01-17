import csv
import numpy as np
from src.heft.feature_selection.fcbf import FCBF
from src.heft.feature_selection.cife import CIFE

X, y = [],[]

with open("data/weather.csv") as f:
    readCSV = csv.reader(f, delimiter=',')
    for (i, line) in enumerate(readCSV):
        X.append(line[:-1])
        y.append(line[len(line) - 1])
        if i == 9999:
            break

X = np.array(X)
y = np.array(y)

F1 = FCBF().select(X, y)
F2 = CIFE().select(X, y)

print("FCBF: Selected {0} feature(s) out of {1}: {2}".format(len(F1), len(X[0]), F1))
print("LCSI: Selected {0} feature(s) out of {1}: {2}".format(len(F2), len(X[0]), F2))