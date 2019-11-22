import csv
from src.heft.feature_selection.fcbf import fcbf

X, Y = [], []

with open("oil.csv") as f:
    readCSV = csv.reader(f, delimiter=',')
    for (i, line) in enumerate(readCSV):
        X.append(line[:-1])
        Y.append(line[len(line) - 1])
        if i > 500:
            break

print(len(X))
print(len(Y))

z = fcbf(X, Y, 0.05)
print(z.shape)
