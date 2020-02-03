import numpy as np

def pearson(X, y, num_feats):
    correlations = []
    # calculate the correlation with y for each feature
    for i in range(len(X[0])):
        cor = np.corrcoef(X[:,i], y)[0, 1]
        correlations.append(cor)
    # replace NaN with 0
    correlations = [0 if np.isnan(i) else i for i in correlations]
    # feature name
    features = np.argsort(np.abs(correlations))[-num_feats:]
    # feature selection? 0 for not select, 1 for select
    support = [True if i in features else False for i in range(len(X[0]))]
    return features, support
