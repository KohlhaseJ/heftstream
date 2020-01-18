import numpy as np
from .base import BaseSelector
from .helper import calculations as c

class FCBF(BaseSelector):
    @staticmethod
    def select(X, y):
        """
        This function implements Fast Correlation Based Filter algorithm

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be discrete
        y: {numpy array}, shape (n_samples,)
            input class labels

        Output
        ------
        F: {numpy array}, shape (n_features,)
            index of selected features, F[0] is the most important feature

        Reference
        ---------
            Yu, Lei and Liu, Huan. "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution." ICML 2003.
        """

        _, n_features = X.shape
        # the default value of delta is 0
        delta = 0

        # t1[:,0] stores index of features, t1[:,1] stores symmetrical uncertainty of features
        t1 = np.zeros((n_features, 2), dtype='object')
        for i in range(n_features):
            f = X[:, i]
            t1[i, 0] = i
            t1[i, 1] = c.su_calculation(f, y)
        s_list = t1[t1[:, 1] > delta, :]
        # index of selected features, initialized to be empty
        F = []
        # Symmetrical uncertainty of selected features
        SU = []
        while len(s_list) != 0:
            # select the largest su inside s_list
            idx = np.argmax(s_list[:, 1])
            # record the index of the feature with the largest su
            fp = X[:, s_list[idx, 0]]
            np.delete(s_list, idx, 0)
            F.append(s_list[idx, 0])
            SU.append(s_list[idx, 1])
            for i in s_list[:, 0]:
                fi = X[:, i]
                if c.su_calculation(fp, fi) >= t1[i, 1]:
                    # construct the mask for feature whose su is larger than su(fp,y)
                    idx = s_list[:, 0] != i
                    idx = np.array([idx, idx])
                    idx = np.transpose(idx)
                    # delete the feature by using the mask
                    s_list = s_list[idx]
                    length = len(s_list)//2
                    s_list = s_list.reshape((length, 2))
        return np.array(F, dtype=int)