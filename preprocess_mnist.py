from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd


if __name__ == '__main__':
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    # normalize
    X = X / np.max(X)

    data = np.c_[X, y]
    print(data)
    df = pd.DataFrame(data)
    df.to_csv('data/MNIST_normalized.csv')
