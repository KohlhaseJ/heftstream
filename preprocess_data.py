import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


def nominal_feature_to_numerical(df, column):
    print(f'Unique values, column {column}: {np.unique(df[column])}')
    df_copy = df.copy()

    le = LabelEncoder()
    labels = le.fit_transform(df_copy[column])
    mappings = {index: label for index, label in enumerate(le.classes_)}

    print(f'Mappings: {mappings}')
    df_copy[column] = labels
    return df_copy


def nominal_features_to_one_hot(df, cols):
    ohe = ce.OneHotEncoder(use_cat_names=True, cols=cols)
    df_one_hot = ohe.fit_transform(df)
    print(df_one_hot.info())
    return df_one_hot


if __name__ == '__main__':
    df = pd.read_csv('data/unpacked/kdd.csv', header=None)
    # print(df.head())

    nominal_feature_cols = [1, 2, 3]
    class_col = 41

    df_numerical_class = nominal_feature_to_numerical(df, class_col)
    df_one_hot = nominal_features_to_one_hot(df_numerical_class, nominal_feature_cols)

    print(df_one_hot.head())

    df_one_hot.to_csv('data/unpacked/kdd_1_hot.csv', index=False)


