import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer, OneHotEncoder
from src.heft.feature_selection.fcbf import FCBF

if __name__ == '__main__':
    names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    df = pd.read_csv('data/unpacked/kdd.csv', header=None, names=names, nrows=1000000)
    print(df.head())
    print(df.describe())

    # remove redundant features
    print(df['num_outbound_cmds'].value_counts())
    """0    1000000"""

    print(df['is_host_login'].value_counts())
    """0    999999
       1    1"""

    df.drop('num_outbound_cmds', axis=1, inplace=True)
    df.drop('is_host_login', axis=1, inplace=True)

    # transform categorical features
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    df['label'] = df['label'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # remove dupes
    df.drop_duplicates(subset=None, keep='first', inplace=True)

    print(df.shape)

    print(df['label'].value_counts())

    # log-scaled distribution of attacks

    # plt.clf()
    # plt.figure(figsize=(12, 8))
    # params = {'axes.titlesize': '18',
    #           'xtick.labelsize': '14',
    #           'ytick.labelsize': '14'}
    # matplotlib.rcParams.update(params)
    # plt.title('Distribution of attacks')
    # # df.plot(kind='barh')
    df['label'].value_counts().apply(np.log).plot(kind='barh')
    #
    # plt.show()

    # standardization
    data = df.values
    X, Y = data[:, 0:39], data[:, 39]

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    # normalization
    normalizer = Normalizer()
    normalized_X = normalizer.fit_transform(scaled_X)

    # df_normalized = pd.DataFrame(data=normalized_X, columns=names_new)

    # encoding
    ohe = OneHotEncoder(handle_unknown='error', n_values='auto', sparse=False, categories='auto')

    Y_enc = Y.copy().reshape(-1, 1)
    Y_enc = ohe.fit_transform(Y_enc)

    print(normalized_X.shape)
    print(Y_enc.shape)

    data_prep = np.hstack((normalized_X, Y_enc))

    print(data_prep.shape)

    names_new = [col for col in names if col not in ['num_outbound_cmds', 'is_host_login', 'label']] + [f'label_{i}' for
                                                                                                        i in range(20)]
    df_prep = pd.DataFrame(data=data_prep, columns=names_new)

    print(df_prep.head())
    print(df_prep.describe())

    ###################################
    # test feature selection without OHE
    ###################################

    print(f'normalized_X: {normalized_X}')
    print(f'Y: {Y}')
    print(f'#features: {len(normalized_X[0])}')

    # fcbf
    # print(normalized_X[:5])
    # z = FCBF(normalized_X[:3000], Y[:3000], **{"delta": 0})
    # print("Selected {0} feature(s) out of {1}: {2}".format(len(z[0]), len(X[0]), z[0]))

    # univariate
    selectkbest = SelectKBest(score_func=f_regression, k=4)
    sfit = selectkbest.fit(normalized_X, Y)
    X_new = selectkbest.transform(normalized_X)
    print(X_new.shape)

    # df_bdc.to_csv('data/unpacked/kdd_prep.csv', index=False)
