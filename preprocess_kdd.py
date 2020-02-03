import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer, OneHotEncoder

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
    df = pd.read_csv('data/unpacked/kdd.csv', header=None, names=names)
    print(df.head())

    # transform categorical features
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    df['label'] = df['label'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    print(df.head())

    df.to_csv('data/unpacked/KDD_PRE.csv', index=False, header=False)
