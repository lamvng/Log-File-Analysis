import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# One-hot encoding categorical features
# Column: string
def one_hot_encode(df, feature):
    loc_last_column = len(df.columns)
    loc_insert = df.columns.get_loc(feature)  # Get original index of the first categorical columns in df
    # drop_first: Make one of the feature goes [0,0,0,...0]
    added_cols = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
    df = pd.concat([df, added_cols], axis=1, sort=False)

    # Move the added one-hot columns to their original index
    moved_df = df[df.columns[loc_last_column:]] # Added one-hot columns
    cols_df = df.columns.tolist()
    cols_features = moved_df.columns.tolist()
    for (count, col_feature) in enumerate(cols_features):
        cols_df.insert(loc_insert + count, cols_df.pop(cols_df.index(col_feature)))
    df = df.reindex(columns=cols_df)

    # Delete categorical columns after being one-hot-encoded
    df.drop([feature], axis=1, inplace=True)
    return df

def label_encode(df, feature):
    df[feature] = df[feature].astype('category').cat.codes
    return df


# Function to classify label into 5 classes:
# Normal, DoS, Probe, U2R, R2L
def classify_label(row):
    normal = ['normal']
    dos = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
    probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    u2r = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    r2l = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']

    if row['label'].lower() in normal:
        return 1
    elif row['label'].lower() in dos:
        return 2
    elif row['label'].lower() in probe:
        return 3
    elif row['label'].lower() in u2r:
        return 4
    elif row['label'].lower() in r2l:
        return 5
    else:
        return 6


# Encode the datatype
def encode(df, one_hot_features=None, label_encoded_features=None):
    # One hot encoding for categorical features: protocol_type, service, flag
    if one_hot_features != None:
        for feature in one_hot_features:
            df = one_hot_encode(df, feature)

    if label_encoded_features != None:
        for feature in label_encoded_features:
            df = label_encode(df, feature)

    # Encode label
    df['attack_type'] = df.apply(lambda row: classify_label(row), axis='columns')

    # Drop unused label columns
    df = df.drop('score', axis='columns')
    df = df.drop('label', axis='columns')
    return df


def normalize_and_split(df):
    # Normalize using MinMaxScaler
    scaler = preprocessing.MinMaxScaler()

    y = df['attack_type'] # Label # Series
    X = df.drop(['attack_type'], axis='columns')
    # X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test