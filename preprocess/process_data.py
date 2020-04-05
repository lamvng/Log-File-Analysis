import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


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
    # Normalize all columns except the last one
    scaler = MinMaxScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])


    y = df['attack_type']  # Label # Series
    X = df.drop(['attack_type'], axis='columns')  # Features # Dataframe


    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2)
    return X_train, X_validate, y_train, y_validate


# Create input data shape for model training
# https://stackoverflow.com/questions/39674713/neural-network-lstm-input-shape-from-dataframe
def create_input(df):
    input_cols = df.columns[:-1]
    output_cols = df.columns[-1]

    # Put your inputs into a single list
    df['single_input_vector'] = df[input_cols].apply(tuple, axis=1).apply(list)
    # Double-encapsulate list so that you can sum it in the next step and keep time steps as separate elements
    df['single_input_vector'] = df.single_input_vector.apply(lambda x: [list(x)])
    # Use .cumsum() to include previous row vectors in the current row list of vectors
    df['cumulative_input_vectors'] = df.single_input_vector.cumsum()

    # If your output is multi-dimensional, you need to capture those dimensions in one object
    # If your output is a single dimension, this step may be unnecessary
    # df['output_vector'] = df[output_cols].apply(tuple, axis=1).apply(list)
    df['output_vector'] = df[output_cols]

    # Pad your sequences so they are the same length
    max_sequence_length = df.cumulative_input_vectors.apply(len).max()
    # Save it as a list
    padded_sequences = pad_sequences(df.cumulative_input_vectors.tolist(), max_sequence_length).tolist()
    df['padded_input_vectors'] = pd.Series(padded_sequences).apply(np.asarray)

    # Extract your training data
    X_train_init = np.asarray(df.padded_input_vectors)
    # Use hstack to and reshape to make the inputs a 3d vector
    X_train = np.hstack(X_train_init).reshape(len(df), max_sequence_length, len(input_cols))
    # y_train = np.hstack(np.asarray(df.output_vector)).reshape(len(df), len(output_cols))
    y_train = np.hstack(np.asarray(df.output_vector))
    return X_train, y_train
