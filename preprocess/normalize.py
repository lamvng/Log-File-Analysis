import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# One-hot encoding for a column of a dataframe
# Column: string
def one_hot_encode(df, one_hot_features):
    loc_last_column = len(df.columns)
    loc_insert = df.columns.get_loc('protocol_type')  # Original index of the first categorical columns in df
    for col_name in one_hot_features:
        # drop_first: Make one of the feature goes [0,0,0,...0]
        feature = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)
        df = pd.concat([df, feature], axis=1, sort=False)

    # Move the added one-hot columns to their original index
    moved_df = df[df.columns[loc_last_column:]] # Added one-hot columns
    cols_df = df.columns.tolist()
    cols_features = moved_df.columns.tolist()
    for (count, col_feature) in enumerate(cols_features):
        cols_df.insert(loc_insert + count, cols_df.pop(cols_df.index(col_feature)))
    df = df.reindex(columns=cols_df)

    # Delete categorical columns after being one-hot-encoded
    for col_name in one_hot_features:
        df.drop([col_name], axis=1, inplace=True)
    return df

def normalize(df):
    # Training and Test
    # One hot encoding for protocol_type, service, flag
    one_hot_features = ['protocol_type', 'service', 'flag']
    df = one_hot_encode(df, one_hot_features)

    # Normalize using MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    Y = df.iloc[:, -2] # Label # Series
    X = df.drop(['label', 'score'], axis='columns') # Training # Dataframe
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return (X_train, y_train), (X_test, y_test)
