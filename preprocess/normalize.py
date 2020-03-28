import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# One-hot encoding for a column of a dataframe
# Column: string
def one_hot_encode(df, one_hot_columns):
    loc_last_column = len(df.columns) - 1  # Index of last column of df
    for col_name in one_hot_columns:
    # drop_first: Make one of the feature goes [0,0,0,...0]
        feature = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)
        df = pd.concat([df, feature], axis=1)

        # Move the one-hotted-column to its original position after concatenating
        cols_df = df.columns.tolist()
        cols_feature = feature.columns.tolist()
        loc_column = df.columns.get_loc(col_name)  # Original ndex of current column of df
        for added_col in cols_feature: # Iterate by column name
            cols_df.insert(loc_column, cols_df.pop(cols_df.index(added_col))) # Pop the current col, then add to the original position
        df = df.reindex(columns=cols_df)

    # Delete categorical columns after being one-hot-encoded
    for col_name in one_hot_columns:
        df.drop([col_name], axis=1, inplace=True)

    return df

def split_train_test(df):

def normalize(df):
    # Training and Test
    # One hot encoding for protocol_type, service, flag
    one_hot_columns = ['protocol_type', 'service', 'flag']
    df = one_hot_encode(df, one_hot_columns)
    scaler = preprocessing.MinMaxScaler()
    Y = df.iloc[:, -2] # Label # Series
    X = df.drop(['label', 'score'], axis='columns') # Training # Dataframe
    X = scaler.fit_transform(X)


