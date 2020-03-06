# Import dataset, return a panda dataframe

import pandas as pd


def load_dataframe(filename):
    # filename = kddcup.data or = kddcup.data_10_percent
    df = pd.read_csv('../data/{}'.format(filename))
    df = df.dropna(inplace=False)  # Drop missing value
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    return df


