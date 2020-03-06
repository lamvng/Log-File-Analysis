# Import dataset, return a panda dataframe

import pandas as pd

def input_data():
    print("Which dataset would you like to test?\n")
    print("1. KDD Cup --------------------------")
    print("2. KDD Cup 10 percent ---------------")

    user_input = input()
    accepted_list = ["1", "2"]
    if user_input not in accepted_list:
        print("Not accepted value\n")
        input_data()

    if user_input == "1":
        filename = "kddcup.data"
    elif user_input == "2":
        filename = "kddcup.data_10_percent"
    else:
        print("Error in input function...")
    return filename

def load_dataframe(filename):
    # filename = kddcup.data or = kddcup.data_10_percent
    filename = input_data()
    df = pd.read_csv('../data/{}'.format(filename))
    df = df.dropna(inplace=False)  # Drop missing value
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    return df


