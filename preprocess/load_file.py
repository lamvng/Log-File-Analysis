# Import dataset, return a panda dataframe

import pandas as pd
import settings


def input_data():
    filename = "./data/nsl_kdd/NSL_KDD-master/KDDTrain+_20Percent.txt"
    return filename


def load_dataframe():
    # filename = kddcup.data or = kddcup.data_10_percent
    # fidf = lename = input_data()
    filename = "KDDTrain+_20Percent.txt"
    df = pd.read_csv('{}/data/nsl_kdd/NSL_KDD-master/KDDTrain+_20Percent.txt'.format(settings.root, filename),
                     names=["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",\
                            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",\
                            "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",\
                            "num_access_files","num_outbound_cmds", "is_host_login", "is_guest_login", "count",\
                            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate"])
    df = df.dropna(inplace=False)  # Drop missing value
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    return df