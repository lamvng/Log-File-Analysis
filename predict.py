import argparse
from tensorflow.keras.models import load_model
from preprocess import load_file, process_data, feature_extract
import settings
from numpy import unique
import json
from datetime import datetime
from termcolor import colored


settings.init()


def predict(df_test):
    print(colored('[+] Loading pre-weighted model...\n', 'green'))
    path = "{}/saved_model/model.h5".format(settings.root)
    loaded_model = load_model(path)

    # Load top columns
    top_columns = []
    with open("{}/results/most_important_features.txt".format(settings.root), "r") as f:
        for line in f:
            column = line.split(':')[0]
            top_columns.append(column)

    # Load and process testing dataset
    df = process_data.encode(df_test, label_encoded_features=['protocol_type', 'service', 'flag'])
    df = process_data.normalize(df)  # Normalize the test dataset
    df = feature_extract.create_dataset(df, top_columns)  # Feature Extraction
    y_test = df['attack_type']  # Label # Series
    X_test = df.drop(['attack_type'], axis='columns')  # Features # Dataframe

    # Predict
    y_predict = loaded_model.predict_classes(X_test, verbose=0)
    return y_predict


# Output
def set_output_verbose(df_test, y_predict):
    for index, pred in enumerate(y_predict):
        if pred != 0:
            if pred == 1:
                print(colored("\n[+] DoS warning:\nPacket {}: {}, service: {}, flag: {}".format(index, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service'], df_test.iloc[index]['flag']), 'red'))
            elif pred == 2:
                print(colored("\n[+] Probing-Scanning warning:\nPacket {}: {}, service: {}, flag: {}".format(index, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service'], df_test.iloc[index]['flag']), 'red'))
            elif pred == 3:
                print(colored("\n[+] User-to-Root Escalation warning:\nPacket {}: {}, service: {}, flag: {}".format(index, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service'], df_test.iloc[index]['flag']), 'red'))
            elif pred == 4:
                print(colored("\n[+] Remote-to-Local Breach warning:\nPacket {}: {}, service: {}, flag: {}".format(index, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service'], df_test.iloc[index]['flag']), 'red'))


def set_output(df_test, y_predict):
    y_unique = unique(y_predict)
    if 1 in y_unique:
        print(colored('\n[+] Warning: DoS packets detected!', 'red'))
    if 2 in y_unique:
        print(colored('\n[+] Warning: Probing-Scanning activities detected!', 'red'))
    if 3 in y_unique:
        print(colored('\n[+] Warning: User-to-Root Escalation detected!', 'red'))
    if 4 in y_unique:
        print(colored('\n[+] Warning: Remote-to-Local Breach detected!', 'red'))


# Output to file:
def output_to_json(df_test, y_predict):
    output = []
    for index, pred in enumerate(y_predict):
        if pred != 0:
            if pred == 1:
                attack_type = 'dos'
            elif pred == 2:
                attack_type = 'probe'
            elif pred == 3:
                attack_type = 'u2r'
            elif pred == 4:
                attack_type = 'r2l'
            dict = {
                "id": index,
                "attack_type": attack_type,
                "protocol_type": df_test.iloc[index]['protocol_type'],
                "service": df_test.iloc[index]['service'],
                "flag": df_test.iloc[index]['flag'],
                "src_bytes": int(df_test.iloc[index]['src_bytes']),
                "dst_bytes": int(df_test.iloc[index]['dst_bytes'])
            }
            output.append(dict)
    now = datetime.now()
    time = now.strftime("%Y-%m-%d_%H:%M:%S")
    dump = json.dumps(output)
    loaded_dump = json.loads(dump)
    with open ("{}/results/results_{}.json".format(settings.root, time), "w") as json_file:
        json_file.write(dump)
    print(colored("\n[+] Analysis output has been saved in JSON format.\n", "green"))


def output_to_csv(df_test, y_predict):
    now = datetime.now()
    time = now.strftime("%Y-%m-%d_%H:%M:%S")
    with open ("{}/results/results_{}.csv".format(settings.root, time), "w") as csv_file:
        for index,pred in enumerate(y_predict):
            if pred != 0:
                if pred == 1:
                    attack_type = 'dos'
                elif pred == 2:
                    attack_type = 'probe'
                elif pred == 3:
                    attack_type = 'u2r'
                elif pred == 4:
                    attack_type = 'r2l'
                csv_file.write("{},{},{},{},{},{},{}\n".format(index,
                                                             attack_type,
                                                             df_test.iloc[index]['protocol_type'],
                                                             df_test.iloc[index]['service'],
                                                             df_test.iloc[index]['flag'],
                                                             df_test.iloc[index]['src_bytes'],
                                                             df_test.iloc[index]['dst_bytes']))
    print(colored("\n[+] Analysis output has been saved in CSV format.\n", "green"))

# Parser
parser = argparse.ArgumentParser(description='Log file analyzer and classifier.')
parser.add_argument('-f', '--file',
                    help='Log file to analyze',
                    required=True,
                    choices=['sample_log_1.csv', 'sample_log_2.csv', 'sample_log_3.csv'])

parser.add_argument('-v', '--verbose',
                    help='Verbose mode',
                    action='store_true')

parser.add_argument('-o',
                    '--output',
                    help='Output type',
                    default='json',
                    choices=['csv', 'json'])


args = vars(parser.parse_args())


sample_log = load_file.load_file(args['file'])
output = args['output']
verbose = args['verbose']


y_predict = predict(sample_log)
sample_log_unencoded = load_file.load_file(args['file'])


# Deal with verbose output
if verbose:
    set_output_verbose(sample_log_unencoded, y_predict)
else:
    set_output(sample_log, y_predict)


# Output to json file:
if output == 'json':
    output_to_json(sample_log_unencoded, y_predict)
elif output == 'csv':
    output_to_csv(sample_log_unencoded, y_predict)

# conda activate mlds
# python3 predict.py --file sample_log_2.csv --output json --verbose