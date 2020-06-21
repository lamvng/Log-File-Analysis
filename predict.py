import argparse
from tensorflow.keras.models import load_model
from preprocess import load_file, process_data, feature_extract
import settings
from numpy import unique
import json

settings.init()


def predict(df_test):
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
                print("\nDoS warning on packet {}: {}, service: {}".format(index+1, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service']))
            elif pred == 2:
                print("\nProbing-Scanning warning on packet {}: {}, service: {}".format(index+1, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service']))
            elif pred == 3:
                print("\nUser-to-Root Escalation warning on packet {}: {}, service: {}".format(index+1, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service']))
            elif pred == 4:
                print("\nRemote-to-Local Breach warning on packet {}: {}, service: {}".format(index+1, df_test.iloc[index]['protocol_type'], df_test.iloc[index]['service']))


def set_output(df_test, y_predict):
    y_unique = unique(y_predict)
    if 1 in y_unique:
        print('\nWarning: DoS packets detected!')
    if 2 in y_unique:
        print('\nWarning: Probing-Scanning activities detected!')
    if 3 in y_unique:
        print('\nWarning: User-to-Root Escalation detected!')
    if 4 in y_unique:
        print('\nWarning: Remote-to-Local Breach detected!')


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
                "protocol_type": df_test.iloc[index]['protocol_type'],
                "service": df_test.iloc[index]['service'],
                "attack_type": attack_type

            }
            output.append(dict)
    return json.dumps(output)

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
                    choices=['txt', 'json'])


args = vars(parser.parse_args())


sample_log = load_file.load_file(args['file'])
output = args['output']
verbose = args['verbose']


y_predict = predict(sample_log)
sample_log_unencoded = load_file.load_file(args['file'])


# Deal with verbose output
if verbose == True:
    set_output_verbose(sample_log_unencoded, y_predict)
else:
    set_output(sample_log, y_predict)


# Output to json file:
if output == 'json':
    file = output_to_json(sample_log_unencoded, y_predict)
    print(file)