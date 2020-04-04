import settings
from preprocess import load_file, process_data, feature_extract
from numpy import argsort


settings.init()


df = load_file.load_dataframe()

# Categorical features: protocol_type, service, flag
df = process_data.encode(df, label_encoded_features=['protocol_type', 'service', 'flag'])

X_train, X_test, y_train, y_test = process_data.normalize_and_split(df)
all_features = X_train.columns.tolist()

# top_columns, top_score = feature_extract.random_forest(X_train, y_train, number_of_features=20)
top_columns, top_score = feature_extract.extra_tree(X_train, y_train, number_of_features=20)

index_order = argsort(top_score)[::-1]  # Return descending top score index
with open('{}/results/most_important_features.txt'.format(settings.root), 'w') as f:
    for i in index_order:
        print(top_columns[i] + ":" + str(top_score[i]), file=f)




