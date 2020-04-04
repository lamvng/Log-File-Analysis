import settings
from preprocess import load_file, process_data, feature_extract
settings.init()


df = load_file.load_dataframe()

# Categorical features: protocol_type, service, flag
df = process_data.encode(df, label_encoded_features=['protocol_type', 'service', 'flag'])

X_train, X_test, y_train, y_test = process_data.normalize_and_split(df)


feature_extract.extra_tree(X_train, y_train)