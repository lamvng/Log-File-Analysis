import settings
from preprocess import load_file, process_data, feature_extract
settings.init()


df = load_file.load_dataframe()

# Categorical features: protocol_type, service, flag
df = process_data.encode(df, label_encoded_features=['protocol_type', 'service', 'flag'])

X_train, X_test, y_train, y_test = process_data.normalize_and_split(df)
all_features = X_train.columns.tolist()

columns, feature_importance_normalized = feature_extract.random_forest(X_train, y_train)
top_columns, top_index = feature_extract.pick_important_features(18, columns, feature_importance_normalized)

stats = [[a, b] for a, b in zip(columns, feature_importance_normalized)]
top_stats = [[a, b] for a, b in zip(top_columns, top_index)]

