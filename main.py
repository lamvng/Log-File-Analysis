import settings
from preprocess import load_file, process_data, feature_extract
from models import models


settings.init()

# Load training dataset
df_train = load_file.load_file()


# Encode categorical features into numeric
# Categorical features: protocol_type, service, flag
df_train = process_data.encode(df_train, label_encoded_features=['protocol_type', 'service', 'flag'])


# Split training and validation data
X_train, X_validate, y_train, y_validate = process_data.normalize_and_split(df_train)
all_features = X_train.columns.tolist()


# Extract most important features
# top_columns, top_score = feature_extract.random_forest(X_train, y_train, number_of_features=20)
top_columns, top_score = feature_extract.extra_tree(X_train, y_train, number_of_features=20)


# Create a dataset after feature extraction
df_train = feature_extract.create_dataset(df_train, top_columns)  # Training set
# X_train, X_validate, y_train, y_validate = process_data.normalize_and_split(df_train)

X_train, y_train = process_data.create_input(df_train)
# Testing set...




models.build_LSTM(X_train, y_train)

# Build model



