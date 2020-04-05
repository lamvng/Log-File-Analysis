import settings
from preprocess import load_file, process_data, feature_extract
from models import models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
# TODO: The dataframe (Pandas) at this step is for training and validation
df_train = feature_extract.create_dataset(df_train, top_columns)  # Training set

scaler = MinMaxScaler()
df_train[df_train.columns[:-1]] = scaler.fit_transform(df_train[df_train.columns[:-1]])

# TODO: These X, y are direct input to LSTM model
X_train, y_train = process_data.create_input(df_train)  # X as a 3D vector


# TODO: Testing set...
# TODO: A wrapper to:
### Training + Validation dataframe
### Transform input for LSTM model
### Split validation and training



models.build_LSTM(X_train, y_train)

# Build model



