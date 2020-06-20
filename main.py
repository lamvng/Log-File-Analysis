import settings
from preprocess import load_file, graph_data, process_data, feature_extract
from models import models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


settings.init()


# Plot activation functions
graph_data.plot_rectified()
graph_data.plot_sigmoid()
graph_data.plot_softmax()


# Load training dataset
# Available dataset: Small_Training_Set.csv, KDDTrain+.csv, KDDTest+.csv
df_train = load_file.load_file('KDDTrain+.csv')
df_test = load_file.load_file('KDDTest+.csv')

# Dataset reporting
print("Training set shape: " + str(df_train.shape))
print("Testing set shape: " + str(df_test.shape))
print("Number of features: " + str(len(df_train.columns) - 2))
print("Number of labels: 2")

print ("\nDatatype:")
print(df_train.dtypes)

# Visualize the dataset before encoding
print("\nTraining dataset before encoding:")
label_train = df_train['label'].value_counts()
print(label_train)
graph_data.plot_train_before_encode(label_train)

print("\nTesting dataset before encoding:")
label_test = df_test['label'].value_counts()
print(label_test)
graph_data.plot_test_before_encode(label_test)


# Encode categorical features into numeric
# Categorical features: protocol_type, service, flag
df_train = process_data.encode(df_train, label_encoded_features=['protocol_type', 'service', 'flag'])
df_test = process_data.encode(df_test, label_encoded_features=['protocol_type', 'service', 'flag'])


# Visualize the dataset after encoding
print("\nTraining dataset after encoding:")
attack_type_train = df_train['attack_type'].value_counts()
print(attack_type_train)
graph_data.plot_train_after_encode(attack_type_train)

print("\nTesting dataset after encoding:")
attack_type_test = df_test['attack_type'].value_counts()
print(attack_type_test)
graph_data.plot_test_after_encode(attack_type_test)



# Split training and validation data
print('\nNormalizing and spliting training data for feature extracion...')
# Split training and validation data
df_train = process_data.normalize(df_train)
y = df_train['attack_type']  # Label # Series
X = df_train.drop(['attack_type'], axis='columns')  # Features # Dataframe
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2)
all_features = X_train.columns.tolist()

print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_validate shape: " + str(X_validate.shape))
print("y_validate shape: " + str(y_validate.shape))


# Extract most important features
# top_columns, top_score = feature_extract.random_forest(X_train, y_train, number_of_features=20)
top_columns, top_score = feature_extract.extract_extra_tree(X_train, y_train, number_of_features=20)


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



