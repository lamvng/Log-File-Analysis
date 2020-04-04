import settings
from preprocess import load_file, process_data
settings.init()


df = load_file.load_dataframe()

df = process_data.encode(df)

(X_train, Y_train), (X_test, Y_test) = process_data.normalize_and_split(df)

