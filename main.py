import settings
from preprocess import load_file, normalize
settings.init()

df = load_file.load_dataframe()

one_hot_features = ['protocol_type', 'service', 'flag']
df = normalize.one_hot_encode(df, one_hot_features)