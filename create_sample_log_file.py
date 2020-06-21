from preprocess import load_file
import settings


settings.init()
sample_path = "{}/data/nsl_kdd/NSL_KDD-master".format(settings.root)

df_test = load_file.load_file('KDDTest+.csv')


df1 = df_test.sample(n=100, random_state=1, axis='rows')
df1.to_csv("{}/sample_log_1.csv".format(sample_path), index=False, header=False)


df2 = df_test.sample(n=200, random_state=2, axis='rows')
df2.to_csv("{}/sample_log_2.csv".format(sample_path), index=False, header=False)

df3 = df_test.sample(n=1000, random_state=3, axis='rows')
df3.to_csv("{}/sample_log_3.csv".format(sample_path), index=False, header=False)

