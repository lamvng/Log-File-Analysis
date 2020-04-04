import matplotlib.pyplot as plt
import seaborn as sns
from preprocess.load_file import load_file
'''
longitude = plt.subplot2grid((4, 2), (0, 0))
latitude = plt.subplot2grid((4, 2), (0, 1))
elevation = plt.subplot2grid((4, 2), (1, 0))
altitude = plt.subplot2grid((4, 2), (1, 1))
clutterheight = plt.subplot2grid((4, 2), (2, 0))
distance = plt.subplot2grid((4, 2), (2, 1))
loss = plt.subplot2grid((4, 2), (3, 0), colspan = 2)

longitude = sns.distplot(df['longitude'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = longitude)
sns.distplot(df['latitude'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = latitude)
sns.distplot(df['elevation'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = elevation)
sns.distplot(df['altitude'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = altitude)
sns.distplot(df['clutterheight'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = clutterheight)
sns.distplot(df['distance'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = distance)
sns.distplot(df['loss'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = loss)
plt.tight_layout()
plt.show()
#sns.pairplot(df[['distance','clutterheight', 'loss']])
'''


# Plotting a Bar Graph to compare the models
def important_features(top_columns, top_score):
    plt.bar(top_columns, top_score)
    plt.xlabel('Feature Labels')
    plt.ylabel('Feature Importances')
    plt.title('Comparison of the most important features')
    plt.show()

