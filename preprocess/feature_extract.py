from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extra_tree(X_train, y_train):
    extra_tree_forest = RandomForestRegressor(n_estimators=100)
    extra_tree_forest.fit(X_train, y_train)
    feature_importance = extra_tree_forest.feature_importances_
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis=0)

    for feature in zip(X_train.columns.tolist(), extra_tree_forest.feature_importances_):
        print(feature)