from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np


def pick_important_features(quantity, columns, feature_importance_normalized):
    # feature_importance_normalized: numpy.ndarray
    # candidates = [[a, b] for a, b in zip(columns, feature_importance_normalized)]
    # Return indexes of max-valued importance
    top_index = np.argpartition(feature_importance_normalized, -quantity)[-quantity:]
    top_columns = []
    top_score = feature_importance_normalized[top_index]
    for i in top_index:
        top_columns.append(columns[i])
    return top_columns, top_score


# https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/
def random_forest(X_train, y_train):
    extra_tree_forest = RandomForestRegressor(n_estimators=100)
    extra_tree_forest.fit(X_train, y_train)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree
                                            in extra_tree_forest.estimators_],
                                           axis=0)  # numpy.ndarray
    top_columns, top_score = pick_important_features(18, X_train.columns.tolist(), feature_importance_normalized)
    return top_columns, top_score




# df.drop(df.columns.difference(['a', 'b']), 1, inplace=True)