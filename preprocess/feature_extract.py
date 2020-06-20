from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import settings
import numpy as np


def pick_important_features(number_of_features, columns, feature_importance_normalized):
    # feature_importance_normalized: numpy.ndarray: Important factor from extra_tree.fit or random_forest.fit
    top_index = np.argpartition(feature_importance_normalized, -number_of_features)[-number_of_features:] # Return indexes of max-valued importance
    top_columns = []
    top_score = feature_importance_normalized[top_index]
    for i in top_index:
        top_columns.append(columns[i])
    '''
    index_order = np.argsort(top_score)[::-1]  # Return descending top score index
    with open('{}/results/most_important_features.txt'.format(settings.root), 'w') as f:
        for i in index_order:
            print(top_columns[i] + ":" + str(top_score[i]), file=f)
    '''
    return top_columns, top_score


def extract_random_forest(X_train, y_train, number_of_features=18):
    random_forest = RandomForestRegressor(n_estimators=100)
    random_forest.fit(X_train, y_train)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree
                                            in random_forest.estimators_],
                                           axis=0)  # numpy.ndarray
    top_columns, top_score = pick_important_features(number_of_features,
                                                     X_train.columns.tolist(),
                                                     feature_importance_normalized)
    return top_columns, top_score


# Reference: https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/
def extract_extra_tree(X_train, y_train, number_of_features=18):
    extra_tree = ExtraTreesRegressor(n_estimators=100)
    extra_tree.fit(X_train, y_train)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree
                                            in extra_tree.estimators_],
                                           axis=0)  # numpy.ndarray
    top_columns, top_score = pick_important_features(number_of_features,
                                                     X_train.columns.tolist(),
                                                     feature_importance_normalized)
    return top_columns, top_score

def create_dataset(df, top_columns):
    top_columns.append('attack_type')
    df = df.drop(df.columns.difference(top_columns), axis=1)
    return df