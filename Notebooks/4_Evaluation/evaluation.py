import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def show_grid_search_results(gcv):
    """ Print grid search results in notebook

    Parameters
    ----------
    gcv : sklearn.model_selection.GridSearchCV
        GridSearchCV object after fit(:)

    Returns
    -------
    pandas.DataFrame
        Grid search results as DataFrame
    """
    return pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score',
                                                     ascending=False)


def load_iris(imbalanced_binary=False):
    """Load iris dataset

    Parameters
    ----------
    imbalanced_binary : bool, default: False
        Load imbalanced and binary version of Iris dataset
    Returns
    -------
    pandas.DataFrame
        Iris dataset as DataFrame
    """
    if imbalanced_binary:
        dataset = pd.read_csv('iris_imbalanced.csv', index_col=[0])
    else:
        dataset = pd.read_csv('iris.csv', index_col=[0])
    return dataset


def generate_iris_imbalanced(file='iris_imbalanced.csv'):
    """Generate imbalanced binary dataset based on Iris dataset

    This function is not needed for the actual lab course. It is only used to
    generate the iris_imbalanced.csv file

    Parameters
    ----------
    file : string
        File to which the dataset is written in CSV format
    """
    from imblearn.datasets import make_imbalance
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length',
                       'petal_width']
    iris = load_iris()
    iris['species'] = LabelEncoder().fit_transform(iris['species'])
    ratio = {0: 2, 1: 10, 2: 50}
    X, y = make_imbalance(iris[feature_columns],
                          iris['species'].values, ratio=ratio)
    y = y < 2
    iris_imbalanced = pd.DataFrame(data=np.concatenate((X, y.reshape(-1, 1)),
                                                       axis=1),
                                   columns=feature_columns)
    iris_imbalanced.to_csv(file)
