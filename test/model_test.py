import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from evoltree.dataset_configs import get_config, load_dataset
from evoltree.evolution_strategy import evolution_strategy_tracked
from evoltree.model_tree import ModelTree
from sklearn.linear_model import LinearRegression, Lasso

if __name__ == "__main__":
    lamb = 100
    mu = 10
    n_generations = 10
    depth = 2

    config = get_config("qsar")
    X, y = load_dataset(config, classification=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=2)

    # params = {"model": LinearRegression, "model_params": {}}
    params = {"model": Lasso, "model_params": {"alpha": 0.2, "max_iter": 1000000, "tol": 0.1}}
    tree, log = evolution_strategy_tracked(config, ModelTree, params, X_train, y_train,
                                           lamb, mu, n_generations, depth, n_jobs=8)

    print(tree)
    print(f"Train MSE: {- tree.evaluate(X_train, y_train)}")
    print(f"Test MSE: {- tree.evaluate(X_test, y_test)}")
