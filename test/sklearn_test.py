from sklearn.model_selection import train_test_split, cross_validate

from evoltree.dataset_configs import get_config, load_dataset
from evoltree.evolution_strategy import evolution_strategy_tracked
from evoltree.regression_tree import RegressionTree

if __name__ == "__main__":
    lamb = 100
    mu = 10
    n_generations = 10
    depth = 4

    config = get_config("qsar")
    X, y = load_dataset(config, classification=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=2)

    print(f"Fitting Regression Tree...")
    tree = RegressionTree(config=config, depth=depth, lamb=100, mu=10, n_generations=100, n_jobs=8)
    tree = tree.fit(X_train, y_train)
    print(f"Done fitting Regression Tree.")

    print(tree)

    print(f"Train MSE: {- tree.evaluate(X_train, y_train)}")
    # print(f"Cross validation: {cross_validate(tree, X, y, cv=5, scoring='neg_mean_squared_error')}")
    print(f"Test MSE: {- tree.evaluate(X_test, y_test)}")

