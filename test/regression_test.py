from sklearn.model_selection import train_test_split

from evoltree.dataset_configs import get_config, load_dataset
from evoltree.evolution_strategy import evolution_strategy_tracked
from evoltree.regression_tree import RegressionTree

if __name__ == "__main__":
    lamb = 100
    mu = 10
    n_generations = 1000
    depth = 4

    config = get_config("qsar")
    X, y = load_dataset(config, classification=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=2)

    tree, log = evolution_strategy_tracked(config, RegressionTree, {}, X_train, y_train, lamb, mu, n_generations, depth, n_jobs=8)

    print(tree)
    print(f"Train MSE: {- tree.evaluate(X_train, y_train)}")
    print(f"Test MSE: {- tree.evaluate(X_test, y_test)}")
