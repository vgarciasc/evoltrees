from sklearn.model_selection import train_test_split

from evoltree.dataset_configs import get_config, load_dataset
from evoltree.evolution_strategy import evolution_strategy_tracked
from evoltree.classification_tree import ClassificationTree

if __name__ == "__main__":
    lamb = 100
    mu = 10
    n_generations = 10
    depth = 2

    config = get_config("drybean")
    X, y = load_dataset(config)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=2, stratify=y_test)

    tree, log = evolution_strategy_tracked(config, ClassificationTree, {}, X_train, y_train, lamb, mu, n_generations,
                                           depth, n_jobs=8)

    print(f"Train accuracy: {tree.evaluate(X_train, y_train)}")
    print(f"Test accuracy: {tree.evaluate(X_test, y_test)}")
