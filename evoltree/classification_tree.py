import numpy as np
from evoltree.tree import Tree


class ClassificationTree(Tree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return ClassificationTree(self.config, self.depth, self.attributes.copy(),
                                  self.thresholds.copy(), self.labels.copy())

    def evaluate(self, X, y):
        pred = self.predict_batch(X)
        return np.mean(pred == y)

    def optimize_leaves(self, X, y):
        pred_leaves_idx = [self.get_leaf(x) for x in X]
        pred_leaves_labels = [[y[i] for i in range(len(y)) if pred_leaves_idx[i] == j] for j in range(2 ** self.depth)]

        for i in range(2 ** self.depth):
            if len(pred_leaves_labels[i]) == 0:
                self.labels[i] = np.argmax(np.bincount(y))
            else:
                self.labels[i] = np.argmax(np.bincount(pred_leaves_labels[i]))

    @staticmethod
    def generate_random(config, depth, params=None, X=None):
        if "attr_metadata" not in config:
            if X is not None:
                config["attr_metadata"] = [(np.min(X_i), np.max(X_i)) for X_i in np.transpose(X.astype(np.float32))]
            else:
                raise ValueError("Should pass X to generate_random to determine min and max values of attributes")

        attributes = []
        thresholds = []
        labels = []

        for i in range(2 ** depth):
            attributes.append(np.random.randint(config['n_attributes']))
            thresholds.append(np.random.uniform(config['attr_metadata'][attributes[-1]][0],
                                                config['attr_metadata'][attributes[-1]][1]))
            labels.append(0)

        attributes = np.array(attributes, dtype=np.int64)
        thresholds = np.array(thresholds, dtype=np.float64)
        labels = np.array(labels, dtype=np.int64)

        return ClassificationTree(config, depth, attributes, thresholds, labels)
