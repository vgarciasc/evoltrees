import numpy as np
from sklearn.linear_model import LinearRegression

from evoltree.tree import Tree


class ModelTree(Tree):
    def __init__(self, *args, **kwargs):
        self.model = LinearRegression
        super().__init__(*args, **kwargs)

    def copy(self):
        return ModelTree(self.config, self.depth, self.attributes.copy(),
                         self.thresholds.copy(), self.labels.copy())

    def evaluate(self, X, y):
        pred = [self.predict(x).predict([x])[0] for x in X]
        return - np.mean((pred - y) ** 2)

    def optimize_leaves(self, X, y):
        pred_leaves_idx = [self.get_leaf(x) for x in X]
        pred_leaves_inputs = [[X[i] for i in range(len(y)) if pred_leaves_idx[i] == j] for j in range(2 ** self.depth)]
        pred_leaves_labels = [[y[i] for i in range(len(y)) if pred_leaves_idx[i] == j] for j in range(2 ** self.depth)]

        for i in range(2 ** self.depth):
            self.labels[i] = self.model()

            if len(pred_leaves_labels[i]) == 0:
                self.labels[i].fit(X, y)
            else:
                self.labels[i].fit(pred_leaves_inputs[i], pred_leaves_labels[i])

    @staticmethod
    def generate_random(config, depth, X=None):
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
        labels = [LinearRegression() for _ in range(2 ** depth)]

        return ModelTree(config, depth, attributes, thresholds, labels)

    def __str__(self):
        stack = [(0, 0, self.depth - 1)]
        output = ""

        while len(stack) > 0:
            node_id, leaf_id, depth = stack.pop()
            output += "-" * (self.depth - depth) + " "

            if depth == -1:
                model = self.labels[leaf_id]
                output += f"y = {model.intercept_:.5f} "
                for i in range(len(model.coef_)):
                    output += f"+ {model.coef_[i]:.5f} * x{i} "
            else:
                output += self.config['attributes'][self.attributes[node_id]]
                output += " <= "
                output += '{:.5f}'.format(self.thresholds[node_id])

                stack.append((node_id + 2 ** depth, leaf_id + 2 ** depth, depth - 1))
                stack.append((node_id + 1, leaf_id, depth - 1))
            output += "\n"

        return output