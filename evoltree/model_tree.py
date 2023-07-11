import numpy as np
from sklearn.linear_model import LinearRegression

from evoltree.tree import Tree
from evoltree.tree_evaluation import get_leaf, get_leaves_dataset
from evoltree.fitness_functions import calc_mse


class ModelTree(Tree):
    def __init__(self, model, model_params, *args, **kwargs):
        self.model = model
        self.model_params = model_params

        super().__init__(*args, **kwargs)

    def copy(self):
        return ModelTree(self.model, self.model_params, self.config, self.depth,
                         self.attributes.copy(), self.thresholds.copy(), self.labels.copy(), self.fitness_fn)

    def predict(self, X):
        try:
            l = get_leaves_dataset(X, self.attributes, self.thresholds, self.depth)
            return [l_i.predict([x_i])[0] for x_i, l_i in zip(X, self.labels[np.argmax(l, axis=1)])]
        except ValueError:
            return self.labels[get_leaf(X, self.attributes, self.thresholds, self.depth)].predict([X])[0]

    def optimize_leaves(self, X, y):
        pred_leaves_idx = [self.get_leaf(x) for x in X]
        pred_leaves_inputs = [[X[i] for i in range(len(y)) if pred_leaves_idx[i] == j] for j in range(2 ** self.depth)]
        pred_leaves_labels = [[y[i] for i in range(len(y)) if pred_leaves_idx[i] == j] for j in range(2 ** self.depth)]

        for i in range(2 ** self.depth):
            self.labels[i] = self.model(**self.model_params)

            if len(pred_leaves_labels[i]) == 0:
                self.labels[i].fit(X, y)
            else:
                self.labels[i].fit(pred_leaves_inputs[i], pred_leaves_labels[i])

    @staticmethod
    def generate_random(config, depth, params={}, X=None):
        if "attr_metadata" not in config:
            if X is not None:
                config["attr_metadata"] = [(np.min(X_i), np.max(X_i)) for X_i in np.transpose(X.astype(np.float32))]
            else:
                raise ValueError("Should pass X to generate_random to determine min and max values of attributes")

        model = params.get("model", LinearRegression)
        model_params = params.get("model_params", {})
        fitness_fn = params.get("fitness_fn", calc_mse)

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
        labels = np.array([model(**model_params) for _ in range(2 ** depth)])

        return ModelTree(model, model_params, config, depth, attributes, thresholds, labels, fitness_fn)

    def __str__(self):
        stack = [(0, 0, self.depth - 1)]
        output = ""

        while len(stack) > 0:
            node_id, leaf_id, depth = stack.pop()
            output += "-" * (self.depth - depth) + " "

            if depth == -1:
                model = self.labels[leaf_id]
                if type(model) == LinearRegression:
                    output += f"y = {model.intercept_:.5f} "
                    for i in range(len(model.coef_)):
                        output += f"+ {model.coef_[i]:.5f} * x{i} "
                else:
                    output += str(model)
            else:
                output += self.config['attributes'][self.attributes[node_id]]
                output += " <= "
                output += '{:.5f}'.format(self.thresholds[node_id])

                stack.append((node_id + 2 ** depth, leaf_id + 2 ** depth, depth - 1))
                stack.append((node_id + 1, leaf_id, depth - 1))
            output += "\n"

        return output