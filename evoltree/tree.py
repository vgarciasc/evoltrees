import numpy as np

from evoltree.tree_evaluation import get_leaf, get_leaves_dataset
from evoltree.evolution_strategy import evolution_strategy

class Tree:
    def __init__(self, config, depth, attributes=None, thresholds=None, labels=None,
                 fitness_fn=None, lamb=100, mu=10, n_generations=100, n_jobs=1):

        self.config = config
        self.depth = depth

        self.fitness_fn = fitness_fn
        self.attributes = attributes
        self.thresholds = thresholds
        self.labels = labels

        self.lamb = lamb
        self.mu = mu
        self.n_generations = n_generations
        self.n_jobs = n_jobs

        self.log = None

    def fit(self, X, y):
        tree, log = evolution_strategy(self.config, self.__class__, {}, X, y,
                                       self.lamb, self.mu, self.n_generations,
                                       self.depth, self.n_jobs)

        self.attributes = tree.attributes
        self.thresholds = tree.thresholds
        self.labels = tree.labels
        self.log = log

        return tree

    def predict(self, X):
        try:
            l = get_leaves_dataset(X, self.attributes, self.thresholds, self.depth)
            return self.labels[np.argmax(l, axis=1)]
        except ValueError:
            return self.labels[get_leaf(X, self.attributes, self.thresholds, self.depth)]

    def evaluate(self, X, y):
        pred = self.predict(X)
        return self.fitness_fn(y, pred)

    def get_leaf(self, x):
        curr_depth = self.depth - 1
        node_idx = 0
        leaf_idx = 0

        while curr_depth >= 0:
            if x[self.attributes[node_idx]] <= self.thresholds[node_idx]:
                node_idx += 1
            else:
                node_idx += 2 ** curr_depth
                leaf_idx += 2 ** curr_depth
            curr_depth -= 1

        return leaf_idx

    def mutate(self):
        operator = np.random.choice(["attribute", "threshold"])
        if operator == "attribute":
            self._mutate_attribute()
        elif operator == "threshold":
            self._mutate_threshold()

    def _mutate_attribute(self):
        node_idx = np.random.randint(2 ** self.depth - 1)
        self.attributes[node_idx] = np.random.randint(self.config['n_attributes'])
        self.thresholds[node_idx] = np.random.uniform(self.config['attr_metadata'][self.attributes[node_idx]][0],
                                                      self.config['attr_metadata'][self.attributes[node_idx]][1])

    def _mutate_threshold(self):
        node_idx = np.random.randint(2 ** self.depth - 1)
        self.thresholds[node_idx] = np.random.uniform(self.config['attr_metadata'][self.attributes[node_idx]][0],
                                                      self.config['attr_metadata'][self.attributes[node_idx]][1])

    def __str__(self):
        stack = [(0, 0, self.depth - 1)]
        output = ""

        while len(stack) > 0:
            node_id, leaf_id, depth = stack.pop()
            output += "-" * (self.depth - depth) + " "

            if depth == -1:
                output += str(self.labels[leaf_id])
            else:
                output += self.config['attributes'][self.attributes[node_id]]
                output += " <= "
                output += '{:.5f}'.format(self.thresholds[node_id])

                stack.append((node_id + 2 ** depth, leaf_id + 2 ** depth, depth - 1))
                stack.append((node_id + 1, leaf_id, depth - 1))
            output += "\n"

        return output

