import numpy as np

def calc_mse(y, y_pred):
    return - np.mean((y - y_pred) ** 2)

def calc_accuracy(y, y_pred):
    return np.mean(y == y_pred)