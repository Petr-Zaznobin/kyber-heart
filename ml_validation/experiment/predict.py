import numpy as np

from .types import Function, XType, YPredType


def batch_predict(func: Function, X: XType, num_classes: int, batch_size: int) -> YPredType:
    y_pred = np.empty((len(X), num_classes), dtype=np.bool_)
    for begin in range(0, len(X), batch_size):
        end = min(begin + batch_size, len(X))
        y_pred[begin:end] = func(X[begin:end])
    return y_pred
