import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import metrics

from .types import YType


def get_metrics(y_true: YType, y_pred: YType, names: list[str]) -> \
        tuple[pd.DataFrame, dict[str, npt.NDArray[np.int64]]]:
    report = metrics.classification_report(
            y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
    matrices = metrics.multilabel_confusion_matrix(y_true, y_pred)
    name_to_matrix = {}
    for name, matrix in zip(names, matrices):
        name_to_matrix[name] = matrix
    return pd.DataFrame(report), name_to_matrix
