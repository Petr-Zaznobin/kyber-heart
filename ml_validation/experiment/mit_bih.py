from datetime import datetime
from pathlib import Path
from typing import Callable, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml_validation.experiment.metrics import get_metrics
from ml_validation.experiment.report import Report

from ..version import __version__
from .types import XType, YType

Indexes: TypeAlias = npt.NDArray[np.int32]

TrainDataset = list[tuple[XType, Indexes, YType]]
TestDataset = list[tuple[XType, Indexes]]

Algorithm = Callable[[TestDataset], list[YType]]
Trainer = Callable[[TrainDataset], Algorithm]


class Experiment:
    _dataset = "MIT-BIH"

    def __init__(self, name: str, authors: list[str], path_dir: Path, description: str = "") -> None:
        self._start = datetime.now().astimezone()
        self.name = name
        self.description = description
        self.authors = authors
        self.url: str = ""
        self._X, df = Experiment._load_data(path_dir)
        self._Y = df["class"].to_numpy(dtype=np.bool_)
        self._records = df["record"].to_numpy(dtype=np.int32)
        assert np.all(self._records == np.sort(self._records))

        self._indexes: list[Indexes] = []
        self._classes: list[YType] = []
        self._record_to_index: dict[int, int] = {}
        for index, (record, group) in enumerate(df.groupby("record")):
            self._indexes.append(group["index"].to_numpy(dtype=np.int32))
            self._classes.append(group["class"].to_numpy(dtype=np.bool_))
            self._record_to_index[record] = index

        assert len(self._X) == 23
        assert len(self._indexes) == 23
        assert len(self._classes) == 23
        assert len(self._record_to_index) == 23

    def validate(self, trainer: Trainer) -> Report:
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        y_pred = np.empty_like(self._Y)

        for train_indexes, test_indexes in sgkf.split(self._Y, self._Y, self._records):
            train = self._get_indexes(self._records[train_indexes])
            test = self._get_indexes(self._records[test_indexes])
            assert sorted(train + test) == list(range(len(self._X)))
            assert np.all(test_indexes == np.sort(test_indexes))

            algorithm = self._fit(trainer, train)
            y_pred_list = self._predict(algorithm, test)
            y_pred[test_indexes] = np.concatenate(y_pred_list)

        table, matrices = get_metrics(self._Y, y_pred, ["NORM", "NOT_NORM"])
        return Report(
            version=__version__,
            start=self._start,
            end=datetime.now().astimezone(),
            dataset=self._dataset,
            task_type=None,
            name=self.name,
            description=self.description,
            authors=self.authors,
            metrics=table,
            matrices=matrices,
            url=self.url
        )

    def _fit(self, trainer: Trainer, indexes: list[int]) -> Algorithm:
        dataset = [(self._X[i], self._indexes[i], self._classes[i]) for i in indexes]
        return trainer(dataset)

    def _predict(self, algorithm: Algorithm, indexes: list[int]) -> list[YType]:
        dataset = [(self._X[i], self._indexes[i]) for i in indexes]
        return algorithm(dataset)

    @staticmethod
    def _load_data(path_dir: Path) -> tuple[list[XType], pd.DataFrame]:
        X_npy = np.load(path_dir / "mit_bih_af_signal.npy")
        X: list[XType] = []
        for i, x in enumerate(X_npy):
            X.append(x if i != 11 else x[:, :8_325_000])

        Y = np.load(path_dir / "mit_bih_af_meta.npy")
        Y[Y[:, 2] > 1, 2] = 1
        return X, pd.DataFrame(Y, columns=["record", "index", "class"])

    def _get_indexes(self, records: npt.NDArray[np.int32]) -> list[int]:
        return [self._record_to_index[r] for r in np.unique(records)]


def start_experiment(
        name: str,
        authors: str | list[str],
        path_dir: Path | str = "datasets",
        description: str = "") -> Experiment:
    if isinstance(authors, str):
        authors = [authors]
    if isinstance(path_dir, str):
        path_dir = Path(path_dir)
    return Experiment(name=name, authors=authors, path_dir=path_dir, description=description)
