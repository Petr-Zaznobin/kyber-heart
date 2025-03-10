from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_validation.experiment.metrics import get_metrics
from ml_validation.experiment.report import Report

from ..version import __version__
from .predict import batch_predict
from .split_data import train_val_split
from .types import Function, TrainValData, XType, YType


class Experiment:
    _dataset = "Three bases"

    def __init__(
        self,
        name: str,
        authors: list[str],
        path: Path,
        description: str = "",
    ) -> None:
        self._start = datetime.now().astimezone()
        self.name = name
        self.description = description
        self.authors = authors
        self.url: str = ""

        X, Y, meta = Experiment._load_data(path)

        groups = meta["Patient ID in source database"]
        self._X_train, self._X_test, self._Y_train, self._Y_test, self._meta_train, self._meta_test = \
            train_val_split(X, Y, meta, groups, 0.2, shuffle=False, random_state=42)

    def get_data(self) -> tuple[XType, YType]:
        return self._X_train, self._Y_train

    def get_meta(self) -> pd.DataFrame:
        return self._meta_train

    def get_train_val_split(
        self,
        test_size: Optional[float | int] = None,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> TrainValData:
        meta = self.get_meta()
        return train_val_split(
            self._X_train,
            self._Y_train,
            meta,
            groups=meta["Patient ID in source database"],
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

    def validate(self, func: Function, batch_size: int = 1) -> Report:
        classes = ["1", "2", "3", "4", "7"]
        y_pred = batch_predict(func, self._X_test, len(classes), batch_size)
        table, matrices = get_metrics(self._Y_test, y_pred, classes)
        return Report(
            version=__version__,
            start=self._start,
            end=datetime.now().astimezone(),
            dataset=self._dataset,
            task_type="5 classes",
            name=self.name,
            description=self.description,
            authors=self.authors,
            metrics=table,
            matrices=matrices,
            url=self.url
        )

    @staticmethod
    def _load_data(path: Path) -> tuple[XType, YType, pd.DataFrame]:
        X = np.load(path / "three_bases_X.npy").astype(np.float32) / 1000
        Y = np.load(path / "three_bases_Y.npy")[:, [0, 1, 2, 3, 6]]
        meta = pd.read_csv(path / "three_bases_meta.csv", index_col=0)

        assert len(X) == len(Y)
        assert len(X) == len(meta)

        zero = np.load(path / "three_bases_zero_indexes.npy")
        noise = np.load(path / "three_bases_noise_indexes.npy")

        good = np.ones(len(X), dtype=np.bool_)
        good[zero] = False
        good[noise] = False

        return X[good], Y[good], meta.iloc[good]


def start_experiment(
    name: str,
    authors: str | list[str],
    path_dir: Path | str = "datasets",
    description: str = "",
) -> Experiment:
    if isinstance(authors, str):
        authors = [authors]
    if isinstance(path_dir, str):
        path_dir = Path(path_dir)

    return Experiment(
        name=name,
        authors=authors,
        path=path_dir,
        description=description,
    )
