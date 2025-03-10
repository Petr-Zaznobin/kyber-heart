import ast
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils
from sklearn.model_selection import GroupKFold

from ..version import __version__
from .metrics import get_metrics
from .predict import batch_predict
from .report import Report
from .split_data import maybe_shuffe, split_by_indexes, train_val_split
from .types import Function, TrainValData, XType, YType


class Experiment:
    _dataset = "PTB-XL"

    def __init__(
        self,
        name: str,
        authors: list[str],
        path: Path,
        task_type: str,
        test_fold: int,
        description: str = "",
    ) -> None:
        self._start = datetime.now().astimezone()
        self.name = name
        self.description = description
        self.authors = authors
        self.url: str = ""
        self._task_type = task_type

        X = np.load(path / "ptb_xl.npy").astype(np.float32) / 1000
        meta = Experiment._load_metadata(path)
        assert len(X) == len(meta)

        self._X_train, self._meta_train, self._X_test, self._meta_test = (
            Experiment._split(X, meta, test_fold)
        )

        scp = pd.read_csv(path / "scp_statements.csv", index_col=0)
        self._classes = Experiment._get_classes(scp, task_type)

        scp_to_class = self._read_scp_to_class(scp, task_type)
        self._Y_train = self._get_classes_from_meta(
            self._meta_train, scp_to_class
        )
        self._Y_test = self._get_classes_from_meta(
            self._meta_test, scp_to_class
        )

    def get_data(self) -> tuple[XType, YType]:
        return self._X_train, self._Y_train

    def _get_split_by_mask_val(self, mask_val: npt.NDArray[np.bool_]) -> TrainValData:
        meta = self.get_meta()
        mask_train = np.logical_not(mask_val)
        return self._X_train[mask_train], self._X_train[mask_val], \
            self._Y_train[mask_train], self._Y_train[mask_val], \
            meta[mask_train], meta[mask_val]

    def get_train_val_folds(
        self,
        n_splits: int = 10,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> Iterator[TrainValData]:
        gkf = GroupKFold(n_splits=n_splits)
        groups = self.get_meta().patient_id

        indexes = np.arange(len(self._X_train), dtype=np.int32)
        arrays = (self._X_train, self._Y_train, groups, indexes)
        if shuffle:
            arrays = sklearn.utils.shuffle(*arrays, random_state=random_state)
        indexes = arrays[-1]

        for train_idx, val_idx in gkf.split(*arrays[:3]):
            yield split_by_indexes(
                self._X_train,
                self._Y_train,
                self.get_meta(),
                indexes[train_idx],
                indexes[val_idx]
            )

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
            groups=meta.patient_id,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

    def get_train_val_by_database_fold(
        self,
        val_fold: Optional[int] = None,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> TrainValData:
        folds = self.get_meta().strat_fold.to_numpy(dtype=np.int32)
        if val_fold is None:
            val_fold = int(folds.max())
        result = self._get_split_by_mask_val(folds == val_fold)
        return maybe_shuffe(result, shuffle, random_state)

    def get_meta(self) -> pd.DataFrame:
        return self._meta_train

    def validate(self, func: Function, batch_size: int = 1) -> Report:
        classes = self._classes
        if len(classes) == 1:
            classes = ["NOT_" + classes[0], classes[0]]
        y_pred = batch_predict(
            func, self._X_test, len(self._classes), batch_size
        )
        table, matrices = get_metrics(self._Y_test, y_pred, classes)
        return Report(
            version=__version__,
            start=self._start,
            end=datetime.now().astimezone(),
            dataset=self._dataset,
            task_type=self._task_type,
            name=self.name,
            description=self.description,
            authors=self.authors,
            metrics=table,
            matrices=matrices,
            url=self.url
        )

    @property
    def classes(self) -> list[str]:
        return self._classes.copy()

    @staticmethod
    def _get_classes(scp: pd.DataFrame, task_type: str) -> list[str]:
        if task_type == "binary":
            return ["NORM"]
        elif task_type == "superclasses":
            return ["NORM", "MI", "STTC", "CD", "HYP"]
        elif task_type == "all":
            return scp.index.to_list()
        else:
            raise RuntimeError(f"Wrong type of task: {task_type}")

    def _read_scp_to_class(
        self, scp: pd.DataFrame, task_type: str
    ) -> dict[str, int]:
        if task_type == "binary":
            return {"NORM": 0}
        elif task_type == "superclasses":
            return self._read_scp_to_class_superclasses(scp)
        elif task_type == "all":
            return {c: i for i, c in enumerate(self._classes)}
        else:
            raise RuntimeError(f"Wrong type of task: {task_type}")

    def _read_scp_to_class_superclasses(
        self, scp: pd.DataFrame
    ) -> dict[str, int]:
        scp = scp[scp.diagnostic == 1]
        scp_to_class = {}
        for d, c in zip(scp.index, scp.diagnostic_class):
            scp_to_class[d] = self._classes.index(c)
        return scp_to_class

    def _get_classes_from_meta(
        self, meta: pd.DataFrame, scp_to_class: dict[str, int]
    ) -> YType:
        Y = np.zeros((len(meta), len(self._classes)), dtype=np.bool_)
        for i, codes in enumerate(meta.scp_codes):
            for code in codes:
                if code in scp_to_class:
                    Y[i, scp_to_class[code]] = True
        return Y

    @staticmethod
    def _load_metadata(path: Path) -> pd.DataFrame:
        meta = pd.read_csv(path / "ptbxl_database.csv", index_col="ecg_id")
        meta.scp_codes = meta.scp_codes.apply(lambda x: ast.literal_eval(x))
        return meta

    @staticmethod
    def _split(
        X: XType, meta: pd.DataFrame, test_fold: int
    ) -> tuple[XType, pd.DataFrame, XType, pd.DataFrame]:
        X_train = X[np.where(meta.strat_fold != test_fold)]
        meta_train = meta[meta.strat_fold != test_fold]
        X_test = X[np.where(meta.strat_fold == test_fold)]
        meta_test = meta[meta.strat_fold == test_fold]
        return X_train, meta_train, X_test, meta_test


def start_experiment(
    name: str,
    authors: str | list[str],
    path_dir: Path | str = "datasets",
    task_type: str = "superclasses",
    test_fold: int = 10,
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
        task_type=task_type,
        test_fold=test_fold,
        description=description,
    )
