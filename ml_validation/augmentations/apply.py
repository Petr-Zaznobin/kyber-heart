from typing import Any, Collection, Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .augmentations import Augmentation

__all__ = ["Compose", "RandomApply", "SampleGenerator"]


class Compose(Augmentation):
    def __init__(self, augmentations: Iterable[Augmentation]) -> None:
        super().__init__()

        self._augmentations = augmentations

    def apply(self, X: npt.NDArray[np.float32], inplace: bool = False) -> npt.NDArray[np.float32]:
        X = X if inplace else X.copy()
        for augmentation in self._augmentations:
            X = augmentation.apply(X, inplace=True)

        return X

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: " + ";".join(
            augmentation.__repr__() for augmentation in self._augmentations
        )


class RandomApply(Augmentation):
    def __init__(
        self,
        augmentations: Sequence[Augmentation],
        weights: Optional[Collection[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._augmentations = augmentations

        if weights is not None:
            assert len(self._augmentations) == len(weights)

        self._weights = None if weights is None else np.array(weights, dtype=np.float32)
        if self._weights is not None:
            self._weights /= self._weights.sum()

        self._random_generator = np.random.default_rng(seed)

    def __call__(self, X: npt.NDArray[np.float32], inplace: bool = False) -> npt.NDArray[np.float32]:
        indexes = self._random_generator.choice(
            a=len(self._augmentations),
            size=len(X),
            p=self._weights)

        Xs_augmented = X if inplace else X.copy()
        for i, x in enumerate(Xs_augmented):
            augmentation = self._augmentations[indexes[i]]
            augmentation.apply(x, inplace=True)

        return Xs_augmented


class SampleGenerator:
    def __init__(
        self,
        augmentation: Augmentation,
        seed: Optional[int] = None,
    ) -> None:
        self._augmentation = augmentation
        self._random_generator = np.random.default_rng(seed)

    def __call__(
        self, X: npt.NDArray[np.float32], Y: npt.NDArray[Any], n_samples: int = 1
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[Any]]:
        sample_indexes = self._random_generator.choice(len(X), size=n_samples)

        Xs_augmented = self._augmentation.apply(X[sample_indexes])
        Ys_augmented = Y[sample_indexes].copy()

        return Xs_augmented, Ys_augmented
