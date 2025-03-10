from typing import Optional

import numpy as np
import numpy.typing as npt

from . import functional as F

__all__ = [
    "GaussianNoise",
    "Scaling",
    "RandomMasking",
    "Permutation",
    "Inversion",
    "TimeInversion",
    "TimeWarp",
    "RandomScaling",
]


class Augmentation:
    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    def __call__(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return self.apply(X, inplace=inplace)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Identity(Augmentation):
    def __init__(self) -> None:
        super().__init__()

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return X if inplace else X.copy()


class GaussianNoise(Augmentation):
    def __init__(
        self, loc: float = 0.0, scale: float = 0.01, seed: Optional[int] = None
    ) -> None:
        super().__init__()

        self._random_generator = np.random.default_rng(seed=seed)

        self._loc = loc
        self._scale = scale

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.add_gauss_noise(
            X,
            loc=self._loc,
            scale=self._scale,
            random_generator=self._random_generator,
            inplace=inplace,
        )


class Scaling(Augmentation):
    def __init__(self, scaling_factor: float = 1.1) -> None:
        super().__init__()

        self._scaling_factor = scaling_factor

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.apply_scaling(
            X, scaling_factor=self._scaling_factor, inplace=inplace
        )


class RandomScaling(Augmentation):
    def __init__(
        self, low: float = 0.9, high: float = 1.1, seed: Optional[int] = None
    ) -> None:
        super().__init__()
        assert low <= high

        self._random_generator = np.random.default_rng(seed=seed)

        self._low = low
        self._high = high

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.apply_random_scaling(
            X,
            low=self._low,
            high=self._high,
            random_generator=self._random_generator,
            inplace=inplace,
        )


class RandomMasking(Augmentation):
    def __init__(
        self,
        duration: int,
        value: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._random_generator = np.random.default_rng(seed=seed)

        self._duration = duration
        self._value = value

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.apply_random_masking(
            X,
            duration=self._duration,
            value=self._value,
            random_generator=self._random_generator,
            inplace=inplace,
        )


class Permutation(Augmentation):
    def __init__(
        self, seed: Optional[int] = None
    ) -> None:
        super().__init__()

        self._random_generator = np.random.default_rng(seed=seed)

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.apply_permutations(
            X,
            random_generator=self._random_generator,
            inplace=inplace,
        )


class Inversion(Augmentation):
    def __init__(self) -> None:
        super().__init__()

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.apply_inversion(X, inplace=inplace)


class TimeInversion(Augmentation):
    def __init__(self) -> None:
        super().__init__()

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.apply_time_inversion(X, inplace=inplace)


class TimeWarp(Augmentation):
    def __init__(self, warp_ratio: float = 0.95, seed: Optional[int] = None) -> None:
        super().__init__()

        self._random_generator = np.random.default_rng(seed=seed)

        self._warp_ratio = warp_ratio

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False
    ) -> npt.NDArray[np.float32]:
        return F.apply_time_warping(X, self._warp_ratio, self._random_generator, inplace=inplace)
