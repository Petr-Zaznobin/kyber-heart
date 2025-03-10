import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


def add_gauss_noise(
    X: npt.NDArray[np.float32],
    loc: float,
    scale: float,
    random_generator: np.random.Generator,
    inplace: bool = False,
) -> npt.NDArray[np.float32]:
    noise = random_generator.normal(loc, scale, X.shape).astype(np.float32)

    return np.add(X, noise, out=X) if inplace else np.add(X, noise, out=noise)


def apply_time_inversion(
    X: npt.NDArray[np.float32], inplace: bool = False
) -> npt.NDArray[np.float32]:
    X = X if inplace else X.copy()

    np.copyto(X, np.flip(X, axis=-1))

    return X


def apply_scaling(
    X: npt.NDArray[np.float32], scaling_factor: float, inplace: bool = False
) -> npt.NDArray[np.float32]:
    return np.multiply(X, scaling_factor, out=(X if inplace else None))


def apply_random_scaling(
    X: npt.NDArray[np.float32],
    low: float,
    high: float,
    random_generator: np.random.Generator,
    inplace: bool = False
) -> npt.NDArray[np.float32]:
    scaling_factor = random_generator.uniform(low, high, X.shape[:-1]).astype(np.float32)
    return np.multiply(X, scaling_factor[..., None], out=(X if inplace else None))


def apply_inversion(
    X: npt.NDArray[np.float32], inplace: bool = False
) -> npt.NDArray[np.float32]:
    return apply_scaling(X, scaling_factor=-1, inplace=inplace)


def apply_random_masking(
    X: npt.NDArray[np.float32],
    duration: int,
    value: float,
    random_generator: np.random.Generator,
    inplace: bool = False,
) -> npt.NDArray[np.float32]:
    T = X.shape[-1]
    high_begin = T - duration + 1
    assert high_begin > 0

    X = X if inplace else X.copy()
    X_view = X.view()
    X_view.shape = (-1, T)
    begins = random_generator.integers(0, high_begin, size=X_view.shape[0])

    for x, begin in zip(X_view, begins):
        x[begin:begin + duration] = value

    return X


def apply_time_warping(
    X: npt.NDArray[np.float32],
    warp_ratio: float,
    random_generator: np.random.Generator,
    inplace: bool = False,
) -> npt.NDArray[np.float32]:
    assert warp_ratio > 0, "Warp ratio must be greater than 0"
    if warp_ratio > 1:
        warp_ratio = 1 / warp_ratio

    T = X.shape[-1]
    C = X.shape[-2] if len(X.shape) > 1 else 1
    base_time = np.arange(T)
    warped_time = base_time * warp_ratio

    view = X.view()
    view.shape = (-1, C, T)

    end_diff = base_time[-1] - warped_time[-1]
    shifts = end_diff * random_generator.random(len(view), dtype=np.float32)

    X_augmented = X if inplace else np.empty_like(X)
    view_out = X_augmented.view()
    view_out.shape = view.shape

    for x, x_out, shift in zip(view, view_out, shifts):
        func = interp1d(base_time, x, assume_sorted=True)
        np.copyto(x_out, func(warped_time + shift))

    return X_augmented


def apply_permutations(
    X: npt.NDArray[np.float32],
    random_generator: np.random.Generator,
    inplace: bool = False,
) -> npt.NDArray[np.float32]:
    X_augmented = X if inplace else X.copy()
    view = X_augmented.view()
    view.shape = (-1, X.shape[-2], X.shape[-1])

    for x in view:
        random_generator.shuffle(x)

    return X_augmented
