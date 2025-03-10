from typing import Callable, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

XType: TypeAlias = npt.NDArray[np.float32]
YType: TypeAlias = npt.NDArray[np.bool_]

YPredType: TypeAlias = npt.NDArray[np.bool_]
Function: TypeAlias = Callable[[XType], YPredType]

TrainValData = tuple[XType, XType, YType, YType, pd.DataFrame, pd.DataFrame]
