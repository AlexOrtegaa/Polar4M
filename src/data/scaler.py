import numpy as np

import torch

class IQRScaler:
    def __init__(
            self,
    ) -> None:

        self._shift = None
        self._scale = None

    def iqr_fit(
            self,
            data,
    ) -> None:

        self._shift = np.percentile(data, 50)
        self._scale = np.percentile(data, 75) - np.percentile(data, 25)

        return

    def iqr_transform(
            self,
            data,
    ) -> torch.Tensor:
        denom = 1 if self._scale == 0 else self._scale

        data_normalized = (data - self._shift) / denom
        return data_normalized



class MapMaxMinScaler:

    def __init__(
            self,
    ) -> None:

        self._max_value = None
        self._min_value = None

    def max_min_fit(
            self,
            data,
    ) -> None:

        self._max_value = np.max(data, axis=(1,2), keepdims=True)
        self._min_value = np.min(data, axis=(1,2), keepdims=True)

        return

    def max_min_transform(
            self,
            data,
    ) -> torch.Tensor:
        denom = self._max_value - self._min_value
        denom[denom == 0] = 1

        data_normalized = (data - self._min_value) / denom
        return data_normalized
