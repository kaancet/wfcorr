import numbers
import numpy as np
from dataclasses import dataclass
from collections.abc import Sequence
from temporaldata import Interval, RegularTimeSeries


@dataclass
class RegionActivity(RegularTimeSeries):
    region_name: str
    dff: np.memmap
    spatial_mask: np.ndarray

    @classmethod
    def from_npy(cls, dff_in: np.memmap, sampling_freq: float = 25) -> "RegionActivity":
        """_summary_

        Args:
            path (Path): _description_

        Returns:
            SignalDFF: _description_
        """
        _sampling_rate = sampling_freq

        dt = 1 / sampling_freq
        _t = dt * np.arange(len())

        return cls(dff=dff_in, sampling_rate=_sampling_rate, domain=Interval(_t[0], _t[-1]))

    def __getitem__(self, idx: int | Sequence[int]) -> np.ndarray:
        if isinstance(idx, numbers.Integral):
            idx = [idx]

        elif isinstance(idx, Sequence) and not isinstance(idx, (str, bytes)):
            if not all(isinstance(i, numbers.Integral) for i in idx):
                raise TypeError("All indices must be integers")

        else:
            raise TypeError("Index must be an integer or a sequence of integers")

        return self.dff[idx, :, :]
