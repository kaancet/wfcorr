import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence

from .region import RegionActivity


@dataclass
class FCMatrixStack:
    """
    Output of SlidingWindowFC.compute().

    Attributes
    ----------
    z_matrices : np.ndarray
        Shape (n_windows, n_regions, n_regions) — Fisher z-transformed FC.
    r_matrices : np.ndarray
        Shape (n_windows, n_regions, n_regions) — raw Pearson r (before z-transform).

    """

    z_matrices: np.ndarray
    r_matrices: np.ndarray
    window_centers: np.ndarray
    region_names: List[str]


class SlidingWindowFC:
    """
    Compute per-window Pearson correlation matrices with Fisher z-transform.

    Parameters
    ----------
    window_size : int
        Number of frames per window (paper: 20 frames = 1.0 s).
    step_size : int
        Number of frames between window starts (paper: 5 frames = 0.25 s).
    """

    def __init__(self, window_size: int = 20, step_size: int = 5) -> None:
        self.window_size = window_size
        self.step_size = step_size

    def _make_windows(self, traces: np.ndarray) -> np.ndarray:
        """
        Slice traces into overlapping windows.

        Parameters
        ----------
        traces : np.ndarray, shape (n_regions, n_bins)

        Returns
        -------
        np.ndarray, shape (n_windows, n_regions, window_size)
        """
        _, n_bins = traces.shape
        W, S = self.window_size, self.step_size
        starts = np.arange(0, n_bins - W + 1, S)
        windows = np.stack([traces[:, s : s + W] for s in starts], axis=0)
        window_centers = windows[starts + W // 2]

        return window_centers, windows  # (n_windows, n_regions, W)

    @staticmethod
    def _pearson_matrix(win: np.ndarray) -> np.ndarray:
        """
        Pearson correlation matrix for one window.

        Parameters
        ----------
        win : np.ndarray, shape (n_regions, W)

        Returns
        -------
        np.ndarray, shape (n_regions, n_regions)
        """
        # np.corrcoef handles NaN rows gracefully with a warning;
        # suppress and propagate NaN explicitly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.corrcoef(win)

    def compute(
        self,
        regions: Sequence[RegionActivity],
    ) -> FCMatrixStack:
        """
        Run the sliding-window FC analysis.

        Parameters
        ----------
        regions : sequence of RegionActivity

        Returns
        -------
        FCMatrixStack
        """

        # Stack traces: (n_regions, n_bins)
        traces = np.stack([r.get_mean() for r in regions], axis=0)

        region_names = [r.region_name for r in regions]

        window_centers, windows = self._make_windows(traces)  # (n_windows, n_regions, W)
        n_windows = windows.shape[0]

        r_matrices = np.full((n_windows, len(region_names), len(region_names)), np.nan)

        for i, win in enumerate(windows):
            r_matrices[i] = self._pearson_matrix(win)

        z_matrices = fisher_z(r_matrices)

        return FCMatrixStack(
            z_matrices=z_matrices,
            r_matrices=r_matrices,
            window_centers=window_centers,
            node_names=region_names,
        )


def fisher_z(r: np.ndarray) -> np.ndarray:
    """
    Apply Fisher z-transform: z = arctanh(r).

    Clips r to (-1+eps, 1-eps) to avoid ±inf on the diagonal.
    """
    eps = 1e-7
    r_clipped = np.clip(r, -1 + eps, 1 - eps)
    return np.arctanh(r_clipped)


def fisher_z_inv(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher z: r = tanh(z)."""
    return np.tanh(z)
