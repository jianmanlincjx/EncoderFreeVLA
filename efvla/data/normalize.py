"""q01/q99 normalisation shared by training and evaluation.

state: all 8 dims mapped [q01, q99] -> [-1, 1] then clipped.
actions: dims 0..5 mapped the same way; gripper (dim 6) is kept as is (dataset values are exactly -1/+1).
"""
import numpy as np

GRIPPER_DIM = 6


def _scale(stats_part, dims):
    q01 = np.asarray(stats_part["q01"], np.float64)[dims]
    q99 = np.asarray(stats_part["q99"], np.float64)[dims]
    span = q99 - q01
    assert (span > 1e-8).all(), f"degenerate quantile span {span}"
    return q01, span


class Normalizer:
    def __init__(self, stats: dict):
        self.stats = stats
        self.a_dims = np.arange(6)
        self.a_q01, self.a_span = _scale(stats["actions"], self.a_dims)
        self.s_q01, self.s_span = _scale(stats["state"], np.arange(8))

    def norm_actions(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, np.float64).copy()
        a[..., :6] = np.clip(2.0 * (a[..., :6] - self.a_q01) / self.a_span - 1.0, -1.0, 1.0)
        return a.astype(np.float32)

    def unnorm_actions(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, np.float64).copy()
        a[..., :6] = (np.clip(a[..., :6], -1.0, 1.0) + 1.0) / 2.0 * self.a_span + self.a_q01
        return a.astype(np.float32)

    def norm_state(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, np.float64)
        return np.clip(2.0 * (s - self.s_q01) / self.s_span - 1.0, -1.0, 1.0).astype(np.float32)


def binarize_gripper(a: np.ndarray) -> np.ndarray:
    """Model outputs a continuous gripper value; the env expects -1 (open) / +1 (close). sign(0) would give 0."""
    a = np.asarray(a, np.float32).copy()
    a[..., GRIPPER_DIM] = np.where(a[..., GRIPPER_DIM] > 0, 1.0, -1.0)
    return a
