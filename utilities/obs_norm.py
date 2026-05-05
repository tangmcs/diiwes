"""Running observation normalization."""

from __future__ import annotations

import numpy as np


class ObsNormalizer:
    """Running mean/variance normalizer using batch Welford updates."""

    def __init__(self, shape: int | tuple[int, ...], epsilon: float = 1e-8) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.epsilon = float(epsilon)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def normalize(
        self,
        obs: np.ndarray,
        *,
        mean: np.ndarray | None = None,
        var: np.ndarray | None = None,
    ) -> np.ndarray:
        m = self.mean if mean is None else mean
        v = self.var if var is None else var
        return (obs - m) / np.sqrt(v + self.epsilon)

    def get_mean_var(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean.copy(), self.var.copy()

    def update_batch(self, observations: np.ndarray) -> None:
        batch = np.asarray(observations, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def get_state(self) -> dict[str, np.ndarray | float]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }
