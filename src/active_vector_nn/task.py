from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class NoisyLatentARStream:
    """Piecewise stationary latent AR(1) process with additive observation noise."""

    def __init__(
        self,
        regimes: Sequence[tuple[float, int]],
        dim: int,
        noise_std: float,
        seed: int,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if noise_std < 0.0 or not math.isfinite(noise_std):
            raise ValueError("noise_std must be finite and nonnegative")
        if not regimes:
            raise ValueError("at least one regime is required")

        schedule: list[float] = []
        for rho, steps in regimes:
            if not -1.0 < rho < 1.0:
                raise ValueError("every rho must be strictly between -1 and 1")
            if steps <= 0:
                raise ValueError("every regime must contain at least one step")
            schedule.extend([float(rho)] * int(steps))

        self.dim = int(dim)
        self.noise_std = float(noise_std)
        self._schedule = tuple(schedule)
        self._index = 0
        self._rng = np.random.default_rng(seed)
        self._latent = np.zeros(self.dim, dtype=float)

    @property
    def total_steps(self) -> int:
        return len(self._schedule)

    def step(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self._index >= len(self._schedule):
            raise StopIteration

        rho = self._schedule[self._index]
        innovation_scale = math.sqrt(1.0 - rho * rho)
        self._latent = (
            rho * self._latent
            + innovation_scale * self._rng.normal(size=self.dim)
        )
        observation = self._latent + self.noise_std * self._rng.normal(size=self.dim)
        self._index += 1
        return observation.copy(), self._latent.copy(), rho


def fit_ridge_readout(
    features: np.ndarray, targets: np.ndarray, ridge: float = 1e-6
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features and targets must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    if x.shape[0] == 0 or x.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("features and targets must be nonempty")
    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("features and targets must be finite")

    gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=float)
    rhs = x.T @ y
    return np.linalg.solve(gram, rhs)


def apply_readout(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("features and weights must be two-dimensional")
    if x.shape[1] != w.shape[0]:
        raise ValueError("feature dimension must match readout input dimension")
    return x @ w
