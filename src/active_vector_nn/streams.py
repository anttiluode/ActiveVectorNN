from __future__ import annotations

import math

import numpy as np


class AR1Stream:
    """Stationary equal-variance vector AR(1) stream."""

    def __init__(self, rho: float, dim: int, seed: int = 0):
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must be strictly between -1 and 1")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.rho = float(rho)
        self.dim = int(dim)
        self._innovation_scale = math.sqrt(1.0 - self.rho * self.rho)
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(self.dim, dtype=float)

    def reset(self) -> None:
        self._state.fill(0.0)

    def step(self) -> np.ndarray:
        noise = self._rng.normal(size=self.dim)
        self._state = self.rho * self._state + self._innovation_scale * noise
        return self._state.copy()
