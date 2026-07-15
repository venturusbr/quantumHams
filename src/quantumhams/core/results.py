from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """State vectors and measurement probabilities sampled over time."""

    times: np.ndarray
    states: np.ndarray
    backend: str = "unknown"

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.states) ** 2
