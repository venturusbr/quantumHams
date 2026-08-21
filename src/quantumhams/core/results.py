from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """State vectors and optional computational-basis samples over time."""

    times: np.ndarray
    states: np.ndarray
    backend: str = "unknown"
    samples: np.ndarray | None = None

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.states) ** 2

    @property
    def shots(self) -> int | None:
        """Number of measurement shots, or ``None`` when sampling was not requested."""

        return None if self.samples is None else int(self.samples.shape[1])

    @property
    def bitstrings(self) -> np.ndarray | None:
        """Measurement outcomes as an array shaped ``(len(times), shots)``."""

        if self.samples is None:
            return None
        n_qubits = (self.states.shape[1] - 1).bit_length()
        formatter = np.vectorize(
            lambda sample: format(sample, f"0{n_qubits}b"), otypes=[f"U{n_qubits}"]
        )
        return np.asarray(formatter(self.samples), dtype=str)

    @property
    def counts(self) -> tuple[dict[str, int], ...] | None:
        """Computational-basis counts for every requested evolution time."""

        bitstrings = self.bitstrings
        if bitstrings is None:
            return None
        return tuple(dict(Counter(row.tolist())) for row in bitstrings)

    def sample(self, shots: int, *, seed: int | None = None) -> EvolutionResult:
        """Return this result with finite-shot computational-basis measurements."""

        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer.")
        rng = np.random.default_rng(seed)
        samples = np.vstack(
            [
                rng.choice(probability.size, size=shots, p=probability / probability.sum())
                for probability in self.probabilities
            ]
        )
        return EvolutionResult(
            times=self.times,
            states=self.states,
            backend=self.backend,
            samples=samples,
        )
