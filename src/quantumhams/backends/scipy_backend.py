from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from quantumhams.core.pauli import PauliHamiltonian
from quantumhams.core.results import EvolutionResult


class ScipyBackend:
    """Portable CPU backend based on SciPy sparse linear algebra."""

    name = "scipy"

    def evolve(
        self, hamiltonian: PauliHamiltonian, initial_state: np.ndarray, times: np.ndarray
    ) -> EvolutionResult:
        sample_times = np.asarray(times, dtype=float).reshape(-1)
        if sample_times.size == 0 or np.any(sample_times < 0) or np.any(np.diff(sample_times) < 0):
            raise ValueError("times must be a non-empty, non-negative, sorted sequence.")
        matrix = hamiltonian.matrix(sparse_output=True)
        states = np.vstack(
            [expm_multiply((-1j * time) * matrix, initial_state) for time in sample_times]
        )
        return EvolutionResult(times=sample_times, states=states, backend=self.name)
