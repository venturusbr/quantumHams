from __future__ import annotations

import warnings

import numpy as np

from quantumhams.backends.cudaq_backend import CudaqBackend, is_available
from quantumhams.backends.scipy_backend import ScipyBackend
from quantumhams.core.pauli import PauliHamiltonian
from quantumhams.core.results import EvolutionResult


class AutoBackend:
    """Prefer CUDA-Q Dynamics and safely fall back to SciPy when unavailable."""

    name = "auto"

    def evolve(
        self, hamiltonian: PauliHamiltonian, initial_state: np.ndarray, times: np.ndarray
    ) -> EvolutionResult:
        if is_available():
            try:
                return CudaqBackend().evolve(hamiltonian, initial_state, times)
            except (ImportError, RuntimeError, OSError) as error:
                warnings.warn(
                    f"CUDA-Q execution failed ({error}); falling back to SciPy.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return ScipyBackend().evolve(hamiltonian, initial_state, times)
