from __future__ import annotations

from typing import Protocol

import numpy as np

from quantumhams.core.pauli import PauliHamiltonian
from quantumhams.core.results import EvolutionResult


class SimulationBackend(Protocol):
    """Execution backend contract."""

    name: str

    def evolve(
        self, hamiltonian: PauliHamiltonian, initial_state: np.ndarray, times: np.ndarray
    ) -> EvolutionResult: ...
