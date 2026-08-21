from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from quantumhams.backends import get_backend
from quantumhams.core.pauli import PauliHamiltonian
from quantumhams.core.results import EvolutionResult
from quantumhams.core.states import basis_state, normalize_state


class HamiltonianModel(ABC):
    """Extension point implemented by every physical Hamiltonian model."""

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Number of effective two-level systems."""

    @abstractmethod
    def pauli_hamiltonian(self) -> PauliHamiltonian:
        """Return the backend-neutral Pauli decomposition."""

    def matrix(self, *, sparse: bool = True) -> sparse.csr_matrix | np.ndarray:
        return self.pauli_hamiltonian().matrix(sparse_output=sparse)

    def evolve(
        self,
        initial_state: str | int | np.ndarray,
        times: Sequence[float] | np.ndarray,
        *,
        backend: str = "auto",
        shots: int | None = None,
        seed: int | None = None,
    ) -> EvolutionResult:
        if isinstance(initial_state, (str | int)):
            state = basis_state(initial_state, self.n_qubits)
        else:
            state = normalize_state(initial_state, self.n_qubits)
        result = get_backend(backend).evolve(self.pauli_hamiltonian(), state, np.asarray(times))
        if shots is None:
            return result
        return result.sample(shots, seed=seed)
