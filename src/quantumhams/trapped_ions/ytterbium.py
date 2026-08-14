from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantumhams.core.model import HamiltonianModel
from quantumhams.core.pauli import PauliHamiltonian, PauliTerm


@dataclass(frozen=True, slots=True)
class YtterbiumIonChain:
    """Effective positions of ytterbium ions in a trap."""

    positions: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[0] == 0:
            raise ValueError("positions must be a non-empty two-dimensional array.")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError("Ion positions must be unique.")
        object.__setattr__(self, "positions", positions)

    @property
    def n_ions(self) -> int:
        return int(self.positions.shape[0])

    @classmethod
    def linear(cls, n_ions: int, spacing: float) -> YtterbiumIonChain:
        if n_ions <= 0 or spacing <= 0:
            raise ValueError("n_ions and spacing must be positive.")
        return cls(np.array([[index * spacing] for index in range(n_ions)]))


@dataclass(frozen=True, slots=True)
class YtterbiumMolmerSorensenHamiltonian(HamiltonianModel):
    r"""Effective long-range Ising Hamiltonian for a Yb trapped-ion chain.

    H = Σ_{i<j} J_ij X_i X_j + B Σ_i Z_i.
    """

    chain: YtterbiumIonChain
    coupling_matrix: np.ndarray
    transverse_field: float | np.ndarray = 0.0

    def __post_init__(self) -> None:
        couplings = np.asarray(self.coupling_matrix, dtype=float)
        expected = (self.chain.n_ions, self.chain.n_ions)
        if couplings.shape != expected or not np.allclose(couplings, couplings.T):
            raise ValueError(f"coupling_matrix must be symmetric with shape {expected}.")
        if not np.allclose(np.diag(couplings), 0.0):
            raise ValueError("coupling_matrix diagonal must be zero.")
        object.__setattr__(self, "coupling_matrix", couplings)

    @property
    def n_qubits(self) -> int:
        return self.chain.n_ions

    @classmethod
    def power_law(
        cls,
        chain: YtterbiumIonChain,
        coupling: float,
        exponent: float,
        transverse_field: float | np.ndarray = 0.0,
    ) -> YtterbiumMolmerSorensenHamiltonian:
        if exponent < 0:
            raise ValueError("exponent must be non-negative.")
        matrix = np.zeros((chain.n_ions, chain.n_ions))
        for left in range(chain.n_ions):
            for right in range(left + 1, chain.n_ions):
                distance = np.linalg.norm(chain.positions[left] - chain.positions[right])
                matrix[left, right] = matrix[right, left] = coupling / distance**exponent
        return cls(chain, matrix, transverse_field)

    def pauli_hamiltonian(self) -> PauliHamiltonian:
        fields = np.broadcast_to(np.asarray(self.transverse_field, dtype=float), (self.n_qubits,))
        terms = [PauliTerm(field, {index: "Z"}) for index, field in enumerate(fields) if field]
        for left in range(self.n_qubits):
            for right in range(left + 1, self.n_qubits):
                coupling = self.coupling_matrix[left, right]
                if coupling:
                    terms.append(PauliTerm(coupling, {left: "X", right: "X"}))
        return PauliHamiltonian(self.n_qubits, terms)
