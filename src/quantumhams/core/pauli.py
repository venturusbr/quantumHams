"""Backend-neutral Pauli representation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import reduce

import numpy as np
from scipy import sparse

_PAULI = {
    "I": sparse.csr_matrix(np.eye(2, dtype=complex)),
    "X": sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    "Y": sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    "Z": sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
}


def _kron_csr(
    left: sparse.csr_matrix,
    right: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Compute a Kronecker product while preserving the CSR matrix type."""
    return sparse.csr_matrix(sparse.kron(left, right, format="csr"))


@dataclass(frozen=True, slots=True)
class PauliTerm:
    """A coefficient multiplied by a tensor product of Pauli operators."""

    coefficient: complex
    operators: Mapping[int, str]

    def __post_init__(self) -> None:
        normalized = {int(index): operator.upper() for index, operator in self.operators.items()}
        if any(index < 0 for index in normalized):
            raise ValueError("Qubit indices must be non-negative.")
        invalid = set(normalized.values()) - {"X", "Y", "Z"}
        if invalid:
            raise ValueError(f"Unsupported Pauli operators: {sorted(invalid)}")
        object.__setattr__(self, "operators", normalized)

    def matrix(self, n_qubits: int) -> sparse.csr_matrix:
        if self.operators and max(self.operators) >= n_qubits:
            raise ValueError("A Pauli term references a qubit outside the system.")
        factors = [_PAULI[self.operators.get(index, "I")] for index in range(n_qubits)]
        identity: sparse.csr_matrix = sparse.csr_matrix([[1.0 + 0.0j]])
        product = reduce(_kron_csr, factors, identity)
        return (self.coefficient * product).tocsr()


@dataclass(frozen=True, slots=True)
class PauliHamiltonian:
    """Finite-dimensional qubit Hamiltonian represented by Pauli terms."""

    n_qubits: int
    terms: tuple[PauliTerm, ...]

    def __init__(self, n_qubits: int, terms: Iterable[PauliTerm] = ()) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "terms", tuple(terms))

    def matrix(self, *, sparse_output: bool = True) -> sparse.csr_matrix | np.ndarray:
        dimension = 2**self.n_qubits
        result = sparse.csr_matrix((dimension, dimension), dtype=complex)
        for term in self.terms:
            result = result + term.matrix(self.n_qubits)
        return result if sparse_output else result.toarray()
