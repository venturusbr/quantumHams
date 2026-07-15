from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantumhams.core.model import HamiltonianModel
from quantumhams.core.pauli import PauliHamiltonian, PauliTerm


@dataclass(frozen=True, slots=True)
class NeutralAtomRegister:
    """Positions of neutral atoms, commonly expressed in micrometres."""

    positions: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] not in (1, 2, 3):
            raise ValueError(
                "positions must have shape (n_atoms, dimension), dimension in {1,2,3}."
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite.")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError("Atom positions must be unique.")
        object.__setattr__(self, "positions", positions)

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

    @classmethod
    def linear(cls, n_atoms: int, spacing: float) -> NeutralAtomRegister:
        if n_atoms <= 0 or spacing <= 0:
            raise ValueError("n_atoms and spacing must be positive.")
        return cls(np.array([[index * spacing, 0.0] for index in range(n_atoms)]))

    @classmethod
    def square(cls, rows: int, columns: int, spacing: float) -> NeutralAtomRegister:
        if rows <= 0 or columns <= 0 or spacing <= 0:
            raise ValueError("rows, columns, and spacing must be positive.")
        return cls(
            np.array(
                [
                    [column * spacing, row * spacing]
                    for row in range(rows)
                    for column in range(columns)
                ]
            )
        )


@dataclass(frozen=True, slots=True)
class RubidiumRydbergHamiltonian(HamiltonianModel):
    r"""Effective two-level Rydberg Hamiltonian.

    H = Ω/2 Σ X_i - Σ Δ_i n_i + Σ_{i<j} C6/r_ij^6 n_i n_j,
    where n_i = (I - Z_i)/2.
    """

    atom_register: NeutralAtomRegister
    rabi_frequency: float
    detuning: float | np.ndarray
    c6: float

    @property
    def n_qubits(self) -> int:
        return self.atom_register.n_atoms

    def pauli_hamiltonian(self) -> PauliHamiltonian:
        detuning = np.broadcast_to(np.asarray(self.detuning, dtype=float), (self.n_qubits,))
        terms: list[PauliTerm] = []
        constant = 0.0
        for index, delta in enumerate(detuning):
            terms.append(PauliTerm(self.rabi_frequency / 2, {index: "X"}))
            constant -= delta / 2
            terms.append(PauliTerm(delta / 2, {index: "Z"}))
        for left in range(self.n_qubits):
            for right in range(left + 1, self.n_qubits):
                distance = np.linalg.norm(
                    self.atom_register.positions[left] - self.atom_register.positions[right]
                )
                interaction = self.c6 / distance**6
                constant += interaction / 4
                terms.extend(
                    [
                        PauliTerm(-interaction / 4, {left: "Z"}),
                        PauliTerm(-interaction / 4, {right: "Z"}),
                        PauliTerm(interaction / 4, {left: "Z", right: "Z"}),
                    ]
                )
        if constant:
            terms.append(PauliTerm(constant, {}))
        return PauliHamiltonian(self.n_qubits, terms)
