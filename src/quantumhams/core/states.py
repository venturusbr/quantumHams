from __future__ import annotations

import numpy as np


def basis_state(value: str | int, n_qubits: int) -> np.ndarray:
    """Create a normalized computational-basis state vector."""
    dimension = 2**n_qubits
    if isinstance(value, str):
        if len(value) != n_qubits or set(value) - {"0", "1"}:
            raise ValueError(f"Expected a {n_qubits}-character bit string.")
        index = int(value, 2)
    else:
        index = value
    if not 0 <= index < dimension:
        raise ValueError("Basis-state index is outside the Hilbert space.")
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


def normalize_state(state: np.ndarray, n_qubits: int) -> np.ndarray:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    if vector.size != 2**n_qubits:
        raise ValueError("Initial state has an incompatible dimension.")
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("Initial state cannot be the zero vector.")
    return vector / norm
