import numpy as np
import pytest

from quantumhams import PauliHamiltonian, PauliTerm


def test_single_x_matrix() -> None:
    matrix = PauliHamiltonian(1, [PauliTerm(2.0, {0: "X"})]).matrix(sparse_output=False)
    np.testing.assert_allclose(matrix, [[0, 2], [2, 0]])


def test_invalid_pauli() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        PauliTerm(1.0, {0: "A"})


def test_invalid_qubit_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        PauliHamiltonian(0)
