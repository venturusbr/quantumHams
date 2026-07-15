import numpy as np
import pytest

from quantumhams.core.states import basis_state, normalize_state


def test_basis_state() -> None:
    np.testing.assert_array_equal(
        basis_state("10", 2),
        np.array([0, 0, 1, 0], dtype=complex),
    )


def test_basis_state_accepts_integer_index() -> None:
    state = basis_state(3, 2)

    np.testing.assert_array_equal(
        state,
        np.array([0, 0, 0, 1], dtype=complex),
    )
    assert state.dtype == complex
    assert np.linalg.norm(state) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value",
    [
        "0",  # Too short.
        "000",  # Too long.
        "0a",  # Invalid character.
        "12",  # Invalid binary digit.
        "-1",  # Invalid characters.
    ],
)
def test_basis_state_rejects_invalid_bit_strings(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="Expected a 2-character bit string",
    ):
        basis_state(value, 2)


@pytest.mark.parametrize("index", [-1, 4, 5])
def test_basis_state_rejects_index_outside_hilbert_space(
    index: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Basis-state index is outside the Hilbert space",
    ):
        basis_state(index, 2)


def test_normalize_state_flattens_and_normalizes_input() -> None:
    state = np.array([[3.0], [4.0]])

    normalized = normalize_state(state, n_qubits=1)

    np.testing.assert_allclose(
        normalized,
        np.array([0.6, 0.8], dtype=complex),
    )
    assert normalized.shape == (2,)
    assert normalized.dtype == complex
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


def test_normalize_state_preserves_normalized_complex_state() -> None:
    state = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)

    normalized = normalize_state(state, n_qubits=1)

    np.testing.assert_allclose(normalized, state)


def test_normalize_state_rejects_incompatible_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="Initial state has an incompatible dimension",
    ):
        normalize_state(np.ones(3), n_qubits=2)


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(2),
        np.array([0.0, 0.0j]),
        np.zeros((2, 1)),
    ],
)
def test_normalize_state_rejects_zero_vector(
    state: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="Initial state cannot be the zero vector",
    ):
        normalize_state(state, n_qubits=1)
