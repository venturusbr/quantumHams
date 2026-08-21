import numpy as np
import pytest

from quantumhams import NeutralAtomRegister, RubidiumRydbergHamiltonian
from quantumhams.backends import backend_status, get_backend
from quantumhams.backends.cudaq_backend import is_available


def test_scipy_evolution_preserves_norm() -> None:
    model = RubidiumRydbergHamiltonian(NeutralAtomRegister.linear(1, 1.0), 1.0, 0.0, 0.0)
    result = model.evolve("0", [0.0, 0.5, 1.0])
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_evolution_can_return_shots_and_state_vectors() -> None:
    model = RubidiumRydbergHamiltonian(NeutralAtomRegister.linear(1, 1.0), 0.0, 0.0, 0.0)
    result = model.evolve("0", [0.0, 1.0], backend="scipy", shots=20, seed=7)

    assert result.states.shape == (2, 2)
    assert result.samples is not None
    assert result.samples.shape == (2, 20)
    assert result.bitstrings is not None
    assert result.bitstrings.tolist() == [["0"] * 20, ["0"] * 20]
    assert result.counts == ({"0": 20}, {"0": 20})
    assert result.shots == 20


def test_evolution_without_shots_is_backward_compatible() -> None:
    model = RubidiumRydbergHamiltonian(NeutralAtomRegister.linear(1, 1.0), 0.0, 0.0, 0.0)
    result = model.evolve("0", [0.0], backend="scipy")

    assert result.samples is None
    assert result.bitstrings is None
    assert result.counts is None
    assert result.shots is None


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_evolution_rejects_invalid_shots(shots: object) -> None:
    model = RubidiumRydbergHamiltonian(NeutralAtomRegister.linear(1, 1.0), 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="positive integer"):
        model.evolve("0", [0.0], backend="scipy", shots=shots)  # type: ignore[arg-type]


def test_auto_backend_is_available() -> None:
    assert get_backend("auto").name == "auto"
    assert backend_status()["scipy"] is True


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_backend("other")


def test_cudaq_missing_error() -> None:
    if not is_available():
        with pytest.raises(RuntimeError, match="not installed"):
            get_backend("cudaq")
