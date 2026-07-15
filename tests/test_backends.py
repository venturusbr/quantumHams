import numpy as np
import pytest

from quantumhams import NeutralAtomRegister, RubidiumRydbergHamiltonian
from quantumhams.backends import backend_status, get_backend
from quantumhams.backends.cudaq_backend import is_available


def test_scipy_evolution_preserves_norm() -> None:
    model = RubidiumRydbergHamiltonian(NeutralAtomRegister.linear(1, 1.0), 1.0, 0.0, 0.0)
    result = model.evolve("0", [0.0, 0.5, 1.0])
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)


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
