from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import quantumhams.backends.cudaq_backend as cudaq_backend
from quantumhams.backends.cudaq_backend import CudaqBackend, _validate_times
from quantumhams.core.pauli import PauliHamiltonian, PauliTerm


class FakeOperator:
    def __init__(self, expression: str) -> None:
        self.expression = expression

    def __mul__(self, other: Any) -> FakeOperator:
        expression = other.expression if isinstance(other, FakeOperator) else repr(other)
        return FakeOperator(f"({self.expression}*{expression})")

    def __rmul__(self, other: Any) -> FakeOperator:
        return FakeOperator(f"({other!r}*{self.expression})")

    def __add__(self, other: Any) -> FakeOperator:
        expression = other.expression if isinstance(other, FakeOperator) else repr(other)
        return FakeOperator(f"({self.expression}+{expression})")


class FakeState:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.complex128)

    @classmethod
    def from_data(cls, data: Any) -> FakeState:
        if hasattr(data, "get"):
            data = data.get()
        return cls(np.asarray(data))

    def to_numpy(self) -> np.ndarray:
        return self._data


class FakeCudaq:
    class IntermediateResultSave:
        ALL = "all"

    class Schedule:
        def __init__(self, times: list[float], labels: list[str]) -> None:
            self.times = times
            self.labels = labels

    class RungeKuttaIntegrator:
        pass

    State = FakeState
    spin = SimpleNamespace(
        x=lambda index: FakeOperator(f"X{index}"),
        y=lambda index: FakeOperator(f"Y{index}"),
        z=lambda index: FakeOperator(f"Z{index}"),
        i=lambda index: FakeOperator(f"I{index}"),
    )

    def __init__(self) -> None:
        self.target: str | None = None
        self.call: dict[str, Any] | None = None

    def set_target(self, target: str) -> None:
        self.target = target

    def evolve(
        self, hamiltonian: Any, dimensions: Any, schedule: Any, state: Any, **kwargs: Any
    ) -> Any:
        self.call = {
            "hamiltonian": hamiltonian,
            "dimensions": dimensions,
            "schedule": schedule,
            "state": state,
            **kwargs,
        }
        print(f"State: {state.to_numpy()}")
        states = [FakeState(state.to_numpy()) for _ in schedule.times]
        result = SimpleNamespace(_intermediate_states=states)

        def intermediate_states(self):
            return self._intermediate_states

        result.intermediate_states = MethodType(intermediate_states, result)

        return result


def test_cudaq_evolve_calls_native_dynamics_api() -> None:
    fake = FakeCudaq()
    backend = CudaqBackend(cudaq_module=fake)
    hamiltonian = PauliHamiltonian(1, [PauliTerm(1.0, {0: "X"})])

    result = backend.evolve(hamiltonian, np.array([1.0, 0.0]), np.array([0.0, 0.5, 1.0]))

    assert fake.target == "dynamics"
    assert fake.call is not None
    assert fake.call["dimensions"] == {0: 2}
    assert fake.call["store_intermediate_results"] == "all"
    assert result.backend == "cudaq"
    assert result.states.shape == (3, 2)


def test_cudaq_reverses_indices_to_preserve_basis_order() -> None:
    fake = FakeCudaq()
    backend = CudaqBackend(cudaq_module=fake)
    hamiltonian = PauliHamiltonian(2, [PauliTerm(1.0, {0: "X", 1: "Z"})])

    operator = backend.to_spin_operator(hamiltonian)

    assert "X1" in operator.expression
    assert "Z0" in operator.expression


def test_is_available_reflects_cudaq_module_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cudaq_backend,
        "find_spec",
        lambda name: object(),
    )

    assert cudaq_backend.is_available() is True

    monkeypatch.setattr(
        cudaq_backend,
        "find_spec",
        lambda name: None,
    )

    assert cudaq_backend.is_available() is False


@pytest.mark.parametrize(
    "times",
    [
        np.array([]),
        np.array([-0.1, 0.0]),
        np.array([0.0, 1.0, 0.5]),
    ],
)
def test_validate_times_rejects_invalid_sequences(
    times: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty, non-negative, sorted",
    ):
        _validate_times(times)


def test_validate_times_flattens_and_converts_to_float() -> None:
    result = _validate_times(
        np.array([[0, 1, 2]], dtype=int),
    )

    np.testing.assert_array_equal(
        result,
        np.array([0.0, 1.0, 2.0]),
    )
    assert result.dtype == float
    assert result.shape == (3,)


def test_constructor_rejects_missing_cudaq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cudaq_backend,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA-Q is not installed",
    ):
        CudaqBackend()


def test_constructor_imports_cudaq_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeCudaq()

    monkeypatch.setattr(
        cudaq_backend,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        cudaq_backend,
        "import_module",
        lambda name: fake,
    )

    backend = CudaqBackend(
        target="custom-target",
    )

    assert backend._cudaq is fake
    assert backend._target == "custom-target"


def test_to_spin_operator_combines_multiple_terms() -> None:
    backend = CudaqBackend(
        cudaq_module=FakeCudaq(),
    )

    hamiltonian = PauliHamiltonian(
        3,
        [
            PauliTerm(
                2.0,
                {
                    0: "X",
                    1: "Y",
                },
            ),
            PauliTerm(
                -0.5,
                {
                    2: "Z",
                },
            ),
        ],
    )

    operator = backend.to_spin_operator(
        hamiltonian,
    )

    # quantumHams index 0 becomes CUDA-Q index 2.
    assert "X2" in operator.expression

    # quantumHams index 1 remains index 1.
    assert "Y1" in operator.expression

    # quantumHams index 2 becomes CUDA-Q index 0.
    assert "Z0" in operator.expression

    # Both Hamiltonian terms were added.
    assert "+" in operator.expression


def test_to_spin_operator_returns_zero_identity_for_empty_hamiltonian() -> None:
    backend = CudaqBackend(
        cudaq_module=FakeCudaq(),
    )

    operator = backend.to_spin_operator(
        PauliHamiltonian(
            n_qubits=1,
            terms=[],
        ),
    )

    assert operator.expression == "(0.0*I0)"


def test_state_from_numpy_uses_numpy_when_cupy_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CudaqBackend(
        cudaq_module=FakeCudaq(),
    )

    def fake_import(name: str) -> Any:
        if name == "cupy":
            raise ImportError

        raise AssertionError(
            f"Unexpected import: {name}",
        )

    monkeypatch.setattr(
        cudaq_backend,
        "import_module",
        fake_import,
    )

    source = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    state = backend._state_from_numpy(
        source,
    )
    result = state.to_numpy()

    assert result.dtype == np.complex128
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(
        result,
        np.array(
            [1.0, 0.0],
            dtype=np.complex128,
        ),
    )


def test_state_from_numpy_uses_cupy_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CudaqBackend(
        cudaq_module=FakeCudaq(),
    )

    calls: list[np.ndarray] = []

    class FakeCupy:
        @staticmethod
        def asarray(
            data: np.ndarray,
        ) -> np.ndarray:
            calls.append(data)
            return data.copy()

    monkeypatch.setattr(
        cudaq_backend,
        "import_module",
        lambda name: FakeCupy,
    )

    state = backend._state_from_numpy(
        np.array([0.0, 1.0]),
    )

    assert len(calls) == 1

    np.testing.assert_array_equal(
        state.to_numpy(),
        np.array(
            [0.0, 1.0],
            dtype=np.complex128,
        ),
    )


def test_state_to_numpy_uses_to_numpy_method() -> None:
    state = FakeState(
        np.array(
            [[1.0, 0.0]],
            dtype=np.complex64,
        ),
    )

    result = CudaqBackend._state_to_numpy(
        state,
    )

    assert result.dtype == np.complex128
    assert result.shape == (2,)

    np.testing.assert_array_equal(
        result,
        np.array(
            [1.0, 0.0],
            dtype=np.complex128,
        ),
    )


def test_state_to_numpy_supports_legacy_array_compatible_state() -> None:
    class LegacyState:
        def __array__(
            self,
            dtype: Any = None,
        ) -> np.ndarray:
            return np.asarray(
                [[1.0, 0.0]],
                dtype=dtype,
            )

    result = CudaqBackend._state_to_numpy(
        LegacyState(),
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [1.0, 0.0],
            dtype=np.complex128,
        ),
    )
    assert result.shape == (2,)


def test_evolve_rejects_wrong_initial_state_dimension() -> None:
    backend = CudaqBackend(
        cudaq_module=FakeCudaq(),
    )

    hamiltonian = PauliHamiltonian(
        n_qubits=2,
        terms=[],
    )

    with pytest.raises(
        ValueError,
        match="dimension 2; expected 4",
    ):
        backend.evolve(
            hamiltonian,
            initial_state=np.array(
                [1.0, 0.0],
            ),
            times=np.array([0.0]),
        )


def test_evolve_uses_supplied_integrator_and_target() -> None:
    fake = FakeCudaq()
    custom_integrator = object()

    backend = CudaqBackend(
        cudaq_module=fake,
        target="dynamics-custom",
        integrator=custom_integrator,
    )

    hamiltonian = PauliHamiltonian(
        1,
        [
            PauliTerm(
                1.0,
                {
                    0: "X",
                },
            ),
        ],
    )

    backend.evolve(
        hamiltonian,
        initial_state=np.array(
            [1.0, 0.0],
        ),
        times=np.array(
            [0.0, 0.5],
        ),
    )

    assert fake.target == "dynamics-custom"
    assert fake.call is not None

    assert fake.call["integrator"] is custom_integrator
    assert fake.call["schedule"].labels == ["t"]
    assert fake.call["observables"] == []
    assert fake.call["collapse_operators"] == []


def test_evolve_uses_default_runge_kutta_integrator() -> None:
    fake = FakeCudaq()

    backend = CudaqBackend(
        cudaq_module=fake,
    )

    hamiltonian = PauliHamiltonian(
        n_qubits=1,
        terms=[],
    )

    backend.evolve(
        hamiltonian,
        initial_state=np.array(
            [1.0, 0.0],
        ),
        times=np.array([0.0]),
    )

    assert fake.call is not None
    assert isinstance(
        fake.call["integrator"],
        FakeCudaq.RungeKuttaIntegrator,
    )


def test_evolve_builds_correct_dimensions_for_multiple_qubits() -> None:
    fake = FakeCudaq()

    backend = CudaqBackend(
        cudaq_module=fake,
    )

    hamiltonian = PauliHamiltonian(
        n_qubits=3,
        terms=[],
    )

    initial_state = np.zeros(
        8,
        dtype=np.complex128,
    )
    initial_state[0] = 1.0

    backend.evolve(
        hamiltonian,
        initial_state=initial_state,
        times=np.array([0.0]),
    )

    assert fake.call is not None
    assert fake.call["dimensions"] == {
        0: 2,
        1: 2,
        2: 2,
    }


def test_evolve_raises_when_intermediate_states_are_missing() -> None:
    fake = FakeCudaq()

    def evolve_without_states(
        hamiltonian: Any,
        dimensions: Any,
        schedule: Any,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        result = SimpleNamespace(_intermediate_states=None)

        def intermediate_states(self):
            return self._intermediate_states

        result.intermediate_states = MethodType(intermediate_states, result)

        return result

    fake.evolve = evolve_without_states

    backend = CudaqBackend(
        cudaq_module=fake,
    )

    hamiltonian = PauliHamiltonian(
        n_qubits=1,
        terms=[],
    )

    with pytest.raises(
        RuntimeError,
        match="did not return intermediate states",
    ):
        backend.evolve(
            hamiltonian,
            initial_state=np.array(
                [1.0, 0.0],
            ),
            times=np.array([0.0]),
        )


def test_evolve_raises_for_unexpected_number_of_states() -> None:
    fake = FakeCudaq()

    def evolve_with_too_few_states(
        hamiltonian: Any,
        dimensions: Any,
        schedule: Any,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        result = SimpleNamespace(
            _intermediate_states=[
                FakeState(
                    [1.0, 0.0],
                ),
            ],
        )

        def intermediate_states(self):
            return self._intermediate_states

        result.intermediate_states = MethodType(intermediate_states, result)

        return result

    fake.evolve = evolve_with_too_few_states

    backend = CudaqBackend(
        cudaq_module=fake,
    )

    hamiltonian = PauliHamiltonian(
        n_qubits=1,
        terms=[],
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected state array shape",
    ):
        backend.evolve(
            hamiltonian,
            initial_state=np.array(
                [1.0, 0.0],
            ),
            times=np.array(
                [0.0, 0.5],
            ),
        )


def test_evolve_raises_for_unexpected_state_vector_dimension() -> None:
    fake = FakeCudaq()

    def evolve_with_wrong_state_dimension(
        hamiltonian: Any,
        dimensions: Any,
        schedule: Any,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        result = SimpleNamespace(
            _intermediate_states=[
                FakeState(
                    [1.0, 0.0, 0.0],
                ),
            ],
        )

        def intermediate_states(self):
            return self._intermediate_states

        result.intermediate_states = MethodType(intermediate_states, result)

        return result

    fake.evolve = evolve_with_wrong_state_dimension

    backend = CudaqBackend(
        cudaq_module=fake,
    )

    hamiltonian = PauliHamiltonian(
        n_qubits=1,
        terms=[],
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected state array shape",
    ):
        backend.evolve(
            hamiltonian,
            initial_state=np.array(
                [1.0, 0.0],
            ),
            times=np.array([0.0]),
        )
