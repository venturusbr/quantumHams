from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np

from quantumhams.core.pauli import PauliHamiltonian
from quantumhams.core.results import EvolutionResult


def is_available() -> bool:
    """Return whether the optional CUDA-Q Python package can be imported."""

    return find_spec("cudaq") is not None


def _validate_times(times: np.ndarray) -> np.ndarray:
    sample_times = np.asarray(times, dtype=float).reshape(-1)
    if sample_times.size == 0 or np.any(sample_times < 0) or np.any(np.diff(sample_times) < 0):
        raise ValueError("times must be a non-empty, non-negative, sorted sequence.")
    return sample_times


class CudaqBackend:
    """GPU-accelerated closed-system evolution through CUDA-Q Dynamics.

    The backend converts the backend-neutral :class:`PauliHamiltonian` into a
    CUDA-Q spin operator and solves the Schrödinger equation with
    ``cudaq.evolve`` on the ``dynamics`` target. Intermediate states are saved
    at every requested time and returned as NumPy state vectors.

    Args:
        target: CUDA-Q target used for dynamics evolution.
        integrator: Optional CUDA-Q integrator instance. By default CUDA-Q's
            ``RungeKuttaIntegrator`` is used.
        cudaq_module: Internal dependency-injection hook used by tests.
    """

    name = "cudaq"

    def __init__(
        self,
        *,
        target: str = "dynamics",
        integrator: Any | None = None,
        cudaq_module: Any | None = None,
    ) -> None:
        if cudaq_module is None and not is_available():
            raise RuntimeError(
                "CUDA-Q is not installed. Install quantumhams[cudaq] or use backend='scipy'."
            )
        self._cudaq = cudaq_module or import_module("cudaq")
        self._target = target
        self._integrator = integrator

    def to_spin_operator(self, hamiltonian: PauliHamiltonian) -> Any:
        """Convert a quantumHams Pauli decomposition to a CUDA-Q spin operator.

        quantumHams treats qubit 0 as the left-most tensor factor, whereas
        CUDA-Q state arrays use qubit 0 as the least-significant/right-most
        factor. Reversing the operator indices preserves quantumHams basis-state
        ordering in returned NumPy arrays.
        """

        cudaq = self._cudaq
        result: Any = None
        for term in hamiltonian.terms:
            operator: Any = term.coefficient
            for index, symbol in sorted(term.operators.items()):
                cudaq_index = hamiltonian.n_qubits - 1 - index
                operator = operator * getattr(cudaq.spin, symbol.lower())(cudaq_index)
            result = operator if result is None else result + operator
        return result if result is not None else 0.0 * cudaq.spin.i(0)

    def _state_from_numpy(self, state: np.ndarray) -> Any:
        contiguous = np.ascontiguousarray(state, dtype=np.complex128)
        try:
            cupy = import_module("cupy")
        except ImportError:
            data: Any = contiguous
        else:
            data = cupy.asarray(contiguous)
        return self._cudaq.State.from_data(data)

    @staticmethod
    def _state_to_numpy(state: Any) -> np.ndarray:
        # if hasattr(state, "to_numpy"):
        #     array = state.to_numpy()
        # else:  # Compatibility with older CUDA-Q State objects.
        #     array = np.asarray(state)
        array = state.to_numpy() if hasattr(state, "to_numpy") else np.asarray(state)
        return np.asarray(array, dtype=np.complex128).reshape(-1)

    def evolve(
        self, hamiltonian: PauliHamiltonian, initial_state: np.ndarray, times: np.ndarray
    ) -> EvolutionResult:
        """Evolve a pure state natively with CUDA-Q Dynamics."""

        sample_times = _validate_times(times)
        expected_dimension = 2**hamiltonian.n_qubits
        state_vector = np.asarray(initial_state, dtype=np.complex128).reshape(-1)
        if state_vector.size != expected_dimension:
            raise ValueError(
                f"initial_state has dimension {state_vector.size}; expected {expected_dimension}."
            )

        cudaq = self._cudaq
        cudaq.set_target(self._target)
        schedule = cudaq.Schedule(sample_times.tolist(), ["t"])
        dimensions = {index: 2 for index in range(hamiltonian.n_qubits)}
        native_hamiltonian = self.to_spin_operator(hamiltonian)
        native_state = self._state_from_numpy(state_vector)
        integrator = self._integrator or cudaq.RungeKuttaIntegrator()

        result = cudaq.evolve(
            native_hamiltonian,
            dimensions,
            schedule,
            native_state,
            observables=[],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.ALL,
            integrator=integrator,
        )

        native_states = result.intermediate_states()
        if native_states is None:
            raise RuntimeError(
                "CUDA-Q did not return intermediate states. Ensure the installed CUDA-Q "
                "version supports IntermediateResultSave.ALL."
            )
        states = np.vstack([self._state_to_numpy(state) for state in native_states])
        if states.shape != (sample_times.size, expected_dimension):
            raise RuntimeError(
                "CUDA-Q returned an unexpected state array shape: "
                f"{states.shape}; expected {(sample_times.size, expected_dimension)}."
            )
        return EvolutionResult(times=sample_times, states=states, backend=self.name)
