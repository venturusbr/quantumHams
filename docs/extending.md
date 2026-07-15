# Extending quantumHams

1. Create a technology-specific package under `src/quantumhams/`.
2. Implement a class derived from `HamiltonianModel`.
3. Expose `n_qubits` and `pauli_hamiltonian()`.
4. Represent the physics using `PauliTerm` objects.
5. Add analytical unit tests for small systems and numerical tests for invariants.
6. Export the stable public class from `quantumhams.__init__`.

Backends consume only `PauliHamiltonian`, so a new physical model does not need backend-specific code. A new backend implements the `SimulationBackend` protocol and is registered in `backends/registry.py`.
