# Changelog

## 0.1.0 - Unreleased

- Initial extensible Hamiltonian model API.
- Rubidium Rydberg neutral-atom model.
- Ytterbium trapped-ion effective Ising model.
- SciPy CPU simulation and optional CUDA-Q operator adapter.

## 0.2.0

- Implement native time evolution with `cudaq.evolve` and the CUDA-Q Dynamics target.
- Preserve quantumHams basis ordering when converting Pauli operators to CUDA-Q.
- Add automatic runtime fallback to SciPy when CUDA-Q is missing or unusable.
- Record the backend used in `EvolutionResult.backend`.
