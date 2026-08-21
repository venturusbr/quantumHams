# Changelog

## 0.3.0 - 2026-08-21

- Add optional `shots` and `seed` arguments to `HamiltonianModel.evolve()`.
- Add sampled outcomes, bitstrings, counts, and shot metadata to `EvolutionResult` while
  preserving state-vector results and backward compatibility.

## 0.2.0

- Implement native time evolution with `cudaq.evolve` and the CUDA-Q Dynamics target.
- Preserve quantumHams basis ordering when converting Pauli operators to CUDA-Q.
- Add automatic runtime fallback to SciPy when CUDA-Q is missing or unusable.
- Record the backend used in `EvolutionResult.backend`.

## 0.1.0 - Unreleased

- Initial extensible Hamiltonian model API.
- Rubidium Rydberg neutral-atom model.
- Ytterbium trapped-ion effective Ising model.
- SciPy CPU simulation and optional CUDA-Q operator adapter.
