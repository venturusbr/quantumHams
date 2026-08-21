#!/usr/bin/env python3
"""Sample one- and three-qubit superpositions with both Hamiltonian models.

Examples
--------
Run all four examples on the CPU with 1,000 shots:

    python examples/superposition_shots.py --compute cpu --shots 1000

Use CUDA-Q when available and otherwise fall back to SciPy:

    python examples/superposition_shots.py --compute auto --shots 1000
"""

from __future__ import annotations

import argparse
from typing import Literal

import numpy as np

from quantumhams import (
    EvolutionResult,
    NeutralAtomRegister,
    RubidiumRydbergHamiltonian,
    YtterbiumIonChain,
    YtterbiumMolmerSorensenHamiltonian,
)

ComputeBackend = Literal["cpu", "gpu", "auto"]


def select_backend(compute: ComputeBackend) -> str:
    """Map a user-facing compute choice to a quantumHams backend."""
    return {"cpu": "scipy", "gpu": "cudaq", "auto": "auto"}[compute]


def neutral_atom_superposition(
    n_qubits: int,
    *,
    backend: str,
    shots: int,
    seed: int,
) -> EvolutionResult:
    """Create an equal superposition using independent Rydberg Rabi drives."""
    omega = 2.0 * np.pi
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(n_atoms=n_qubits, spacing=1.0),
        rabi_frequency=omega,
        detuning=0.0,
        c6=0.0,
    )

    # exp(-i * Omega * t * X / 2)|0> is an equal superposition at this time.
    final_time = np.pi / (2.0 * omega)
    return model.evolve(
        initial_state="0" * n_qubits,
        times=[final_time],
        backend=backend,
        shots=shots,
        seed=seed,
    )


def trapped_ion_superposition(
    n_qubits: int,
    *,
    backend: str,
    shots: int,
    seed: int,
) -> EvolutionResult:
    """Evolve an equal superposition under a Mølmer–Sørensen Hamiltonian."""
    coupling = 2.0 * np.pi
    model = YtterbiumMolmerSorensenHamiltonian.power_law(
        chain=YtterbiumIonChain.linear(n_ions=n_qubits, spacing=1.0),
        coupling=coupling,
        exponent=0.0,
        transverse_field=0.0,
    )

    # |+>^n is an eigenstate of every X_i X_j term, so the interaction only
    # adds a global phase and its computational-basis probabilities stay uniform.
    dimension = 2**n_qubits
    initial_state = np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)
    return model.evolve(
        initial_state=initial_state,
        times=[1.0 / (8.0 * coupling)],
        backend=backend,
        shots=shots,
        seed=seed,
    )


def print_result(technology: str, n_qubits: int, result: EvolutionResult) -> None:
    """Print exact probabilities alongside finite-shot measurement counts."""
    probabilities = result.probabilities[-1]
    counts = result.counts
    if counts is None:
        raise RuntimeError("The evolution result does not contain shot samples.")

    print(f"\n{technology}: {n_qubits} qubit{'s' if n_qubits != 1 else ''}")
    print(f"Backend: {result.backend}")
    print("Basis state  Exact probability  Shot count")
    for basis_index, probability in enumerate(probabilities):
        bitstring = format(basis_index, f"0{n_qubits}b")
        print(f"|{bitstring}>       {probability:17.6f}  {counts[-1].get(bitstring, 0):10d}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one- and three-qubit superpositions with both technologies."
    )
    parser.add_argument(
        "--compute",
        choices=("cpu", "gpu", "auto"),
        default="auto",
        help="CPU uses SciPy, GPU uses CUDA-Q, and auto tries CUDA-Q then SciPy.",
    )
    parser.add_argument("--shots", type=int, default=1_000, help="Measurements per example.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shot sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.shots <= 0:
        raise ValueError("--shots must be a positive integer.")

    backend = select_backend(args.compute)
    print("quantumHams superposition shot-sampling example")
    print("================================================")
    print(f"Requested compute: {args.compute}")
    print(f"Shots per case  : {args.shots}")

    for n_qubits in (1, 3):
        neutral_result = neutral_atom_superposition(
            n_qubits, backend=backend, shots=args.shots, seed=args.seed
        )
        print_result("Neutral atom", n_qubits, neutral_result)

        ion_result = trapped_ion_superposition(
            n_qubits, backend=backend, shots=args.shots, seed=args.seed
        )
        print_result("Trapped ion", n_qubits, ion_result)


if __name__ == "__main__":
    main()
