#!/usr/bin/env python3
"""Generate a two-qubit Bell state with quantumHams.

Examples
--------
Neutral atoms on the CPU:

    python examples/bell_state.py \
        --technology neutral-atom \
        --compute cpu

Trapped ions using CUDA-Q:

    python examples/bell_state.py \
        --technology trapped-ion \
        --compute gpu

Automatically use CUDA-Q when available, otherwise SciPy:

    python examples/bell_state.py \
        --technology neutral-atom \
        --compute auto

Use a quantumHams repository outside the current directory:

    python examples/bell_state.py \
        --quantumhams-path /path/to/quantumHams \
        --technology trapped-ion \
        --compute cpu
"""

from __future__ import annotations

import argparse
from typing import Literal

import numpy as np

from quantumhams import (
    NeutralAtomRegister,
    RubidiumRydbergHamiltonian,
    YtterbiumIonChain,
    YtterbiumMolmerSorensenHamiltonian,
)

Technology = Literal["neutral-atom", "trapped-ion"]
ComputeBackend = Literal["cpu", "gpu", "auto"]


def select_backend(compute: ComputeBackend) -> str:
    """Map the requested infrastructure to a quantumHams backend."""
    backend_map = {
        "cpu": "scipy",
        "gpu": "cudaq",
        "auto": "auto",
    }

    return backend_map[compute]


def normalize_global_phase(
    state: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Remove an irrelevant global phase for easier visualization."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)

    nonzero_indices = np.flatnonzero(
        np.abs(state) > tolerance,
    )

    if len(nonzero_indices) == 0:
        return state

    first_amplitude = state[nonzero_indices[0]]
    phase = np.exp(-1j * np.angle(first_amplitude))

    return state * phase


def state_fidelity(
    state: np.ndarray,
    target: np.ndarray,
) -> float:
    """Calculate the pure-state fidelity |<target|state>|²."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    target = np.asarray(target, dtype=np.complex128).reshape(-1)

    state = state / np.linalg.norm(state)
    target = target / np.linalg.norm(target)

    return float(np.abs(np.vdot(target, state)) ** 2)


def create_neutral_atom_bell_state(
    backend: str,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Create the symmetric Bell state with two Rydberg atoms.

    The target state is:

        (|01> + |10>) / sqrt(2)

    Under strong Rydberg blockade, |11> is energetically suppressed. Starting
    from |00>, the global Rabi drive couples |00> to the symmetric single-
    excitation state.

    For the quantumHams convention

        H_drive = (Omega / 2) * (X_0 + X_1),

    the effective coupling between |00> and the symmetric Bell state is

        Omega / sqrt(2).

    Therefore, the ideal Bell-state preparation time is

        t_final = pi / (sqrt(2) * Omega).
    """

    omega = 2.0 * np.pi

    # A strong interaction suppresses the doubly excited |11> state.
    blockade_interaction = 100.0 * omega

    spacing = 1.0

    # Since V = C6 / r^6 and r = 1, C6 equals the desired blockade energy.
    c6 = blockade_interaction * spacing**6

    register = NeutralAtomRegister.linear(
        n_atoms=2,
        spacing=spacing,
    )

    hamiltonian = RubidiumRydbergHamiltonian(
        atom_register=register,
        rabi_frequency=omega,
        detuning=0.0,
        c6=c6,
    )

    final_time = np.pi / (np.sqrt(2.0) * omega)

    times = np.linspace(
        0.0,
        final_time,
        samples,
    )

    result = hamiltonian.evolve(
        initial_state="00",
        times=times,
        backend=backend,
    )

    target = np.array(
        [
            0.0,
            1.0 / np.sqrt(2.0),
            1.0 / np.sqrt(2.0),
            0.0,
        ],
        dtype=np.complex128,
    )

    return result.states[-1], target, result.backend


def create_trapped_ion_bell_state(
    backend: str,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Create a Bell state with a two-ion XX interaction.

    Starting from |00>, the Hamiltonian

        H = J X_0 X_1

    produces

        |psi(t)> = cos(Jt)|00> - i sin(Jt)|11>.

    At

        t_final = pi / (4J),

    the final state is

        (|00> - i|11>) / sqrt(2).
    """

    coupling = 2.0 * np.pi

    chain = YtterbiumIonChain.linear(
        n_ions=2,
        spacing=1.0,
    )

    coupling_matrix = np.array(
        [
            [0.0, coupling],
            [coupling, 0.0],
        ],
        dtype=float,
    )

    hamiltonian = YtterbiumMolmerSorensenHamiltonian(
        chain=chain,
        coupling_matrix=coupling_matrix,
        transverse_field=0.0,
    )

    final_time = np.pi / (4.0 * coupling)

    times = np.linspace(
        0.0,
        final_time,
        samples,
    )

    result = hamiltonian.evolve(
        initial_state="00",
        times=times,
        backend=backend,
    )

    target = np.array(
        [
            1.0 / np.sqrt(2.0),
            0.0,
            0.0,
            -1.0j / np.sqrt(2.0),
        ],
        dtype=np.complex128,
    )

    return result.states[-1], target, result.backend


def print_state_vector(
    state: np.ndarray,
    *,
    title: str,
) -> None:
    """Print a two-qubit state vector with basis labels."""
    basis_labels = (
        "|00>",
        "|01>",
        "|10>",
        "|11>",
    )

    state = np.asarray(
        state,
        dtype=np.complex128,
    ).reshape(-1)

    print(f"\n{title}")
    print("-" * len(title))

    for label, amplitude in zip(
        basis_labels,
        state,
        strict=True,
    ):
        probability = float(np.abs(amplitude) ** 2)

        print(
            f"{label}: "
            f"{amplitude.real:+.8f} "
            f"{amplitude.imag:+.8f}j"
            f"    probability={probability:.8f}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create a two-qubit Bell state using a local quantumHams repository."),
    )

    parser.add_argument(
        "--technology",
        choices=(
            "neutral-atom",
            "trapped-ion",
        ),
        default="neutral-atom",
        help="Quantum-computer technology to simulate.",
    )

    parser.add_argument(
        "--compute",
        choices=(
            "cpu",
            "gpu",
            "auto",
        ),
        default="auto",
        help=(
            "Execution infrastructure: CPU uses SciPy, GPU uses CUDA-Q, "
            "and auto attempts CUDA-Q before falling back to SciPy."
        ),
    )

    parser.add_argument(
        "--quantumhams-path",
        default=".",
        help=(
            "Path to the local quantumHams repository root. The default is the current directory."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=101,
        help="Number of time points in the evolution.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.samples < 2:
        raise ValueError("--samples must be at least 2.")

    backend = select_backend(
        args.compute,
    )

    print("quantumHams Bell-state example")
    print("==============================")
    print(f"Technology       : {args.technology}")
    print(f"Requested compute: {args.compute}")
    print(f"Requested backend: {backend}")

    if args.technology == "neutral-atom":
        final_state, target, actual_backend = create_neutral_atom_bell_state(
            backend=backend,
            samples=args.samples,
        )
        target_name = "(|01> + |10>) / sqrt(2)"
    else:
        final_state, target, actual_backend = create_trapped_ion_bell_state(
            backend=backend,
            samples=args.samples,
        )
        target_name = "(|00> - i|11>) / sqrt(2)"

    fidelity = state_fidelity(
        final_state,
        target,
    )

    print(f"Actual backend   : {actual_backend}")
    print(f"Target Bell state: {target_name}")

    print_state_vector(
        final_state,
        title="Raw final state vector",
    )

    phase_normalized_state = normalize_global_phase(
        final_state,
    )

    print_state_vector(
        phase_normalized_state,
        title="Final state vector with global phase removed",
    )

    print("\nCompact NumPy state vector")
    print("--------------------------")
    print(
        np.array2string(
            final_state,
            precision=8,
            suppress_small=True,
        )
    )

    print(f"\nBell-state fidelity: {fidelity:.10f}")

    if fidelity < 0.95:
        print(
            "\nWarning: the fidelity is below 0.95. "
            "For the neutral-atom model, increase the blockade interaction "
            "or numerically optimize the final evolution time."
        )


if __name__ == "__main__":
    main()
