import numpy as np
import pytest

from quantumhams import (
    YtterbiumIonChain,
    YtterbiumMolmerSorensenHamiltonian,
)


def test_chain_converts_positions_to_float_array_and_reports_size() -> None:
    chain = YtterbiumIonChain(
        positions=[[0], [2], [5]],
    )

    np.testing.assert_array_equal(
        chain.positions,
        np.array([[0.0], [2.0], [5.0]]),
    )
    assert chain.positions.dtype == float
    assert chain.n_ions == 3


@pytest.mark.parametrize(
    "positions",
    [
        [],
        [0.0, 1.0, 2.0],
    ],
)
def test_chain_rejects_empty_or_non_2d_positions(
    positions: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="positions must be a non-empty two-dimensional array",
    ):
        YtterbiumIonChain(
            positions=np.asarray(positions),
        )


def test_chain_rejects_duplicate_positions() -> None:
    with pytest.raises(
        ValueError,
        match="Ion positions must be unique",
    ):
        YtterbiumIonChain(
            positions=[[0.0], [0.0]],
        )


@pytest.mark.parametrize(
    ("n_ions", "spacing"),
    [
        (0, 1.0),
        (-1, 1.0),
        (2, 0.0),
        (2, -1.0),
    ],
)
def test_linear_chain_rejects_non_positive_arguments(
    n_ions: int,
    spacing: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="n_ions and spacing must be positive",
    ):
        YtterbiumIonChain.linear(
            n_ions=n_ions,
            spacing=spacing,
        )


def test_linear_chain_positions() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=3,
        spacing=1.5,
    )

    np.testing.assert_allclose(
        chain.positions,
        np.array(
            [
                [0.0],
                [1.5],
                [3.0],
            ],
        ),
    )


def test_hamiltonian_accepts_valid_coupling_matrix() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=2,
        spacing=1.0,
    )
    couplings = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
        ],
    )

    model = YtterbiumMolmerSorensenHamiltonian(
        chain=chain,
        coupling_matrix=couplings,
    )

    assert model.n_qubits == 2
    assert model.coupling_matrix.dtype == float
    np.testing.assert_array_equal(
        model.coupling_matrix,
        couplings,
    )


def test_hamiltonian_rejects_wrong_coupling_matrix_shape() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=2,
        spacing=1.0,
    )

    with pytest.raises(
        ValueError,
        match=r"symmetric with shape \(2, 2\)",
    ):
        YtterbiumMolmerSorensenHamiltonian(
            chain=chain,
            coupling_matrix=np.zeros((3, 3)),
        )


def test_hamiltonian_rejects_non_symmetric_matrix() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=2,
        spacing=1.0,
    )
    couplings = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ],
    )

    with pytest.raises(
        ValueError,
        match="coupling_matrix must be symmetric",
    ):
        YtterbiumMolmerSorensenHamiltonian(
            chain=chain,
            coupling_matrix=couplings,
        )


def test_hamiltonian_rejects_nonzero_diagonal() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=2,
        spacing=1.0,
    )

    with pytest.raises(
        ValueError,
        match="coupling_matrix diagonal must be zero",
    ):
        YtterbiumMolmerSorensenHamiltonian(
            chain=chain,
            coupling_matrix=np.eye(2),
        )


def test_power_law_rejects_negative_exponent() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=2,
        spacing=1.0,
    )

    with pytest.raises(
        ValueError,
        match="exponent must be non-negative",
    ):
        YtterbiumMolmerSorensenHamiltonian.power_law(
            chain=chain,
            coupling=1.0,
            exponent=-0.1,
        )


def test_power_law_couplings() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=3,
        spacing=2.0,
    )

    model = YtterbiumMolmerSorensenHamiltonian.power_law(
        chain=chain,
        coupling=8.0,
        exponent=1.0,
    )

    expected = np.array(
        [
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 4.0],
            [2.0, 4.0, 0.0],
        ],
    )

    np.testing.assert_allclose(
        model.coupling_matrix,
        expected,
    )


def test_power_law_zero_exponent_produces_uniform_couplings() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=3,
        spacing=7.0,
    )

    model = YtterbiumMolmerSorensenHamiltonian.power_law(
        chain=chain,
        coupling=2.5,
        exponent=0.0,
    )

    expected = np.array(
        [
            [0.0, 2.5, 2.5],
            [2.5, 0.0, 2.5],
            [2.5, 2.5, 0.0],
        ],
    )

    np.testing.assert_allclose(
        model.coupling_matrix,
        expected,
    )


def test_pauli_hamiltonian_contains_scalar_field_and_couplings() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=3,
        spacing=1.0,
    )
    coupling_matrix = np.array(
        [
            [0.0, 1.5, 0.0],
            [1.5, 0.0, -2.0],
            [0.0, -2.0, 0.0],
        ],
    )

    model = YtterbiumMolmerSorensenHamiltonian(
        chain=chain,
        coupling_matrix=coupling_matrix,
        transverse_field=0.25,
    )

    hamiltonian = model.pauli_hamiltonian()

    assert hamiltonian.n_qubits == 3
    assert [(term.coefficient, term.operators) for term in hamiltonian.terms] == [
        (0.25, {0: "Z"}),
        (0.25, {1: "Z"}),
        (0.25, {2: "Z"}),
        (1.5, {0: "X", 1: "X"}),
        (-2.0, {1: "X", 2: "X"}),
    ]


def test_pauli_hamiltonian_supports_individual_transverse_fields() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=3,
        spacing=1.0,
    )

    model = YtterbiumMolmerSorensenHamiltonian(
        chain=chain,
        coupling_matrix=np.zeros((3, 3)),
        transverse_field=np.array([1.0, 0.0, -0.5]),
    )

    terms = model.pauli_hamiltonian().terms

    assert [(term.coefficient, term.operators) for term in terms] == [
        (1.0, {0: "Z"}),
        (-0.5, {2: "Z"}),
    ]


def test_pauli_hamiltonian_skips_zero_couplings_and_fields() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=2,
        spacing=1.0,
    )

    model = YtterbiumMolmerSorensenHamiltonian(
        chain=chain,
        coupling_matrix=np.zeros((2, 2)),
        transverse_field=0.0,
    )

    hamiltonian = model.pauli_hamiltonian()

    assert hamiltonian.n_qubits == 2
    assert hamiltonian.terms == ()


def test_pauli_hamiltonian_rejects_incompatible_field_shape() -> None:
    chain = YtterbiumIonChain.linear(
        n_ions=3,
        spacing=1.0,
    )

    model = YtterbiumMolmerSorensenHamiltonian(
        chain=chain,
        coupling_matrix=np.zeros((3, 3)),
        transverse_field=np.array([1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="could not be broadcast"):
        model.pauli_hamiltonian()
