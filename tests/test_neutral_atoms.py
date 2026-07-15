import numpy as np
import pytest

from quantumhams import NeutralAtomRegister, RubidiumRydbergHamiltonian


def test_square_register() -> None:
    register = NeutralAtomRegister.square(2, 3, 5.0)
    assert register.positions.shape == (6, 2)
    assert register.n_atoms == 6


def test_one_atom_rydberg_hamiltonian() -> None:
    model = RubidiumRydbergHamiltonian(NeutralAtomRegister.linear(1, 1.0), 2.0, 4.0, 0.0)
    expected = np.array([[0, 1], [1, -4]], dtype=complex)
    np.testing.assert_allclose(model.matrix(sparse=False), expected)


def test_duplicate_atoms_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        NeutralAtomRegister(np.array([[0.0, 0.0], [0.0, 0.0]]))


def test_register_converts_positions_to_float_and_reports_size() -> None:
    register = NeutralAtomRegister(
        positions=[[0], [2], [5]],
    )

    np.testing.assert_array_equal(
        register.positions,
        np.array(
            [
                [0.0],
                [2.0],
                [5.0],
            ],
        ),
    )
    assert register.positions.dtype == float
    assert register.n_atoms == 3


@pytest.mark.parametrize(
    "positions",
    [
        np.array([]),
        np.array([0.0, 1.0]),
        np.zeros((2, 4)),
    ],
)
def test_register_rejects_invalid_position_shapes(
    positions: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="positions must have shape",
    ):
        NeutralAtomRegister(positions)


@pytest.mark.parametrize(
    "positions",
    [
        [[0.0, np.nan]],
        [[0.0, np.inf]],
        [[0.0, -np.inf]],
    ],
)
def test_register_rejects_non_finite_positions(
    positions: list[list[float]],
) -> None:
    with pytest.raises(
        ValueError,
        match="positions must be finite",
    ):
        NeutralAtomRegister(positions)


def test_register_rejects_duplicate_positions() -> None:
    with pytest.raises(
        ValueError,
        match="Atom positions must be unique",
    ):
        NeutralAtomRegister(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        )


@pytest.mark.parametrize(
    ("n_atoms", "spacing"),
    [
        (0, 1.0),
        (-1, 1.0),
        (2, 0.0),
        (2, -1.0),
    ],
)
def test_linear_register_rejects_non_positive_arguments(
    n_atoms: int,
    spacing: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="n_atoms and spacing must be positive",
    ):
        NeutralAtomRegister.linear(
            n_atoms=n_atoms,
            spacing=spacing,
        )


def test_linear_register_positions() -> None:
    register = NeutralAtomRegister.linear(
        n_atoms=3,
        spacing=1.5,
    )

    np.testing.assert_allclose(
        register.positions,
        np.array(
            [
                [0.0, 0.0],
                [1.5, 0.0],
                [3.0, 0.0],
            ],
        ),
    )


@pytest.mark.parametrize(
    ("rows", "columns", "spacing"),
    [
        (0, 2, 1.0),
        (-1, 2, 1.0),
        (2, 0, 1.0),
        (2, -1, 1.0),
        (2, 2, 0.0),
        (2, 2, -1.0),
    ],
)
def test_square_register_rejects_non_positive_arguments(
    rows: int,
    columns: int,
    spacing: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="rows, columns, and spacing must be positive",
    ):
        NeutralAtomRegister.square(
            rows=rows,
            columns=columns,
            spacing=spacing,
        )


def test_square_register_positions_follow_row_major_order() -> None:
    register = NeutralAtomRegister.square(
        rows=2,
        columns=2,
        spacing=2.5,
    )

    np.testing.assert_allclose(
        register.positions,
        np.array(
            [
                [0.0, 0.0],
                [2.5, 0.0],
                [0.0, 2.5],
                [2.5, 2.5],
            ],
        ),
    )


def test_hamiltonian_reports_number_of_qubits() -> None:
    register = NeutralAtomRegister.linear(
        n_atoms=4,
        spacing=1.0,
    )

    model = RubidiumRydbergHamiltonian(
        atom_register=register,
        rabi_frequency=1.0,
        detuning=0.0,
        c6=0.0,
    )

    assert model.n_qubits == 4


def test_scalar_detuning_is_broadcast_to_all_atoms() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(
            n_atoms=2,
            spacing=2.0,
        ),
        rabi_frequency=4.0,
        detuning=3.0,
        c6=0.0,
    )

    terms = model.pauli_hamiltonian().terms

    x_terms = [term for term in terms if term.operators in ({0: "X"}, {1: "X"})]
    detuning_terms = [term for term in terms[:4] if "Z" in term.operators.values()]
    identity_terms = [term for term in terms if term.operators == {}]

    assert [(term.coefficient, term.operators) for term in x_terms] == [
        (2.0, {0: "X"}),
        (2.0, {1: "X"}),
    ]

    assert [(term.coefficient, term.operators) for term in detuning_terms] == [
        (1.5, {0: "Z"}),
        (1.5, {1: "Z"}),
    ]

    assert len(identity_terms) == 1
    assert identity_terms[0].coefficient == pytest.approx(-3.0)


def test_individual_detunings_are_applied_per_atom() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(
            n_atoms=2,
            spacing=1.0,
        ),
        rabi_frequency=0.0,
        detuning=np.array([2.0, -4.0]),
        c6=0.0,
    )

    terms = model.pauli_hamiltonian().terms

    # The first four terms correspond to the drive and detuning terms.
    assert [(term.coefficient, term.operators) for term in terms[:4]] == [
        (0.0, {0: "X"}),
        (1.0, {0: "Z"}),
        (0.0, {1: "X"}),
        (-2.0, {1: "Z"}),
    ]

    identity_terms = [term for term in terms if term.operators == {}]

    assert len(identity_terms) == 1
    assert identity_terms[0].coefficient == pytest.approx(1.0)


def test_incompatible_detuning_shape_is_rejected() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(
            n_atoms=3,
            spacing=1.0,
        ),
        rabi_frequency=1.0,
        detuning=np.array([1.0, 2.0]),
        c6=0.0,
    )

    with pytest.raises(
        ValueError,
        match="could not be broadcast",
    ):
        model.pauli_hamiltonian()


def test_two_atom_interaction_pauli_decomposition() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister(
            [
                [0.0],
                [2.0],
            ],
        ),
        rabi_frequency=0.0,
        detuning=0.0,
        c6=64.0,
    )

    # distance = 2, so C6 / distance**6 = 64 / 64 = 1.
    terms = model.pauli_hamiltonian().terms

    interaction_terms = terms[-4:]

    assert [(term.coefficient, term.operators) for term in interaction_terms] == [
        (-0.25, {0: "Z"}),
        (-0.25, {1: "Z"}),
        (0.25, {0: "Z", 1: "Z"}),
        (0.25, {}),
    ]


def test_interaction_uses_euclidean_distance_in_two_dimensions() -> None:
    register = NeutralAtomRegister(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ],
    )

    model = RubidiumRydbergHamiltonian(
        atom_register=register,
        rabi_frequency=0.0,
        detuning=0.0,
        c6=5.0**6,
    )

    # The distance is 5, so the interaction strength is exactly 1.
    interaction_terms = model.pauli_hamiltonian().terms[-4:]

    assert [(term.coefficient, term.operators) for term in interaction_terms] == [
        (-0.25, {0: "Z"}),
        (-0.25, {1: "Z"}),
        (0.25, {0: "Z", 1: "Z"}),
        (0.25, {}),
    ]


def test_interaction_uses_euclidean_distance_in_three_dimensions() -> None:
    register = NeutralAtomRegister(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
        ],
    )

    # The distance is 3, so selecting C6 = 3**6 gives interaction 1.
    model = RubidiumRydbergHamiltonian(
        atom_register=register,
        rabi_frequency=0.0,
        detuning=0.0,
        c6=3.0**6,
    )

    interaction_terms = model.pauli_hamiltonian().terms[-4:]

    assert [(term.coefficient, term.operators) for term in interaction_terms] == [
        (-0.25, {0: "Z"}),
        (-0.25, {1: "Z"}),
        (0.25, {0: "Z", 1: "Z"}),
        (0.25, {}),
    ]


def test_single_atom_with_zero_detuning_has_no_identity_term() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(
            n_atoms=1,
            spacing=1.0,
        ),
        rabi_frequency=2.0,
        detuning=0.0,
        c6=100.0,
    )

    terms = model.pauli_hamiltonian().terms

    assert [(term.coefficient, term.operators) for term in terms] == [
        (1.0, {0: "X"}),
        (0.0, {0: "Z"}),
    ]

    assert not any(term.operators == {} for term in terms)


def test_three_atom_model_creates_each_pair_once() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(
            n_atoms=3,
            spacing=1.0,
        ),
        rabi_frequency=0.0,
        detuning=0.0,
        c6=1.0,
    )

    terms = model.pauli_hamiltonian().terms

    zz_terms = [term for term in terms if len(term.operators) == 2]

    assert [term.operators for term in zz_terms] == [
        {0: "Z", 1: "Z"},
        {0: "Z", 2: "Z"},
        {1: "Z", 2: "Z"},
    ]

    # Atom distances are 1, 2 and 1.
    np.testing.assert_allclose(
        [term.coefficient for term in zz_terms],
        [
            1.0 / 4.0,
            1.0 / 256.0,
            1.0 / 4.0,
        ],
    )


def test_negative_c6_produces_negative_interaction() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(
            n_atoms=2,
            spacing=1.0,
        ),
        rabi_frequency=0.0,
        detuning=0.0,
        c6=-4.0,
    )

    interaction_terms = model.pauli_hamiltonian().terms[-4:]

    assert [(term.coefficient, term.operators) for term in interaction_terms] == [
        (1.0, {0: "Z"}),
        (1.0, {1: "Z"}),
        (-1.0, {0: "Z", 1: "Z"}),
        (-1.0, {}),
    ]


def test_two_atom_matrix_matches_number_operator_form() -> None:
    model = RubidiumRydbergHamiltonian(
        atom_register=NeutralAtomRegister.linear(
            n_atoms=2,
            spacing=1.0,
        ),
        rabi_frequency=0.0,
        detuning=0.0,
        c6=8.0,
    )

    matrix = model.matrix(sparse=False)

    # With no drive or detuning, only |11> receives the interaction energy.
    expected = np.diag(
        [
            0.0,
            0.0,
            0.0,
            8.0,
        ],
    ).astype(complex)

    np.testing.assert_allclose(
        matrix,
        expected,
    )
