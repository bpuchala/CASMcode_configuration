import math

import numpy as np
import pytest

import libcasm.configuration as config
import libcasm.xtal as xtal


def test_simple_cubic_binary_supercell(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    T = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.Supercell(prim, T)
    S = supercell.superlattice.column_vector_matrix()
    assert math.isclose(S[0, 0], 2.0)
    assert math.isclose(S[0, 1], 1.0)
    assert math.isclose(S[1, 1], 1.0)
    assert math.isclose(S[2, 2], 1.0)
    assert supercell.n_sites == 2
    assert supercell.n_unitcells == 2
    assert supercell.n_occupants() == [2, 2]
    assert supercell.occ_dof() == [["A", "B"], ["A", "B"]]
    assert np.allclose(
        supercell.coordinate_frac(),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).transpose(),
    )
    assert np.allclose(
        supercell.coordinate_cart(),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).transpose(),
    )
    assert supercell.sublattice_indices() == [0, 0]
    assert (
        supercell.unitcell_indices() == np.array([[0, 0, 0], [1, 0, 0]]).transpose()
    ).all()


def test_simple_cubic_binary_supercell_compare(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    T1 = np.array(
        [
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell1 = config.Supercell(prim, T1)

    T1b = np.array(
        [
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell1b = config.Supercell(prim, T1b)

    T2 = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]
    )
    supercell2 = config.Supercell(prim, T2)

    T3 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 2],
        ]
    )
    supercell3 = config.Supercell(prim, T3)

    # test left-handed lattice
    T4 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -2],
        ]
    )
    supercell4 = config.Supercell(prim, T4)

    # test left-handed permuted vectors
    T5 = np.array(
        [
            [0, -1, 0],
            [0, 0, 1],
            [2, 0, 0],
        ]
    )
    supercell5 = config.Supercell(prim, T5)

    assert supercell2 > supercell1
    assert supercell3 > supercell1
    assert supercell3 > supercell2

    assert supercell1 < supercell2
    assert supercell1 < supercell3
    assert supercell2 < supercell3

    assert supercell1 != supercell2
    assert supercell1 == supercell1b

    expected_S_canonical = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    canonical_supercell = config.make_canonical_supercell(supercell1)
    assert canonical_supercell == supercell3

    assert config.is_canonical_supercell(supercell1) is False
    assert config.make_canonical_supercell(supercell1) == supercell3

    assert config.is_canonical_supercell(supercell2) is False
    assert config.make_canonical_supercell(supercell2) == supercell3

    assert config.is_canonical_supercell(supercell3) is True
    assert config.make_canonical_supercell(supercell3) == supercell3

    assert config.is_canonical_supercell(supercell4) is False
    assert config.make_canonical_supercell(supercell4) == supercell3

    assert config.is_canonical_supercell(supercell5) is False
    assert config.make_canonical_supercell(supercell5) == supercell3

    assert np.allclose(
        canonical_supercell.superlattice.column_vector_matrix(), expected_S_canonical
    )

    equivalent_supercells = config.make_equivalent_supercells(supercell1)
    for scel in equivalent_supercells:
        print(scel.superlattice.column_vector_matrix())
    assert len(equivalent_supercells) == 3

    data = dict()
    data[supercell1] = "supercell1"
    assert len(data) == 1
    assert supercell1 in data
    assert data[supercell1] == "supercell1"
    assert supercell1b in data
    assert data[supercell1b] == "supercell1"
    assert supercell2 not in data

    data[supercell2] = "supercell2"
    assert len(data) == 2
    assert supercell2 in data
    assert data[supercell2] == "supercell2"


def test_supercell_io(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    T1 = np.array(
        [
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell1 = config.Supercell(prim, T1)

    # Test to_dict
    data = supercell1.to_dict()
    assert isinstance(data, dict)
    assert (data["transformation_matrix_to_supercell"] == T1).all()

    # Test from_dict
    supercell_set = config.SupercellSet(prim)
    supercell1_in = config.Supercell.from_dict(data=data, supercells=supercell_set)
    assert isinstance(supercell1_in, config.Supercell)
    assert supercell1_in == supercell1
    assert len(supercell_set) == 1

    # Test print
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        print(supercell1)
    out = f.getvalue()
    assert "transformation_matrix_to_super" in out


@pytest.mark.parametrize(
    "lattice_matrix,T",
    [
        # Simple cubic, diagonal supercell
        (
            np.eye(3),
            np.diag([2, 2, 2]),
        ),
        # Simple cubic, non-diagonal supercell
        (
            np.eye(3),
            np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]]),
        ),
        # FCC primitive lattice, diagonal supercell
        (
            np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]).T,
            np.diag([2, 2, 2]),
        ),
        # FCC primitive lattice, non-diagonal supercell
        (
            np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]).T,
            np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
        # BCC primitive lattice, diagonal supercell
        (
            np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]).T,
            np.diag([2, 2, 2]),
        ),
    ],
)
def test_generic_group(lattice_matrix, T):
    """Test that Supercell.generic_group() gives the same multiplication
    table as Supercell.symgroup()."""
    lattice = xtal.Lattice(lattice_matrix)
    coordinate_frac = np.array([[0.0, 0.0, 0.0]]).T
    xtal_prim = xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=[["A", "B"]],
        occupants={},
    )
    prim = config.Prim(xtal_prim)
    supercell = config.Supercell(prim, T)

    g = supercell.generic_group()
    sg = supercell.symgroup()

    assert g.size() == sg.size
    assert g.multiplication_table == sg.multiplication_table

    g2 = supercell.generic_group_v2()
    assert g2.size() == sg.size
    assert g2.multiplication_table == sg.multiplication_table
