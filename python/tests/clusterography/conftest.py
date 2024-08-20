import numpy as np
import pytest

import libcasm.xtal as xtal
import libcasm.xtal.lattices as xtal_lattices


@pytest.fixture
def tetragonal_lattice():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 1.0, 0.0],  # a
            [0.0, 0.0, 2.0],  # c
        ]
    ).transpose()
    return xtal.Lattice(lattice_column_vector_matrix)


@pytest.fixture
def simple_cubic_binary_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # a
            [0.0, 1.0, 0.0],  # a
            [0.0, 0.0, 1.0],  # a
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["A", "B"]]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def ZrO_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [3.233986856383, 0.000000000000, 0.000000000000],  # a
            [-1.616993428191, 2.800714773133, 0.000000000000],  # a
            [0.000000000000, 0.000000000000, 5.168678340000],  # c
        ]
    ).transpose()
    lattice = xtal.Lattice(lattice_column_vector_matrix)

    # Basis sites positions, as columns of a matrix,
    # in fractional coordinates with respect to the lattice vectors
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0],
            [1.0 / 3.0, 2.0 / 3.0, 3.0 / 4.0],
        ]
    ).transpose()

    # Occupation degrees of freedom (DoF)
    occupants = {}
    occ_dof = [["Zr"], ["Zr"], ["O", "Va"], ["O", "Va"]]

    # Local continuous degrees of freedom (DoF)
    local_dof = []

    # Global continuous degrees of freedom (DoF)
    global_dof = []

    return xtal.Prim(
        lattice=lattice,
        coordinate_frac=coordinate_frac,
        occ_dof=occ_dof,
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=occupants,
    )


@pytest.fixture
def FCC_with_interstitials_prim():
    coordinate_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.5, 0.5, 0.5],
        ]
    ).transpose()

    occ_dof = [["A"], ["Itet"], ["Itet"], ["Ioct"]]

    return xtal.make_primitive_prim(
        xtal.Prim(
            lattice=xtal_lattices.FCC(a=4.0),
            coordinate_frac=coordinate_frac,
            occ_dof=occ_dof,
        )
    )
