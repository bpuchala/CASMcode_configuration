import numpy as np
import pytest

import libcasm.xtal as xtal


@pytest.fixture
def BCC_binary_GLstrain_disp_prim():
    # Lattice vectors
    lattice_column_vector_matrix = np.array(
        [
            [-1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0],  # a
            [1.0 / 2.0, -1.0 / 2.0, 1.0 / 2.0],  # a
            [1.0 / 2.0, 1.0 / 2.0, -1.0 / 2.0],  # a
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
    disp_dof = xtal.DoFSetBasis("disp")  # Atomic displacement
    local_dof = [
        [disp_dof],  # local DoF, basis site b=0
    ]

    # Global continuous degrees of freedom (DoF)
    GLstrain_dof = xtal.DoFSetBasis("GLstrain")  # Green-Lagrange strain metric
    global_dof = [GLstrain_dof]

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
def ZrO_prim_GLstrain_disp(ZrO_prim):
    prim = ZrO_prim
    disp_dof = xtal.DoFSetBasis("disp")  # Atomic displacement
    local_dof = [
        [disp_dof],
        [disp_dof],
        [disp_dof],
        [disp_dof],
    ]

    GLstrain_dof = xtal.DoFSetBasis("GLstrain")  # Green-Lagrange strain metric
    global_dof = [GLstrain_dof]

    return xtal.Prim(
        lattice=prim.lattice(),
        coordinate_frac=prim.coordinate_frac(),
        occ_dof=prim.occ_dof(),
        local_dof=local_dof,
        global_dof=global_dof,
        occupants=prim.occupants(),
    )
