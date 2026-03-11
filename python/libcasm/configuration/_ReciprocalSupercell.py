from fractions import Fraction
from math import gcd, lcm
from typing import Optional, Union

import numpy as np

from libcasm.clexulator import DoFSpace
from libcasm.group import (
    get_all_subgroup_orbits,
    get_cyclic_subgroup_orbits,
)
from libcasm.irreps import IrrepDecomposition, IrrepInfo
from libcasm.xtal import (
    UnitCellIndexConverter,
)

from ._configuration import (
    Configuration,
    Prim,
    Supercell,
    SupercellSymOp,
    make_dof_space_rep,
    make_symgroup,
)
from ._misc import pretty


def _get_verbosity_level(verbosity: Optional[str] = None) -> int:
    if verbosity is None or verbosity == "none":
        verbosity_level = 0
    elif verbosity == "standard":
        verbosity_level = 1
    elif verbosity == "verbose":
        verbosity_level = 2
    else:
        raise ValueError(f"Invalid verbosity option: {verbosity}")
    return verbosity_level


def _normalize_column(
    basis: np.ndarray,
    i_col: int,
    tol: float = 1e-8,
):
    norm = np.linalg.norm(basis[:, i_col])
    if norm > tol:
        basis[:, i_col] /= norm
    else:
        basis[:, i_col] = np.zeros(basis.shape[0], dtype=basis.dtype)


def _make_unique(matrices: list[np.ndarray], tol: float = 1e-8) -> list[np.ndarray]:
    """Return a list of unique matrices from the input list, within a tolerance.

    Parameters
    ----------
    matrices: list[np.ndarray]
        List of square numpy arrays (matrices) to filter for uniqueness.
    tol: float = 1e-8
        Tolerance for considering two matrices as equal.

    Returns
    -------
    unique_matrices: list[np.ndarray]
        List of unique matrices from the input list.
    """
    unique_matrices = []
    for M in matrices:
        is_unique = True
        for M_unique in unique_matrices:
            if np.allclose(M, M_unique, atol=tol):
                is_unique = False
                break
        if is_unique:
            unique_matrices.append(M)
    return unique_matrices


def _close(group_ops: list[SupercellSymOp]) -> list[SupercellSymOp]:
    """Close a list of SupercellSymOp under composition.

    Parameters
    ----------
    group_ops: list[SupercellSymOp]
        List of SupercellSymOp objects to close under multiplication.

    Returns
    -------
    closed_group_ops: list[SupercellSymOp]
        List of SupercellSymOp objects forming a closed subgroup.
    """

    final_ops = [x.copy() for x in group_ops]
    i = 0
    while i < len(final_ops):
        op_i = final_ops[i]
        j = 0
        while j < len(final_ops):
            op_j = final_ops[j]
            op_product = op_i * op_j
            if op_product not in final_ops:
                final_ops.append(op_product)
            j += 1
        i += 1
    return final_ops


class AxisIrrepInfo:
    """Information about symmetry-adapted axes from an irrep decomposition.

    Attributes
    ----------
    orbit_index: int
        Index of the k-point orbit this axis belongs to.
    kpoint_index: int
        Index of the k-point whose subspace this axis belongs to.
    kpoint_irreps_index: int
        Index into `kpoint_irreps` list of the irrep this axis belongs to.
    irrep_char_index: int
        An index identifying irreps which have the same characters.
    """

    def __init__(
        self,
        orbit_index: int,
        kpoint_index: int,
        kpoint_irreps_index: int,
        irrep_char_index: int,
    ):
        self.orbit_index = orbit_index
        self.kpoint_index = kpoint_index
        self.kpoint_irreps_index = kpoint_irreps_index
        self.irrep_char_index = irrep_char_index


def _get_irrep_char_index(
    irrep: IrrepInfo,
    kpoint_irreps: list[IrrepInfo],
    axis_irrep_info: list[AxisIrrepInfo],
    next_irrep_char_index: int,
) -> int:
    """Helper to get a unique index for all irreps with the same characters."""
    found_index = -1
    for _i, _axis_info in enumerate(axis_irrep_info):
        _irrep = kpoint_irreps[_axis_info.kpoint_irreps_index]
        if len(_irrep.characters) != len(irrep.characters):
            continue
        if np.allclose(
            _irrep.characters,
            irrep.characters,
            atol=1e-8,
        ):
            found_index = _axis_info.irrep_char_index
            break
    if found_index == -1:
        irrep_char_index = next_irrep_char_index
        next_irrep_char_index += 1
    else:
        irrep_char_index = found_index

    return (irrep_char_index, next_irrep_char_index)


class SupercellKpoints:
    """K-point generation, symmetry orbits, and little groups for a particular
    Supercell."""

    def __init__(self, supercell: Supercell, tol: float = 1e-5):
        self.supercell = supercell
        """libcasm.configuration.Supercell: The supercell for which commensurate
        k-points are generated.

        This Supercell supplies the superlattice, factor group (symmetry operations),
        and unitcell index converters used during k-point enumeration and symmetry
        orbit construction.
        """

        self.tol = tol
        """float: Tolerance used when matching k-points under periodic boundary
        conditions.

        The tolerance is passed to the primitive-lattice periodic displacement routine
        to decide when two k-points are equivalent up to reciprocal-lattice
        translations.
        """

        # Supercell info
        self.site_sublattice_indices = self.supercell.sublattice_indices()
        """list[int]: Sublattice indices for each site in the supercell."""

        self.site_linear_unitcell_indices = self.supercell.linear_unitcell_indices()
        """list[int]: Linear unitcell indices for each site in the supercell."""

        self.site_coordinate_cart = self.supercell.coordinate_cart()
        """np.ndarray: Array of site Cartesian coordinates with shape (3, n_sites)."""

        # primitive and supercell reciprocal lattices
        self.reciprocal_prim_lattice = supercell.prim.xtal_prim.lattice().reciprocal()
        """libcasm.xtal.Lattice: Reciprocal lattice of the primitive cell lattice."""

        self.reciprocal_superlattice = supercell.superlattice.reciprocal()
        """libcasm.xtal.Lattice: Reciprocal lattice of the supercell lattice.

        This forms the primitive cell for the grid of k-points commensurate with the
        real-space supercell: kpoint_value = S_recip @ kpoint_index, where `S_recip` is
        the column vector matrix of `reciprocal_superlattice` and `kpoint_index` is an
        integer column vector.
        """

        # placeholders populated by build methods
        self.index_converter = None
        """xtal.UnitCellIndexConverter: Enumerates k-points commensurate with the
        supercell.

        The converter is constructed with `transformation_matrix_to_super = T_recip`
        where `T_recip` is the integer matrix found by solving
        ``S_recip @ T_recip = P_recip`` (reciprocal lattice of the primitive cell
        column matrix). It enumerates k-points commensurate with the supercell.
        """

        self.indices = None
        """np.ndarray: Array of k-point indices with shape (3,N), where each column
        gives the coordinates of a k-point as multiples of the of reciprocal cell of
        the superrcell's superlattice.
        .

        Each column is a shape (3,) array giving the indices / fraction coordinates
        of the k-points, according to `self.coordinates = S_recip @ self.indices`,
        where `S_recip` is the column vector matrix of
        :py:attr:`reciprocal_superlattice`. K-points are ordered as generated by
        :py:attr:`index_converter`.
        """

        self.coordinates = None
        """np.ndarray: Array of k-points with shape (3, N), where each column is a
        k-point.

        Each column is a shape (3,) array giving the coordinates of the
        k-points, according to `self.coordinates = S_recip @ self.indices`, where
        `S_recip` is the column vector matrix of :py:attr:`reciprocal_superlattice`.
        K-points are ordered as generated by :py:attr:`index_converter`.
        """

        self.coordinates_frac = None
        """np.ndarray: Array of k-points in primitive reciprocal lattice fractional
        coordinates, with shape (3, N).

        Each column ``coordinates_frac[:, i]`` is the fractional coordinate vector
        ``q`` such that ``coordinates[:, i] == B_prim @ q``, where ``B_prim`` is the
        column vector matrix of :py:attr:`reciprocal_prim_lattice`. These are the ``q``
        values accepted by :func:`~libcasm.configuration.min_supercell_for_kpoint`.
        """

        self.neg_kpoint_index = []
        """list[int | None]: For each k-point index i, the index of −k in
        `self.coordinates` (mod primitive reciprocal lattice), or None if not found.

        For TRIM points where k ≡ −k (mod primitive reciprocal lattice),
        ``neg_kpoint_index[i] == i``.
        """

        self.orbits = []
        """list[list[int]]: Symmetry orbits of k-point indices under the supercell
        factor group.

        Each orbit is a list of integer indices referring to columns of
        `self.coordinates`. Orbits enumerate k-points that are symmetry-equivalent
        under the supercell's factor group.
        """

        self.equivalence_map = []
        """list[list[libcasm.configuration.SupercellSymOp]]: List of SupercellSymOp
        operations mapping the first k-point in each orbit to the other k-points in
        the orbit.

        Each element ``equivalence_map[i_orbit][j]`` is a factor group
        SupercellSymOp (``translation_index() == 0``) that maps
        ``coordinates[:, orbits[i_orbit][0]]`` to
        ``coordinates[:, orbits[i_orbit][j]]``.
        """

        self.little_groups = []
        """list[list[libcasm.configuration.SupercellSymOp]]: Little groups for each
        k-point in `self.coordinates`.

        The little group for each k-point is a list of symmetry operations from the
        supercell factor group that leave the k-point invariant.
        """

        self.orbit_independent_indices = []
        """list[list[int]]: For each orbit, the positions j within the orbit whose
        real-valued plane wave subspace is independent of all earlier members.

        Position 0 (the prototype) is always included. A later position j is excluded
        when ``coordinates[:, orbits[i_orbit][j]]`` is the negative (mod primitive
        reciprocal lattice) of an already-independent k-point in the same orbit,
        because the real-valued plane wave basis for k and −k spans the same subspace.
        """

        self._S_recip_inv = None
        """np.ndarray: Inverse of the reciprocal superlattice column vector matrix,
        with shape (3, 3).

        Used to invert a Cartesian k-point back to integer coordinates.
        """

        self._kpoint_permutations = None
        """np.ndarray: Permutation arrays for each supercell factor group element,
        with shape (n_sc_fg, n_kpts).

        ``_kpoint_permutations[sc_fg_idx, k]`` gives the index of the k-point that
        k-point ``k`` maps to under the point-group matrix of supercell factor group
        element ``sc_fg_idx``, where ``sc_fg_idx`` is the value returned by
        ``SupercellSymOp.supercell_factor_group_index()``. Built by
        :func:`_build_kpoint_permutations`.
        """

        # build data
        self._build_kpoints()
        self._build_kpoint_permutations()
        self._build_kpoint_orbits()
        self._build_little_groups()

    def _build_kpoints(self):
        """Populate self.coordinates, self.indices and self.index_converter."""
        S_recip = self.reciprocal_superlattice.column_vector_matrix()
        T_recip = self.supercell.transformation_matrix_to_super.T
        self.index_converter = UnitCellIndexConverter(
            transformation_matrix_to_super=T_recip,
        )

        kpt_indices_list = []
        for i in range(self.index_converter.total_unitcells()):
            kpt_indices = self.index_converter.unitcell(i)
            # Ensure column-vector shape (3, 1) for each index
            kpt_col = np.asarray(kpt_indices).reshape(3, -1)
            kpt_indices_list.append(kpt_col)

        if kpt_indices_list:
            # Stack index column-vectors into a 3 x N integer array
            self.indices = np.hstack(kpt_indices_list).astype(int)
        else:
            self.indices = np.empty((3, 0), dtype=int)

        # Compute k-points by applying S_recip to the integer index columns
        self.coordinates = pretty(S_recip @ self.indices)

        # Compute fractional coordinates in the primitive reciprocal lattice
        B_prim = self.reciprocal_prim_lattice.column_vector_matrix()
        B_prim_inv = np.linalg.inv(B_prim)
        self.coordinates_frac = pretty(B_prim_inv @ self.coordinates)

        # Precompute inverse for O(1) k-point lookup
        self._S_recip_inv = np.linalg.inv(S_recip)

        # For each k-point, find the index of its negative (mod primitive reciprocal
        # lattice), or None if not found
        n_kpts = self.coordinates.shape[1]
        neg_kpoint_index = []
        for i in range(n_kpts):
            idx = self._get_kpoint_index(-self.coordinates[:, i])
            neg_kpoint_index.append(idx if idx != i else None)
        self.neg_kpoint_index = neg_kpoint_index
        return

    def _get_kpoint_index(self, kpt: np.ndarray) -> int:
        """Find column index of equivalent k-point in self.coordinates."""
        idx_float = self._S_recip_inv @ kpt
        idx_int = np.round(idx_float).astype(int)
        return int(self.index_converter.linear_unitcell_index(idx_int))

    def _build_kpoint_permutations(self):
        """Build permutation arrays for each unique factor group element.

        For each supercell factor group element (indexed by
        supercell_factor_group_index), precomputes the permutation of k-point indices.
        """
        n_kpts = self.coordinates.shape[1]
        elements = self.supercell.factor_group.elements
        n_sc_fg = len(elements)
        self._kpoint_permutations = np.empty((n_sc_fg, n_kpts), dtype=int)
        for sc_fg_idx, op in enumerate(elements):
            matrix = op.matrix()
            for k in range(n_kpts):
                self._kpoint_permutations[sc_fg_idx, k] = self._get_kpoint_index(
                    matrix @ self.coordinates[:, k]
                )

    def _build_kpoint_orbits(self):
        """Populate self.orbits using symmetry operations of the supercell."""
        if self.coordinates is None:
            raise RuntimeError("coordinates must be built before computing orbits")

        n_kpts = self.coordinates.shape[1]
        found = [False] * n_kpts
        kpoint_orbits = []
        for i in range(n_kpts):
            if found[i]:
                continue
            orbit = [i]
            found[i] = True
            for perm in self._kpoint_permutations:
                kpt_index = int(perm[i])
                if not found[kpt_index]:
                    orbit.append(kpt_index)
                    found[kpt_index] = True
            orbit.sort()
            kpoint_orbits.append(orbit)
        self.orbits = kpoint_orbits
        return

    def _build_little_groups(self):
        """Populate self.little_groups and self.equivalence_map using symmetry
        operations of the supercell."""
        if self.coordinates is None:
            raise RuntimeError("coordinates must be built before computing orbits")

        n_kpts = self.coordinates.shape[1]

        # Build lookup tables from k-point index to orbit position
        orbit_index_of = {}
        position_in_orbit = {}
        for i_orbit, orbit in enumerate(self.orbits):
            for j, kpt_idx in enumerate(orbit):
                orbit_index_of[kpt_idx] = i_orbit
                position_in_orbit[kpt_idx] = j

        prototype_indices = [orbit[0] for orbit in self.orbits]

        little_groups = [[] for _ in range(n_kpts)]
        equivalence_map = [[None] * len(orbit) for orbit in self.orbits]

        it = SupercellSymOp.begin(self.supercell)
        end = SupercellSymOp.end(self.supercell)
        while it != end:
            op = it.copy()
            sc_fg_idx = op.supercell_factor_group_index()
            perm = self._kpoint_permutations[sc_fg_idx]
            is_factor_group_op = op.translation_index() == 0

            # Use precomputed permutation to find little group members
            for kpt_index_init in range(n_kpts):
                if perm[kpt_index_init] == kpt_index_init:
                    little_groups[kpt_index_init].append(op)

            # Use factor group ops to build equivalence_map (maps prototype k-point
            # in each orbit to each other k-point in the orbit)
            if is_factor_group_op:
                for i_orbit, proto_idx in enumerate(prototype_indices):
                    dest_idx = perm[proto_idx]
                    j_dest = position_in_orbit.get(dest_idx)
                    if j_dest is not None and equivalence_map[i_orbit][j_dest] is None:
                        equivalence_map[i_orbit][j_dest] = op

            it.next()

        self.little_groups = little_groups
        self.equivalence_map = equivalence_map

        # For each orbit, determine which positions j contribute an independent
        # real-valued plane wave subspace. Position j is dependent if k_j is the
        # negative of an already-independent k-point, because the real-valued cos/sin
        # basis for k and −k spans the same subspace.
        orbit_independent_indices = []
        for orbit in self.orbits:
            independent = [0]
            covered = {orbit[0]}
            neg = self.neg_kpoint_index[orbit[0]]
            if neg is not None:
                covered.add(neg)
            for j in range(1, len(orbit)):
                if orbit[j] in covered:
                    continue
                independent.append(j)
                covered.add(orbit[j])
                neg = self.neg_kpoint_index[orbit[j]]
                if neg is not None:
                    covered.add(neg)
            orbit_independent_indices.append(independent)
        self.orbit_independent_indices = orbit_independent_indices


class SupercellDoF:
    """Data objects used for building bases for a particular Supercell and degree of
    freedom type."""

    def __init__(self, supercell: Supercell, dof_key: str):
        self.supercell = supercell
        """libcasm.configuration.Supercell: The supercell."""

        self.dof_key = dof_key
        """str: Degree of freedom key for which DoF spaces are constructed."""

        # Supercell info
        self.site_sublattice_indices = self.supercell.sublattice_indices()
        """list[int]: Sublattice indices for each site in the supercell."""

        self.site_linear_unitcell_indices = self.supercell.linear_unitcell_indices()
        """list[int]: Linear unitcell indices for each site in the supercell."""

        self.dof_space = None
        """libcasm.clexulator.DoFSpace: DoF space for the entire supercell, with
        identity basis.
        """

        self.dof_id = None
        """list[list[int]]: DoF ID list, specifying the unique combinations of
        (sublattice index, dof component index) in the DoF space.

        DoF in the DoF space that have the same DoF ID differ only by a lattice
        translation.
        """

        self.axis_linear_unitcell_index = None
        """list[int]: The linear unitcell index associated with each row of the DoF
        space basis."""

        self.axis_sublattice_index = None
        """list[int]: The sublattice index associated with each row of the DoF space
        basis."""

        self.axis_dof_id = None
        """list[int]: The DoF ID associated with each row of the DoF space basis."""

        # build data
        self._build_dof_space()

    def _build_dof_space(self):
        """Build the DoF space for this supercell with identity basis and build the
        unique DoF id list."""
        _default_config = Configuration(supercell=self.supercell)
        self.dof_space, _ = _default_config.make_dof_space(
            dof_key=self.dof_key,
            symmetry_adapted=False,
            exclude_homogeneous_modes=False,
            include_default_occ_modes=True,
        )

        # Get the DoF space for the given dof_key in this supercell
        dof_space = self.dof_space
        total_dim = dof_space.basis.shape[0]
        basis_linear_site_index = dof_space.axis_info.linear_site_index
        basis_dof_component_index = dof_space.axis_info.dof_component_index

        # Unique DoF per primitive cell as (sublattice index, dof component index) pairs
        dof_id = []
        axis_linear_unitcell_index = []
        axis_sublattice_index = []
        axis_dof_id = []
        for i in range(total_dim):
            lsi = basis_linear_site_index[i]
            u = self.site_linear_unitcell_indices[lsi]
            axis_linear_unitcell_index.append(u)

            b = self.site_sublattice_indices[lsi]
            axis_sublattice_index.append(b)

            c = basis_dof_component_index[i]
            test_dof_id = [b, c]

            found = False
            for j, unique_dof_id in enumerate(dof_id):
                if test_dof_id == unique_dof_id:
                    axis_dof_id.append(j)
                    found = True
                    break
            if not found:
                dof_id.append(test_dof_id)
                axis_dof_id.append(len(dof_id) - 1)
        self.dof_id = dof_id
        self.axis_linear_unitcell_index = axis_linear_unitcell_index
        self.axis_sublattice_index = axis_sublattice_index
        self.axis_dof_id = axis_dof_id


class DiscreteFourierTransform:
    """Discrete Fourier transform functionality for a SupercellKpoints instance."""

    def __init__(
        self,
        supercell_kpoints: SupercellKpoints,
        supercell_dof: SupercellDoF,
    ):
        self.supercell_kpoints = supercell_kpoints
        """SupercellKpoints: The k-point data for this transform."""

        self.supercell_dof = supercell_dof
        """SupercellDoF: The DoF data for this transform."""

        # Convenience references
        self.supercell = supercell_kpoints.supercell
        self.dof_key = supercell_dof.dof_key
        self.dof_space = supercell_dof.dof_space
        self.dof_id = supercell_dof.dof_id
        self.axis_dof_id = supercell_dof.axis_dof_id
        self.axis_linear_unitcell_index = supercell_dof.axis_linear_unitcell_index
        self.site_sublattice_indices = supercell_kpoints.site_sublattice_indices
        self.site_linear_unitcell_indices = (
            supercell_kpoints.site_linear_unitcell_indices
        )
        self.coordinates = supercell_kpoints.coordinates

        self.dX = None
        """np.ndarray: Cached array for local_delta_value computation."""

        self._build_dft_matrices()

    def _build_dft_matrices(self):
        r"""Build DFT matrices

        Notes
        -----

        We want to evaluate the discrete Fourier transform for each DoF ID over all
        unitcells in the supercell. The DFT is defined as:

        .. math::

            X_{di} \;=\; \sum_{u} e^{-i \vec{k}_i\cdot\vec{R}_u}\,
            e^{-i \vec{k}_i\cdot\vec{\tau}_d}\; x_{du}

        where:

        - X_{di} is the discrete Fourier transform at k-point i for DoF ID d
        - x_{du} is the DoF value at unitcell u for DoF ID
        - \vec{R}_u is the lattice vector of unitcell u
        - \vec{\tau}_d is the basis site coordinate of DoF ID d in the primitive cell

        We can write this using numpy as:

        ... code-block:: python

            for d in range(n_dof):
                X[d] = dft_phase[d] * (dft_M @ x[d])

        where:

        - dft_phase[d]: np.ndarray of shape (n_kpts,), with elements
          ``np.exp(-1j * np.dot(kpoints[:, i_k], tau_d))``
        - dft_M: np.ndarray of shape (n_kpts, n_unitcells), with elements
          ``np.exp(-1j * np.dot(kpoints[:, i_k], R_u))``
        - x: np.ndarray of shape (n_unitcells, n_dof), with columns containing the DoF
          values for the d-th DoF ID in each unitcell. This can be constructed from a
          Configuration or DoF values vector corresponding to the supercell's DoFSpace.
        - X: np.ndarray of shape (n_kpts, n_dof), with columns containing the DFT
          coefficients for the d-th DoF ID at each k-point.


        """
        n_unitcells = self.supercell.n_unitcells
        n_dof_id = len(self.dof_id)

        # Supercell info
        unitcell_index_converter = self.supercell.unitcell_index_converter

        # Prim info
        xtal_prim = self.supercell.prim.xtal_prim
        L_prim = xtal_prim.lattice().column_vector_matrix()
        sublat_coordinate_cart = xtal_prim.coordinate_cart()

        # Build M_dft and M_idft
        M_dft = np.zeros((n_unitcells, n_unitcells), dtype=complex)
        dft_phase = np.zeros((n_dof_id, n_unitcells), dtype=complex)
        M_idft = np.zeros((n_unitcells, n_unitcells), dtype=complex)
        idft_phase = np.zeros((n_dof_id, n_unitcells), dtype=complex)
        for i_row in range(n_unitcells):
            k = self.coordinates[:, i_row]

            # Build M_dft
            for i_col in range(n_unitcells):
                R = L_prim @ unitcell_index_converter.unitcell(i_col)
                M_dft[i_row, i_col] = np.exp(-1j * np.dot(k, R))

            # Build M_idft
            for i_col in range(n_unitcells):
                R = L_prim @ unitcell_index_converter.unitcell(i_col)
                M_idft[i_row, i_col] = np.exp(1j * np.dot(k, R))

            # Build dft_phase
            for i_dof, dof_id in enumerate(self.dof_id):
                b = dof_id[0]
                tau = sublat_coordinate_cart[:, b]
                dft_phase[i_dof, i_row] = np.exp(-1j * np.dot(k, tau))

            # Build idft_phase
            for i_dof, dof_id in enumerate(self.dof_id):
                b = dof_id[0]
                tau = sublat_coordinate_cart[:, b]
                idft_phase[i_dof, i_row] = np.exp(1j * np.dot(k, tau))

        self.dft_M = M_dft
        self.dft_phase = dft_phase
        self.idft_M = M_idft
        self.idft_phase = idft_phase
        return

    def make_x(
        self,
        src: Union[Configuration, np.ndarray],
    ) -> np.ndarray:
        """Make the real space input for the discrete fourier transform.

        Parameters
        ----------
        src: Union[Configuration, np.ndarray]
            The source of DoF values, either a Configuration or a DoF values vector for
            the associated DoFSpace, as generated by
            :meth:`Configuration.dof_values_vector()`.

        Returns
        -------
        x: np.ndarray
            A shape (n_unitcells, n_dof_id) array of DoF values organized for DFT.
            Each column corresponds to a unique DoF ID, in the order specified by
            :py:attr:`dof_id`, and each row corresponds to a unitcell in the supercell
            (using the linear unitcell index).
        """
        n_unitcells = (self.supercell.n_unitcells,)
        n_dof_id = len(self.dof_id)

        if isinstance(src, Configuration):
            v = src.dof_values_vector(dof_space=self.dof_space)
        elif isinstance(src, np.ndarray):
            v = src
            if v.shape != (len(self.axis_dof_id),):
                raise ValueError("Error in SupercellKpoints.make_x: invalid src shape")
        else:
            raise ValueError("Error in SupercellKpoints.make_x: invalid src type")

        x = np.zeros((n_unitcells, n_dof_id), dtype=complex)
        for i, d in enumerate(self.axis_dof_id):
            lsi = self.axis_linear_unitcell_index[i]
            x[lsi, d] = v[lsi]
        return x

    def resolve_dof_values_vector(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Make a DoF values vector corresponding to a set of DoF values organized for
        DFT.

        Parameters
        ----------
        x: np.ndarray
            A shape (n_unitcells, n_dof_id) complex-valued array of DoF values
            organized for DFT. Each column corresponds to a unique DoF ID, in the order
            specified by :py:attr:`dof_id`, and each row corresponds to a unitcell in
            the supercell (using the linear unitcell index). Imaginary parts are
            ignored. For occupation DoFs, values are not rounded.

        Returns
        -------
        v: np.ndarray
            A shape (n_dof_space,) array of DoF values corresponding to the DoFSpace.
        """
        v = np.zeros((len(self.axis_dof_id),), dtype=float)
        for i, d in enumerate(self.axis_dof_id):
            lsi = self.axis_linear_unitcell_index[i]
            v[i] = x[lsi, d].real
        return v

    def resolve_occupation(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Make an occupation array corresponding to a set of DoF values organized for
        DFT.

        Parameters
        ----------
        x: np.ndarray
            A shape (n_unitcells, n_dof_id) complex-valued array of DoF values
            organized for DFT. Each column corresponds to a unique DoF ID, in the order
            specified by :py:attr:`dof_id`, and each row corresponds to a unitcell in
            the supercell (using the linear unitcell index). Imaginary parts are
            ignored. For occupation DoFs, real-valued entries are rounded to zero or
            one.

        Returns
        -------
        occupation: np.ndarray
            A shape (n_sites,) array of integer occupation values for each site in the
            supercell.
        """

        n_rows = max(self.dof_space.axis_info.dof_component_index) + 1
        n_cols = self.supercell.n_sites

        occupation = np.zeros((n_cols,), dtype=int)
        V = np.zeros((n_rows, n_cols), dtype=float)
        v = self.resolve_dof_values_vector(x=x)
        for i in range(len(v)):
            lsi = self.dof_space.axis_info.linear_site_index[i]
            c = self.dof_space.axis_info.dof_component_index[i]
            V[c, lsi] = round(v[lsi])

            if int(round(v[i])) == 1:
                occupation[lsi] = c

        # Check that the sum of each column is approximately 1:
        col_sums = np.sum(V, axis=0)

        # Get a list of columns that do not sum to 1 within tolerance:
        invalid_cols = [
            j for j in range(n_cols) if not np.isclose(col_sums[j], 1.0, atol=1e-5)
        ]
        if invalid_cols:
            raise ValueError(
                f"Error in SupercellKpoints.resolve_occupation: "
                f"occupation values do not sum to 1 for sites "
                f"with linear indices {invalid_cols}"
            )

        return occupation

    def resolve_local_dof_values(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Make a local DoF values array corresponding to a set of DoF values organized
        for DFT.

        Parameters
        ----------
        x: np.ndarray
            A shape (n_unitcells, n_dof_id) complex-valued array of DoF values
            organized for DFT. Each column corresponds to a unique DoF ID, in the order
            specified by :py:attr:`dof_id`, and each row corresponds to a unitcell in
            the supercell (using the linear unitcell index). Imaginary parts are
            ignored.

        Returns
        -------
        local_dof_values: np.ndarray
            A shape (m, n_sites) array of float local DoF values, where m is the
            maximum number of DoF components.
        """

        n_rows = max(self.dof_space.axis_info.dof_component_index) + 1
        n_cols = self.supercell.n_sites
        local_dof_values = np.zeros((n_rows, n_cols), dtype=float)
        v = self.resolve_dof_values_vector(x=x)
        for i in range(len(v)):
            lsi = self.dof_space.axis_info.linear_site_index[i]
            c = self.dof_space.axis_info.dof_component_index[i]
            local_dof_values[c, lsi] = v[i]

        return local_dof_values

    def resolve_config(
        self,
        x: np.ndarray,
        background: Optional[Configuration] = None,
    ) -> Configuration:
        """Make a Configuration corresponding to a set of DoF values organized for DFT.

        Parameters
        ----------
        x: np.ndarray
            A shape (n_unitcells, n_dof_id) complex-valued array of DoF values
            organized for DFT. Each column corresponds to a unique DoF ID, in the order
            specified by :py:attr:`dof_id`, and each row corresponds to a unitcell in
            the supercell (using the linear unitcell index). Imaginary parts are
            ignored. For occupation DoFs, real-valued entries are rounded to zero or
            one.
        background: Optional[Configuration] = None
            An optional Configuration to copy and use as a background for DoF values not
            specified in `x`. If None (default), the Configuration is initialized
            from scratch with zero values for all DoFs.

        Returns
        -------
        config: Configuration
            A Configuration object with DoF values set according to `x`.
        """

        if background is not None:
            config = background.copy()
        else:
            config = Configuration(supercell=self.supercell)

        # Convert `x` to DoF values
        if self.dof_key == "occ":
            occupation = self.resolve_occupation(x=x)
            config.set_occupation(occupation)

        else:
            local_dof_values = self.resolve_dof_values_vector(x=x)
            config.set_local_dof_values(
                key=self.dof_key,
                dof_values=local_dof_values,
            )

        return config

    def dft(
        self,
        x: np.ndarray,
    ):
        """Compute the discrete Fourier transform of DoF values over the supercell.

        Parameters
        ----------
        x: np.ndarray
            A shape (n_dof_id, n_unitcells) array of DoF values organized for DFT.
            Each row corresponds to a unique DoF ID, in the order specified by
            :py:attr:`dof_id`, and each column corresponds to a unitcell in the
            supercell (using the linear unitcell index).

        Returns
        -------
        X: np.ndarray
            A shape (n_dof_id, n_kpts) array of DFT coefficients. Each row
            corresponds to a unique DoF ID, in the order specified by :py:attr:`dof_id`,
            and each column corresponds to a k-point commensurate with the supercell, in
            the order specified by :py:attr:`coordinates`.
        """
        n_unitcells = self.supercell.n_unitcells
        n_dof_id = len(self.dof_id)
        X = np.zeros((n_unitcells, n_dof_id), dtype=complex)
        for d in range(n_dof_id):
            X[d] = self.dft_phase[d] * (self.dft_M @ x[d])
        return X

    def idft(
        self,
        X: np.ndarray,
    ):
        """Compute the inverse discrete Fourier transform of DFT coefficients over
        the supercell.

        Parameters
        ----------
        X: np.ndarray
            A shape (n_dof_id, n_kpts) array of DFT coefficients. Each row
            corresponds to a unique DoF ID, in the order specified by :py:attr:`dof_id`,
            and each column corresponds to a k-point commensurate with the supercell, in
            the order specified by :py:attr:`coordinates`.

        Returns
        -------
        x: np.ndarray
            A shape (n_dof_id, n_unitcells) array of DoF values organized for DFT.
            Each row corresponds to a unique DoF ID, in the order specified by
            :py:attr:`dof_id`, and each column corresponds to a unitcell in the
            supercell (using the linear unitcell index).
        """
        n_unitcells = self.supercell.n_unitcells
        n_dof_id = len(self.dof_id)
        x = np.zeros((n_unitcells, n_dof_id), dtype=complex)
        for d in range(n_dof_id):
            x[d] = self.idft_phase[d] * (self.idft_M @ X[d])
        return x

    def local_delta_value(
        self,
        l: int,  # noqa: E741
        old_value: Union[int, np.ndarray],
        new_value: Union[int, np.ndarray],
    ):
        """Compute the change in DFT coefficients X for a change in DoF value at site l.

        Parameters
        ----------
        l: int
            Linear site index in the supercell where the DoF value is changed.
        old_value: Union[int, np.ndarray]
            The old DoF value at site l. For occupation DoFs, this is an integer
            specifying the occupied component index. For other DoFs, this is a
            np.ndarray of shape (n_components,) specifying the old local DoF values.
        new_value: Union[int, np.ndarray]
            The new DoF value at site l. For occupation DoFs, this is an integer
            specifying the occupied component index. For other DoFs, this is a
            np.ndarray of shape (n_components,) specifying the new local DoF values.

        Returns
        -------
        dX: np.ndarray
            A shape (n_dof_id, n_kpts) array of changes in DFT coefficients due to
            the change in DoF value at site l. This variable is cached for efficiency
            and updated in place on subsequent calls. Make a copy before storing or
            modifying the returned array.
        """

        if self.dX is None:
            n_unitcells = self.supercell.n_unitcells
            n_dof_id = len(self.dof_id)
            self.dX = np.zeros((n_unitcells, n_dof_id), dtype=complex)
        else:
            self.dX.fill(0.0)

        self.update_X(
            l=l,
            old_value=old_value,
            new_value=new_value,
            X=self.dX,
        )
        return self.dX

    def update_X(
        self,
        l: int,  # noqa: E741
        old_value: Union[int, np.ndarray],
        new_value: Union[int, np.ndarray],
        X: np.ndarray,
    ):
        """Update DFT coefficients X for a change in DoF value at site l.

        Parameters
        ----------
        l: int
            Linear site index in the supercell where the DoF value is changed.
        old_value: Union[int, np.ndarray]
            The old DoF value at site l. For occupation DoFs, this is an integer
            specifying the occupied component index. For other DoFs, this is a
            np.ndarray of shape (n_components,) specifying the old local DoF values.
        new_value: Union[int, np.ndarray]
            The new DoF value at site l. For occupation DoFs, this is an integer
            specifying the occupied component index. For other DoFs, this is a
            np.ndarray of shape (n_components,) specifying the new local DoF values.
        X: np.ndarray
            A shape (n_dof_id, n_kpts) array of DFT coefficients to be updated in place.

        Returns
        -------
        X: np.ndarray
            The updated DFT coefficients array.
        """
        b = self.site_sublattice_indices[l]
        u = self.site_linear_unitcell_indices[l]

        if self.dof_key == "occ":
            for d in range(len(self.dof_id)):
                if self.dof_id[d][0] != b:
                    continue
                c = self.dof_id[d][1]
                if c == old_value:
                    old = 1
                    new = 0
                elif c == new_value:
                    old = 0
                    new = 1
                else:
                    continue
                X[d] += self.dft_phase[d] * self.dft_M[:, u] * (new - old)
        else:
            for d in range(len(self.dof_id)):
                if self.dof_id[d][0] != b:
                    continue
                c = self.dof_id[d][1]
                X[d] += (
                    self.dft_phase[d] * self.dft_M[:, u] * (new_value[c] - old_value[c])
                )
        return X


def make_plane_wave_basis(
    supercell_kpoints: SupercellKpoints,
    supercell_dof: SupercellDoF,
    kpoint_index: int,
    as_complex: bool = False,
) -> np.ndarray:
    """Make a plane wave basis for a set of k-points

    Parameters
    ----------
    supercell_kpoints: SupercellKpoints
        The SupercellKpoints instance containing k-point data.
    supercell_dof: SupercellDoF
        The SupercellDoF instance containing DoF space data.
    kpoint_index: int
        The index into `supercell_kpoints.coordinates` for which to construct the basis.
    as_complex: bool = False
        Whether to construct a complex-valued basis (True) or a real-valued basis
        (False, default).

    Returns
    -------
    basis: np.ndarray
        A shape (n_dof, n_basis) matrix. Each column is a basis vector, and
        n_basis is 2 * n_kpts for a real-valued basis (or less if some basis
        vectors are 0), or n_kpts for a complex-valued basis. Rows have the
        same meaning as rows in the :class:`libcasm.clexulator.DoFSpace` for
        the given `dof_key`. For example, if the DoF space is for atomic
        displacements, then rows correspond to Cartesian displacement components
        for each site in the supercell in the prim DoF basis. For a real-valued
        basis, each k-point contributes two sequential basis vectors: one with
        cosine phases and one with sine phases. For a complex-valued basis, each
        k-point contributes one basis vector with complex exponential phases.
    """
    kpt_value = supercell_kpoints.coordinates[:, kpoint_index]
    tol = 1e-8

    # Get the DoF space for the given dof_key in this supercell
    total_dim = supercell_dof.dof_space.basis.shape[0]
    basis_linear_site_index = supercell_dof.dof_space.axis_info.linear_site_index

    n_dof = len(supercell_dof.dof_id)

    if as_complex:
        # Complex-valued basis case:
        basis_complex = np.zeros((total_dim, n_dof), dtype=complex)

        i_col = 0
        for i_dof in range(n_dof):
            for i_row in range(total_dim):
                if supercell_dof.axis_dof_id[i_row] != i_dof:
                    continue
                i_n = basis_linear_site_index[i_row]
                R_n = supercell_kpoints.site_coordinate_cart[:, i_n]
                phase = np.exp(1j * np.dot(kpt_value, R_n))

                basis_complex[i_row, i_col] = phase
            i_col += 1
        basis = basis_complex
    else:

        # Real-valued basis case:
        basis_real = np.zeros((total_dim, n_dof * 2), dtype=float)

        i_col = 0
        for i_dof in range(n_dof):
            for i_row in range(total_dim):
                if supercell_dof.axis_dof_id[i_row] != i_dof:
                    continue
                i_n = basis_linear_site_index[i_row]
                R_n = supercell_kpoints.site_coordinate_cart[:, i_n]
                phase_cos = np.cos(np.dot(kpt_value, R_n))
                phase_sin = np.sin(np.dot(kpt_value, R_n))

                # Fill the cosine component
                basis_real[i_row, i_col] = phase_cos

                # Fill the sine component
                basis_real[i_row, i_col + 1] = phase_sin

            # Deal with TRIM points where sin- part and cos- part may be linearly
            # dependent.

            # Normalize the cos- part
            _normalize_column(basis=basis_real, i_col=i_col, tol=tol)

            # Orthogonalize the pair (Gram-Schmidt)
            proj_sin_on_cos = np.dot(
                basis_real[:, i_col + 1],
                basis_real[:, i_col],
            )
            basis_real[:, i_col + 1] -= proj_sin_on_cos * basis_real[:, i_col]

            # Normalize the sin- part
            _normalize_column(basis=basis_real, i_col=i_col + 1, tol=tol)

            i_col += 2

        basis = basis_real

    # Normalize the non-zero valued basis vectors
    for i_col in range(basis.shape[1]):
        _normalize_column(basis=basis, i_col=i_col, tol=tol)

    column_norms = np.linalg.norm(basis, axis=0)
    active_mask = column_norms > 1e-8

    # Remove zero columns:
    basis = basis[:, active_mask]
    dim = basis.shape[1]

    if not np.allclose(basis.T @ basis, np.eye(dim), atol=1e-8):
        raise ValueError("Error in make_plane_wave_basis: basis is not orthonormal.")

    return basis


def make_kpoint_irreps(
    supercell_kpoints: SupercellKpoints,
    supercell_dof: SupercellDoF,
    kpoint_index: int,
    symmetrization: str = "complete",
    verbosity: Optional[str] = None,
):
    """Make irreducible representations for a set of k-points

    Parameters
    ----------
    supercell_kpoints: SupercellKpoints
        The SupercellKpoints instance containing k-point data.
    supercell_dof: SupercellDoF
        The SupercellDoF instance containing DoF space data.
    kpoint_index: int
        The index into `supercell_kpoints.coordinates` for which to construct the
        irreps.
    symmetrization: str = "complete"
        The symmetrization method to use when constructing irreps. Options are
        "none", "fast", and "complete". See the documentation for
        :class:`libcasm.irreps.IrrepDecomposition` for details.
    verbosity: Optional[str] = None
        The verbosity level for logging the irrep decomposition. Options are None,
        "none", "standard", and "verbose".

    Returns
    -------
    irrep_decomposition: libcasm.irreps.IrrepDecomposition
        The irrep decomposition for the given k-points and dof_key.
    dof_space: libcasm.clexulator.DoFSpace
        The DoFSpace for the `kpts` with plane wave basis as initially constructed.
        The `irrep_decomposition` is constructed from the matrix representation
        acting on the basis of this DoF space.
    """
    verbosity_level = _get_verbosity_level(verbosity)

    if verbosity_level > 0:
        print(
            "Constructing irreps for k-point index "
            f"{kpoint_index} with symmetrization method '{symmetrization}'...",
            flush=True,
        )
        print(
            "Constructing plane wave basis...",
            flush=True,
        )

    # Make DoF space with plane wave basis
    basis = make_plane_wave_basis(
        supercell_kpoints=supercell_kpoints,
        supercell_dof=supercell_dof,
        kpoint_index=kpoint_index,
        as_complex=False,
    )

    dof_space = DoFSpace(
        dof_key=supercell_dof.dof_key,
        xtal_prim=supercell_kpoints.supercell.prim.xtal_prim,
        transformation_matrix_to_super=supercell_kpoints.supercell.transformation_matrix_to_super,
        basis=basis,
    )

    if verbosity_level > 0:
        print(
            "Constructing matrix representation for the little group of the k-point...",
            flush=True,
        )

    matrix_rep = make_dof_space_rep(
        group=supercell_kpoints.little_groups[kpoint_index],
        dof_space=dof_space,
    )
    if symmetrization == "none":
        if verbosity_level > 0:
            print(
                "No symmetrization selected; skipping subgroup orbit construction.",
                flush=True,
            )
        subgroup_orbits = None
    elif symmetrization == "fast":
        if verbosity_level > 0:
            print(
                "Constructing cyclic subgroup orbits for fast symmetrization...",
                flush=True,
            )
        symgroup = make_symgroup(supercell_kpoints.little_groups[kpoint_index])
        subgroup_orbits = get_cyclic_subgroup_orbits(
            symgroup=symgroup,
        )
    elif symmetrization == "complete":
        if verbosity_level > 0:
            print(
                "Constructing all subgroup orbits for complete symmetrization...",
                flush=True,
            )
        symgroup = make_symgroup(supercell_kpoints.little_groups[kpoint_index])
        subgroup_orbits = get_all_subgroup_orbits(
            symgroup=symgroup,
        )
    else:
        raise ValueError(f"Invalid symmetrization option: {symmetrization}")

    if verbosity_level > 0:
        print(
            f"Plane wave basis shape={basis.shape}, "
            f"Matrix rep size={len(matrix_rep)}, "
            f"Matrix rep element shape={matrix_rep[0].shape}",
            flush=True,
        )
        print(
            "Performing irrep decomposition...",
            flush=True,
        )
    # matrix_rep acts in the d-dimensional plane wave subspace;
    # init_subspace defaults to identity in that d-dimensional space.
    irrep_decomposition = IrrepDecomposition(
        matrix_rep=matrix_rep,
        allow_complex=False,
        subgroup_orbits=subgroup_orbits,
        verbosity=verbosity,
    )
    if verbosity_level > 0:
        print(
            f"Constructed {len(irrep_decomposition.irreps)} irreps for k-point index "
            f"{kpoint_index}.",
            flush=True,
        )
        B_sub = irrep_decomposition.symmetry_adapted_subspace
        print(
            f"Symmetry-adapted subspace shape (plane wave coords): {B_sub.shape}",
            flush=True,
        )
        print()

    return (irrep_decomposition, dof_space)


def make_unique_kpoint_irreps(
    supercell_kpoints: SupercellKpoints,
    supercell_dof: SupercellDoF,
    symmetrization: str = "complete",
    verbosity: Optional[str] = None,
) -> DoFSpace:
    """Make a symmetry-adapted DoF space for the entire supercell for a
    particular degree of freedom type.

    Notes
    -----
    This method constructs irreducible representations for each k-point orbit
    in the supercell, and combines the symmetry-adapted bases from each irrep
    decomposition into a full symmetry-adapted DoF space for the supercell.

    Note that this method results in symmetry-adapted modes that are restricted to the
    plane wave subspace for each k-point and do not mix with each other. This may
    result in axes that are less symmetrized than if the symmetry-adapted modes were
    constructed using the full supercell symmetry from the start.

    Parameters
    ----------
    supercell_kpoints: SupercellKpoints
        The SupercellKpoints instance containing k-point data.
    supercell_dof: SupercellDoF
        The SupercellDoF instance containing DoF space data.
    symmetrization: str = "complete"
        The symmetrization method to use when constructing irreps. Options are
        "none", "fast", and "complete". See the documentation for
        :class:`libcasm.irreps.IrrepDecomposition` for details. Note that a "complete"
        symmetrization in this case is only for the subspace associated with a single
        k-point and not an orbit of k-points.
    verbosity: Optional[str] = None
        The verbosity level for logging the irrep decomposition. Options are None,
        "none", "standard", and "verbose".

    Returns
    -------
    kpoint_irreps: list[libcasm.irreps.IrrepInfo]
        The list of irreps constructed for one prototype k-point in each k-point
        orbit.
    dof_space: libcasm.clexulator.DoFSpace
        The k-point by k-point symmetry-adapted DoF space for the given dof_key.
    axis_irrep_info: list[AxisIrrepInfo]
        Information about each axis in the final symmetry-adapted basis.
    """
    verbosity_level = _get_verbosity_level(verbosity)

    # Get the full DoF space for the supercell
    full_dof_space = supercell_dof.dof_space

    ## Iterate over k-point orbits to build irreps and symmetry-adapted bases ##

    # Irreps generated from a single k-point in each orbit
    kpoint_irreps: list[IrrepInfo] = []

    # Bases generated by applying the equivalence map to expand the symmetry-adapted
    # basis from the prototype k-point in each orbit to the other k-points in the
    # orbit. Combined at the end to form the full symmetry-adapted DoF space.
    orbit_adapted_bases: list[np.ndarray] = []

    # Information about each axis in the final symmetry-adapted basis
    axis_irrep_info = []
    next_irrep_char_index = 0

    # For each k-point orbit, build irreps for the plane wave basis from the first
    # k-point in the orbit, then use the equivalence map to extend the
    # symmetry-adapted basis to the other k-points in the orbit.
    for i_orbit, orbit in enumerate(supercell_kpoints.orbits):
        if verbosity_level > 0:
            print(
                "Processing k-point orbit "
                f"{i_orbit + 1}/{len(supercell_kpoints.orbits)}...",
                flush=True,
            )

        # Get irreps for first k-point in orbit
        kpoint_index = orbit[0]
        irrep_decomp, dof_space = make_kpoint_irreps(
            supercell_kpoints=supercell_kpoints,
            supercell_dof=supercell_dof,
            kpoint_index=kpoint_index,
            symmetrization=symmetrization,
            verbosity=verbosity,
        )

        # symmetry_adapted_subspace is in plane wave subspace coordinates (d-dim);
        # project back to the full DoF space (768-dim) via the plane wave basis.
        B = dof_space.basis @ irrep_decomp.symmetry_adapted_subspace
        B = pretty(B)

        # Matrix reps for independent orbit members (skipping −k partners)
        independent_indices = supercell_kpoints.orbit_independent_indices[i_orbit]
        independent_ops = [
            supercell_kpoints.equivalence_map[i_orbit][j] for j in independent_indices
        ]
        equivalence_map_reps = make_dof_space_rep(
            group=independent_ops,
            dof_space=full_dof_space,
        )

        # For each irrep in the decomposition, extend basis to full orbit and
        # store axis irrep info
        i_axis = 0
        for i_irrep, irrep in enumerate(irrep_decomp.irreps):

            d = irrep.irrep_dim

            # Apply equivalence map ops for independent orbit members to generate
            # orthogonal subspaces. −k partners are skipped because the real-valued
            # plane wave basis for k and −k spans the same subspace.
            kpoint_adapted_basis = B[:, i_axis : i_axis + d]
            # Sanity check: columns of kpoint_adapted_basis should be orthonormal
            inner = kpoint_adapted_basis.T @ kpoint_adapted_basis
            if not np.allclose(inner, np.eye(d), atol=1e-8):
                raise ValueError(
                    "Error in make_unique_kpoint_irreps: kpoint_adapted_basis "
                    "columns are not orthonormal."
                )
            orbit_cols = [kpoint_adapted_basis]
            orbit_cols_kpoint_index = [kpoint_index] * d
            for j, D in enumerate(equivalence_map_reps):
                if j == 0:
                    continue
                transformed = pretty(D @ kpoint_adapted_basis)
                # Sanity check: transformed subspace should be orthogonal to prototype
                overlap = kpoint_adapted_basis.T @ transformed
                if not np.allclose(overlap, np.zeros_like(overlap), atol=1e-8):
                    raise ValueError(
                        "Error in make_unique_kpoint_irreps: equivalence map "
                        "transformed subspace is not orthogonal to prototype subspace."
                    )
                orbit_cols.append(transformed)
                transformed_kpoint_index = orbit[independent_indices[j]]
                orbit_cols_kpoint_index.extend([transformed_kpoint_index] * d)
            orbit_adapted_basis = np.hstack(orbit_cols)

            orbit_adapted_bases.append(orbit_adapted_basis)

            # Find unique index for this irrep's characters
            irrep_char_index, next_irrep_char_index = _get_irrep_char_index(
                irrep=irrep,
                kpoint_irreps=kpoint_irreps,
                axis_irrep_info=axis_irrep_info,
                next_irrep_char_index=next_irrep_char_index,
            )

            # Store axis irrep info
            for j in range(orbit_adapted_basis.shape[1]):
                kpt = orbit_cols_kpoint_index[j]
                neg_kpt = supercell_kpoints.neg_kpoint_index[kpt]
                if neg_kpt is None:
                    x = [kpt]
                else:
                    x = sorted([kpt, neg_kpt])
                axis_irrep_info.append(
                    AxisIrrepInfo(
                        orbit_index=i_orbit,
                        kpoint_index=x,
                        kpoint_irreps_index=len(kpoint_irreps),
                        irrep_char_index=irrep_char_index,
                    )
                )

            # Store k-point irrep
            kpoint_irreps.append(irrep)
            i_axis += d

    # Combine all symmetry-adapted bases from each irrep decomposition. Each
    # orbit-adapted basis should be orthogonal to all previously accumulated bases.
    symmetry_adapted_basis = orbit_adapted_bases[0]
    for basis in orbit_adapted_bases[1:]:
        overlap = symmetry_adapted_basis.T @ basis
        if not np.allclose(overlap, np.zeros_like(overlap), atol=1e-8):
            raise ValueError(
                "Error in make_unique_kpoint_irreps: orbit-adapted basis is not "
                "orthogonal to previously accumulated symmetry-adapted basis."
            )
        symmetry_adapted_basis = np.hstack([symmetry_adapted_basis, basis])

    dof_space = DoFSpace(
        dof_key=supercell_dof.dof_key,
        xtal_prim=supercell_kpoints.supercell.prim.xtal_prim,
        transformation_matrix_to_super=supercell_kpoints.supercell.transformation_matrix_to_super,
        basis=symmetry_adapted_basis,
    )
    return (
        kpoint_irreps,
        dof_space,
        axis_irrep_info,
    )


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) with ``a*x + b*y = g = gcd(|a|, |b|)``, ``g >= 0``."""
    if a < 0:
        g, x, y = _extended_gcd(-a, b)
        return g, -x, y
    if b < 0:
        g, x, y = _extended_gcd(a, -b)
        return g, x, -y
    if b == 0:
        return a, 1, 0
    g, x, y = _extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def _row_reduce_to_smith(h: np.ndarray) -> tuple[np.ndarray, int]:
    """Find unimodular ``R`` such that ``R @ h == [g, 0, 0]^T``.

    Parameters
    ----------
    h: np.ndarray
        Integer column vector of shape (3,).

    Returns
    -------
    R: np.ndarray
        3x3 unimodular integer matrix (det = +1) with ``R @ h == [g, 0, 0]^T``.
    g: int
        ``gcd(|h[0]|, |h[1]|, |h[2]|) >= 0``.
    """
    h1, h2, h3 = int(h[0]), int(h[1]), int(h[2])

    # Step 1: eliminate h[1] using rows 0 and 1
    if h1 == 0 and h2 == 0:
        R1 = np.eye(3, dtype=int)
        g12 = 0
    else:
        g12, a, b = _extended_gcd(h1, h2)
        R1 = np.array(
            [
                [a, b, 0],
                [-(h2 // g12), h1 // g12, 0],
                [0, 0, 1],
            ],
            dtype=int,
        )

    # Step 2: eliminate h[2] using rows 0 and 2 of (R1 @ h) = [g12, 0, h3]
    if g12 == 0 and h3 == 0:
        R2 = np.eye(3, dtype=int)
        g = 0
    else:
        g, a, b = _extended_gcd(g12, h3)
        R2 = np.array(
            [
                [a, 0, b],
                [0, 1, 0],
                [-(h3 // g), 0, g12 // g],
            ],
            dtype=int,
        )

    R = R2 @ R1
    return R, g


def min_supercell_for_kpoint(
    prim: Prim,
    q: np.ndarray,
    max_denominator: int = 10000,
) -> Supercell:
    """Find the minimum volume supercell commensurate with a given k-point.

    A k-point ``q`` (in primitive reciprocal lattice fractional coordinates) is
    commensurate with supercell transformation matrix ``T`` if and only if
    ``T^T @ q`` is an integer vector. This function finds the ``T`` with
    minimum ``|det(T)|`` satisfying that condition.

    Writing each component in lowest terms as ``qi = pi / di``, with
    ``lcm = lcm(d1, d2, d3)`` and integer numerators ``hi = pi * lcm / di``,
    the minimum volume is ``D' = lcm / gcd(lcm, gcd(h1, h2, h3))``.

    Parameters
    ----------
    prim: Prim
        The primitive cell.
    q: array-like of float
        K-point in primitive reciprocal lattice fractional coordinates, as a
        shape (3,) array. Each component must be rational (or sufficiently close
        to a rational with denominator <= ``max_denominator``).
    max_denominator: int = 10000
        Maximum denominator used when approximating each component of ``q`` as
        a rational number via ``fractions.Fraction.limit_denominator``.

    Returns
    -------
    supercell: Supercell
        The minimum volume :class:`~libcasm.configuration.Supercell` commensurate
        with the given k-point.
    """
    q = np.asarray(q, dtype=float)

    # Express q = h / D: integer vector h, integer scalar D = lcm of denominators
    fracs = [Fraction(float(qi)).limit_denominator(max_denominator) for qi in q]
    D = lcm(fracs[0].denominator, fracs[1].denominator, fracs[2].denominator)
    h = np.array(
        [f.numerator * (D // f.denominator) for f in fracs],
        dtype=int,
    )

    g = gcd(abs(int(h[0])), abs(int(h[1])), abs(int(h[2])))
    D_prime = D // gcd(D, g) if g != 0 else 1  # minimum supercell volume

    if D_prime == 1:
        T = np.eye(3, dtype=int)
    else:
        # Find unimodular R such that R @ h = [g, 0, 0]^T
        R, _ = _row_reduce_to_smith(h)

        # T^T = diag(D', 1, 1) @ R  →  T = R^T @ diag(D', 1, 1)
        T = (np.diag([D_prime, 1, 1]) @ R).T

    return Supercell(prim=prim, transformation_matrix_to_super=T)
