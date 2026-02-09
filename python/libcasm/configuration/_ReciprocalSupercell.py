from typing import Optional, Union

import numpy as np

from libcasm.clexulator import DoFSpace
from libcasm.irreps import IrrepDecomposition, IrrepInfo, make_invariant_subspace
from libcasm.xtal import (
    UnitCellIndexConverter,
    min_periodic_displacement,
)

from ._configuration import (
    Configuration,
    Supercell,
    SupercellSymOp,
    make_dof_space_rep,
)


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
    kpoint_irreps_index: int
        Index of into `kpoint_irreps` list of the irrep this axis belongs to.
    irrep_char_index: int
        An index identifying irreps which have the same characters.
    """

    def __init__(
        self,
        orbit_index: int,
        kpoint_irreps_index: int,
        irrep_char_index: int,
    ):
        self.orbit_index = orbit_index
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
    for _i, _irrep in enumerate(kpoint_irreps):
        if len(_irrep.characters) != len(irrep.characters):
            continue
        if np.allclose(
            _irrep.characters,
            irrep.characters,
            atol=1e-8,
        ):
            found_index = axis_irrep_info[_i].irrep_char_index
            break
    if found_index == -1:
        irrep_char_index = next_irrep_char_index
        next_irrep_char_index += 1
    else:
        irrep_char_index = found_index

    return (irrep_char_index, next_irrep_char_index)


class ReciprocalSupercell:
    """K-point generation, discrete fourier transforms, etc. for a particular Supercell
    and degree of freedom type."""

    def __init__(self, supercell: Supercell, dof_key: str, tol: float = 1e-5):
        self.supercell = supercell
        """libcasm.configuration.Supercell: The supercell for which commensurate 
        k-points are generated.

        This Supercell supplies the superlattice, factor group (symmetry operations),
        and unitcell index converters used during k-point enumeration and symmetry
        orbit construction.
        """

        self.dof_key = dof_key
        """str: Degree of freedom key for which DoF spaces are constructed."""

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
        self.kpoints_converter = None
        """xtal.UnitCellIndexConverter: Enumerates k-points commensurate with the 
        supercell.

        The converter is constructed with `transformation_matrix_to_super = T_recip` 
        where `T_recip` is the integer matrix found by solving 
        ``S_recip @ T_recip = P_recip`` (reciprocal lattice of the primitive cell 
        column matrix). It enumerates k-points commensurate with the supercell.
        """

        self.kpoint_indices = None
        """np.ndarray: Integer unit-cell index column vectors for each k-point with 
        shape (3, N).

        Each column is a 3x1 integer column vector giving the index in the
        supercell unit-cell index basis that maps to the corresponding column in
        `self.kpoints` via `k = S_recip @ index`.
        """

        self.kpoints = None
        """np.ndarray: Array of k-points with shape (3, N), each column is a k-point.

        The k-points are obtained as S_recip @ index for every index enumerated
        by `self.kpoints_converter`. Columns correspond 1:1 with `self.kpoint_indices`.
        """

        self.kpoint_orbits = []
        """list[list[int]]: Symmetry orbits of k-point indices under the supercell 
        factor group.

        Each orbit is a list of integer indices referring to columns of `self.kpoints`.
        Orbits enumerate k-points that are symmetry-equivalent under the supercell's
        factor group.
        """

        self.kpoint_equivalence_map = []
        """list[list[libcasm.xtal.SymOp]: List of symmetry operations mapping the first
        k-point in each orbit to the other k-points in the orbit.
        """

        self.supercell_factor_group = []
        """list[libcasm.configuration.SupercellSymOp]: Symmetry operations of the
        supercell factor group."""

        self.little_groups = []
        """list[list[libcasm.configuration.SupercellSymOp]]: Little groups for each
        k-point in `self.kpoints`.
        
        The little group for each k-point is a list of symmetry operations from the
        supercell factor group that leave the k-point invariant.
        """

        # self.little_group_shifts = []
        # """list[list[np.ndarray]]: List of shift vectors for each operation in the
        # little groups.
        #
        # Each shift vector corresponds to the translation component of the associated
        # symmetry operation in the little group.
        # """

        # self.dof_space: Optional[DoFSpace] = None
        # """Optional[libcasm.clexulator.DoFSpace]: Degree of freedom space with
        # identity basis for the last call to `make_bloch_basis`.
        #
        # This is populated when `make_bloch_basis` is called and can be used to
        # interpret the basis vectors returned by that method.
        # """

        self.dof_space = None
        """libcasm.clexulator.DoFSpace]: DoF space the entire supercell, with identity
        basis.
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

        self.irrep_decomposition_by_orbit: Optional[list[IrrepDecomposition]] = None
        """Optional[list[libcasm.irreps.IrrepDecomposition]]: List of irrep 
        decompositions for each k-point orbit for the last call of 
        `make_symmetry_adapted_dof_space`.
        
        This is populated when `make_symmetry_adapted_dof_space` is called and can be 
        used to interpret the symmetry-adapted basis returned by that method. These
        are irreps constructed on bases for the corresponding DoFSpace stored in
        `dof_space_by_orbit`.
        """

        self.dof_space_by_orbit: Optional[list[DoFSpace]] = None
        """Optional[list[libcasm.clexulator.DoFSpace]]: List of DoF spaces for each
        k-point orbit for the last call of `make_symmetry_adapted_dof_space`.
        
        This is populated when `make_symmetry_adapted_dof_space` is called.
        """

        # build data
        self._build_dof_space()
        self._build_kpoints()
        self._build_kpoint_orbits()
        self._build_little_groups()
        # self._build_dft_matrices()

    def _build_dof_space(self) -> DoFSpace:
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

            c = basis_dof_component_index[lsi]
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

    def _build_kpoints(self):
        """Populate self.kpoints, self.kpoint_indices and self.kpoints_converter."""
        S_recip = self.reciprocal_superlattice.column_vector_matrix()
        T_recip = self.supercell.transformation_matrix_to_super.T
        self.kpoints_converter = UnitCellIndexConverter(
            transformation_matrix_to_super=T_recip,
        )

        kpt_indices_list = []
        for i in range(self.kpoints_converter.total_unitcells()):
            kpt_indices = self.kpoints_converter.unitcell(i)
            # Ensure column-vector shape (3, 1) for each index
            kpt_col = np.asarray(kpt_indices).reshape(3, -1)
            kpt_indices_list.append(kpt_col)

        if kpt_indices_list:
            # Stack index column-vectors into a 3 x N integer array
            self.kpoint_indices = np.hstack(kpt_indices_list).astype(int)
        else:
            self.kpoint_indices = np.empty((3, 0), dtype=int)

        # Compute k-points by applying S_recip to the integer index columns
        self.kpoints = S_recip @ self.kpoint_indices
        return

    def _get_kpoint_index(self, kpt: np.array) -> int:
        """Find index of kpt in self.kpoints using periodic displacement on primitive
        lattice."""
        for i in range(self.kpoints.shape[1]):
            d = min_periodic_displacement(
                lattice=self.reciprocal_prim_lattice,
                r1=kpt,
                r2=self.kpoints[:, i],
            )
            if np.linalg.norm(d) < self.tol:
                return i
        return -1

    def _build_kpoint_orbits(self):
        """Populate self.kpoint_orbits using symmetry operations of the supercell."""
        if self.kpoints is None:
            raise RuntimeError("kpoints must be built before computing orbits")

        found = [False] * self.kpoints.shape[1]
        kpoint_orbits = []
        kpoint_equivalence_map = []
        elements = self.supercell.factor_group.elements
        for i in range(self.kpoints.shape[1]):
            if found[i]:
                continue
            kpt = self.kpoints[:, i]
            orbit = [i]
            equivalence_map = [elements[0].copy()]
            found[i] = True
            for op in elements:
                matrix = op.matrix()
                kpt_transformed = matrix @ kpt
                kpt_index = self._get_kpoint_index(kpt_transformed)
                if kpt_index == -1:
                    raise ValueError(
                        "Transformed k-point not found in list of k-points"
                    )
                if not found[kpt_index]:
                    orbit.append(kpt_index)
                    equivalence_map.append(op.copy())
                    found[kpt_index] = True
            kpoint_orbits.append(orbit)
            kpoint_equivalence_map.append(equivalence_map)
        self.kpoint_orbits = kpoint_orbits
        self.kpoint_equivalence_map = kpoint_equivalence_map

        # Build supercell factor group list of SupercellSymOp objects
        op = SupercellSymOp.begin(self.supercell)
        end = SupercellSymOp.end(self.supercell)
        while op != end:
            if op.translation_index() == 0:
                self.supercell_factor_group.append(op.copy())
            op.next()

        return

    def _build_little_groups(self):
        """Populate self.little_groups using symmetry operations of the supercell."""
        if self.kpoints is None:
            raise RuntimeError("kpoints must be built before computing orbits")

        little_groups = []
        for i in range(self.kpoints.shape[1]):
            little_group = []

            kpt = self.kpoints[:, i]

            it = SupercellSymOp.begin(self.supercell)
            end = SupercellSymOp.end(self.supercell)
            while it != end:
                op = it.copy()
                matrix = op.to_symop().matrix()
                kpt_transformed = matrix @ kpt
                kpt_index = self._get_kpoint_index(kpt_transformed)

                if kpt_index == -1:
                    raise ValueError(
                        "Transformed k-point not found in list of k-points"
                    )
                if kpt_index == i:
                    little_group.append(op)
                it.next()

            print(f"Little group for k-point {i} has {len(little_group)} elements.")
            little_groups.append(little_group)
        self.little_groups = little_groups

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
            k = self.kpoints[:, i_row]

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
            the order specified by :py:attr:`kpoints`.
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
            the order specified by :py:attr:`kpoints`.

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

    def make_bloch_basis(
        self,
        kpoint_index: int,
        as_complex: bool = False,
    ) -> np.ndarray:
        """Make a Bloch plane wave basis for a set of k-points

        Parameters
        ----------
        kpoint_index: int
            The index into `self.kpoints` for which to construct the basis.
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
        kpt_value = self.kpoints[:, kpoint_index]
        tol = 1e-8

        # Get the DoF space for the given dof_key in this supercell
        total_dim = self.dof_space.basis.shape[0]
        basis_linear_site_index = self.dof_space.axis_info.linear_site_index

        n_dof = len(self.dof_id)

        if as_complex:
            # Complex-valued basis case:
            basis_complex = np.zeros((total_dim, n_dof), dtype=complex)

            i_col = 0
            for i_dof in range(n_dof):
                for i_row in range(total_dim):
                    if self.axis_dof_id[i_row] != i_dof:
                        continue
                    i_n = basis_linear_site_index[i_row]
                    R_n = self.site_coordinate_cart[:, i_n]
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
                    if self.axis_dof_id[i_row] != i_dof:
                        continue
                    i_n = basis_linear_site_index[i_row]
                    R_n = self.site_coordinate_cart[:, i_n]
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

            print("Basis shape (as constructed):", basis.shape)

        # Normalize the non-zero valued basis vectors
        for i_col in range(basis.shape[1]):
            _normalize_column(basis=basis, i_col=i_col, tol=tol)

        column_norms = np.linalg.norm(basis, axis=0)
        active_mask = column_norms > 1e-8

        # Remove zero columns:
        basis = basis[:, active_mask]
        dim = basis.shape[1]
        print("Final basis shape (after removing zero columns):", basis.shape)

        q, r = np.linalg.qr(basis)
        rank = np.linalg.matrix_rank(r)
        print("Basis rank:", rank)

        if not np.allclose(basis.T @ basis, np.eye(dim), atol=1e-8):
            print("basis.T @ basis:")
            x = basis.T @ basis
            x[np.abs(x) < 1e-8] = 0.0
            print(x)
            raise ValueError("Error in make_bloch_basis: basis is not orthonormal.")

        return basis

    def make_kpoint_irreps(
        self,
        kpoint_index: int,
        symmetrization: str = "complete",
        verbosity: str = None,
    ):
        """Make irreducible representations for a set of k-points

        Parameters
        ----------
        kpoint_index: int
            The index into `self.kpoints` for which to construct the irreps.
        dof_key: str
            The degree of freedom key for which to construct the irreps.
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
            The DoFSpace for the `kpts` with Bloch basis as initially constructed.
            The `irrep_decomposition` is constructed from the matrix representation
            acting on the basis of this DoF space.
        """
        print("Begin make_kpoint_irreps for kpoint index:", kpoint_index)

        # Make DoF space with Bloch basis
        print("Construct Bloch basis...")
        basis = self.make_bloch_basis(
            kpoint_index=kpoint_index,
            as_complex=False,
        )
        print("Bloch basis shape:", basis.shape)
        dof_space = DoFSpace(
            dof_key=self.dof_key,
            xtal_prim=self.supercell.prim.xtal_prim,
            transformation_matrix_to_super=self.supercell.transformation_matrix_to_super,
            basis=basis,
        )

        full_space_matrix = True
        if full_space_matrix:
            print("Construct Bloch basis matrix representation...")
            matrix_rep = make_dof_space_rep(
                group=self.little_groups[kpoint_index],
                dof_space=self.dof_space,
            )
            print("Matrix rep size:", len(matrix_rep))
            print("matrix size:", matrix_rep[0].shape)

            irrep_decomposition = IrrepDecomposition(
                matrix_rep=matrix_rep,
                symmetrization=symmetrization,
                allow_complex=False,
                init_subspace=basis,
                verbosity=verbosity,
            )

        else:

            dof_space = DoFSpace(
                dof_key=self.dof_key,
                xtal_prim=self.supercell.prim.xtal_prim,
                transformation_matrix_to_super=self.supercell.transformation_matrix_to_super,
                basis=basis,
            )

            print("Construct reduced space Bloch basis matrix representation...")
            matrix_rep = make_dof_space_rep(
                group=self.little_groups[kpoint_index],
                dof_space=dof_space,
            )
            print("Matrix rep size:", len(matrix_rep))
            matrix_rep = _make_unique(matrix_rep)
            print("Unique Matrix rep size:", len(matrix_rep))
            print("matrix size:", matrix_rep[0].shape)

            irrep_decomposition = IrrepDecomposition(
                matrix_rep=matrix_rep,
                symmetrization=symmetrization,
                allow_complex=False,
                verbosity=verbosity,
            )

        print("DONE")
        print()
        print()
        return (irrep_decomposition, dof_space)

    # def make_orbit_irreps(
    #     self,
    #     kpoint_orbit_index: int,
    #     dof_key: str,
    #     symmetrization: str = "complete",
    # ):
    #     """Make irreducible representations for a k-point orbit
    #
    #     Parameters
    #     ----------
    #     kpoint_orbit_index: int
    #         The index into `self.kpoint_orbits` for which to construct the irreps.
    #     dof_key: str
    #         The degree of freedom key for which to construct the irreps.
    #     symmetrization: str = "complete"
    #         The symmetrization method to use when constructing irreps. Options are
    #         "none", "fast", and "complete". See the documentation for
    #         :class:`libcasm.irreps.IrrepDecomposition` for details.
    #
    #     Returns
    #     -------
    #     irrep_decompositions: list[libcasm.irreps.IrrepDecomposition]
    #         The irrep decompositions for the given k-point orbit and dof_key.
    #     dof_spaces: libcasm.clexulator.DoFSpace
    #         The DoFSpace for each k-point in the orbit with Bloch basis as initially
    #         constructed. The `irrep_decompositions` are constructed for the matrix
    #         representations acting on the bases of these DoF spaces.
    #     """
    #     orbit = self.kpoint_orbits[kpoint_orbit_index]
    #     irrep_decompositions = []
    #     dof_spaces = []
    #     for kpoint_index in orbit:
    #         irrep_decomp, dof_space = self.make_kpt_irreps(
    #             kpoint_index=kpoint_index,
    #             dof_key=dof_key,
    #             symmetrization=symmetrization,
    #         )
    #         irrep_decompositions.append(irrep_decomp)
    #         dof_spaces.append(dof_space)
    #     return (irrep_decompositions, dof_spaces)

    def make_symmetry_adapted_dof_space(
        self,
        symmetrization: str = "complete",
    ) -> DoFSpace:
        """Make a symmetry-adapted DoF space for the entire supercell for a
        particular degree of freedom type.

        Notes
        -----
        This method constructs irreducible representations for each k-point orbit
        in the supercell, and combines the symmetry-adapted bases from each irrep
        decomposition into a full symmetry-adapted DoF space for the supercell.

        As a side effect, the list of irrep decompositions for each k-point orbit
        is stored in `self.irrep_decompositions` for later reference.

        Parameters
        ----------
        symmetrization: str = "complete"
            The symmetrization method to use when constructing irreps. Options are
            "none", "fast", and "complete". See the documentation for
            :class:`libcasm.irreps.IrrepDecomposition` for details.

        Returns
        -------
        dof_space: libcasm.clexulator.DoFSpace
            The symmetry-adapted DoF space for the given dof_key.
        kpoint_irreps: list[libcasm.irreps.IrrepInfo]
            The list of irreps constructed for one prototype k-point in each k-point
            orbit.
        axis_irrep_info: list[AxisIrrepInfo]
            Information about each axis in the final symmetry-adapted basis.
        """

        # Get the full DoF space for the supercell
        full_dof_space = self.dof_space

        print("Full DoFSpace basis shape:", full_dof_space.basis.shape)

        # Make matrix representation of supercell factor group on full DoF space
        factor_group = []
        it = SupercellSymOp.begin(self.supercell)
        end = SupercellSymOp.end(self.supercell)
        while it != end:
            if it.translation_index() == 0:
                factor_group.append(it.copy())
            it.next()
        matrix_rep = make_dof_space_rep(
            group=factor_group,
            dof_space=full_dof_space,
        )

        ## Iterate over k-point orbits to build irreps and symmetry-adapted bases ##

        # Irreps generated from a single k-point in each orbit
        kpoint_irreps: list[IrrepInfo] = []

        # Bases generated by applying the supercell factor group to expand the symmetry
        # adapted basis from a single k-point in each orbit to the full orbit. This
        # will be combined at the end to form the full symmetry-adapted DoF space.
        orbit_adapted_bases: list[np.ndarray] = []

        # Information about each axis in the final symmetry-adapted basis
        axis_irrep_info = []
        next_irrep_char_index = 0

        # For each k-point orbit, build irreps for the Bloch basis from the first
        # k-point in the orbit, then apply the supercell factor group to extend the
        # symmetry-adapted basis to the space for the full orbit.
        for i_orbit, orbit in enumerate(self.kpoint_orbits):
            # if i_orbit == 0:
            #     print("Skipping Gamma-point orbit (handled separately)")
            #     continue

            # Get irreps for first k-point in orbit
            kpoint_index = orbit[0]
            irrep_decomp, dof_space = self.make_kpoint_irreps(
                kpoint_index=kpoint_index,
                symmetrization=symmetrization,
            )

            B = irrep_decomp.symmetry_adapted_subspace

            # For each irrep in the decomposition, extend basis to full orbit and
            # store axis irrep info
            i_axis = 0
            for i_irrep, irrep in enumerate(irrep_decomp.irreps):

                d = irrep.irrep_dim
                # print(" Irrep", i_irrep, "dimension:", d)

                # Apply supercell factor group to adapt basis to full symmetry
                kpoint_adapted_basis = B[:, i_axis : i_axis + d]
                orbit_adapted_basis = make_invariant_subspace(
                    matrix_rep=matrix_rep,
                    init_subspace=kpoint_adapted_basis,
                )
                q, r = np.linalg.qr(kpoint_adapted_basis)
                kpt_rank = np.linalg.matrix_rank(r)
                # print("   Rank of kpoint-adapted-basis:", np.linalg.matrix_rank(r))
                q, r = np.linalg.qr(orbit_adapted_basis)
                orbit_rank = np.linalg.matrix_rank(r)
                # print("   Rank of orbit-adapted-basis:", np.linalg.matrix_rank(r))
                orbit_adapted_bases.append(orbit_adapted_basis)

                print(
                    f" Irrep {i_irrep}: "
                    f"Dim={d} KptRank={kpt_rank} OrbitRank={orbit_rank}"
                )

                # Find unique index for this irrep's characters
                irrep_char_index, next_irrep_char_index = _get_irrep_char_index(
                    irrep=irrep,
                    kpoint_irreps=kpoint_irreps,
                    axis_irrep_info=axis_irrep_info,
                    next_irrep_char_index=next_irrep_char_index,
                )

                # Store axis irrep info
                for _ in range(orbit_adapted_basis.shape[1]):
                    axis_irrep_info.append(
                        AxisIrrepInfo(
                            orbit_index=i_orbit,
                            kpoint_irreps_index=len(kpoint_irreps),
                            irrep_char_index=irrep_char_index,
                        )
                    )

                # Store k-point irrep
                kpoint_irreps.append(irrep)
                i_axis += d

        # Combine all symmetry-adapted bases from each irrep decomposition

        symmetry_adapted_basis = np.hstack(orbit_adapted_bases)
        print(
            "Symmetry-adapted basis shape (as constructed):",
            symmetry_adapted_basis.shape,
        )
        q, r = np.linalg.qr(symmetry_adapted_basis)
        rank = np.linalg.matrix_rank(r)
        print(" q.shape:", q.shape)
        print(" Rank of symmetry-adapted basis:", rank)
        symmetry_adapted_basis = q[:, :rank]

        dof_space = DoFSpace(
            dof_key=self.dof_key,
            xtal_prim=self.supercell.prim.xtal_prim,
            transformation_matrix_to_super=self.supercell.transformation_matrix_to_super,
            basis=symmetry_adapted_basis,
        )
        return (
            dof_space,
            kpoint_irreps,
            axis_irrep_info,
        )
