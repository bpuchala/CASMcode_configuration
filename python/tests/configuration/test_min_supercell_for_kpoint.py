"""Tests for min_supercell_for_kpoint."""

import numpy as np
import pytest

import libcasm.configuration as casmconfig


def _is_commensurate(supercell, q):
    """Return True if k-point q (primitive recip fractional coords) is commensurate."""
    T = supercell.transformation_matrix_to_super.astype(float)
    result = T.T @ np.asarray(q)
    return np.allclose(result, np.round(result), atol=1e-8)


def _volume(supercell):
    return abs(round(np.linalg.det(supercell.transformation_matrix_to_super)))


@pytest.fixture
def simple_cubic_prim(simple_cubic_binary_prim):
    return casmconfig.Prim(xtal_prim=simple_cubic_binary_prim)


@pytest.fixture
def fcc_prim(FCC_binary_prim):
    return casmconfig.Prim(xtal_prim=FCC_binary_prim)


class TestGammaPoint:
    """Gamma point (q=[0,0,0]) requires only the primitive cell (volume 1)."""

    def test_volume(self, simple_cubic_prim):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, [0, 0, 0])
        assert _volume(sc) == 1

    def test_commensurate(self, simple_cubic_prim):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, [0, 0, 0])
        assert _is_commensurate(sc, [0, 0, 0])

    def test_identity_T(self, simple_cubic_prim):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, [0, 0, 0])
        assert np.array_equal(sc.transformation_matrix_to_super, np.eye(3, dtype=int))


class TestZoneEdge:
    """k-points along a single axis: [1/n, 0, 0] needs volume n."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_volume(self, simple_cubic_prim, n):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, [1 / n, 0, 0])
        assert _volume(sc) == n

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_commensurate(self, simple_cubic_prim, n):
        q = [1 / n, 0, 0]
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, q)
        assert _is_commensurate(sc, q)


class TestNonAxisKpoints:
    """k-points with two equal nonzero components can be hosted in a non-diagonal
    supercell whose volume equals the common denominator, not its square.

    For q = [1/n, 1/n, 0], D' = n rather than n^2 (diagonal would give n^2).
    """

    @pytest.mark.parametrize(
        "q, expected_volume",
        [
            ([1 / 2, 1 / 2, 0], 2),  # not 4
            ([1 / 3, 1 / 3, 0], 3),  # not 9
            ([1 / 4, 1 / 4, 0], 4),  # not 16
            ([1 / 2, 1 / 2, 1 / 2], 2),  # not 8 — R-point of simple cubic
            ([1 / 3, 1 / 3, 1 / 3], 3),  # not 27
        ],
    )
    def test_volume(self, simple_cubic_prim, q, expected_volume):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, q)
        assert _volume(sc) == expected_volume

    @pytest.mark.parametrize(
        "q",
        [
            [1 / 2, 1 / 2, 0],
            [1 / 3, 1 / 3, 0],
            [1 / 4, 1 / 4, 0],
            [1 / 2, 1 / 2, 1 / 2],
            [1 / 3, 1 / 3, 1 / 3],
        ],
    )
    def test_commensurate(self, simple_cubic_prim, q):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, q)
        assert _is_commensurate(sc, q)


class TestMixedDenominators:
    """k-points whose components have different denominators."""

    @pytest.mark.parametrize(
        "q, expected_volume",
        [
            ([1 / 2, 1 / 3, 0], 6),  # lcm(2,3) = 6, gcd(h)=gcd(3,2,0)=1 → D'=6
            ([1 / 2, 1 / 4, 0], 4),  # lcm(2,4) = 4, h=[2,1,0], gcd=1 → D'=4
            ([1 / 3, 1 / 6, 0], 6),  # lcm(3,6) = 6, h=[2,1,0], gcd=1 → D'=6
            ([2 / 3, 0, 0], 3),  # denominator 3, h=[2,0,0], gcd=2, D'=3/gcd(3,2)=3
            ([2 / 4, 0, 0], 2),  # 2/4 = 1/2, D'=2
        ],
    )
    def test_volume(self, simple_cubic_prim, q, expected_volume):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, q)
        assert _volume(sc) == expected_volume

    @pytest.mark.parametrize(
        "q",
        [
            [1 / 2, 1 / 3, 0],
            [1 / 2, 1 / 4, 0],
            [1 / 3, 1 / 6, 0],
            [2 / 3, 0, 0],
        ],
    )
    def test_commensurate(self, simple_cubic_prim, q):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, q)
        assert _is_commensurate(sc, q)


class TestFCCPrim:
    """Tests using the FCC primitive cell, where the reciprocal lattice is BCC."""

    @pytest.mark.parametrize(
        "q, expected_volume",
        [
            ([0, 0, 0], 1),  # Gamma
            ([1 / 2, 0, 0], 2),  # zone edge
            ([1 / 2, 1 / 2, 0], 2),  # L-point (min vol non-diagonal)
            ([1 / 2, 1 / 2, 1 / 2], 2),  # min vol
            ([1 / 3, 1 / 3, 0], 3),
        ],
    )
    def test_volume(self, fcc_prim, q, expected_volume):
        sc = casmconfig.min_supercell_for_kpoint(fcc_prim, q)
        assert _volume(sc) == expected_volume

    @pytest.mark.parametrize(
        "q",
        [
            [0, 0, 0],
            [1 / 2, 0, 0],
            [1 / 2, 1 / 2, 0],
            [1 / 2, 1 / 2, 1 / 2],
            [1 / 3, 1 / 3, 0],
        ],
    )
    def test_commensurate(self, fcc_prim, q):
        sc = casmconfig.min_supercell_for_kpoint(fcc_prim, q)
        assert _is_commensurate(sc, q)


class TestVerifyWithSupercelKpoints:
    """Verify the k-point appears in SupercellKpoints.coordinates."""

    def _kpoint_in_sc(self, prim, supercell, q):
        """Return True if q (fractional recip coords) is in supercell's k-point grid."""
        B = prim.xtal_prim.lattice().reciprocal().column_vector_matrix()
        k_cart = B @ np.asarray(q)
        sc_kpts = casmconfig.SupercellKpoints(supercell=supercell)
        B_inv = np.linalg.inv(B)
        for i in range(sc_kpts.coordinates.shape[1]):
            diff = sc_kpts.coordinates[:, i] - k_cart
            # allow difference by any reciprocal lattice vector
            diff_frac = B_inv @ diff
            if np.allclose(diff_frac, np.round(diff_frac), atol=1e-8):
                return True
        return False

    @pytest.mark.parametrize(
        "q",
        [
            [0, 0, 0],
            [1 / 2, 0, 0],
            [1 / 2, 1 / 2, 0],
            [1 / 2, 1 / 2, 1 / 2],
            [1 / 3, 0, 0],
            [1 / 4, 1 / 4, 0],
        ],
    )
    def test_kpoint_in_grid(self, simple_cubic_prim, q):
        sc = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, q)
        assert self._kpoint_in_sc(simple_cubic_prim, sc, q)

    def test_roundtrip_via_coordinates_frac(self, simple_cubic_prim):
        """Each q from coordinates_frac should reproduce the same minimum supercell."""
        # Build a moderately sized supercell to get several k-points
        T = np.diag([3, 4, 6])
        sc = casmconfig.Supercell(simple_cubic_prim, T)
        sc_kpts = casmconfig.SupercellKpoints(supercell=sc)

        for i in range(sc_kpts.coordinates_frac.shape[1]):
            q = sc_kpts.coordinates_frac[:, i]
            sc_min = casmconfig.min_supercell_for_kpoint(simple_cubic_prim, q)

            # The minimum supercell must be no larger than the original
            assert _volume(sc_min) <= _volume(sc)

            # The k-point must be commensurate with the minimum supercell
            assert _is_commensurate(sc_min, q)
