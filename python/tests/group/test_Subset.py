import numpy as np
import pytest

import libcasm.configuration as casmconfig
import libcasm.group as casmgroup
import libcasm.xtal.prims as xtal_prims


def test_subset_construction():
    # Define a simple lattice primitive
    prim = casmconfig.Prim(xtal_prim=xtal_prims.FCC(a=4.0, occ_dof=["A", "B"]))

    assert len(prim.factor_group.elements) == 48

    subset = casmgroup.Subset(group=prim.factor_group, indices=set(range(48)))
    assert isinstance(subset, casmgroup.Subset)

    assert subset.is_group
    assert subset.is_normal

    print("Conjugacy classes:")
    conjugacy_classes = prim.factor_group.conjugacy_classes()
    for i, cc in enumerate(conjugacy_classes):
        print(f"{i}: Indices: {cc}")
    print()

    def find_class(j):
        for i, cc in enumerate(conjugacy_classes):
            if j in cc:
                return i
        return -1

    maximal_cyclic_subgroups = subset.maximal_cyclic_subgroups()
    generators = subset.maximal_cyclic_generators()
    assert len(maximal_cyclic_subgroups) == len(generators)
    for i, s in enumerate(maximal_cyclic_subgroups):
        gen = generators[i]
        classes = [find_class(j) for j in s.indices]

        print(f"{i}: Generator: {gen}, Indices: {s.indices}, Classes: {classes}")
        # for j in s.indices:
        #     op = s.group.elements[j]
        #     info = xtal.SymInfo(op, prim.xtal_prim.lattice())
        #     print(f"- Element {j}: {info.brief_cart()}")
    print()

    generators = subset.minimal_generators()
    print("Minimal generators:", generators)
    print()

    generators = {1, 32}
    subset = casmgroup.Subset.from_generators(
        group=prim.factor_group,
        generators=generators,
    )
    print("Generators:", generators)
    print("Indices:", subset.indices)
    print("Size:", len(subset.indices))
    print()


def _make_supercell_subset(xtal_prim, T):
    """Build a Supercell and return its full symgroup as a Subset
    with N_translations set to the number of unit cells."""
    prim = casmconfig.Prim(xtal_prim=xtal_prim)
    scel = casmconfig.Supercell(prim, T)
    return casmgroup.Subset(group=scel.symgroup(), N_translations=scel.n_unitcells)


def _subgroup_index_sets(subgroups):
    """Return a set of frozensets of indices, one per subgroup."""
    return {frozenset(s.indices) for s in subgroups}


@pytest.mark.parametrize(
    "T",
    [
        np.diag([1, 1, 1]),
        np.diag([2, 1, 1]),
        np.diag([2, 2, 1]),
        np.diag([2, 2, 2]),
        np.diag([3, 3, 3]),
        np.diag([2, 3, 4]),
    ],
)
def test_all_subgroups_methods_BCC(BCC_binary_GLstrain_disp_prim, T):
    """Both methods find the same subgroups for BCC supercells."""
    subset_dfs = _make_supercell_subset(BCC_binary_GLstrain_disp_prim, T)
    subset_ns = casmgroup.Subset(
        group=subset_dfs.group, N_translations=subset_dfs.N_translations
    )

    sg_dfs = subset_dfs.all_subgroups(method="depth_first_search", progress="none")
    sg_ns = subset_ns.all_subgroups(method="normal_subgroup")

    assert len(sg_dfs) == len(sg_ns)
    assert _subgroup_index_sets(sg_dfs) == _subgroup_index_sets(sg_ns)


@pytest.mark.parametrize(
    "T",
    [
        np.diag([1, 1, 1]),
        np.diag([2, 1, 1]),
        np.diag([2, 2, 1]),
        np.diag([2, 2, 2]),
        np.diag([3, 3, 3]),
        np.diag([2, 3, 4]),
    ],
)
def test_all_subgroups_methods_ZrO(ZrO_prim_GLstrain_disp, T):
    """Both methods find the same subgroups for ZrO supercells."""
    subset_dfs = _make_supercell_subset(ZrO_prim_GLstrain_disp, T)
    subset_ns = casmgroup.Subset(
        group=subset_dfs.group, N_translations=subset_dfs.N_translations
    )

    sg_dfs = subset_dfs.all_subgroups(method="depth_first_search", progress="none")
    sg_ns = subset_ns.all_subgroups(method="normal_subgroup")

    assert len(sg_dfs) == len(sg_ns)
    assert _subgroup_index_sets(sg_dfs) == _subgroup_index_sets(sg_ns)
