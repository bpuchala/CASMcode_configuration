import libcasm.clusterography as clust
import libcasm.sym_info as sym_info
import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims


def test_cluster_specs():
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    cluster_specs = clust.ClusterSpecs(
        xtal_prim=xtal_prim,
        generating_group=prim_factor_group,
        max_length=[0.0, 0.0, 2.01],
    )
    assert isinstance(cluster_specs, clust.ClusterSpecs)

    orbits = cluster_specs.make_orbits()
    assert len(orbits) == 3
    assert len(orbits[0]) == 1
    assert len(orbits[1]) == 1
    assert len(orbits[2]) == 6

    data = cluster_specs.to_dict()
    print(xtal.pretty_json(data))
    # fmt: off
    expected_data = {
        "generating_group": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                             45, 46, 47],
        "orbit_branch_specs": {
            "0": { "max_length": 0.0 },
            "1": { "max_length": 0.0 },
            "2": { "max_length": 2.01 }
        },
        "site_filter_method": "dof_sites"
    }
    # fmt: on
    assert data == expected_data

    integral_site_coordinate_symgroup_rep = (
        clust.make_integral_site_coordinate_symgroup_rep(
            group_elements=prim_factor_group.elements,
            xtal_prim=xtal_prim,
        )
    )
    cluster_specs_in = clust.ClusterSpecs.from_dict(
        data=data,
        xtal_prim=xtal_prim,
        prim_factor_group=prim_factor_group,
        integral_site_coordinate_symgroup_rep=integral_site_coordinate_symgroup_rep,
    )
    assert cluster_specs_in.generating_group().head_group_index == list(range(48))
    assert cluster_specs_in.max_length() == cluster_specs.max_length()
    assert cluster_specs_in.cutoff_radius() == cluster_specs.cutoff_radius()


def test_cluster_specs_from_dict_1():
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    integral_site_coordinate_symgroup_rep = (
        clust.make_integral_site_coordinate_symgroup_rep(
            group_elements=prim_factor_group.elements,
            xtal_prim=xtal_prim,
        )
    )
    data = {
        "orbit_branch_specs": {
            "2": {"max_length": 3.01},
            "3": {"max_length": 2.01},
        },
        "site_filter_method": "dof_sites",
    }
    cluster_specs_in = clust.ClusterSpecs.from_dict(
        data=data,
        xtal_prim=xtal_prim,
        prim_factor_group=prim_factor_group,
        integral_site_coordinate_symgroup_rep=integral_site_coordinate_symgroup_rep,
    )
    assert cluster_specs_in.max_length() == [0.0, 0.0, 3.01, 2.01]


def test_cluster_specs_from_dict_2():
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    integral_site_coordinate_symgroup_rep = (
        clust.make_integral_site_coordinate_symgroup_rep(
            group_elements=prim_factor_group.elements,
            xtal_prim=xtal_prim,
        )
    )
    phenomenal = clust.Cluster.from_list([[0, 0, 0, 0], [0, 1, 0, 0]])
    custom_generators = [
        clust.ClusterOrbitGenerator(
            prototype=clust.Cluster.from_list([[0, 2, 0, 0], [0, 10, 0, 0]]),
            include_subclusters=True,
        )
    ]
    cluster_specs = clust.ClusterSpecs(
        xtal_prim=xtal_prim,
        generating_group=prim_factor_group,
        max_length=[0.0, 0.0, 3.01, 2.01],
        custom_generators=custom_generators,
        phenomenal=phenomenal,
        cutoff_radius=[0.0, 0.0, 5.01, 4.01],
    )
    data = cluster_specs.to_dict()
    print(xtal.pretty_json(data))

    cluster_specs_in = clust.ClusterSpecs.from_dict(
        data=data,
        xtal_prim=xtal_prim,
        prim_factor_group=prim_factor_group,
        integral_site_coordinate_symgroup_rep=integral_site_coordinate_symgroup_rep,
    )
    assert cluster_specs_in.max_length() == [0.0, 0.0, 3.01, 2.01]
    assert len(cluster_specs_in.custom_generators()) == 1
    assert (
        cluster_specs_in.custom_generators()[0].prototype
        == custom_generators[0].prototype
    )
    assert cluster_specs_in.max_length() == [0.0, 0.0, 3.01, 2.01]
    assert cluster_specs_in.cutoff_radius() == [0.0, 0.0, 5.01, 4.01]


def test_cluster_specs_from_dict_3():
    """Check CASM v1 compatibility."""
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    integral_site_coordinate_symgroup_rep = (
        clust.make_integral_site_coordinate_symgroup_rep(
            group_elements=prim_factor_group.elements,
            xtal_prim=xtal_prim,
        )
    )
    data = {
        "method": "periodic_max_length",
        "params": {
            "orbit_branch_specs": {
                "2": {"max_length": 3.01},
                "3": {"max_length": 2.01},
            },
            "site_filter_method": "dof_sites",
        },
    }
    cluster_specs_in = clust.ClusterSpecs.from_dict(
        data=data,
        xtal_prim=xtal_prim,
        prim_factor_group=prim_factor_group,
        integral_site_coordinate_symgroup_rep=integral_site_coordinate_symgroup_rep,
    )
    assert cluster_specs_in.max_length() == [0.0, 0.0, 3.01, 2.01]


def test_cluster_specs_from_dict_4():
    """Check no orbit_branch_specs warning"""

    import io
    from contextlib import redirect_stdout

    # Test 1, with local DoF -> should give a warning
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    integral_site_coordinate_symgroup_rep = (
        clust.make_integral_site_coordinate_symgroup_rep(
            group_elements=prim_factor_group.elements,
            xtal_prim=xtal_prim,
        )
    )
    data = {}
    f = io.StringIO()
    with redirect_stdout(f):
        cluster_specs_in = clust.ClusterSpecs.from_dict(
            data=data,
            xtal_prim=xtal_prim,
            prim_factor_group=prim_factor_group,
            integral_site_coordinate_symgroup_rep=integral_site_coordinate_symgroup_rep,
        )
    assert cluster_specs_in.max_length() == []
    assert "Warning" in f.getvalue()

    # Test 2, with local DoF and empty orbit_branch_specs -> should give no warning
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    integral_site_coordinate_symgroup_rep = (
        clust.make_integral_site_coordinate_symgroup_rep(
            group_elements=prim_factor_group.elements,
            xtal_prim=xtal_prim,
        )
    )
    data = {
        "orbit_branch_specs": {},
    }
    f = io.StringIO()
    with redirect_stdout(f):
        cluster_specs_in = clust.ClusterSpecs.from_dict(
            data=data,
            xtal_prim=xtal_prim,
            prim_factor_group=prim_factor_group,
            integral_site_coordinate_symgroup_rep=integral_site_coordinate_symgroup_rep,
        )
    assert cluster_specs_in.max_length() == []
    assert "Warning" not in f.getvalue()

    # Test 3, with global DoF only -> Should give no warning
    xtal_prim = xtal_prims.FCC(
        r=1.0, occ_dof=["A"], global_dof=[xtal.DoFSetBasis("Hstrain")]
    )
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    integral_site_coordinate_symgroup_rep = (
        clust.make_integral_site_coordinate_symgroup_rep(
            group_elements=prim_factor_group.elements,
            xtal_prim=xtal_prim,
        )
    )
    data = {}
    f = io.StringIO()
    with redirect_stdout(f):
        cluster_specs_in = clust.ClusterSpecs.from_dict(
            data=data,
            xtal_prim=xtal_prim,
            prim_factor_group=prim_factor_group,
            integral_site_coordinate_symgroup_rep=integral_site_coordinate_symgroup_rep,
        )
    assert cluster_specs_in.max_length() == []
    assert "Warning" not in f.getvalue()


def test_make_periodic_cluster_specs():
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    cluster_specs = clust.make_periodic_cluster_specs(
        xtal_prim=xtal_prim, max_length=[0.0, 0.0, 2.01]
    )
    assert isinstance(cluster_specs, clust.ClusterSpecs)

    orbits = cluster_specs.make_orbits()
    assert len(orbits) == 3
    assert len(orbits[0]) == 1
    assert len(orbits[1]) == 1
    assert len(orbits[2]) == 6


def test_ClusterSpecs_repr():
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    cluster_specs = clust.make_periodic_cluster_specs(
        xtal_prim=xtal_prim, max_length=[0.0, 0.0, 2.01]
    )

    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        print(cluster_specs)
    out = f.getvalue()
    assert "generating_group" in out


def test_local_cluster_specs():
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    symgroup_rep = clust.make_integral_site_coordinate_symgroup_rep(
        prim_factor_group.elements, xtal_prim
    )
    phenomenal = clust.Cluster(
        [
            xtal.IntegralSiteCoordinate.from_list([0, 0, 0, 0]),
            xtal.IntegralSiteCoordinate.from_list([0, 1, 0, 0]),
        ]
    )
    cluster_group = clust.make_cluster_group(
        cluster=phenomenal,
        group=prim_factor_group,
        lattice=xtal_prim.lattice(),
        integral_site_coordinate_symgroup_rep=symgroup_rep,
    )
    cluster_specs = clust.ClusterSpecs(
        xtal_prim=xtal_prim,
        generating_group=cluster_group,
        max_length=[0.0, 0.0],
        phenomenal=phenomenal,
        cutoff_radius=[0.0, 2.01],
    )
    assert isinstance(cluster_specs, clust.ClusterSpecs)

    orbits = cluster_specs.make_orbits()
    assert len(orbits) == 5
    assert len(orbits[0]) == 1
    assert len(orbits[1]) == 4
    assert len(orbits[2]) == 4
    assert len(orbits[3]) == 8
    assert len(orbits[4]) == 2


def test_make_local_cluster_specs():
    xtal_prim = xtal_prims.FCC(r=1.0, occ_dof=["A", "B", "Va"])
    phenomenal_cluster = clust.Cluster(
        [
            xtal.IntegralSiteCoordinate.from_list([0, 0, 0, 0]),
            xtal.IntegralSiteCoordinate.from_list([0, 1, 0, 0]),
        ]
    )
    cluster_specs = clust.make_local_cluster_specs(
        xtal_prim=xtal_prim,
        phenomenal_cluster=phenomenal_cluster,
        max_length=[0.0, 0.0],
        cutoff_radius=[0.0, 2.01],
    )
    assert isinstance(cluster_specs, clust.ClusterSpecs)

    orbits = cluster_specs.make_orbits()
    assert len(orbits) == 5
    assert len(orbits[0]) == 1
    assert len(orbits[1]) == 4
    assert len(orbits[2]) == 4
    assert len(orbits[3]) == 8
    assert len(orbits[4]) == 2


def find_asym_unit_index(sublattice_index, asym_units):
    for asym_unit_index, asym_unit in enumerate(asym_units):
        if sublattice_index in asym_unit:
            return asym_unit_index
    return None


def test_make_custom_cluster_specs(FCC_with_interstitials_prim):
    """Make clusters of FCC oct / tet sites separately, save as custom generators"""

    xtal_prim = FCC_with_interstitials_prim
    prim_factor_group = sym_info.make_factor_group(xtal_prim)

    # symmetrically equivalent sublattices
    # A_sites = [0]
    tet_sites = [1, 2]
    oct_sites = [3]

    ### Make tet clusters, make oct clusters, then combine approach ###

    # Step 1: function to include tet sites only (cutoff = 10.4)
    def tet_site_filter_f(site: xtal.IntegralSiteCoordinate) -> bool:
        return site.sublattice() in tet_sites

    tet_only_cluster_specs = clust.make_custom_cluster_specs(
        xtal_prim=xtal_prim,
        generating_group=prim_factor_group,
        site_filter_f=tet_site_filter_f,
        max_length=[0.0, 0.0, 10.4],
    )
    assert isinstance(tet_only_cluster_specs, clust.ClusterSpecs)
    tet_only_orbits = tet_only_cluster_specs.make_orbits()
    # print([orbit[0] for orbit in tet_only_orbits])
    assert len(tet_only_orbits) == 40

    # Step 2: lambda function to include oct sites only (cutoff = 20.4)
    oct_only_cluster_specs = clust.make_custom_cluster_specs(
        xtal_prim=xtal_prim,
        generating_group=prim_factor_group,
        site_filter_f=lambda site: site.sublattice() in oct_sites,
        max_length=[0.0, 0.0, 20.4],
    )
    oct_only_orbits = oct_only_cluster_specs.make_orbits()
    # print([orbit[0] for orbit in oct_only_orbits])
    assert len(oct_only_orbits) == 78

    # Now combine & also include some clusters of any site (cutoff = 4.01)
    cluster_specs = clust.ClusterSpecs(
        xtal_prim=xtal_prim,
        generating_group=prim_factor_group,
        max_length=[0.0, 0.0, 4.01],
        site_filter_method="all_sites",
        custom_generators=tet_only_cluster_specs.custom_generators()
        + oct_only_cluster_specs.custom_generators(),
    )
    orbits = cluster_specs.make_orbits()
    # print([orbit[0] for orbit in orbits])
    assert len(orbits) == 126

    ### Make all clusters and then filter approach ###
    cluster_specs = clust.ClusterSpecs(
        xtal_prim=xtal_prim,
        generating_group=prim_factor_group,
        max_length=[0.0, 0.0, 20.4],
        site_filter_method="all_sites",
    )
    orbits = cluster_specs.make_orbits()

    def all_tet(proto):
        for site in proto:
            if site.sublattice() not in tet_sites:
                return False
        return True

    def all_oct(proto):
        for site in proto:
            if site.sublattice() not in oct_sites:
                return False
        return True

    include_subclusters = False
    custom_generators = []
    for orbit in orbits:
        proto = orbit[0]
        max_dist = proto.to_dict(xtal_prim).get("max_length")
        if all_tet(proto) and max_dist < 10.4:
            custom_generators.append(
                clust.ClusterOrbitGenerator(proto, include_subclusters)
            )
            continue
        if all_oct(proto) and max_dist < 20.4:
            custom_generators.append(
                clust.ClusterOrbitGenerator(proto, include_subclusters)
            )
            continue
    # print(custom_generators)

    cluster_specs = clust.ClusterSpecs(
        xtal_prim=xtal_prim,
        generating_group=prim_factor_group,
        max_length=[0.0, 0.0, 4.01],
        site_filter_method="all_sites",
        custom_generators=custom_generators,
    )
    orbits = cluster_specs.make_orbits()
    # print([orbit[0] for orbit in orbits])
    assert len(orbits) == 126
