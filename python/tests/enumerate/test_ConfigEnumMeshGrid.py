import numpy as np
import pytest

import libcasm.clexulator as casmclex
import libcasm.configuration as casmconfig
import libcasm.enumerate as casmenum
import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims


def test_ConfigEnumMeshGrid_by_range_FCC_1():
    """Test ConfigEnumMeshGrid.by_range

    Using: FCC, prim unit cell, default background, full basis, num=5,
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    dof_space = casmclex.DoFSpace(
        dof_key="Hstrain",
        xtal_prim=xtal_prim,
    )

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    supercell = supercell_set.add(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)

    for i, configuration in enumerate(
        config_enum.by_range(
            background=background,
            dof_space=dof_space,
            start=-0.1,
            stop=0.1,
            num=5,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    assert len(configs_as_enumerated) == 5**6
    assert len(canonical_configs) == 900


def test_ConfigEnumMeshGrid_by_range_FCC_2():
    """Test ConfigEnumMeshGrid.by_range

    Using:
    - FCC,
    - prim unit cell, default background,
    - full basis,
    - num=5, skip_equivalents=True
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    dof_space = casmclex.DoFSpace(
        dof_key="Hstrain",
        xtal_prim=xtal_prim,
    )

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    supercell = supercell_set.add(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)

    for i, configuration in enumerate(
        config_enum.by_range(
            background=background,
            dof_space=dof_space,
            start=-0.1,
            stop=0.1,
            num=5,
            skip_equivalents=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    assert len(configs_as_enumerated) == 900
    assert len(canonical_configs) == 900


def test_ConfigEnumMeshGrid_by_range_FCC_3():
    """Test ConfigEnumMeshGrid.by_range

    Using:
    - FCC,
    - prim unit cell, default background,
    - full basis,
    - step=0.05, skip_equivalents=True
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    dof_space = casmclex.DoFSpace(
        dof_key="Hstrain",
        xtal_prim=xtal_prim,
    )

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    supercell = supercell_set.add(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)

    for i, configuration in enumerate(
        config_enum.by_range(
            background=background,
            dof_space=dof_space,
            start=-0.1,
            stop=0.101,
            step=0.05,
            skip_equivalents=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    assert len(configs_as_enumerated) == 900
    assert len(canonical_configs) == 900


def test_ConfigEnumMeshGrid_by_grid_coordinates_FCC_1():
    """Test ConfigEnumMeshGrid.by_grid_coordinates

    Using:
    - FCC,
    - prim unit cell, default background,
    - full basis,
    - step=0.05, skip_equivalents=True
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    dof_space = casmclex.DoFSpace(
        dof_key="Hstrain",
        xtal_prim=xtal_prim,
    )

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    supercell = supercell_set.add(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)

    xi = []
    for d in range(dof_space.subspace_dim):
        xi.append(np.linspace(-0.1, 0.1, 5))

    for i, configuration in enumerate(
        config_enum.by_grid_coordinates(
            background=background,
            dof_space=dof_space,
            xi=xi,
            skip_equivalents=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    assert len(configs_as_enumerated) == 900
    assert len(canonical_configs) == 900


def test_ConfigEnumMeshGrid_by_irreducible_wedge_FCC_1():
    """Test ConfigEnumMeshGrid.by_irreducible_wedge

    Using:
    - FCC,
    - prim unit cell, default background,
    - full basis,
    - stop=0.1, num=5, skip_equivalents=default (False)
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    dof_space = casmclex.DoFSpace(
        dof_key="Hstrain",
        xtal_prim=xtal_prim,
    )
    dof_space_analysis_results = casmconfig.dof_space_analysis(
        dof_space=dof_space,
        prim=prim,
        calc_wedges=True,
    )
    symmetry_report = dof_space_analysis_results.symmetry_report

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    supercell = supercell_set.add(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)

    # ~~~ trim_corners=False ~~~
    total = 0
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=5,
            trim_corners=False,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        total += 1
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    assert total == 168750  #  (9 * (5**5)) points/subwedge * 6 subwedges
    assert len(configs_as_enumerated) == 136125  # some subwedge share edges
    assert len(canonical_configs) == 102141  # some edges are not shared but equivalent

    # ~~~ trim_corners=True ~~~
    configs_as_enumerated.clear()
    canonical_configs.clear()
    total = 0
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=5,
            trim_corners=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        total += 1
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    assert total == 12936
    assert len(configs_as_enumerated) == 8418
    assert len(canonical_configs) == 4963

    # ~~~ trim_corners=True, skip_equivalents=True ~~~
    configs_as_enumerated.clear()
    canonical_configs.clear()
    total = 0
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=5,
            skip_equivalents=True,
            trim_corners=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        total += 1
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    assert total == 4963
    assert len(configs_as_enumerated) == 4963
    assert len(canonical_configs) == 4963


def test_ConfigEnumMeshGrid_by_irreducible_wedge_FCC_2():
    """Test ConfigEnumMeshGrid.by_irreducible_wedge

    Using:
    - FCC,
    - conventional unit cell, A2B2 background,
    - full basis,
    - stop=0.1, num=5, skip_equivalents=default (False)
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        local_dof=[xtal.DoFSetBasis("disp")],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    # conventional FCC, A2B2
    supercell = supercell_set.add(
        np.array(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)
    background.set_occ(0, 1)
    background.set_occ(1, 1)

    # fmt: off
    dof_space, symmetry_report = background.make_dof_space(
        dof_key="Hstrain",
        calc_wedges=True,
    )
    # fmt: on

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    total = 0
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=5,
            skip_equivalents=False,
            trim_corners=False,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        total += 1
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))

    # print("total:", total)
    # print("len(configs_as_enumerated):", len(configs_as_enumerated))
    # print("len(canonical_configs):", len(canonical_configs))

    assert total == 202500
    assert len(configs_as_enumerated) == 164025
    assert len(canonical_configs) == 136161


def test_ConfigEnumMeshGrid_by_range_FCC_disp_1():
    """Test ConfigEnumMeshGrid.by_range

    Using:
    - FCC,
    - conventional unit cell, A3B1 background,
    - first irrep disp basis,
    - stop=0.1, num=5, skip_equivalents=False
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        local_dof=[xtal.DoFSetBasis("disp")],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    # conventional FCC, A3B1
    supercell = supercell_set.add(
        np.array(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)
    background.set_occ(0, 1)

    # Enumerate in first irrep:
    # fmt: off
    dof_space = casmclex.DoFSpace(
        dof_key="disp",
        xtal_prim=xtal_prim,
        transformation_matrix_to_super=supercell.transformation_matrix_to_super,
        basis=np.array([
            [ 0.0, 0.0, 0.853553390593274, 0.0, 0.0, -0.35355339059327395, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672624,],  # noqa: E501
            [ 0.0, 0.8535533905932741, 0.0, 0.0, -0.1464466094067265, 0.0, 0.0, -0.353553390593274, 0.0, 0.0, -0.3535533905932735, 0.0,],  # noqa: E501
            [ 0.853553390593274, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672616, 0.0, 0.0, -0.3535533905932737, 0.0, 0.0,],  # noqa: E501
        ]).transpose(),
    )
    # fmt: on

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    for i, configuration in enumerate(
        config_enum.by_range(
            background=background,
            dof_space=dof_space,
            start=-0.1,
            stop=0.1,
            num=5,
            skip_equivalents=False,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))
    #
    # print("len(configs_as_enumerated):", len(configs_as_enumerated))
    # print("len(canonical_configs):", len(canonical_configs))

    assert len(configs_as_enumerated) == 5**3
    assert len(canonical_configs) == 10


def test_ConfigEnumMeshGrid_by_range_FCC_disp_2():
    """Test ConfigEnumMeshGrid.by_range

    Using:
    - FCC,
    - conventional unit cell, A3B1 background,
    - first irrep disp basis,
    - stop=0.1, num=5, skip_equivalents=True
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        local_dof=[xtal.DoFSetBasis("disp")],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    # conventional FCC, A3B1
    supercell = supercell_set.add(
        np.array(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)
    background.set_occ(0, 1)

    # Enumerate in first irrep:
    # fmt: off
    dof_space = casmclex.DoFSpace(
        dof_key="disp",
        xtal_prim=xtal_prim,
        transformation_matrix_to_super=supercell.transformation_matrix_to_super,
        basis=np.array([
            [ 0.0, 0.0, 0.853553390593274, 0.0, 0.0, -0.35355339059327395, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672624,],  # noqa: E501
            [ 0.0, 0.8535533905932741, 0.0, 0.0, -0.1464466094067265, 0.0, 0.0, -0.353553390593274, 0.0, 0.0, -0.3535533905932735, 0.0,],  # noqa: E501
            [ 0.853553390593274, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672616, 0.0, 0.0, -0.3535533905932737, 0.0, 0.0,],  # noqa: E501
        ]).transpose(),
    )
    # fmt: on

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    for i, configuration in enumerate(
        config_enum.by_range(
            background=background,
            dof_space=dof_space,
            start=-0.1,
            stop=0.1,
            num=5,
            skip_equivalents=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))
    #
    # print("len(configs_as_enumerated):", len(configs_as_enumerated))
    # print("len(canonical_configs):", len(canonical_configs))

    assert len(configs_as_enumerated) == 10
    assert len(canonical_configs) == 10


# Mark test as expected to fail since this is currently failing in make_irrep_wedges
# when the number of high symmetry direction orbits is greater than 0 but less than
# the irrep dimension.
@pytest.mark.xfail(reason="Error in make_irrep_wedges")
def test_ConfigEnumMeshGrid_by_irreducible_wedge_FCC_disp_1():
    """Test ConfigEnumMeshGrid.by_irreducible_wedge

    Using:
    - FCC,
    - conventional unit cell, A3B1 background,
    - first irrep disp basis,
    - stop=0.1, num=5, skip_equivalents=False
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        local_dof=[xtal.DoFSetBasis("disp")],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    # conventional FCC, A3B1
    supercell = supercell_set.add(
        np.array(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)
    background.set_occ(0, 1)

    # Enumerate in first irrep:
    # fmt: off
    dof_space = casmclex.DoFSpace(
        dof_key="disp",
        xtal_prim=xtal_prim,
        transformation_matrix_to_super=supercell.transformation_matrix_to_super,
        basis=np.array([
            [ 0.0, 0.0, 0.853553390593274, 0.0, 0.0, -0.35355339059327395, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672624,],  # noqa: E501
            [ 0.0, 0.8535533905932741, 0.0, 0.0, -0.1464466094067265, 0.0, 0.0, -0.353553390593274, 0.0, 0.0, -0.3535533905932735, 0.0,],  # noqa: E501
            [ 0.853553390593274, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672616, 0.0, 0.0, -0.3535533905932737, 0.0, 0.0,],  # noqa: E501
        ]).transpose(),
    )
    # fmt: on

    # Notes:
    # - This is failing in make_irrep_wedges when the number of high symmetry direction
    #   orbits is greater than 0 but less than the irrep dimension.
    dof_space_analysis_results = casmconfig.dof_space_analysis(
        dof_space=dof_space,
        prim=prim,
        configuration=background,
        calc_wedges=True,
        exclude_homogeneous_modes=False,  # TODO: why is this necessary?
    )
    symmetry_report = dof_space_analysis_results.symmetry_report

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    # ~~~ trim_corners=False ~~~
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=5,
            skip_equivalents=False,
            trim_corners=False,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    for i, record in enumerate(canonical_configs):
        print(xtal.pretty_json(record.configuration.to_dict()))

    print("len(configs_as_enumerated):", len(configs_as_enumerated))
    print("len(canonical_configs):", len(canonical_configs))

    assert len(configs_as_enumerated) == 5**3
    assert len(canonical_configs) == 5**3

    # ~~~ trim_corners=False ~~~
    configs_as_enumerated.clear()
    canonical_configs.clear()
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=5,
            skip_equivalents=False,
            trim_corners=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    for i, record in enumerate(canonical_configs):
        print(xtal.pretty_json(record.configuration.to_dict()))

    print("len(configs_as_enumerated):", len(configs_as_enumerated))
    print("len(canonical_configs):", len(canonical_configs))

    assert len(configs_as_enumerated) == 54
    assert len(canonical_configs) == 54  # no subwedge axes are equivalent

    # ~~~ trim_corners=False ~~~
    configs_as_enumerated.clear()
    canonical_configs.clear()
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=5,
            skip_equivalents=True,
            trim_corners=True,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))
    #
    # print("len(configs_as_enumerated):", len(configs_as_enumerated))
    # print("len(canonical_configs):", len(canonical_configs))

    assert len(configs_as_enumerated) == 54
    assert len(canonical_configs) == 54  # no subwedge axes are equivalent


@pytest.mark.xfail(reason="Error in make_irrep_wedges")
def test_ConfigEnumMeshGrid_by_irreducible_wedge_FCC_disp_2():
    """Test ConfigEnumMeshGrid.by_irreducible_wedge

    Using:
    - FCC,
    - conventional unit cell, A3B1 background,
    - first & second irrep disp basis,
    - stop=0.1, num=5, skip_equivalents=False
    """
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B"],
        local_dof=[xtal.DoFSetBasis("disp")],
        global_dof=[xtal.DoFSetBasis("Hstrain")],
    )
    prim = casmconfig.Prim(xtal_prim)
    supercell_set = casmconfig.SupercellSet(prim=prim)
    configs_as_enumerated = casmconfig.ConfigurationSet()
    canonical_configs = casmconfig.ConfigurationSet()

    # conventional FCC, A3B1
    supercell = supercell_set.add(
        np.array(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
    ).supercell
    background = casmconfig.Configuration(supercell)
    background.set_occ(0, 1)

    # Enumerate in first irrep:
    # fmt: off
    dof_space = casmclex.DoFSpace(
        dof_key="disp",
        xtal_prim=xtal_prim,
        transformation_matrix_to_super=supercell.transformation_matrix_to_super,
        basis=np.array( [
            [ 0.0, 0.0, 0.853553390593274, 0.0, 0.0, -0.35355339059327395, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672624,],  # noqa: E501
            [ 0.0, 0.8535533905932741, 0.0, 0.0, -0.1464466094067265, 0.0, 0.0, -0.353553390593274, 0.0, 0.0, -0.3535533905932735, 0.0,],  # noqa: E501
            [ 0.853553390593274, 0.0, 0.0, -0.3535533905932739, 0.0, 0.0, -0.14644660940672616, 0.0, 0.0, -0.3535533905932737, 0.0, 0.0,],  # noqa: E501
            [ 0.0, 0.0, -0.14644660940672624, 0.0, 0.0, -0.353553390593274, 0.0, 0.0, -0.3535533905932738, 0.0, 0.0, 0.853553390593274],  # noqa: E501
            [ 0.0, -0.14644660940672655, 0.0, 0.0, 0.8535533905932737, 0.0, 0.0, -0.35355339059327373, 0.0, 0.0, -0.3535533905932736, 0.0],  # noqa: E501
            [-0.1464466094067263, 0.0, 0.0, -0.35355339059327395, 0.0, 0.0, 0.853553390593274, 0.0, 0.0, -0.35355339059327373, 0.0, 0.0],  # noqa: E501
        ]).transpose(),
    )
    # fmt: on

    dof_space_analysis_results = casmconfig.dof_space_analysis(
        dof_space=dof_space,
        prim=prim,
        configuration=background,
        calc_wedges=True,
        exclude_homogeneous_modes=False,  # TODO: why is this necessary?
    )
    symmetry_report = dof_space_analysis_results.symmetry_report

    config_enum = casmenum.ConfigEnumMeshGrid(
        prim=prim,
        supercell_set=supercell_set,
    )

    total = 0
    for i, configuration in enumerate(
        config_enum.by_irreducible_wedge(
            background=background,
            dof_space=dof_space,
            irreducible_wedge=symmetry_report.irreducible_wedge,
            stop=0.1,
            num=3,
            skip_equivalents=False,
            trim_corners=False,
        )
    ):
        assert isinstance(configuration, casmconfig.Configuration)
        total += 1
        configs_as_enumerated.add(configuration)
        canonical_configs.add(casmconfig.make_canonical_configuration(configuration))

    # for i, record in enumerate(canonical_configs):
    #     print(xtal.pretty_json(record.configuration.to_dict()))
    #
    # print("len(configs_as_enumerated):", len(configs_as_enumerated))
    # print("len(canonical_configs):", len(canonical_configs))

    assert total == 34992  # 3**6 * len(symmetry_report.irreducible_wedge)
    assert len(configs_as_enumerated) == 19575
    assert len(canonical_configs) == 11413
