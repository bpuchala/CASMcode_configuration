# Changelog

All notable changes to `libcasm-configuration` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Added

- Added `libcasm.clusterography.make_custom_cluster_specs`, which takes a custom site filter function to generate a ClusterSpecs with custom_generators generated with the custom site filter. This approach allows creating a ClusterSpecs that is customized, as with a custom site filter, but still works to save/load without to_dict/from_dict needing the custom filter. 


## [2.0a5] - 2024-08-13

### Fixed

- Fixed `libcasm.clusterography.equivalents_info_from_dict`, which was trying to read clusters from the wrong position. This method was not used to read equivalents_info.json for KMC simulations.
- Fixed `libcasm.occ_events.get_occevent_coordinate`, which was using the reverse of the appropriate translation to get the unitcell_index.
- Fixed `to_json(to_json(config::Configuration)` and `read_dof_values` to properly support writing and reading DoF values in both the prim and standard basis.

### Changed

- Changed the `by_supercell` and `by_supercell_list` methods of `libcasm.enumerate.ConfigEnumAllOccupations` to only do default continuous DoF. 
- Changed the `skip_non_canonical` parameter of the `by_linear_site_indices`, `by_integral_site_coordinate`, `by_sublattice`, `by_cluster`, and `by_cluster_list` methods of `libcasm.enumerate.ConfigEnumAllOccupations` to `skip_equivalents`.
- Changed `ConfigEnumAllOccupations` methods that take a `background` configuration to maintain the orientation of continuous DoF.
- Changed `ClusterSpecs.from_dict` to allow reading CASM v1 cluster specs JSON by checking for a "params" attribute and parsing that if it exists.
- Changed `ClusterSpecs.from_dict` to warn if the prim has local DoF but no `"orbit_branch_specs"` attribute is present.
- Changed occ_events `to_json` methods for `occ_events::OccPosition`, `occ_events::OccTrajectory`, and `occ_events::OccEvent` to optionally accept occ_events::OccSystem.

### Added

- Added option to include phenomenal site to local-cluster site distances by passing the phenomenal cluster to `libcasm.clusterography.Cluster.to_dict`.
- Added `libcasm.configuration.Supercell.symgroup_rep`.
- Added copy methods for `libcasm.configuration.SupercellSymOp`.
- Added `Configuration.copy`, `ConfigurationWithProperties.copy`, `SupercellRecord.copy`, `ConfigurationRecord.copy`, `Cluster.copy`, and `OccEvent.copy` methods.
- Added `libcasm.configuration.copy_local_dof_values` and `libcasm.configuration.copy_global_dof_values`
- Added `which_dofs` parameter to `libcasm.configuration.make_invariant_subgroup`
- Added `libcasm.enumerate.SuperConfigEnum`
- Added `libcasm.ConfigEnumAllOccupations.by_supercell_with_continuous_dof`
- Added `__repr__` for `Cluster`, `ClusterOrbitGenerator`, and `ClusterSpecs`.
- Added `to_json` methods for `config::ConfigSpaceAnalysisResults` and `config::DoFSpaceAnalysisResults`.
- Added `Prim.__repr__`, `Supercell.__repr__`, `SupercellRecord.__repr__`, `Configuration.__repr__`, `ConfigurationRecord.__repr__`, `ConfigurationWithProperties.__repr__`, `DoFSpaceAnalysisResults.__repr__`.
- Added `ConfigSpaceAnalysisResults.to_dict` and `DoFSpaceAnalysisResults.to_dict` methods.
- Added `OccSystem.__repr__`, `OccPosition.copy`, `OccPosition.__copy__`, `OccPosition.__deepcopy__`, `OccPosition.__repr__`, `OccEventRep.__repr__`, `OccEventRep.copy`, `OccEventRep.__copy__`, `OccEventRep.__deepcopy__`, `OccEvent.__repr__`.


## [2.0a4] - 2024-07-16

### Added

- Added libcasm.sym_info.make_lattice_point_group
- Added libcasm.configuration.Prim.lattice_point_group
- Added libcasm.configuration.io.tools to get properties from CASM objects
- Added libcasm.configuration.io.spglib to interface with spglib
- Added libcasm.io.symgroup_to_dict_with_group_classification to add group classification from spglib to SymGroup.to_dict data.
- Added libcasm.clusterography.make_local_cluster_group

### Changed

- Changed libcasm.clusterography.make_cluster_group, CASM::clust::make_cluster_group, and CASM::clust::make_cluster_groups so that the head group of the cluster group is set to the head group of the group used to generate the cluster group. In typical use this means the head group of the cluster group is the prim factor group rather than a subgroup, even if a subgroup was used to generate the cluster group.
- Fixed spelling of reservoir (#16) 
- Wheels compiled with numpy>=2.0.0

### Fixed

- Fixed occ_symgroup_rep and atom_position_symgroup_rep in CASM::config::PrimSymInfo and libcasm.configuration.Prim, which were giving the inverse of the documented permutation.


## [2.0a3] - 2024-03-15

### Fixed

- Fixed CASM::config::make_distinct_cluster_sites
- Fixed CASM::config::dof_space_analysis for global DoF with non-primitive unit cell configuration
- Fixed bug in `Configuration.from_structure` that caused error in resulting occupation

### Changed

- Changed libcasm.configuration.Prim.is_atomic, which was returning has_anisotropic_occupants
- Changed libcasm.configuration.is_canonical_configuration arguments for providing a subgroup
- Changed most methods with no arguments that return a value without doing additional work to readonly attributes

### Added

- Added prototype and include_subclusters attributes to libcasm.clusterography.IntegralClusterOrbitGenerator
- Added ConfigEnumAllOccupations, ConfigEnumInfo, ConfigEnumMeshGrid, ScelEnum, make_all_distinct_periodic_perturbations, and make_distinct_cluster_sites to libcasm.enumerate
- Added make_all_super_configurations_by_subsets and make_distinct_super_configurations to libcasm.configuration
- Added `excluded_species` option to `Configuration.to_structure` and `ConfigurationWithProperties.to_structure`
- Added `with_prim_basis` option to `Configuration.to_dict` and `ConfigurationSet.to_dict` to allow output of DoF values in the prim basis, along with "basis" tag which is respected on reading from JSON/dict.
- Added set_order_parameters, make_dof_space, order_parameters, order_parameters_contribution, dof_values_vector, and dof_values_vector_contribution methods to libcasm.configuration.Configuration
- Added CASM::config::cluster_from_index_vector and CASM::config::cluster_from_index_set
- Added `libcasm.irreps.IrrepDecomposition` and `libcasm.irreps.MatrixRepGroup` for generic irreducible space decompositions


### Removed

- Removed `to_canonical_supercell` and `from_canonical_supercell` which can be too easily misinterpreted.


## [2.0a2] - 2023-12-11

### Fixed

- Fix bug in irrep decomposition
- Fix comparison of Configuration with equivalent but distinct supercell

### Added

- Added more irrep decomposition, DoF space analysis, and config space analysis tests
- Added options to config_space_analysis and dof_space_analysis methods to specify default occupation mode on a site or sublattice basis
- Added CASM::config::make_dof_space_rep and libcasm.configuration.make_dof_space_rep
- Added libcasm.configuration.ConfigurationWithProperties, and methods for libcasm.configuration.SupercellSymOp to act on ConfigurationWithProperties
- Added site_index_converter and unitcell_index_converter accessors to libcasm.Supercell.
- Added to_structure and from_structure to libcasm.configuration.Configuration and libcasm.configuration.ConfigurationWithProperties for conversions between atomic structures and configuration
- Added more access to matrix reps from PrimSymInfo in libcasm.configuration.Prim
- Added to_index_list, to_index_set, sort, sorted, is_sorted, __rmul__ to libcasm.clusterography.Cluster
- Added make_periodic_equivalence_map, make_periodic_equivalence_map_indices, make_local_equivalence_map, and make_local_equivalence_map_indices to libcasm.clusterography
- Added to_dict and from_dict methods to libcasm.configuration.Prim

### Changed

- Changed libcasm.clusterography.make_prim_periodic_orbits to make_periodic_orbits

### Deprecated

- Deprecated to_json and from_json methods of libcasm.configuration.Prim


## [2.0a1] - 2023-08-21

This release creates the libcasm-configuration comparison and enumeration module. It includes:

- Classes for representing supercells, configurations, clusters, and occupation events
- Methods for comparing and enumerating unique configurations, clusters, occupation events, and local environments
- Methods for generating orbits of symmetrically equivalent configurations, clusters, and occupation events
- Methods for copying configurations to make sub- or super-configurations
- Methods for generating symmetry groups, and constructing and applying symmetry representations
- Methods for performing irreducible space decompositions and finding symmetry adapted order parameters

This distribution package libcasm-configuration contains several Python packages of use for configuration comparison and enumeration:

- libcasm.sym_info
- libcasm.irreps
- libcasm.clusterography
- libcasm.configuration
- libcasm.occ_events
- libcasm.enumerate

This package may be installed via pip install, using scikit-build, CMake, and pybind11. This release also includes usage examples and API documentation, built using Sphinx.
