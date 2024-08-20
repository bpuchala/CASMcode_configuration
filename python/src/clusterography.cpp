#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/container/stream_io.hh"
#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/configuration/clusterography/ClusterSpecs.hh"
#include "casm/configuration/clusterography/IntegralCluster.hh"
#include "casm/configuration/clusterography/io/json/ClusterSpecs_json_io.hh"
#include "casm/configuration/clusterography/io/json/EquivalentsInfo_json_io.hh"
#include "casm/configuration/clusterography/io/json/IntegralClusterOrbitGenerator_json_io.hh"
#include "casm/configuration/clusterography/io/json/IntegralCluster_json_io.hh"
#include "casm/configuration/clusterography/orbits.hh"
#include "casm/configuration/group/Group.hh"
#include "casm/configuration/sym_info/unitcellcoord_sym_info.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/LinearIndexConverter.hh"
#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/crystallography/UnitCellCoordRep.hh"
#include "casm/crystallography/io/UnitCellCoordIO.hh"
#include "pybind11_json/pybind11_json.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

xtal::UnitCellCoord site_from_list(std::vector<Index> const &site) {
  if (site.size() != 4) {
    throw std::runtime_error(
        "Error integral sit coordinate from list: list size != 4");
  }
  return xtal::UnitCellCoord(site[0], site[1], site[2], site[3]);
}

std::vector<Index> site_to_list(xtal::UnitCellCoord const &site) {
  std::vector<Index> list;
  for (Index i = 0; i < 4; ++i) {
    list.push_back(site[i]);
  }
  return list;
}

clust::IntegralCluster &append_site(clust::IntegralCluster &cluster,
                                    xtal::UnitCellCoord const &site) {
  cluster.elements().push_back(site);
  return cluster;
}

clust::IntegralCluster &append_site_list(clust::IntegralCluster &cluster,
                                         std::vector<Index> const &site) {
  cluster.elements().push_back(site_from_list(site));
  return cluster;
}

clust::IntegralCluster make_cluster_from_list(
    std::vector<std::vector<Index>> const &list) {
  clust::IntegralCluster cluster;
  for (auto const &site : list) {
    append_site_list(cluster, site);
  }
  return cluster;
}

struct FilterBySublatticeIndex {
  FilterBySublatticeIndex(
      std::shared_ptr<xtal::BasicStructure const> const &prim,
      std::set<Index> sublattice_indices)
      : m_prim(prim), m_included_sublattices(sublattice_indices) {}

  bool operator()(xtal::Site const &site) const {
    xtal::UnitCellCoord unitcellcoord = xtal::UnitCellCoord::from_coordinate(
        *m_prim, site, m_prim->lattice().tol());
    Index sublattice_index = unitcellcoord.sublattice();
    return m_included_sublattices.find(sublattice_index) !=
           m_included_sublattices.end();
  }

  std::shared_ptr<xtal::BasicStructure const> m_prim;
  std::set<Index> m_included_sublattices;
};

clust::ClusterSpecs make_cluster_specs(
    std::shared_ptr<xtal::BasicStructure const> const &prim,
    std::shared_ptr<clust::SymGroup const> const &generating_group,
    std::vector<double> max_length,
    std::vector<clust::IntegralClusterOrbitGenerator> custom_generators,
    std::string site_filter_method,
    std::optional<std::set<Index>> included_sublattices,
    std::optional<clust::IntegralCluster> phenomenal,
    bool include_phenomenal_sites, std::vector<double> cutoff_radius) {
  clust::ClusterSpecs cluster_specs(prim, generating_group);
  cluster_specs.max_length = max_length;
  cluster_specs.custom_generators = custom_generators;

  if (site_filter_method == "dof_sites") {
    cluster_specs.site_filter = clust::dof_sites_filter();
  } else if (site_filter_method == "alloy_sites") {
    cluster_specs.site_filter = clust::alloy_sites_filter;
  } else if (site_filter_method == "all_sites") {
    cluster_specs.site_filter = clust::all_sites_filter;
  } else if (site_filter_method == "selected_sites") {
    //cluster_specs.site_filter = FilterBySublatticeIndex(prim, included_sublattices);
    if (included_sublattices.has_value()) {
        cluster_specs.site_filter =
        FilterBySublatticeIndex(prim, *included_sublattices);
    }
    else {
        std::stringstream ss;
        ss << "Error in make_cluster_specs: site_filter_method="
        << site_filter_method << " is selected_sites but included_sublattices is not provided";
        throw std::runtime_error(ss.str());
    }
  }
  else {
    std::stringstream ss;
    ss << "Error in make_cluster_specs: site_filter_method="
       << site_filter_method << " is not recognized";
    throw std::runtime_error(ss.str());
  }
  cluster_specs.site_filter_method = site_filter_method;

  cluster_specs.phenomenal = phenomenal;
  cluster_specs.include_phenomenal_sites = include_phenomenal_sites;
  cluster_specs.cutoff_radius = cutoff_radius;
  return cluster_specs;
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(_clusterography, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Cluster classes and methods

        libcasm.clusterography
        --------------------

        The libcasm.clusterography package contains data structures and methods for generating clusters, orbits of clusters, and neighborhoods.

    )pbdoc";
  py::module::import("libcasm.xtal");

  py::class_<clust::IntegralCluster>(m, "Cluster",
                                     R"pbdoc(
      A cluster of IntegralSiteCoordinate
      )pbdoc")
      .def(py::init<std::vector<xtal::UnitCellCoord> const &>(),
           py::arg("sites"),
           R"pbdoc(

      .. rubric:: Special Methods

      The multiplication operator ``X = lhs * rhs`` can be used to apply
      :class:`libcasm.xtal.IntegralSiteCoordinateRep` to Cluster:

      - ``X=Cluster``, ``lhs=IntegralSiteCoordinateRep``, ``rhs=Cluster``: Copy and
        transform all sites in the cluster, returning the cluster of transformed sites,
        such that `X.site(i) == lhs * rhs.site(i)`, for all valid `i`.

      Translate a Cluster using operators ``+``, ``-``, ``+=``, ``-=``:

      .. code-block:: Python

          import numpy as np
          from libcasm.clusterography import Cluster

          # construct Cluster
          cluster = Cluster.from_list(
              [
                  [0, 0, 0, 0], # [b, i, j, k]
                  [0, 1, 0, 0],
              ]
          )
          translation = np.array([0, 0, 1])

          # translate via `+=`:
          cluster += translation

          # translate via `-=`:
          cluster -= translation

          # copy & translate via `+`:
          translated_cluster = cluster + translation

          # copy & translate via `-`:
          translated_cluster = cluster - translation


      Sort Cluster by lexicographical order of their sites using ``<``, ``<=``,
      ``>``, ``>=``, and compare using ``==``, ``!=``:

      .. code-block:: Python

          import numpy as np
          from libcasm.clusterography import Cluster

          # construct Cluster
          A = Cluster.from_list(
              [
                  [0, 0, 0, 0], # [b, i, j, k]
                  [0, 1, 0, 0],
              ]
          )
          B = Cluster.from_list(
              [
                  [0, 0, 0, 0], # [b, i, j, k]
                  [0, 0, 1, 0],
              ]
          )

          assert A < B
          assert A <= B
          assert A <= A
          assert B > A
          assert B >= A
          assert B >= B
          assert A == A
          assert B == B
          assert A != B

      Additional methods:

      - ``for site in cluster``: Iterate over sites
        (:class:`~libcasm.xtal.IntegralSiteCoordinate`) in the cluster.
      - ``if site in cluster``: Check if a cluster contains a site.
      - ``len(cluster)``: Get the number of sites in a cluster.
      - ``site = cluster[i]``: Get the `i`-th site in a cluster (indices
        start at 0).
      - Cluster may be copied with
        :func:`Cluster.copy <libcasm.clusterography.Cluster.copy>`,
        `copy.copy`, or `copy.deepcopy`.

      .. rubric:: Constructor

      Parameters
      ----------
      sites : List[libcasm.xtal.IntegralSiteCoordinate]
          List of sites in the cluster. May be empty.
      )pbdoc")
      .def_static("from_list", &make_cluster_from_list, py::arg("sites"),
                  R"pbdoc(
      Construct a Cluster from a list of sites (as `[b, i, j, k]`)

      Parameters
      ----------
      sites : List[List[int]]
          List of sites (as [b, i, j, k]) in the cluster. May be empty.
      )pbdoc")
      .def(
          "sites",
          [](clust::IntegralCluster const &cluster) {
            return cluster.elements();
          },
          "Returns the cluster sites")
      .def("append", &append_site, "Append to cluster sites")
      .def("append", &append_site_list, "Append to cluster sites")
      .def(
          "to_list",
          [](clust::IntegralCluster const &cluster) {
            std::vector<std::vector<Index>> list;
            for (auto const &site : cluster.elements()) {
              list.push_back(site_to_list(site));
            }
            return list;
          },
          "Returns the cluster sites as a list of sites (as `[b, i, j, k]`)")
      .def("to_index_list", &clust::to_index_vector,
           "Convert Cluster to a list of linear site indices in a supercell")
      .def("to_index_set", &clust::to_index_set,
           "Convert Cluster to a set of linear site indices in a supercell")
      .def(
          "site",
          [](clust::IntegralCluster const &cluster, Index i) {
            return cluster.element(i);
          },
          "Returns the i-th site in the cluster")
      .def(
          "size",
          [](clust::IntegralCluster const &cluster) {
            return cluster.elements().size();
          },
          "Returns number of sites in the cluster")
      .def("sort", &clust::IntegralCluster::sort, "Sort cluster sites")
      .def("sorted", &clust::IntegralCluster::sorted,
           "Return a copy of the cluster with sites sorted")
      .def("is_sorted", &clust::IntegralCluster::is_sorted,
           "Return True if the cluster sites are sorted; False otherwise.")
      .def("__len__", &clust::IntegralCluster::size)
      .def(
          "__getitem__",
          [](clust::IntegralCluster const &cluster, Index i) {
            return cluster.element(i);
          },
          "Returns the i-th site in the cluster")
      .def(
          "__iter__",
          [](clust::IntegralCluster const &cluster) {
            return py::make_iterator(cluster.begin(), cluster.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def(
          "__add__",
          [](clust::IntegralCluster const &cluster,
             Eigen::Vector3l const &translation) {
            return cluster + translation;
          },
          "Translate a cluster by adding unit cell indices")
      .def(
          "__iadd__",
          [](clust::IntegralCluster &cluster,
             Eigen::Vector3l const &translation) {
            return cluster += translation;
          },
          "Translate a cluster by adding unit cell indices")
      .def(
          "__sub__",
          [](clust::IntegralCluster const &cluster,
             Eigen::Vector3l const &translation) {
            return cluster - translation;
          },
          "Translate a cluster by subtracting unit cell indices")
      .def(
          "__isub__",
          [](clust::IntegralCluster &cluster,
             Eigen::Vector3l const &translation) {
            return cluster -= translation;
          },
          "Translate a cluster by subtracting unit cell indices")
      .def(
          "__rmul__",
          [](clust::IntegralCluster const &cluster,
             xtal::UnitCellCoordRep const &site_rep) {
            return clust::copy_apply(site_rep, cluster);
          },
          "Transform a :class:`~libcasm.clusterography.Cluster`")
      .def(py::self < py::self, "Compares via lexicographical order of sites")
      .def(py::self <= py::self, "Compares via lexicographical order of sites")
      .def(py::self > py::self, "Compares via lexicographical order of sites")
      .def(py::self >= py::self, "Compares via lexicographical order of sites")
      .def(py::self == py::self, "True if clusters are equal")
      .def(py::self != py::self, "True if clusters are not equal")
      .def(
          "copy",
          [](clust::IntegralCluster const &self) {
            return clust::IntegralCluster(self);
          },
          R"pbdoc(
          Returns a copy of the Cluster.
          )pbdoc")
      .def("__copy__",
           [](clust::IntegralCluster const &self) {
             return clust::IntegralCluster(self);
           })
      .def("__deepcopy__",
           [](clust::IntegralCluster const &self, py::dict) {
             return clust::IntegralCluster(self);
           })
      .def("__repr__",
           [](clust::IntegralCluster const &self) {
             std::stringstream ss;
             jsonParser json;
             json["sites"] = jsonParser::array();
             for (auto const &site : self.elements()) {
               json["sites"].push_back(site);
             }
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             xtal::BasicStructure const &prim) -> clust::IntegralCluster {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;
            jsonParser json{data};
            return jsonConstructor<clust::IntegralCluster>::from_json(json,
                                                                      prim);
          },
          R"pbdoc(
          Construct an Cluster from a Python dict

          Parameters
          ----------
          data : dict
              The serialized Cluster

          xtal_prim : libcasm.xtal.Prim
              The :class:`libcasm.xtal.Prim`

          Returns
          -------
          cluster : libcasm.clusterography.Cluster
              The Cluster
          )pbdoc",
          py::arg("data"), py::arg("prim"))
      .def(
          "to_dict",
          [](clust::IntegralCluster const &cluster,
             xtal::BasicStructure const &prim,
             std::optional<clust::IntegralCluster> phenomenal)
              -> nlohmann::json {
            jsonParser json;
            to_json(cluster, json, prim, phenomenal);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the Cluster as a Python dict

          Parameters
          ----------
          xtal_prim : libcasm.xtal.Prim
              The :class:`libcasm.xtal.Prim`
          phenomenal : Optional[Cluster] = None
              For local-clusters, the phenomenal cluster may optionally be
              provided to include the distances from the phenomenal sites to
              the local-cluster sites in the output.

          Returns
          -------
          data : dict
              The Cluster as a Python dict
          )pbdoc",
          py::arg("xtal_prim"), py::arg("phenomenal") = std::nullopt);

  py::class_<clust::IntegralClusterOrbitGenerator>(m, "ClusterOrbitGenerator",
                                                   R"pbdoc(
      A cluster of IntegralSiteCoordinate, and optionally subclusters
      )pbdoc")
      .def(py::init<clust::IntegralCluster const &, bool>(),
           py::arg("prototype"), py::arg("include_subclusters") = true,
           R"pbdoc(

      .. rubric:: Constructor

      Parameters
      ----------
      prototype: Cluster
          A prototype cluster
      include_subclusters: bool
          If True, include subcluster orbits
      )pbdoc")
      //
      .def_readwrite("prototype",
                     &clust::IntegralClusterOrbitGenerator::prototype, R"pbdoc(
                     Cluster: A prototype cluster.
                     )pbdoc")
      .def_readwrite("include_subclusters",
                     &clust::IntegralClusterOrbitGenerator::include_subclusters,
                     R"pbdoc(
                     bool: If True, include subcluster orbits.
                     )pbdoc")
      .def("__repr__",
           [](clust::IntegralClusterOrbitGenerator const &self) {
             std::stringstream ss;
             jsonParser json;
             json["include_subclusters"] = self.include_subclusters;
             json["sites"] = jsonParser::array();
             for (auto const &site : self.prototype.elements()) {
               json["sites"].push_back(site);
             }
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_list",
          [](const nlohmann::json &data, xtal::BasicStructure const &prim)
              -> std::vector<clust::IntegralClusterOrbitGenerator> {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;
            jsonParser json{data};
            InputParser<std::vector<clust::IntegralClusterOrbitGenerator>>
                parser(json, prim);
            std::runtime_error error_if_invalid{
                "Error in "
                "libcasm.clusterography.ClusterOrbitGenerator.from_list"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return *parser.value;
          },
          R"pbdoc(
          Construct a list of ClusterOrbitGenerator from a list of Python dict

          Parameters
          ----------
          data : list[dict]
              The serialized list of ClusterOrbitGenerator

          xtal_prim : libcasm.xtal.Prim
              The :class:`libcasm.xtal.Prim`

          Returns
          -------
          orbit_generators : list[ClusterOrbitGenerator]
              The orbit generators
          )pbdoc",
          py::arg("data"), py::arg("prim"))
      .def(
          "to_dict",
          [](clust::IntegralClusterOrbitGenerator const &orbit_generator,
             xtal::BasicStructure const &prim) -> nlohmann::json {
            jsonParser json;
            to_json(orbit_generator, json, prim);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the ClusterOrbitGenerator as a Python dict

          Parameters
          ----------
          xtal_prim : libcasm.xtal.Prim
              The :class:`libcasm.xtal.Prim`

          Returns
          -------
          data : dict
              The ClusterOrbitGenerator as a Python dict
          )pbdoc",
          py::arg("xtal_prim"));

  py::class_<clust::ClusterSpecs>(m, "ClusterSpecs",
                                  R"pbdoc(
      Specifications for generating orbits of clusters
      )pbdoc")
      .def(py::init(&make_cluster_specs),
           R"pbdoc(

      .. rubric:: Constructor

      Parameters
      ----------
      xtal_prim : libcasm.xtal.Prim
          The :class:`libcasm.xtal.Prim`
      generating_group: libcasm.sym_info.SymGroup
          The group used to generate orbits of equivalent clusters
      max_length: list[float]=[]
          The maximum site-to-site distance to allow in clusters, by number
          of sites in the cluster. Example: `[0.0, 0.0, 5.0, 4.0]` specifies
          that pair clusters up to distance 5.0 and triplet clusters up to
          distance 4.0 should be included. The null cluster and point
          cluster values (elements 0 and 1) are arbitrary.
      custom_generators: list[ClusterOrbitGenerator]=[]
          Specifies clusters that should be uses to construct orbits
          regardless of the max_length or cutoff_radius parameters
      site_filter_method: str = "dof_sites"
          Names a method for selecting which sites are included in clusters.
          Options are:

              "dof_sites": (default)
                  Include sites with any continuous degrees of freedom (DoF)
                  or >1 allowed occupant DoF
              "alloy_sites":
                  Include all sites with >1 allowed occupant DoF
              "all_sites":
                  Include all sites
              "selected_sites":
                  Include sites with sublattice index in included_sublattices
      included_sublattices: Optional[set[int]] = None
          For "selected_sites" `site_filter_method`, specifies the sublattice
          indices to include in the generated clusters.
      phenomenal: Optional[Cluster] = None
          For local clusters, specifies the sites about which local-clusters
          are generated.
      include_phenomenal_sites: bool = False
          For local clusters, if True, the phenomenal cluster sites are
          included in the local-clusters.
      cutoff_radius: list[float]
          For local clusters, the maximum distance of sites from any
          phenomenal cluster site to include in the local environment, by
          number of sites in the cluster. The null cluster value
          (element 0) is arbitrary.
      )pbdoc",
           py::arg("xtal_prim"), py::arg("generating_group"),
           py::arg("max_length") = std::vector<double>{},
           py::arg("custom_generators") =
               std::vector<clust::IntegralClusterOrbitGenerator>{},
           py::arg("site_filter_method") = std::string("dof_sites"),
           py::arg("included_sublattices") = std::nullopt,
           py::arg("phenomenal") = std::nullopt,
           py::arg("include_phenomenal_sites") = false,
           py::arg("cutoff_radius") = std::vector<double>{})
      .def(
          "xtal_prim",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.prim;
          },
          "Return the prim.")
      .def(
          "generating_group",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.generating_group;
          },
          "Return the generating group.")
      .def(
          "max_length",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.max_length;
          },
          "Return the `max_length` list.")
      .def(
          "custom_generators",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.custom_generators;
          },
          "Return the `custom_generators` list.")
      .def(
          "site_filter_method",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.site_filter_method;
          },
          "Return the `site_filter_method`.")
      .def(
          "phenomenal",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.phenomenal;
          },
          "Return the phenomenal cluster if it is present, else None")
      .def(
          "include_phenomenal_sites",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.include_phenomenal_sites;
          },
          "Return the `include_phenomenal_sites` parameter")
      .def(
          "cutoff_radius",
          [](clust::ClusterSpecs const &cluster_specs) {
            return cluster_specs.cutoff_radius;
          },
          "Return the `cutoff_radius` list")
      .def(
          "make_orbits",
          [](clust::ClusterSpecs const &cluster_specs) {
            // construct
            std::vector<std::set<clust::IntegralCluster>> _orbits;
            if (cluster_specs.phenomenal.has_value()) {
              // local clusters
              auto symgroup_rep = sym_info::make_unitcellcoord_symgroup_rep(
                  cluster_specs.generating_group->element, *cluster_specs.prim);
              _orbits = make_local_orbits(
                  cluster_specs.prim, symgroup_rep, cluster_specs.site_filter,
                  cluster_specs.max_length, cluster_specs.custom_generators,
                  cluster_specs.phenomenal.value(), cluster_specs.cutoff_radius,
                  cluster_specs.include_phenomenal_sites);
            } else {
              // prim periodic clusters
              auto generating_group_unitcellcoord_symgroup_rep =
                  sym_info::make_unitcellcoord_symgroup_rep(
                      cluster_specs.generating_group->element,
                      *cluster_specs.prim);
              _orbits = make_prim_periodic_orbits(
                  cluster_specs.prim,
                  generating_group_unitcellcoord_symgroup_rep,
                  cluster_specs.site_filter, cluster_specs.max_length,
                  cluster_specs.custom_generators);
            }

            // copy
            std::vector<std::vector<clust::IntegralCluster>> orbits;
            for (Index i = 0; i < _orbits.size(); ++i) {
              orbits.emplace_back(_orbits[i].begin(), _orbits[i].end());
            }
            return orbits;
          },
          R"pbdoc(
          Construct cluster orbits

          Returns
          -------
          orbits: list[list[Cluster]]
              A list of cluster orbits, `orbits[i]` is the i-th orbit. If a
              phenomenal cluster is included in the ClusterSpecs, the resulting
              orbits are local-cluster orbits, otherwise they are periodic.
         )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             std::shared_ptr<xtal::BasicStructure const> const &prim,
             std::shared_ptr<clust::SymGroup const> const &prim_factor_group,
             std::vector<xtal::UnitCellCoordRep> const
                 &unitcellcoord_symgroup_rep) -> clust::ClusterSpecs {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;
            jsonParser json{data};

            // compatibility to read CASM v1 ClusterSpecs
            if (json.contains("params")) {
              json = json["params"];
            }
            InputParser<clust::ClusterSpecs> parser(
                json, prim, prim_factor_group, unitcellcoord_symgroup_rep);
            std::runtime_error error_if_invalid{
                "Error in libcasm.clusterography.ClusterSpecs.from_dict"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return *parser.value;
          },
          R"pbdoc(
          Construct ClusterSpecs from a Python dict

          See the CASM documentation for the `ClusterSpecs format`_.

          .. _`ClusterSpecs format`: https://prisms-center.github.io/CASMcode_docs/formats/casm/clusterography/ClusterSpecs/

          Parameters
          ----------
          data : dict
              The ClusterSpecs as a Python dict

          xtal_prim : libcasm.xtal.Prim
              The :class:`libcasm.xtal.Prim`

          prim_factor_group: libcasm.sym_info.SymGroup
              The prim factor group

          integral_site_coordinate_symgroup_rep: list[libcasm.xtal.IntegralSiteCoordinateRep]
              Symmetry representation for IntegralSiteCoordinate tranformation

          Returns
          -------
          cluster_specs : ClusterSpecs
              The ClusterSpecs object
          )pbdoc",
          py::arg("data"), py::arg("xtal_prim"), py::arg("prim_factor_group"),
          py::arg("integral_site_coordinate_symgroup_rep"))
      .def(
          "to_dict",
          [](clust::ClusterSpecs const &cluster_specs) -> nlohmann::json {
            jsonParser json;
            to_json(cluster_specs, json);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the ClusterSpecs as a Python dict

          Returns
          -------
          data : dict
              The ClusterSpecs as a Python dict
          )pbdoc")
      .def("__repr__", [](clust::ClusterSpecs const &self) {
        std::stringstream ss;
        jsonParser json;
        to_json(self, json);
        ss << json;
        return ss.str();
      });

  m.def(
      "make_integral_site_coordinate_symgroup_rep",
      [](std::vector<xtal::SymOp> const &group_elements,
         xtal::BasicStructure const &xtal_prim) {
        return sym_info::make_unitcellcoord_symgroup_rep(group_elements,
                                                         xtal_prim);
      },
      R"pbdoc(
      Construct a group representation of IntegralSiteCoordinateRep

      Parameters
      ----------
      group_elements: list[libcasm.xtal.SymOp]
          Symmetry group elements

      xtal_prim: libcasm.xtal.Prim
          The Prim structure

      Returns
      -------
      integral_site_coordinate_symgroup_rep: list[IntegralSiteCoordinateRep]
          Symmetry group representation for transforming IntegralSiteCoordinate.
        )pbdoc",
      py::arg("group_elements"), py::arg("xtal_prim"));

  m.def(
      "make_periodic_orbit",
      [](clust::IntegralCluster const &orbit_element,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        std::set<clust::IntegralCluster> orbit =
            make_prim_periodic_orbit(orbit_element, unitcellcoord_symgroup_rep);
        return std::vector<clust::IntegralCluster>(orbit.begin(), orbit.end());
      },
      R"pbdoc(
      Construct an orbit of Cluster

      The orbit of Cluster is all distinct Cluster that are equivalent
      under the provided symmetry group, including one element for all
      Cluster that are equivalent according to prim translational symmetry.

      Parameters
      ----------
      orbit_element : OccEvent
          One Cluster in the orbit

      integral_site_coordinate_symgroup_rep: list[IntegralSiteCoordinateRep]
          Symmetry group representation.

      Returns
      -------
      orbit : list[Cluster]
          The orbit of Cluster
      )pbdoc",
      py::arg("orbit_element"),
      py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "make_periodic_equivalence_map",
      [](std::vector<clust::IntegralCluster> const &orbit,
         std::shared_ptr<clust::SymGroup const> const &symgroup,
         xtal::Lattice const &lattice,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        std::set<clust::IntegralCluster> _orbit(orbit.begin(), orbit.end());
        return clust::make_prim_periodic_equivalence_map(
            _orbit, symgroup, lattice.lat_column_mat(),
            unitcellcoord_symgroup_rep);
      },
      R"pbdoc(
      Construct the equivalence map for an orbit of Cluster

      The orbit of Cluster is all distinct Cluster that are equivalent
      under the provided symmetry group, including one element for all
      Cluster that are equivalent according to prim translational symmetry.

      The equivalence map specifies the group elements that map the prototype orbit
      element (the first element) onto the `i`-th orbit element.

      Parameters
      ----------
      orbit : list[Cluster]
          The orbit of Cluster
      symgroup: libcasm.sym_info.SymGroup
          The group that generated the orbit. Typically the prim factor group, or it
          may be a subgroup.
      lattice: libcasm.xtal.Lattice
          The lattice, used to find translations.
      integral_site_coordinate_symgroup_rep: list[libcasm.xtal.IntegralSiteCoordinateRep]
          Symmetry group representation of `symgroup`.

      Returns
      -------
      equivalence_map : list[list[libcasm.xtal.SymOp]]
          The elements of `equivalence_map[i]` are the SymOp that map the prototype
          orbit element (the first element) onto the `i`-th orbit element, up to a
          permutation of cluster sites.
      )pbdoc",
      py::arg("orbit"), py::arg("symgroup"), py::arg("lattice"),
      py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "make_periodic_equivalence_map_indices",
      [](std::vector<clust::IntegralCluster> const &orbit,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        std::set<clust::IntegralCluster> _orbit(orbit.begin(), orbit.end());
        return clust::make_prim_periodic_equivalence_map_indices(
            _orbit, unitcellcoord_symgroup_rep);
      },
      R"pbdoc(
      Construct the equivalence map indices for an orbit of Cluster

      The orbit of Cluster is all distinct Cluster that are equivalent
      under the provided symmetry group, including one element for all
      Cluster that are equivalent according to prim translational symmetry.

      The equivalence map specifies the group elements that map the prototype orbit
      element (the first element) onto the `i`-th orbit element.

      Parameters
      ----------
      orbit : list[Cluster]
          The orbit of Cluster
      integral_site_coordinate_symgroup_rep: list[libcasm.xtal.IntegralSiteCoordinateRep]
          Representation of the symmetry group used to generate the orbit.

      Returns
      -------
      equivalence_map_indices : list[list[int]]
          The elements of `equivalence_map[i]` are the indices in the symmetry group
          elements list of the SymOp that map the prototype orbit element (the first
          element) onto the `i`-th orbit element, up to a translation and a permutation
          of cluster sites.
      )pbdoc",
      py::arg("orbit"), py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "make_cluster_group",
      [](clust::IntegralCluster cluster,
         std::shared_ptr<clust::SymGroup const> const &group,
         xtal::Lattice const &lattice,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        return make_cluster_group(cluster, group, lattice.lat_column_mat(),
                                  unitcellcoord_symgroup_rep);
      },
      R"pbdoc(
      Construct the subgroup which leaves a Cluster invariant

      Parameters
      ----------
      cluster : Cluster
          The Cluster that remains invariant after transformation by
          subgroup elements.

      group: libcasm.sym_info.SymGroup
          The super group.

      lattice: xtal.Lattice
          The lattice.

      integral_site_coordinate_symgroup_rep: list[xtal.IntegralSiteCoordinateRep]
          Representation of `group` for transforming IntegralSiteCoordinateRep.

      Returns
      -------
      cluster_group : libcasm.sym_info.SymGroup
          The subgroup which leaves the cluster invariant. Elements may differ
          from elements of `group` by a translation. The head group of
          `cluster_group` is set to `group`.
      )pbdoc",
      py::arg("cluster"), py::arg("group"), py::arg("lattice"),
      py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "make_local_cluster_group",
      [](clust::IntegralCluster cluster,
         std::shared_ptr<clust::SymGroup const> const &phenomenal_group,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        return make_local_cluster_group(cluster, phenomenal_group,
                                        unitcellcoord_symgroup_rep);
      },
      R"pbdoc(
      Construct the subgroup which leaves a local Cluster invariant

      Parameters
      ----------
      cluster : Cluster
          The local cluster that remains invariant after transformation by
          subgroup elements.

      phenomenal_group: libcasm.sym_info.SymGroup
          The symmetry group that generates an orbit of the local cluster.

      integral_site_coordinate_symgroup_rep: list[libcasm.xtal.IntegralSiteCoordinateRep]
          Symmetry group representation of `phenomenal_group`.

      Returns
      -------
      cluster_group : libcasm.sym_info.SymGroup
          The subgroup which leaves the cluster invariant. Elements may not
          differ from elements of `group` by a translation. The head group of
          `cluster_group` is set to the head group of `phenomenal_group`.
      )pbdoc",
      py::arg("cluster"), py::arg("phenomenal_group"),
      py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "make_local_orbit",
      [](clust::IntegralCluster const &orbit_element,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        std::set<clust::IntegralCluster> orbit =
            make_local_orbit(orbit_element, unitcellcoord_symgroup_rep);
        return std::vector<clust::IntegralCluster>(orbit.begin(), orbit.end());
      },
      R"pbdoc(
      Construct an orbit of local Cluster

      Local clusters are clusters defined relative to another, phenomenal cluster. For
      example, the phenomenal cluster might be the cluster of sites where occupants
      exchange during a diffusional hop, and the local clusters are the nearby clusters
      of sites whose occupation changes the migration barrier for the hopping atoms.

      The orbit of local Cluster is all distinct Cluster that are equivalent
      under the provided phenomenal symmetry group, which depends on the symmetry of
      the phenomenal cluster and the property of interest.

      Parameters
      ----------
      orbit_element : OccEvent
          One Cluster in the orbit

      integral_site_coordinate_symgroup_rep: list[IntegralSiteCoordinateRep]
          Representation of the phenomenal symmetry group.

      Returns
      -------
      orbit : list[Cluster]
          The orbit of Cluster
      )pbdoc",
      py::arg("orbit_element"),
      py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "make_local_equivalence_map",
      [](std::vector<clust::IntegralCluster> const &orbit,
         std::shared_ptr<clust::SymGroup const> const &phenomenal_group,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        std::set<clust::IntegralCluster> _orbit(orbit.begin(), orbit.end());
        return clust::make_local_equivalence_map(_orbit, phenomenal_group,
                                                 unitcellcoord_symgroup_rep);
      },
      R"pbdoc(
      Construct the equivalence map for an orbit of local Cluster

      Local clusters are clusters defined relative to another, phenomenal cluster. For
      example, the phenomenal cluster might be the cluster of sites where occupants
      exchange during a diffusional hop, and the local clusters are the nearby clusters
      of sites whose occupation changes the migration barrier for the hopping atoms.

      The orbit of local Cluster is all distinct Cluster that are equivalent
      under the provided phenomenal symmetry group, which depends on the symmetry of
      the phenomenal cluster and the property of interest.

      The equivalence map specifies the phenomenal group elements that map the
      prototype orbit element (the first element) onto the `i`-th orbit element.

      Parameters
      ----------
      orbit : list[Cluster]
          The orbit of local Cluster
      phenomenal_group: libcasm.sym_info.SymGroup
          The symmetry group that generated the orbit of local cluster.
      integral_site_coordinate_symgroup_rep: list[libcasm.xtal.IntegralSiteCoordinateRep]
          Symmetry group representation of `phenomenal_group`.

      Returns
      -------
      equivalence_map : list[list[libcasm.xtal.SymOp]]
          The elements of `equivalence_map[i]` are the SymOp that map the prototype
          orbit element (the first element) onto the `i`-th orbit element, up to a
          permutation of cluster sites.
      )pbdoc",
      py::arg("orbit"), py::arg("phenomenal_group"),
      py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "make_local_equivalence_map_indices",
      [](std::vector<clust::IntegralCluster> const &orbit,
         std::vector<xtal::UnitCellCoordRep> const
             &unitcellcoord_symgroup_rep) {
        std::set<clust::IntegralCluster> _orbit(orbit.begin(), orbit.end());
        return clust::make_local_equivalence_map_indices(
            _orbit, unitcellcoord_symgroup_rep);
      },
      R"pbdoc(
      Construct the equivalence map indices for an orbit of local Cluster

      Local clusters are clusters defined relative to another, phenomenal cluster. For
      example, the phenomenal cluster might be the cluster of sites where occupants
      exchange during a diffusional hop, and the local clusters are the nearby clusters
      of sites whose occupation changes the migration barrier for the hopping atoms.

      The orbit of local Cluster is all distinct Cluster that are equivalent
      under the provided phenomenal symmetry group, which depends on the symmetry of
      the phenomenal cluster and the property of interest.

      The equivalence map specifies the phenomenal group elements that map the prototype
      orbit element (the first element) onto the `i`-th orbit element.

      Parameters
      ----------
      orbit : list[Cluster]
          The orbit of local Cluster
      integral_site_coordinate_symgroup_rep: list[IntegralSiteCoordinateRep]
          Representation of the phenomenal symmetry group.

      Returns
      -------
      equivalence_map_indices : list[list[int]]
          The elements of `equivalence_map[i]` are the indices in the cluster symmetry
          group elements list of the SymOp that map the prototype orbit element (the
          first element) onto the `i`-th orbit element, up to a permutation
          of cluster sites.
      )pbdoc",
      py::arg("orbit"), py::arg("integral_site_coordinate_symgroup_rep"));

  m.def(
      "equivalents_info_from_dict",
      [](const nlohmann::json &data,
         xtal::BasicStructure const &prim) -> py::tuple {
        // print errors and warnings to sys.stdout
        py::scoped_ostream_redirect redirect;

        jsonParser json{data};
        InputParser<clust::EquivalentsInfo> parser(json, prim);
        std::runtime_error error_if_invalid{
            "Error in equivalents_info_from_dict"};
        report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
        return py::make_tuple(parser.value->phenomenal_clusters,
                              parser.value->equivalent_generating_op_indices);
      },
      R"pbdoc(
    Read the contents of an "equivalents_info.json" file

    The "equivalents_info.json" file is written when a local basis set
    is generated to specify the orientation and translational position
    of the phenomenal cluster and local basis set.

    Parameters
    ----------
    data : dict
        The serialized equivalents info

    xtal_prim : libcasm.xtal.Prim
        The :class:`libcasm.xtal.Prim`

    Returns
    -------
    phenomenal_clusters, equivalent_generating_op_indices :

        phenomenal_clusters : list[Cluster]
            The phenomenal clusters of the local basis sets

        equivalent_generating_op_indices : list[int]
            Indices of the factor group operations that
            generate the phenomenal clusters from the
            prototype.

    )pbdoc",
      py::arg("data"), py::arg("prim"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
