#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <utility>

#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/configuration/group/Group.hh"
#include "casm/configuration/group/io/json/GenericGroup_json_io.hh"
#include "casm/configuration/group/subgroups.hh"
#include "casm/global/pybind11_helpers.hh"
#include "casm/global/threads.hh"
#include "pybind11_json/pybind11_json.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// GenericGroup

std::shared_ptr<group::GenericGroup> make_generic_group(
    std::optional<group::MultiplicationTable> multiplication_table,
    std::optional<std::shared_ptr<group::GenericGroup const>> head_group,
    std::optional<std::set<Index>> head_group_index) {
  if (head_group.has_value()) {
    if (!head_group_index.has_value()) {
      throw std::runtime_error(
          "Error in make_generic_group: head_group_index must be provided "
          "when head_group is provided");
    }
    return std::make_shared<group::GenericGroup>(*head_group,
                                                 *head_group_index);
  }
  if (!multiplication_table.has_value()) {
    throw std::runtime_error(
        "Error in make_generic_group: multiplication_table must be provided "
        "when head_group is not provided");
  }
  return std::make_shared<group::GenericGroup>(*multiplication_table);
}

std::shared_ptr<group::GenericGroup const> make_generic_group_subgroup(
    std::shared_ptr<group::GenericGroup const> const &group,
    std::set<Index> const &head_group_index) {
  std::shared_ptr<group::GenericGroup const> head_group_ptr;
  if (!group->head_group_ptr) {
    head_group_ptr = group;
  } else {
    head_group_ptr = group->head_group_ptr;
  }
  return std::make_shared<group::GenericGroup>(head_group_ptr,
                                               head_group_index);
}

group::Subset make_subset(std::shared_ptr<group::GenericGroup const> group,
                          std::optional<std::set<Index>> indices,
                          std::optional<Index> N_translations) {
  if (!indices.has_value()) {
    std::set<Index> all_indices;
    for (Index i = 0; i < group->size(); ++i) {
      all_indices.insert(i);
    }
    indices = all_indices;
  }

  return group::Subset(group, *indices, N_translations);
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(_group, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Group theory utilities

        libcasm.group
        ----------------

        The libcasm.group package contains group theory utilities.

    )pbdoc";

  py::class_<group::GenericGroup, std::shared_ptr<group::GenericGroup>>(
      m, "GenericGroup",
      R"pbdoc(
      Data structure holding the group multiplication table and derived properties.

      The :class:`~libcasm.group.GenericGroup` class does not store the elements of
      the group, but it does store the multiplication table and inverse element table.
      The GenericGroup class may represent a head group, or a subgroup.

      When representing a subgroup, the head group can be obtained as a shared pointer
      from :func:`~libcasm.group.GenericGroup.head_group`. Pointers to subgroups are not
      stored by the head group. For subgroups, there is a list of indices indicating which
      element in the head group each subgroup element corresponds to
      (:func:`~libcasm.group.GenericGroup.head_group_index`).
      )pbdoc")
      .def(py::init(&make_generic_group), R"pbdoc(

          .. rubric:: Constructor

          Notes
          -----

          To construct a head group, provide only `multiplication table`. To
          construct a subgroup, use `head_group` and `head_group_index` or,
          alternatively, use :func:`~libcasm.group.GenericGroup.make_subgroup`.

          Parameters
          ----------
          multiplication_table: Optional[list[list[int]]]
              The multiplication table element
              `multiplication_table[i][j] == k` represents that
              ``elements[k] == elements[i] * elements[j]``.

              The multiplication table must be square, represent a closed
              group, and be consistent with having identity operation as the
              first element.

          head_group: Optional[GenericGroup] = None
              The head group if this is a subgroup, else None. If None, then
              this is a head group.
          head_group_index: Optional[set[int]] = None
              Indices of elements in the head group (which may or may not be
              `self`) to include in a subgroup. Only used if `head_group` is
              not None.
          )pbdoc",
           py::arg("multiplication_table") = std::nullopt,
           py::arg("head_group") = std::nullopt,
           py::arg("head_group_index") = std::nullopt)
      .def("size", &group::GenericGroup::size, R"pbdoc(
          Returns the size of the group

          Returns
          -------
          size: int
              The number of elements in the group.
          )pbdoc")
      .def("make_subgroup", &make_generic_group_subgroup, R"pbdoc(
          Make a subgroup

          Parameters
          ----------
          head_group_index: set[int]
              Indices of elements in the head group (which may or may not be
              `self`) to include in a subgroup.
          )pbdoc",
           py::arg("head_group_index"))
      .def_property_readonly(
          "multiplication_table",
          [](std::shared_ptr<group::GenericGroup const> const &group) {
            return group->multiplication_table;
          },
          R"pbdoc(
          list[list[int]]: The multiplication table.

          The multiplication table element `multiplication_table[i][j] == k`
          represents that
          ``np.allclose(elements[k], elements[i] @ elements[j]) == True``.
          )pbdoc")
      .def(
          "conjugacy_classes",
          [](std::shared_ptr<group::GenericGroup const> const &group) {
            std::vector<std::vector<Index>> conjugacy_classes;
            for (Index i = 0; i < group->size(); ++i) {
              Index cc = group->class_of(i);
              if (cc >= conjugacy_classes.size()) {
                conjugacy_classes.resize(cc + 1);
              }
              conjugacy_classes[cc].push_back(i);
            }
            return conjugacy_classes;
          },
          R"pbdoc(
          Returns the conjugacy classes

          Returns
          -------
          conjugacy_classes: list[list[int]]
              ``conjugacy_classes[i]`` is a list of the indices of elements in
              the `i`-th class."
          )pbdoc")
      .def(
          "mult",
          [](std::shared_ptr<group::GenericGroup const> const &group, Index i,
             Index j) { return group->mult(i, j); },
          py::arg("i"), py::arg("j"),
          R"pbdoc(
          Returns the index of the element product.

          Parameters
          ----------
          i: int
              lhs element index.
          j: int
              rhs element index.

          Returns
          -------
          k: int
              The index ``k == multiplication_table[i][j]``.
          )pbdoc")
      .def_property_readonly(
          "is_subgroup",
          [](std::shared_ptr<group::GenericGroup const> const &group) {
            return group->head() != nullptr;
          },
          R"pbdoc(
          bool: True if this is a subgroup, False otherwise.
          )pbdoc")
      .def_property_readonly(
          "head_group",
          [](std::shared_ptr<group::GenericGroup const> const &group) {
            std::optional<std::shared_ptr<group::GenericGroup const>>
                head_group;
            if (group->head() != nullptr) {
              head_group = group->head();
            }
            return head_group;
          },
          R"pbdoc(
          Optional[GenericGroup]: The head group if this is a subgroup, else None.
          )pbdoc")
      .def_property_readonly(
          "head_group_index",
          [](std::shared_ptr<group::GenericGroup const> const &group) {
            return group->head_group_index;
          },
          R"pbdoc(
          list[int]: The list of head group indices (guaranteed sorted)

          If this is the head group, then
          ``group.head_group_index == [0, 1, 2, ...]``.

          If this is a sub group, then ``subgroup.element[i]`` is the same element
          as ``subgroup.head_group.element[subgroup.head_group_index[i]]``.

          )pbdoc")
      .def_property_readonly(
          "inverse_index",
          [](std::shared_ptr<group::GenericGroup const> const &group) {
            return group->inverse_index;
          },
          R"pbdoc(
          list[int]: The list of inverse indices

          Represents that ``group.element[group.inverse_index[i]]`` is the
          inverse of ``group.element[i]``.
          )pbdoc")
      .def(
          "inv",
          [](std::shared_ptr<group::GenericGroup const> const &group, Index i) {
            return group->inv(i);
          },
          py::arg("i"), R"pbdoc(
          Returns the index of the inverse of an element

          Parameters
          ----------
          i: int
              The element index.

          Returns
          -------
          i_inverse: int
              The index the inverse of the `i`-th element.
          )pbdoc")
      .def(
          "class_of",
          [](std::shared_ptr<group::GenericGroup const> const &group, Index i) {
            return group->class_index[i];
          },
          py::arg("i"), R"pbdoc(
          Returns the index of the conjugacy class containing an element

          Parameters
          ----------
          i: int
              The element index.

          Returns
          -------
          i_class: int
              The index the conjugacy class containing the `i`-th element.
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;
            jsonParser json{data};
            InputParser<std::shared_ptr<group::GenericGroup const>> parser(
                json);
            std::runtime_error error_if_invalid{
                "Error in libcasm.group.GenericGroup.from_dict"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return std::move(*parser.value);
          },
          R"pbdoc(
          Construct a GenericGroup from a Python dict.

          Parameters
          ----------
          data : dict
              The dict representation.

          Returns
          -------
          group: libcasm.group.GenericGroup
              The GenericGroup
          )pbdoc",
          py::arg("data"))
      .def(
          "to_dict",
          [](std::shared_ptr<group::GenericGroup const> const &group) {
            jsonParser json;
            to_json(group, json);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the GenericGroup as a Python dict

          Returns
          -------
          data : dict
              The dict representation.

          )pbdoc");

  py::class_<group::Subset>(m, "Subset", R"pbdoc(
      Data structure specifying a subset of group elements, as indices, which
      may or may not form a subgroup.

      )pbdoc")
      .def(py::init(&make_subset), R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          group: GenericGroup
              The group this subset is part of. Must be the head group.
          indices: Optional[set[int]] = None
              Indices of the elements in the group forming the subset. If None,
              all elements in the group are included.
          N_translations: Optional[int] = None
              Size of the translation subgroup T = {0, ..., N_translations-1}.
              Required when using ``all_subgroups(method="normal_subgroup")``.
          )pbdoc",
           py::arg("group"), py::arg("indices") = std::nullopt,
           py::arg("N_translations") = std::nullopt)
      .def_static(
          "from_generators",
          [](std::shared_ptr<group::GenericGroup const> const &group,
             std::set<Index> const &generators) {
            py::scoped_ostream_redirect redirect;
            return group::Subset::from_generators(group, generators);
          },
          R"pbdoc(
          Construct a subset from subgroup generators

          Parameters
          ----------
          group: GenericGroup
              The group this subset is part of. Must be the head group.
          generators: set[int]
              Indices of elements in the head group that generate the
              a subgroup under closure by multiplication.
          )pbdoc",
          py::arg("group"), py::arg("generators"))
      .def_property_readonly(
          "indices",
          [](group::Subset const &subset) { return subset.indices(); },
          R"pbdoc(
          set[int]: The indices of elements in the head group that form this subset.
          )pbdoc")
      .def_property_readonly(
          "N_translations",
          [](group::Subset const &subset) -> std::optional<Index> {
            return subset.N_translations();
          },
          R"pbdoc(
          Optional[int]: Size of the translation subgroup T = {0,...,N-1},
          if set. Required for ``all_subgroups(method="normal_subgroup")``.
          )pbdoc")
      .def_property_readonly(
          "group", [](group::Subset const &subset) { return subset.group(); },
          R"pbdoc(
          GenericGroup: The head group this subset belongs to.
          )pbdoc")
      .def_property_readonly(
          "is_group",
          [](group::Subset const &subset) { return subset.is_group(); },
          R"pbdoc(
          Return True if the subset is closed under multiplication and inverses.
          )pbdoc")
      .def_property_readonly(
          "is_abelian_group",
          [](group::Subset const &subset) { return subset.is_abelian_group(); },
          R"pbdoc(
        Return True if the subset is an abelian group (a commutative group where
        `a*b == b*a` for all `a`, `b` in the subset).
        )pbdoc")
      .def_property_readonly(
          "is_normal",
          [](group::Subset const &subset) { return subset.is_normal(); },
          R"pbdoc(
          Return True if the subset is normal (invariant under conjugation).

          A subset, :math:`N`, is called normal if it is invariant under
          conjugation by elements of the head group, :math:`G`. This means for
          every element :math:`g` in :math:`G` and every element :math:`n` in
          :math:`N`, the element :math:`g*n*g^{-1}` is an element in :math:`N`.
          )pbdoc")
      .def(
          "close", [](group::Subset const &subset) { return subset.close(); },
          R"pbdoc(
          Extend this subset to be the closure under group multiplication
          and return the result
          )pbdoc")
      .def(
          "extend",
          [](group::Subset const &subset, group::Subset const &other) {
            return subset.extend(other);
          },
          py::arg("other"), R"pbdoc(
          Extend this subset by adding indices from `other` and return the
          result.
          )pbdoc")
      .def(
          "extend_and_close",
          [](group::Subset const &subset, group::Subset const &other) {
            return subset.extend_and_close(other);
          },
          py::arg("other"), R"pbdoc(
          Extend this subset with `other` then close under multiplication and
          return the result.
          )pbdoc")
      .def(
          "is_proper_subset_of",
          [](group::Subset const &subset, group::Subset const &other) {
            return subset.is_proper_subset_of(other);
          },
          py::arg("other"), R"pbdoc(
          Return True if this subset is a proper subset of `other`.
          )pbdoc")
      .def(
          "is_subset_of",
          [](group::Subset const &subset, group::Subset const &other) {
            return subset.is_subset_of(other);
          },
          py::arg("other"), R"pbdoc(
          Return True if this subset is a subset of `other` (allowing equality).
          )pbdoc")
      .def(
          "left_cosets",
          [](group::Subset const &subset) -> std::vector<group::Subset> {
            return subset.left_cosets();
          },
          R"pbdoc(
          Return a list of left cosets of this subset.

          Each left coset is returned as a Subset containing the indices of
          elements in the head group that form the coset.

          Returns
          -------
          left_cosets: list[Subset]
              The list of left cosets.
          )pbdoc")
      .def(
          "right_cosets",
          [](group::Subset const &subset) -> std::vector<group::Subset> {
            return subset.right_cosets();
          },
          R"pbdoc(
          Return a list of right cosets of this subset.

          Each right coset is returned as a Subset containing the indices of
          elements in the head group that form the coset.

          Returns
          -------
          right_cosets: list[Subset]
              The list of right cosets.
          )pbdoc")
      .def(
          "cyclic_subgroups",
          [](group::Subset const &subset) -> std::vector<group::Subset> {
            return subset.cyclic_subgroups();
          },
          R"pbdoc(
          Return a list of the unique cyclic subgroups (each as a Subset).

          These are the unique cyclic subgroups generated by elements of the
          subset.

          Returns
          -------
          cyclic_subgroups: list[Subset]
              The list of unique cyclic subgroups.
          )pbdoc")
      .def(
          "cyclic_generators",
          [](group::Subset const &subset) -> std::vector<Index> {
            return subset.cyclic_generators();
          },
          R"pbdoc(
          Return the generators of the `cyclic_subgroups` for this subset.

          These are the elements that generated the unique cyclic subgroups.

          Returns
          -------
          generators: list[int]
              The indices of the elements that generate the cyclic subgroups.
          )pbdoc")
      .def(
          "cyclic_subgroup_orbits",
          [](group::Subset const &subset) -> group::SubgroupOrbitVec {
            return to_vector_of_orbit_vec(subset.cyclic_subgroups());
          },
          R"pbdoc(
          Return a list of cyclic subgroup orbits

          Returns
          -------
          subgroup_orbits: list[list[list[int]]]
                The list ``subgroup_orbits[i][j]`` contains the indices of
                elements in the head group forming the `j`-th subgroup in the
                `i`-th orbit of equivalent cyclic subgroups.
          )pbdoc")
      .def(
          "maximal_cyclic_subgroups",
          [](group::Subset const &subset) -> std::vector<group::Subset> {
            return subset.maximal_cyclic_subgroups();
          },
          R"pbdoc(
          Return a list of maximal cyclic subgroups (each as a Subset).

          These are the cyclic subgroups generated by elements of the subset
          that are not a subgroup of any other cyclic subgroup of the subset.

          Returns
          -------
          maximal_cyclic_subgroups: list[Subset]
              The list of maximal cyclic subgroups.
          )pbdoc")
      .def(
          "maximal_cyclic_generators",
          [](group::Subset const &subset) -> std::vector<Index> {
            return subset.maximal_cyclic_generators();
          },
          R"pbdoc(
          Return the generators of the `maximal_cyclic_subgroups` for this subset.

          All subset elements are included in the cyclic subgroups generated by
          these elements. Specifically, these are the elements that generated the
          maximal cyclic subgroups.

          Returns
          -------
          generators: list[int]
              The indices of the elements that generate the maximal cyclic
              subgroups.
          )pbdoc")
      .def(
          "maximal_cyclic_subgroup_orbits",
          [](group::Subset const &subset) -> group::SubgroupOrbitVec {
            return to_vector_of_orbit_vec(subset.cyclic_subgroups());
          },
          R"pbdoc(
          Return a list of maximal cyclic subgroup orbits

          Returns
          -------
          subgroup_orbits: list[list[list[int]]]
                The list ``subgroup_orbits[i][j]`` contains the indices of
                elements in the head group forming the `j`-th subgroup in the
                `i`-th orbit of equivalent maximal cyclic subgroups.
          )pbdoc")
      .def(
          "minimal_generators",
          [](group::Subset const &subset) -> std::set<Index> {
            return subset.minimal_generators();
          },
          R"pbdoc(
          Return a minimal set of generators for this subset.

          Closure by multiplication starting from these elements can generate
          all elements of the subset. Specifically, these are the first unique
          elements of `maximal_cyclic_generators` that generate the entire
          subset. It may not be the minimum size generating set.
          )pbdoc")
      .def(
          "_all_subgroup_orbits_v1",
          [](group::Subset const &subset) -> group::SubgroupOrbitVec {
            // -- WARNING: Do not set py::scoped_ostream_redirect here --
            //    May cause a deadlock if there is output to std::cout on
            //    threads.
            //
            // py::scoped_ostream_redirect redirect;

            if (subset.indices().size() != subset.group()->size()) {
              throw std::runtime_error(
                  "Subset._all_subgroup_orbits_v1 is only implemented for the "
                  "full group subset.");
            }

            std::set<group::SubgroupOrbit> orbits_set =
                group::make_all_subgroups(*subset.group());

            // convert from sets to vectors for Python

            group::SubgroupOrbitVec result;
            for (auto const &orbit : orbits_set) {
              std::vector<std::vector<Index>> orbit_vec;
              for (auto const &subgroup : orbit) {
                std::vector<Index> subgroup_vec(subgroup.begin(),
                                                subgroup.end());
                orbit_vec.push_back(subgroup_vec);
              }
              result.push_back(orbit_vec);
            }

            return result;
          },
          R"pbdoc(
          Return a list of all subgroups (each as a Subset) of this subset.

          Uses a depth-first search for combinations of subset elements to use as
          subgroup generators.

          Parameters
          ----------
          n_subtrees: int = 100
              The number of subtrees to divide the search tree into.
          progress_callback: Optional[Callable[[int, int], None]] = None
              A callback function which takes two int arguments: the number of
              finished subtrees and the total number of subgroups found so far.
              This is called each time a task is finished. The default prints
              progress to standard output.

          Returns
          -------
          subgroups: list[Subset]
              The list of all subgroups found.
          )pbdoc")
      .def(
          "_has_all_subgroups",
          [](group::Subset const &subset) -> bool {
            return subset.has_all_subgroups();
          },
          R"pbdoc(
          Return True if all subgroups have been found.
          )pbdoc")
      .def(
          "_all_subgroups",
          [](group::Subset const &subset, Index n_subtrees, std::string method,
             std::optional<std::function<void(Index, Index)>>
                 progress_callback_f) -> std::vector<group::Subset> {
            if (method == "normal_subgroup") {
              return subset.all_subgroups(n_subtrees, method);
            }

            // method == "depth_first_search" (default)
            // -- WARNING: Do not set py::scoped_ostream_redirect here --
            //    May cause a deadlock if there is output to std::cout on
            //    threads.
            //
            // py::scoped_ostream_redirect redirect;

            if (!progress_callback_f.has_value()) {
              progress_callback_f = group::DefaultProgressCallback(n_subtrees);
            }

            // Use the reusable helper which sets a temporary SIGINT handler,
            // releases the Python GIL while running, and restores the old
            // handler on exit or exception.
            std::vector<group::Subset> const &result = run_with_sigint_handler(
                [&]() -> std::vector<group::Subset> const & {
                  return subset.all_subgroups(n_subtrees, method,
                                              *progress_callback_f);
                });

            return result;
          },
          R"pbdoc(
          Return all subgroups of this subset

          Parameters
          ----------
          n_subtrees: int = 100
              Number of subtrees (used by method="depth_first_search" only).
          method: str = "depth_first_search"
              Which algorithm to use:

              - "depth_first_search": depth-first search over generator
                combinations (multithreaded, general purpose).
              - "normal_subgroup": exploits the G = T.F extension structure
                where T = {0,...,N_translations-1}. Requires ``N_translations``
                to be set on the Subset. Faster for large supercell groups.
          progress_callback: Optional[Callable] = None
              Progress callback (used by method="depth_first_search" only).

          Returns
          -------
          subgroups: list[Subset]
              The list of all subgroups found.
          )pbdoc",
          py::arg("n_subtrees") = 100, py::arg("method") = "depth_first_search",
          py::arg("progress_callback") = std::nullopt)
      .def(
          "all_subgroup_generators",
          [](group::Subset const &subset) -> std::vector<std::set<Index>> {
            if (!subset.has_all_subgroups()) {
              throw std::runtime_error(
                  "Subset.all_subgroups_generators is only available after "
                  "calling Subset.all_subgroups.");
            }
            return subset.all_subgroups_generators();
          },
          R"pbdoc(
          Return a list of subgroup generators for each subgroup

          Notes
          -----

          - This subset should be a group
          - This information is only available after calling `all_subgroups`

          Returns
          -------
          generators: list[set[int]]
                Subgroup generators for each of the subgroups found by
                `all_subgroups`.
          )pbdoc")
      .def_property_readonly(
          "is_simple_group",
          [](group::Subset const &subset) -> bool {
            if (!subset.has_all_subgroups()) {
              throw std::runtime_error(
                  "Subset.is_simple_group is only available after "
                  "calling Subset.all_subgroups.");
            }
            return subset.is_simple_group();
          },
          R"pbdoc(
          Return True if this subset is a simple group
          )pbdoc")
      .def(
          "all_subgroup_orbits",
          [](group::Subset const &subset) -> group::SubgroupOrbitVec {
            if (!subset.has_all_subgroups()) {
              throw std::runtime_error(
                  "Subset.all_subgroup_orbits is only available after "
                  "calling Subset.all_subgroups.");
            }
            return to_vector_of_orbit_vec(subset.all_subgroups());
          },
          R"pbdoc(
          Return a list of subgroup orbits

          Notes
          -----

          - This subset should be a group
          - This information is only available after calling `all_subgroups`

          Returns
          -------
          subgroup_orbits: list[list[set[int]]]
                Subgroup orbits found by `all_subgroups`. The list
                ``subgroup_orbits[i][j]`` contains the indices of elements in
                the head group forming the `j`-th subgroup in the `i`-th orbit
                of equivalent subgroups.
          )pbdoc")
      .def("__eq__", [](group::Subset const &a,
                        group::Subset const &b) { return a == b; })
      .def("__ne__", [](group::Subset const &a,
                        group::Subset const &b) { return a != b; })
      .def("__lt__",
           [](group::Subset const &a, group::Subset const &b) { return a < b; })
      .def("__hash__", [](group::Subset const &subset) {
        std::set<Index> const &indices = subset.indices();
        size_t seed = 0;
        for (long x : indices) {
          seed ^= std::hash<long>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
      });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
