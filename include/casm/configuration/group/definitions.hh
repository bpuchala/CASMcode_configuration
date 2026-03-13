// The `casm/configuration/group` module supports groups, orbits, and algorithms
//
// Primarily, this purpose of this module is to provide:
// - Group: a simple group data structure
//   - Includes multiplication_table, inverse, and group-subgroup relationships
// - make_orbit: function to construct and sort an orbit of objects
// - make_equivalence_map: function to construct an equivalence map
// - make_cyclic_subgroups: function to make cyclic (small) subgroups
// - make_all_subgroups_by_cyclic_join: combines cyclic subgroups to form all
// subgroups
// - make_invariant_subgroups: make the invariant subgroup of each element of
//   an orbit
//
// Allowed dependencies:
// - CASMcode_global

#ifndef CASM_group_definitions
#define CASM_group_definitions

#include <iomanip>
#include <memory>
#include <set>
#include <vector>

#include "casm/casm_io/Log.hh"
#include "casm/external/Eigen/Dense"

namespace CASM {
namespace group {

typedef long Index;

typedef std::vector<Eigen::MatrixXd> MatrixRep;
typedef std::set<Index> GroupIndices;
typedef std::set<GroupIndices> GroupIndicesOrbit;
typedef std::set<GroupIndicesOrbit> GroupIndicesOrbitSet;
typedef std::vector<std::vector<std::vector<Index>>> GroupIndicesOrbitVec;

typedef std::set<Index> SubgroupIndices;
typedef std::set<SubgroupIndices> SubgroupOrbit;
typedef std::set<SubgroupOrbit> SubgroupOrbitSet;
typedef std::vector<std::vector<std::vector<Index>>> SubgroupOrbitVec;

struct GenericGroup;

template <typename ElementType>
struct Group;

/// \brief Specifies the multiplication table for a group
///
/// Defined according to:
///
///     element[k] == element[i] * element[j],
///     k = multiplication_table[i][j]
///
typedef std::vector<std::vector<Index>> MultiplicationTable;

inline void append_time(Log &log, int n_newlines) {
  if (log.print()) {
    log.ostream() << " - Time: " << std::setprecision(6) << log.time_s()
                  << " (s)";
    for (int i = 0; i < n_newlines; ++i) {
      log.ostream() << std::endl;
    }
  }
}

}  // namespace group
}  // namespace CASM

#endif
