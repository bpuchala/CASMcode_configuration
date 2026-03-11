#ifndef CASM_group_subgroups
#define CASM_group_subgroups

#include <algorithm>
#include <iterator>
#include <map>
#include <shared_mutex>

#include "casm/casm_io/container/stream_io.hh"
#include "casm/configuration/group/Group.hh"
#include "casm/configuration/group/definitions.hh"
#include "casm/container/Counter.hh"
#include "casm/global/threads.hh"

namespace CASM {
namespace group {

namespace {
inline void print_indices(std::set<Index> const &indices) {
  std::cout << "{ ";
  Index idx = 0;
  for (auto const &i : indices) {
    if (idx != 0) {
      std::cout << ", ";
    }
    std::cout << i;
    ++idx;
  }
  std::cout << "}";
}

}  // namespace

/// \brief Return all cyclic subgroups
std::set<SubgroupOrbit> make_cyclic_subgroups(GenericGroup const &group);

/// \brief Return all subgroups
std::set<SubgroupOrbit> make_all_subgroups(GenericGroup const &group);

/// \brief Make the invariant subgroup for each orbit element, as
///     indices of group elements
std::vector<SubgroupIndices> make_invariant_subgroups(
    std::vector<std::vector<Index>> const &equivalence_map,
    GenericGroup const &group);

/// \brief Utility class generating a subgroup from known generators
struct MakeSubgroupFromGenerators {
  /// \brief Elements generated upon closure
  std::set<Index> indices;

  /// \brief True if element in head group is member of subgroup
  std::vector<bool> member;

  /// \brief The generating elements
  std::set<Index> generators;

  /// \brief Constructor
  ///
  /// \param group The head group, with multiplication table to use
  /// \param generators Indices of known subgroup generators
  MakeSubgroupFromGenerators(GenericGroup const &group)
      : member(group.size(), false) {}

  /// \brief Constructor
  ///
  /// \param group The head group, with multiplication table to use
  /// \param generators Indices of known subgroup generators
  MakeSubgroupFromGenerators(GenericGroup const &group,
                             std::set<Index> const &_generators)
      : member(group.size(), false), generators(_generators) {
    /// Add identity to the result:
    indices.insert(0);
    member[0] = true;

    /// Queue of products - start with identity
    std::set<Index> queue;
    queue.insert(0);

    _close(group, queue);
  }

  void add_generator(GenericGroup const &group, Index k) {
    if (member[k]) {
      return;
    }
    generators.insert(k);

    std::set<Index> queue;

    if (indices.size()) {
      /// Products of the existing elements and the new generator
      /// to get new elements which are added to the queue
      Index prod;
      for (Index i : indices) {
        prod = group.mult(i, k);
        if (!indices.count(prod)) {
          queue.insert(prod);
        }
      }

    } else {
      /// Add identity to the result:
      indices.insert(0);
      member[0] = true;

      /// Queue of products - start with identity
      queue.insert(0);
    }

    _close(group, queue);
  }

 private:
  void _close(GenericGroup const &group, std::set<Index> &queue) {
    /// Products of the new elements and all generators
    Index prod;
    do {
      auto begin = queue.begin();
      Index x = *begin;
      queue.erase(*begin);

      // For each generator:
      for (auto j : generators) {
        // Check x * j
        prod = group.mult(x, j);
        if (!member[prod]) {
          indices.insert(prod);
          member[prod] = true;
          queue.insert(prod);
        }
        // Check x * j^-1
        prod = group.mult(x, group.inv(j));
        if (!member[prod]) {
          indices.insert(prod);
          member[prod] = true;
          queue.insert(prod);
        }
      }

    } while (queue.size());
  }
};

/// \brief Return true if `a` is a subset of `b` (may be equal)
inline bool is_subset_of(std::set<Index> const &a, std::set<Index> const &b) {
  if (a.size() > b.size()) {
    return false;
  }

  // Both sets are ordered; use std::includes to check if `this` contains
  // all elements of `other`.
  return std::includes(b.begin(), b.end(), a.begin(), a.end());
}

/// \brief Return true if `a` is a proper subset of `b` (may not be equal)
inline bool is_proper_subset_of(std::set<Index> const &a,
                                std::set<Index> const &b) {
  if (a.size() == b.size()) {
    return false;
  }
  return is_subset_of(a, b);
}

/// \brief Utility class finding a minimal set of generators
struct MakeUniqueCyclicSubgroups {
  /// Unique cyclic subgroups
  std::vector<std::set<Index>> cyclic_subgroups;

  /// Elements which generate the corresponding unique cyclic subgroup
  std::vector<Index> generators;

  /// \brief Constructor
  ///
  /// \param group The head group, with multiplication table to use
  /// \param indices Indices of subset elements
  MakeUniqueCyclicSubgroups(GenericGroup const &group,
                            std::set<Index> const &indices) {
    std::map<std::set<Index>, Index> tmp_subgroups;
    for (Index x : indices) {
      std::set<Index> new_cycle;
      Index prod = x;
      while (new_cycle.insert(prod).second) {
        prod = group.mult(x, prod);
      }

      tmp_subgroups.emplace(new_cycle, x);
    }

    for (auto const &pair : tmp_subgroups) {
      cyclic_subgroups.push_back(pair.first);
      generators.push_back(pair.second);
    }
  }
};

/// \brief Utility class finding a minimal set of generators
struct MakeMaximalCyclicSubgroups {
  /// Cyclic subgroups which are not contained by another cyclic subgroup
  std::vector<std::set<Index>> maximal_cyclic_subgroups;

  /// Elements which generate the corresponding maximal cyclic subgroup
  std::vector<Index> generators;

  /// \brief Constructor
  ///
  /// \param group The head group, with multiplication table to use
  /// \param indices Indices of subset elements
  MakeMaximalCyclicSubgroups(GenericGroup const &group,
                             std::set<Index> const &indices) {
    for (Index x : indices) {
      std::set<Index> new_cycle;
      Index prod = x;
      while (new_cycle.insert(prod).second) {
        prod = group.mult(x, prod);
      }

      _update(x, new_cycle);
    }
  }

 private:
  void _update(Index new_generator, std::set<Index> const &new_cycle) {
    Index i = 0;
    while (i != maximal_cyclic_subgroups.size()) {
      auto const &existing = maximal_cyclic_subgroups[i];
      if (is_subset_of(new_cycle, existing)) {
        return;
      }
      if (is_proper_subset_of(existing, new_cycle)) {
        if (i != maximal_cyclic_subgroups.size() - 1) {
          generators[i] = std::move(generators.back());
          maximal_cyclic_subgroups[i] = maximal_cyclic_subgroups.back();
        }
        generators.pop_back();
        maximal_cyclic_subgroups.pop_back();
      } else {
        ++i;
      }
    }
    generators.push_back(new_generator);
    maximal_cyclic_subgroups.push_back(new_cycle);
  }
};

/// \brief Utility class finding a minimal set of generators
struct MakeMinimalSubsetGenerators {
  /// \brief A minimal set of generators of the subset
  std::set<Index> generators;

  /// Elements generated upon closure - may include elements not in the
  /// original subset if the subset is not a group
  std::set<Index> indices;

  /// \brief Constructor
  ///
  /// \param group The head group, with multiplication table to use
  /// \param indices Indices of subset elements
  MakeMinimalSubsetGenerators(GenericGroup const &group,
                              std::set<Index> const &subset_indices) {
    MakeMaximalCyclicSubgroups f(group, subset_indices);

    // Make a lookup table for conjugacy classes included in the new subgroup
    std::vector<bool> classes_included;
    for (Index i : subset_indices) {
      Index cc = group.class_of(i);
      if (cc >= classes_included.size()) {
        classes_included.resize(cc + 1);
      }
      classes_included[cc] = false;
    }

    // Loop over maximal cyclic subgroup generators,
    // adding the ones that add new conjugacy classes first
    MakeSubgroupFromGenerators x(group);
    for (Index i = 0; i < f.generators.size(); ++i) {
      // Update classes included in the subgroup already
      for (Index j : x.indices) {
        classes_included[group.class_of(j)] = true;
      }

      // Select best next generator
      std::optional<Index> best_generator = std::nullopt;
      Index best_n_new_class = 0;
      for (Index k_index = 0; k_index < f.generators.size(); ++k_index) {
        Index k = f.generators[k_index];
        if (x.generators.count(k)) {
          continue;
        }
        if (x.indices.count(k)) {
          continue;
        }
        Index n_new_class = 0;
        for (Index j : f.maximal_cyclic_subgroups[k_index]) {
          if (!classes_included[group.class_of(j)]) {
            n_new_class++;
          }
        }
        if (!best_generator.has_value()) {
          best_generator = k;
          best_n_new_class = n_new_class;
        } else if (n_new_class > best_n_new_class) {
          best_generator = k;
          best_n_new_class = n_new_class;
        }
      }

      if (!best_generator.has_value()) {
        break;
        // throw std::runtime_error(
        //     "Error in MakeMinimalSubsetGenerators: logic error");
      }

      x.add_generator(group, *best_generator);
    }
    generators = x.generators;
    indices = x.indices;
  }
};

struct DefaultProgressCallback {
  Index n_subtrees;

  DefaultProgressCallback(Index _n_subtrees) : n_subtrees(_n_subtrees) {}

  void operator()(Index n_finished_tasks, Index subgroups_size) {
    long double percent =
        (static_cast<long double>(n_finished_tasks) * 100.0L) /
        static_cast<long double>(n_subtrees);
    std::cout << "\r- Subgroups found: " << subgroups_size << " | "
              << std::fixed << std::setprecision(0) << percent << "% finished"
              << std::flush;
  }
};

/// \brief Multi-threaded method to find all subgroups
///
/// This method:
/// 1. Finds all cyclic subgroups and stores the elements which generate unique
///    cyclic subgroups as the candidate generators.
/// 2. Performs a depth-first search over combinations of the candidate
///    generators to find all subgroups.
///
/// Nodes in the tree are combinations of candidate generators, sorted in
/// lexicographical order. At each node, the subgroup generated by the
/// combination of generators is found by adding a generator to the subgroup
/// found at the parent node.
///
/// Results are stored in a map with keys being the set of subgroup indices and
/// values being the set of subgroup generators.
///
/// If the subgroup is new, it is added to the results and the search
/// continues deeper. Generators that are already in the subgroup can be
/// skipped while moving deeper along a branch.
///
/// If the subgroup is not new, the search skips to the next branch. Because of
/// the lexicographical ordering of the search tree, finding an existing
/// subgroup at a node guarantees that all the subgroups generated by any
/// combination of generators deeper along that branch of the tree have
/// already been found.
///
/// The depth-first search is parallelized by breaking the tree into subtrees
/// and assigning each subtree to a thread. Each thread performs a depth-first
/// search over its assigned subtree, storing found subgroups in a shared map
/// which is protected by a mutex. Most of the time spent in the threads is
/// consumed by subgroup generation and checking if a subgroup is new, so the
/// threads need read-only access most of the time and write access only when
/// a new subgroup is found.
///
/// Positions in the tree are represented by a vector of iterators into the
/// candidate generators. Each iterator corresponds to a level in the tree,
/// and points to the next candidate generator to be added at that level. The
/// beginning and end position of each subtree is determined using an
/// approximately equal division of the total number of positions in the tree,
/// based on indexing the positions lexicographically. By finding the ln of the
/// total number of positions, the ln(index) of the beginning of the i-th
/// subtree can be calculated, and the position corresponding to that ln(index)
/// can be found determinstically. Beginning and end positions for each subtree
/// are limited to combinations with a certain maximum size, max_k.
///
/// Each subtree is assigned to a thread in a thread pool, with the number of
/// subtrees set to ~100 or ~1000 to avoid blocking while waiting for a long
/// thread. Naturally, initial branches are deeper and have more subgroups, so
/// the final threads are very unlikely to take much longer than the first
/// threads.
///
struct MakeAllSubgroupsFromGenerators {
  /// \brief The head group
  std::shared_ptr<GenericGroup const> group;

  /// \brief The subgroup for which subgroups are being found
  std::set<Index> indices;

  /// \brief Generators for unique cyclic subgroups
  std::set<Index> candidate_generators;

  /// \brief Map type: (subgroup indices) -> (generators)
  typedef std::map<std::set<Index>, std::set<Index>> map_type;

  /// \brief Map iterator type
  typedef map_type::iterator map_iterator;

  /// \brief All subgroups, as the map (indices) -> (generators)
  ///
  /// This also includes the input subgroup itself (which may be the full
  /// group).
  map_type subgroups;

  // /// \brief All subgroups, organized in orbits
  // std::shared_ptr<std::set<SubgroupOrbit>> subgroup_orbits;

 private:
  // Mutex for access to subgroups map
  std::shared_mutex m_subgroups_mutex;

  /// \brief Return the number of subgroups
  map_type::size_type subgroups_size() {
    std::shared_lock lock(m_subgroups_mutex);
    return subgroups.size();
  }

  /// \brief Return the number of subgroups with given indices
  ///
  /// \param indices The subgroup indices
  ///
  map_type::size_type subgroups_count(std::set<Index> const &node_indices) {
    std::shared_lock lock(m_subgroups_mutex);
    return subgroups.count(node_indices);
  }

  /// \brief Emplace a subgroups with given indices and generators
  ///
  /// \param indices The subgroup indices
  /// \param generators The subgroup generators
  ///
  std::pair<map_type::iterator, bool> subgroups_emplace(
      std::set<Index> const &node_indices,
      std::set<Index> const &node_generators) {
    std::unique_lock lock(m_subgroups_mutex);
    return subgroups.emplace(node_indices, node_generators);
  }

 public:
  // typedefs
  typedef MakeSubgroupFromGenerators node_type;
  typedef std::set<Index>::const_iterator member_iterator;

  // Debug print function
  void print_position(std::vector<member_iterator> &iters,
                      std::vector<node_type> &nodes) {
    Log &log = CASM::log();
    log.indent() << "---" << std::endl;
    log.indent() << "# subgroups: " << subgroups_size() << std::endl;
    log.indent() << "Current position:" << std::endl;
    log.indent() << "- nodes: ";
    for (auto const &n : nodes) {
      print_indices(n.generators);
    }
    log.indent() << std::endl;
    log.indent() << "- iters: ";
    print_iters(iters);
    log.indent() << std::endl << std::endl;
  };

  /// \brief Given a position in the tree (iters), initialize nodes so that
  ///     the position is created next time `make_next_node` is called.
  void initialize_nodes(std::vector<member_iterator> const &iters,
                        std::vector<node_type> &nodes) {
    nodes.clear();
    auto it = iters.begin();
    auto end = iters.end();
    while (true) {
      Index value = **it;
      ++it;
      if (it == end) {
        break;
      }
      // Create next node
      if (nodes.size() == 0) {
        // New root - construct new empty node
        nodes.emplace_back(*group);
      } else {
        // Copy an existing node
        auto const &last_node = nodes.back();
        nodes.emplace_back(last_node);
      }
      auto &next_node = nodes.back();

      // Add the new generator - will generate all elements in the subgroup
      // auto next_it = iters.back();
      next_node.add_generator(*group, value);
    }
  }

  /// \brief Given a position in the tree (iters),
  ///      set nodes to the corresponding vector
  void set_nodes(std::vector<member_iterator> const &iters,
                 std::vector<node_type> &nodes) {
    nodes.clear();
    for (auto it = iters.begin(); it != iters.end(); ++it) {
      // Create next node
      if (nodes.size() == 0) {
        // New root - construct new empty node
        nodes.emplace_back(*group);
      } else {
        // Copy an existing node
        auto const &last_node = nodes.back();
        nodes.emplace_back(last_node);
      }
      auto &next_node = nodes.back();

      // Add the new generator - will generate all elements in the subgroup
      // auto next_it = iters.back();
      next_node.add_generator(*group, **it);
    }
  }

  /// \brief Compare two positions in the tree
  ///
  /// - This uses the fact that indices are sorted so we can compare set
  /// iterators by comparing the underlying value the iterator points at.
  /// - An `iters` with size 0 is treated as the `end` position.
  bool less_than(std::vector<member_iterator> const &iters_a,
                 std::vector<member_iterator> const &iters_b) {
    if (iters_a.size() == 0) {
      return false;
    }
    if (iters_b.size() == 0) {
      return true;
    }

    auto a_it = iters_a.begin();
    auto a_end = iters_a.end();
    auto b_it = iters_b.begin();
    auto b_end = iters_b.end();

    while (a_it != a_end && b_it != b_end) {
      if (**a_it == **b_it) {
        ++a_it;
        ++b_it;
      } else {
        return (**a_it < **b_it);
      }
    }
    if (a_it == a_end && b_it != b_end) {
      return true;
    }
    return false;
  }

  /// \brief Advance the position in the tree if the current position is not
  /// valid
  ///
  /// - This modifies iters and nodes
  void advance_if_invalid(std::vector<member_iterator> &local_iters,
                          std::vector<node_type> &local_nodes) {
    // --- Identify next position in depth-first search ---

    // Index debug_print_count = 0;

    // Backtrack over exhausted iterators before preparing the next node.
    // If the last iterator is `candidate_generators.end()` we should pop it
    // (and the corresponding node if present) and continue backtracking. This
    // prevents dereferencing `candidate_generators.end()` below when the subset
    // has only the identity element (or when deeper levels are exhausted).
    do {
      // debug_print_count++;
      // if (debug_print_count >= 10000) {
      //   // print_position();
      //   // std::cout << "Total checks made: " << total_checks_made <<
      //   std::endl; debug_print_count = 0;
      // }
      while (local_nodes.size() > 0 && local_iters.size() > 0 &&
             local_iters.back() == candidate_generators.end()) {
        // Backtracking over exhausted iterator
        local_nodes.pop_back();
        local_iters.pop_back();
        ++(local_iters.back());
      }
      if (local_nodes.size() == 0) {
        if (local_iters.back() == candidate_generators.end()) {
          local_iters.clear();
        }
        return;
      }

      auto const &last_node = local_nodes.back();
      auto next_it = local_iters.back();

      // Check if the next generator is already in the subgroup indices
      if (!last_node.indices.count(*next_it)) {
        // Generator not already in subgroup indices, proceed
        break;
      }

      // Generator already in subgroup indices, advance iterator
      ++(local_iters.back());
    } while (true);
  }

  /// \brief Make the next node
  ///
  /// After advancing to the next valid position, and checking that the
  /// termination criteria has not been reached, this is used to create a new
  /// node at the current position
  void make_next_node(std::vector<member_iterator> &local_iters,
                      std::vector<node_type> &local_nodes) {
    // Create next node
    if (local_nodes.size() == 0) {
      // New root - construct new empty node
      local_nodes.emplace_back(*group);
    } else {
      // Copy an existing node
      auto const &last_node = local_nodes.back();
      local_nodes.emplace_back(last_node);
    }
    auto &next_node = local_nodes.back();

    // Add the new generator - will generate all elements in the subgroup
    auto next_it = local_iters.back();
    next_node.add_generator(*group, *next_it);
  }

  /// \brief Debug print function for subtree positions
  void print_iters(std::vector<member_iterator> const &local_iters) {
    std::cout << "{ ";
    Index idx = 0;
    for (auto const &it : local_iters) {
      if (idx != 0) {
        std::cout << ", ";
      }
      if (it != candidate_generators.end()) {
        std::cout << *it;
      } else {
        std::cout << "end";
      }
      ++idx;
    }
    std::cout << "}";
  }

  // Debug print function for node generators
  void print_node_generators(node_type const &local_node) {
    print_indices(local_node.generators);
  }

  // Debug print function for node indices
  void print_node_indices(node_type const &local_node) {
    print_indices(local_node.indices);
  }

  // Debug print function for inserted subgroup
  void print_insert_info(node_type const &local_node) {
    std::cout << "New subgroup generators: ";
    print_node_generators(local_node);
    std::cout << std::endl;

    std::cout << "New subgroup indices: ";
    print_node_indices(local_node);
    std::cout << std::endl;

    std::cout << "# Subgroups: " << subgroups_size() << std::endl;
    std::cout << std::endl;
  }

  /// \brief Insert the current node into the local subgroup map and increment
  ///     local_iters and local_nodes
  bool insert_local_node(std::vector<member_iterator> &local_iters,
                         std::vector<node_type> &local_nodes) {
    auto &next_node = local_nodes.back();

    // Add the new generator - will generate all elements in the subgroup
    auto next_it = local_iters.back();

    auto indices_it = next_node.indices.begin();
    auto indices_end = next_node.indices.end();

    // Try adding subgroup
    bool successful_insert = false;
    if (!subgroups_count(next_node.indices)) {
      auto emplace_res =
          subgroups_emplace(next_node.indices, next_node.generators);
      successful_insert = emplace_res.second;
    }

    if (successful_insert && next_node.indices.size() != indices.size()) {
      // Proceeding deeper in the search...

      // New subgroup found which is not the full group
      // Keep new node and continue the search deeper
      ++next_it;
      local_iters.push_back(next_it);
    } else {
      // Backtracking...

      // Existing subgroup or full group generated
      // Backtrack before continuing the search
      local_nodes.pop_back();
      ++(local_iters.back());
    }

    return successful_insert;
  }

  /// \brief Task structure
  ///
  /// Defines the beginning and end positions of a subtree in the search tree
  /// in which the worker will search for subgroups.
  struct Task {
    std::vector<member_iterator> begin;
    std::vector<member_iterator> end;

    Task() = default;
    Task(std::vector<member_iterator> _begin, std::vector<member_iterator> _end)
        : begin(std::move(_begin)), end(std::move(_end)) {}

    // Comparison operator for ordering tasks:
    bool operator<(Task const &other) const {
      auto a_it = begin.begin();
      auto a_end = begin.end();
      auto b_it = other.begin.begin();
      auto b_end = other.begin.end();

      while (a_it != a_end && b_it != b_end) {
        if (**a_it < **b_it) {
          return true;
        } else if (**b_it < **a_it) {
          return false;
        }
        ++a_it;
        ++b_it;
      }
      return false;
    }
  };

  /// \brief Result structure
  ///
  /// Since subgroups are stored directly in the main class by the workers,
  /// the result just returns the task that was completed.
  struct Result {
    Task task;

    Result() = default;
    Result(Task _task) : task(std::move(_task)) {}
  };

  // --- Logarithmic combinatorial functions ---

  typedef long double long_double_t;

  // Natural log of nCr using log-gamma function
  long_double_t log_ncr(int n, int r) {
    if (r < 0 || r > n) return -1e18;
    if (r == 0 || r == n) return 0.0;
    return lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1);
  }

  // Computes ln(sum_{i=0}^{max_k} nCr(n, i)) using LogSumExp trick
  long_double_t log_sum_ncr(int n, int max_k) {
    if (max_k < 0) return -1e18;
    std::vector<long_double_t> logs;
    for (int i = 0; i <= max_k; ++i) {
      logs.push_back(log_ncr(n, i));
    }
    long_double_t max_l = *std::max_element(logs.begin(), logs.end());
    long_double_t sum_exp = 0;
    for (long_double_t l : logs) {
      sum_exp += std::exp(l - max_l);
    }
    return max_l + std::log(sum_exp);
  }

  // Finds combination at target_log index in lexicographical power set
  std::vector<Index> get_lex_combination(Index n, Index max_k,
                                         long_double_t target_log,
                                         long_double_t total_log) {
    // If target is at or beyond the total count, return empty set
    if (target_log >= total_log) {
      return {};
    }

    std::vector<Index> result;
    long_double_t current_offset_log = -1e18;
    int start_val = 0;

    while (result.size() < (size_t)max_k) {
      bool found = false;
      for (int v = start_val; v < n; ++v) {
        // Check the node itself (the current prefix)
        // If target is less than or equal to current offset, we've arrived
        if (target_log <= current_offset_log) return result;

        // Calculate branch size (subtree rooted at v)
        Index remaining_k = max_k - (Index)result.size() - 1;
        long_double_t subtree_size_log = log_sum_ncr(n - 1 - v, remaining_k);

        // LogSumExp: log(exp(offset) + exp(subtree))
        long_double_t branch_max_log =
            std::max(current_offset_log, subtree_size_log) +
            std::log(std::exp(current_offset_log -
                              std::max(current_offset_log, subtree_size_log)) +
                     std::exp(subtree_size_log -
                              std::max(current_offset_log, subtree_size_log)));

        if (target_log < branch_max_log) {
          result.push_back(v);
          start_val = v + 1;
          found = true;
          break;
        } else {
          current_offset_log = branch_max_log;
        }
      }
      if (!found) {
        break;
      }
    }
    return result;
  }

  /// \brief Find the position in the tree for the given subtree
  ///
  /// Divides the tree into `n_subtrees` subtrees, and finds the approximate
  /// position of the `subtree`-th subtree. The position vector returned is
  /// limited to combinations of size at most `max_k` to avoid wasting time
  /// because only approximate positions are needed to partition the tree.
  /// The result is determinstic.
  ///
  /// \param candidate_generators The candidate generators for subgroups
  /// \param max_k The maximum size of the position vector to return.
  /// \param subtree The subtree index (0-based). The first subtree is 0, and
  ///     gives position {`candidate_generators.begin()`}. If `subtree` equals
  ///     `n_subtrees`, this returns an empty position vector indicating the
  ///     end of the search.
  /// \param n_subtrees The total number of subtrees to divide the tree into.
  ///
  /// \return The position in the tree as a vector of iterators into
  ///     `candidate_generators`.
  std::vector<member_iterator> subtree_position(Index max_k, Index subtree,
                                                Index n_subtrees) {
    std::vector<Index> res;
    if (subtree == 0) {
      res = {0};  // First generator
    } else if (subtree == n_subtrees) {
      res = {};  // Empty set
    } else {
      // Linear interpolation in the range [1, total_count]
      // Calculation: log(1 + (subtree/n_subtrees)*(total_count - 1))
      // For large values, this is effectively log(subtree/n_subtrees) +
      // total_log
      Index n = candidate_generators.size();
      long_double_t total_log = log_sum_ncr(n, max_k);
      long_double_t current_log =
          std::log((long_double_t)subtree / n_subtrees) + total_log;
      res = get_lex_combination(n, max_k, current_log, total_log);
    }

    /// Convert indices to iterators:
    std::vector<member_iterator> candidate_iters;
    auto it = candidate_generators.begin();
    auto end = candidate_generators.end();
    while (it != end) {
      candidate_iters.push_back(it);
      ++it;
    }

    std::vector<member_iterator> iters;
    for (Index idx : res) {
      iters.push_back(candidate_iters[idx]);
    }
    return iters;
  }

  // ---

  /// \brief Constructor
  ///
  /// Uses the full group as the subgroup for which subgroups are found.
  MakeAllSubgroupsFromGenerators(std::shared_ptr<GenericGroup const> _group)
      : group(_group) {
    for (Index i = 0; i < group->size(); ++i) {
      indices.insert(i);
    }
  }

  /// \brief Constructor
  ///
  /// This assumes that `indices` is subset that is a group.
  MakeAllSubgroupsFromGenerators(std::shared_ptr<GenericGroup const> _group,
                                 std::set<Index> const &_indices)
      : group(_group), indices(_indices) {}

  /// \brief Run the subgroup finding algorithm
  void run(Index n_subtrees = 100,
           std::function<void(Index, Index)> progress_callback = nullptr);
};

/// \brief Data for the quotient group T/S
///
/// Given a group T and a normal subgroup S ≤ T, this struct holds the quotient
/// group T/S along with the coset map and canonical representatives.
struct QuotientGroupData {
  /// \brief The quotient group T/S as a GenericGroup
  std::shared_ptr<GenericGroup> group;

  /// \brief coset_of[t] = coset index of element t in T/S (size = |T|)
  std::vector<Index> coset_of;

  /// \brief rep[c] = index of the canonical representative (smallest element)
  ///     of coset c in T (size = |T/S|)
  std::vector<Index> rep;
};

/// \brief Compute the quotient group T/S for a normal subgroup S of T
///
/// Cosets are enumerated in order of their smallest element, which becomes
/// the canonical representative. Since T is abelian (translation group),
/// left and right cosets coincide.
///
/// \param T_group The group T (must be the head group, not a subgroup)
/// \param S_indices Indices forming a normal subgroup S of T
/// \returns QuotientGroupData: quotient group T/S, coset map, and
///     representatives
inline QuotientGroupData make_quotient_group_data(
    GenericGroup const &T_group, std::set<Index> const &S_indices) {
  Index N_T = T_group.size();
  Index N_S = static_cast<Index>(S_indices.size());
  Index N_TS = N_T / N_S;

  std::vector<Index> coset_of(N_T, -1);
  std::vector<Index> rep;
  rep.reserve(N_TS);

  // Enumerate cosets in order of smallest representative.
  // Iterating t from 0 upward, the first unassigned t is the smallest
  // element of its coset (all smaller coset-mates would have been
  // assigned when their own coset was first encountered).
  Index coset_idx = 0;
  for (Index t = 0; t < N_T; ++t) {
    if (coset_of[t] != -1) continue;
    rep.push_back(t);
    for (Index s : S_indices) {
      coset_of[T_group.mult(t, s)] = coset_idx;
    }
    ++coset_idx;
  }

  // Build quotient multiplication table:
  //   quot_mult[a][b] = coset_of[T.mult(rep[a], rep[b])]
  MultiplicationTable quot_mult(N_TS, std::vector<Index>(N_TS));
  for (Index a = 0; a < N_TS; ++a)
    for (Index b = 0; b < N_TS; ++b)
      quot_mult[a][b] = coset_of[T_group.mult(rep[a], rep[b])];

  return {std::make_shared<GenericGroup>(quot_mult), std::move(coset_of),
          std::move(rep)};
}

/// \brief Find all subgroups using the normal translation subgroup structure
///
/// Exploits the group extension structure G = T.F, where:
/// - G is the full group (e.g. all SupercellSymOp combinations)
/// - T = {0, ..., N_translations - 1} is the normal translation subgroup
/// - F = G/T is the quotient group (e.g. the supercell factor group)
///
/// Elements of G are indexed as:
///   g = f * N_translations + t
/// where f is the F-index (0, ..., |F|-1) and t is the T-index
/// (0, ..., N_translations-1). This is consistent with SupercellSymOp
/// iteration order (translations in inner loop, factor group in outer loop).
///
/// The algorithm:
/// 1. Constructs T and F as standalone GenericGroup objects from G's
///    multiplication table.
/// 2. Finds all subgroups of T and all subgroups of F using
///    MakeAllSubgroupsFromGenerators.
/// 3. Precomputes the conjugation action of F on T:
///    conj_action[f][t] = rep(f) * t * rep(f)^{-1}, where rep(f) = f *
///    N_translations.
/// 4. For each subgroup S of T and subgroup K of F, checks whether K
///    normalizes S (i.e. conj(k, s) in S for all k in K, s in S).
/// 5. For each K-normalizing (K, S) pair, enumerates all valid sections
///    ξ: K → T/S via find_all_sections_brute_force. Each valid ξ defines a
///    distinct subgroup H_ξ ≤ G with H_ξ ∩ T = S and H_ξ/(H_ξ ∩ T) ≅ K.
///
/// Correctness:
/// - Every subgroup H ≤ G satisfies H ∩ T = S for some S ≤ T and projects
///   onto some K ≤ F with K normalizing S, so all subgroups are found.
/// - This holds for both symmorphic (G = T x| F) and non-symmorphic groups.
///
/// Performance:
/// - The section search is brute-force: O(|T/S|^r × |K|²) per (K, S) pair,
///   where r = number of K generators (typically ≤ 3). This can be slow when
///   |T/S| is large (large supercell, small S).
///
struct MakeAllSubgroupsFromNormalSubgroup {
  /// \brief The head group G
  std::shared_ptr<GenericGroup const> group;

  /// \brief Size of the translation subgroup T = {0, ..., N_translations - 1}
  Index N_translations;

  /// \brief Map type: (subgroup indices in G) -> (generators in G)
  typedef std::map<std::set<Index>, std::set<Index>> map_type;

  /// \brief All subgroups found, as (indices in G) -> (generators in G)
  map_type subgroups;

  /// \brief True if G is a semidirect product T x| F (symmorphic case)
  ///
  /// G is a semidirect product if and only if the coset representatives
  /// {f * N_translations : f in F} form a subgroup of G, i.e. the product
  /// of any two coset representatives is also a coset representative:
  ///   group->mult(f1 * N_T, f2 * N_T) % N_T == 0  for all f1, f2 in F.
  ///
  /// run() finds all subgroups regardless of this flag. For symmorphic groups
  /// the 2-cocycle is identically zero, making the section search faster in
  /// practice (only 1-cocycles ξ ∈ Z¹(K, T/S) are valid). The flag is
  /// retained for informational purposes.
  bool is_symmorphic;

  /// \brief Constructor
  ///
  /// \param _group The full group G
  /// \param _N_translations Size of the translation subgroup T. The
  ///     translation subgroup is T = {0, ..., _N_translations - 1}.
  MakeAllSubgroupsFromNormalSubgroup(std::shared_ptr<GenericGroup const> _group,
                                     Index _N_translations)
      : group(_group), N_translations(_N_translations) {
    Index N_T = N_translations;
    Index N_F = group->size() / N_T;
    is_symmorphic = true;
    for (Index f1 = 0; f1 < N_F && is_symmorphic; ++f1)
      for (Index f2 = 0; f2 < N_F && is_symmorphic; ++f2)
        if (group->mult(f1 * N_T, f2 * N_T) % N_T != 0) is_symmorphic = false;
  }

  /// \brief Run the subgroup finding algorithm
  ///
  /// Finds all subgroups of G for both symmorphic and non-symmorphic groups
  /// by enumerating all valid sections ξ: K → T/S for each (K, S) pair.
  /// The brute-force section search has complexity O(|T/S|^r × |K|²) per
  /// pair, where r is the number of generators of K (typically ≤ 3).
  ///
  /// \param n_subtrees_F Number of subtrees for MakeAllSubgroupsFromGenerators
  ///     applied to F = G/T (the quotient group).
  /// \param n_subtrees_T Number of subtrees for MakeAllSubgroupsFromGenerators
  ///     applied to T (the translation subgroup).
  void run(Index n_subtrees_F = 100, Index n_subtrees_T = 100);
};

/// \brief From found subgroups, make a container with maximal proper subgroups
///
/// A maximal proper subgroup is a proper subgroup that is not a proper subset
/// of any other proper subgroup.
///
/// \param indices The indices of the original subgroup (may be the full group).
/// \param subgroups The map of all found subgroups, as (indices) ->
///     (generators).
///
/// \return A map of maximal proper subgroups, as (indices) -> (generators).
///
inline std::map<std::set<Index>, std::set<Index>> make_maximal_proper_subgroups(
    std::set<Index> const &indices,
    std::map<std::set<Index>, std::set<Index>> const &subgroups) {
  std::map<std::set<Index>, std::set<Index>> maximal_proper_subgroups;

  auto it_a = subgroups.begin();
  auto end = subgroups.end();
  for (; it_a != end; ++it_a) {
    std::set<Index> const &indices_a = it_a->first;

    if (indices_a.size() == indices.size()) {
      // Not a proper subgroup
      continue;
    }

    bool found = false;
    auto it_b = subgroups.begin();
    for (; it_b != end; ++it_b) {
      if (it_a == it_b) {
        continue;
      }
      std::set<Index> const &indices_b = it_b->first;
      if (indices_b.size() == indices.size()) {
        // Don't compare to full input subgroup
        continue;
      }

      // Check if indices_a is a subgroup of indices_b
      if (indices_a.size() < indices_b.size() &&
          std::includes(indices_b.begin(), indices_b.end(), indices_a.begin(),
                        indices_a.end())) {
        // a is a proper subset of b; mark a as non-maximal
        found = true;
        break;
      }
    }

    if (!found) {
      maximal_proper_subgroups.insert(*it_a);
    }
  }
  return maximal_proper_subgroups;
}

/// \brief A subset of a head group, for purposes of calculating properties
///
/// By design, this class has the semantics of an immutable object; all
/// modifying operations return a new Subset object rather than modifying
/// the existing object. Some properties are lazily evaluated and cached.
class Subset {
 public:
  /// \brief Constructor (full group)
  ///
  /// \param group The head group
  /// \param N_translations Optional size of the translation subgroup T =
  ///     {0, ..., N_translations-1}. Required for the "normal_subgroup" method
  ///     of all_subgroups().
  Subset(std::shared_ptr<GenericGroup const> group,
         std::optional<Index> N_translations = std::nullopt)
      : Subset(std::move(group),
               Group_impl::_identity_indices_set(group->size()),
               N_translations) {}

  /// \brief Constructor
  ///
  /// \param group The head group
  /// \param indices Indices of elements into `group` that form the subset
  /// \param N_translations Optional size of the translation subgroup T =
  ///     {0, ..., N_translations-1}. Required for the "normal_subgroup" method
  ///     of all_subgroups().
  Subset(std::shared_ptr<GenericGroup const> group, std::set<Index> indices,
         std::optional<Index> N_translations = std::nullopt)
      : m_group(group),
        m_indices(std::move(indices)),
        m_N_translations(N_translations),
        m_is_group(std::nullopt),
        m_is_normal(std::nullopt),
        m_cyclic_generators(std::nullopt),
        m_cyclic_subgroups(std::nullopt),
        m_maximal_cyclic_generators(std::nullopt),
        m_maximal_cyclic_subgroups(std::nullopt) {
    // Validate group is not null
    if (m_group == nullptr) {
      throw std::runtime_error(
          "Error in CASM::group::Subset constructor: group is null.");
    }

    // Validate group is the head group
    if (m_group->head_group_ptr != nullptr) {
      throw std::runtime_error(
          "Error in CASM::group::Subset constructor: "
          "`group` is not the head group.");
    }

    // Validate indices are in valid range
    if (m_indices.size() > 0) {
      Index min_index = *m_indices.begin();
      if (min_index < 0) {
        throw std::runtime_error(
            "Error in CASM::group::Subset constructor: group index out of "
            "range.");
      }

      Index max_index = *m_indices.rbegin();
      if (max_index >= m_group->size()) {
        throw std::runtime_error(
            "Error in CASM::group::Subset constructor: group index out of "
            "range.");
      }
    }
  }

  static Subset from_generators(
      std::shared_ptr<GenericGroup const> const &group,
      std::set<Index> generators) {
    MakeSubgroupFromGenerators x(*group, generators);
    return Subset(group, x.indices);
  }

  /// \brief Access the indices comprising this subset
  std::set<Index> const &indices() const { return m_indices; }

  /// \brief Access the head group this subset belongs to
  std::shared_ptr<GenericGroup const> const &group() const { return m_group; }

  /// \brief Return true if the subset forms a group
  bool is_group() const {
    if (!m_is_group.has_value()) {
      m_is_group = this->_is_group();
    }
    return *m_is_group;
  }

  /// \brief Return true if the subset is an Abelian group
  bool is_abelian_group() const {
    if (!m_is_abelian_group.has_value()) {
      m_is_abelian_group = this->_is_abelian_group();
    }
    return *m_is_abelian_group;
  }

  /// \brief Return true if the subset is normal
  ///
  /// A subset is normal if it is invariant under conjugation by elements of
  /// the head group.
  ///
  /// This returns true if and only if g*n*g^-1 is in the subset for all g
  /// in G and n in N, where G is the head group and N is this subset.
  bool is_normal() const {
    if (!m_is_normal.has_value()) {
      m_is_normal = this->_is_normal();
    }
    return *m_is_normal;
  }

  /// \brief Extend this subset to be the closure under group multiplication
  /// and return the result
  Subset close() const {
    if (!this->is_group()) {
      return this->_naive_close();
    }
    return Subset(*this);
  }

  /// \brief Extend this subset by adding indices from other
  Subset extend(Subset const &other) const {
    std::set<Index> new_indices(m_indices.begin(), m_indices.end());
    new_indices.insert(other.m_indices.begin(), other.m_indices.end());
    return Subset(m_group, new_indices);
  }

  /// \brief Extend and close
  Subset extend_and_close(Subset const &other) const {
    return extend(other).close();
  }

  /// \brief Check if this subset is a proper subset of `other`
  bool is_proper_subset_of(Subset const &other) const {
    if (m_indices.size() >= other.m_indices.size()) {
      return false;
    }

    // Both sets are ordered; use std::includes to check if `other` contains
    // all elements of `this`.
    return std::includes(other.m_indices.begin(), other.m_indices.end(),
                         m_indices.begin(), m_indices.end());
  }

  /// \brief Check if this subset is a proper subset of other or equal
  bool is_subset_of(Subset const &other) const {
    if (m_indices.size() > other.m_indices.size()) {
      return false;
    }

    // Both sets are ordered; use std::includes to check if `this` contains
    // all elements of `other`.
    return std::includes(other.m_indices.begin(), other.m_indices.end(),
                         m_indices.begin(), m_indices.end());
  }

  /// \brief Check if this subgroup is equal to other
  bool operator==(Subset const &other) const {
    if (this->m_group != other.m_group) {
      throw std::runtime_error(
          "Error in CASM::group::Subset operator==: cannot compare subsets of "
          "different groups.");
    }
    return m_indices == other.m_indices;
  }

  /// \brief Check if this subgroup is not equal to other
  bool operator!=(Subset const &other) const {
    if (this->m_group != other.m_group) {
      throw std::runtime_error(
          "Error in CASM::group::Subset operator!=: cannot compare subsets of "
          "different groups.");
    }
    return m_indices != other.m_indices;
  }

  /// \brief Less than comparison using lexicographical ordering of indices
  bool operator<(Subset const &other) const {
    if (this->m_group != other.m_group) {
      throw std::runtime_error(
          "Error in CASM::group::Subset operator<: cannot compare subsets of "
          "different groups.");
    }
    return std::lexicographical_compare(m_indices.begin(), m_indices.end(),
                                        other.m_indices.begin(),
                                        other.m_indices.end());
  }

  std::vector<Subset> const &left_cosets() const;

  std::vector<Subset> const &right_cosets() const;

  /// \brief Construct all cyclic subgroups generated by elements of
  /// this subset
  std::vector<Subset> const &all_cyclic_subgroups() const {
    if (!m_all_cyclic_subgroups.has_value()) {
      this->_make_all_cyclic_subgroups();
    }
    return *m_all_cyclic_subgroups;
  }

  /// \brief Construct the unique cyclic subgroups generated by elements of
  /// this subset
  std::vector<Subset> const &cyclic_subgroups() const {
    std::cout << "-- 1 --" << std::endl;
    if (!m_cyclic_subgroups.has_value()) {
      std::cout << "-- 2 --" << std::endl;
      MakeUniqueCyclicSubgroups x(*m_group, m_indices);

      // Store results
      m_cyclic_subgroups = std::vector<Subset>();
      m_cyclic_generators = std::vector<Index>();

      for (auto const &cycle : x.cyclic_subgroups) {
        m_cyclic_subgroups->emplace_back(m_group, cycle);
      }
      m_cyclic_generators = x.generators;
    }
    std::cout << "-- 3 --" << std::endl;
    return *m_cyclic_subgroups;
  }

  /// \brief Return generators for the unique cyclic subgroups of this subset
  ///
  /// All subset elements are included in the cyclic subgroups generated by
  /// these elements. Specifically, these are the elements that generated
  /// the maximal cyclic subgroups.
  std::vector<Index> const &cyclic_generators() const {
    if (!m_cyclic_generators.has_value()) {
      this->cyclic_subgroups();
    }
    return *m_cyclic_generators;
  }

  /// \brief Construct the maximal cyclic subgroups generated by elements of
  /// this subset
  ///
  /// These are the cyclic subgroups generated by elements of the subset
  /// that are not a subgroup of any other cyclic subgroup of the subset.
  std::vector<Subset> const &maximal_cyclic_subgroups() const {
    if (!m_maximal_cyclic_subgroups.has_value()) {
      MakeMaximalCyclicSubgroups x(*m_group, m_indices);

      // Store results
      m_maximal_cyclic_subgroups = std::vector<Subset>();
      m_maximal_cyclic_generators = std::vector<Index>();

      for (auto const &cycle : x.maximal_cyclic_subgroups) {
        m_maximal_cyclic_subgroups->emplace_back(m_group, cycle);
      }
      m_maximal_cyclic_generators = x.generators;
    }
    return *m_maximal_cyclic_subgroups;
  }

  /// \brief Return generators for the maximal cyclic subgroups of this subset
  ///
  /// All subset elements are included in the cyclic subgroups generated by
  /// these elements. Specifically, these are the elements that generated
  /// the maximal cyclic subgroups.
  std::vector<Index> const &maximal_cyclic_generators() const {
    if (!m_maximal_cyclic_generators.has_value()) {
      this->maximal_cyclic_subgroups();
    }
    return *m_maximal_cyclic_generators;
  }

  /// \brief Return a minimal set of generators for this subset
  ///
  /// Method:
  /// - Find the maximal cyclic subgroups of this subset.
  /// - Iteratively add the generators of those cyclic subgroups that add
  ///   the most new conjugacy classes to the generated set until all
  ///   subset elements are generated.
  ///
  /// Notes:
  /// - This is a small set of generators that generates all elements of the
  ///   subset when closed under multiplication.
  /// - It may not be the minimum set of generators.
  /// - I have not proven if this is always a minimal set of generators
  /// (i.e.,
  ///   removing any generator makes it impossible to generate all elements
  ///   of the subset), but it does generate small sets in practice.
  std::set<Index> const &minimal_generators() const {
    if (!m_minimal_generators.has_value()) {
      MakeMinimalSubsetGenerators x(*m_group, m_indices);
      m_minimal_generators = x.generators;
    }
    return *m_minimal_generators;
  }

  /// \brief Optional size of the translation subgroup T = {0,...,N-1}
  ///
  /// Required for the "normal_subgroup" method of all_subgroups().
  std::optional<Index> const &N_translations() const {
    return m_N_translations;
  }

  /// \brief Check if all subgroups have been computed
  bool has_all_subgroups() const { return m_all_subgroups.has_value(); }

  /// \brief Return all subgroups
  ///
  /// Find all subgroups of this subset.
  ///
  /// Notes:
  /// - This subset should be a group
  ///
  /// \param n_subtrees The number of subtrees to divide the search tree into
  ///     (used by method="depth_first_search" only).
  /// \param method Which algorithm to use:
  ///     - "depth_first_search" (default): multithreaded depth-first search
  ///       over generator combinations. General purpose.
  ///     - "normal_subgroup": exploits the G = T.F extension structure.
  ///       Requires N_translations to be set on the Subset.
  /// \param progress_callback A callback function which takes two Index
  ///     arguments: the number of finished subtrees and the total number of
  ///     subgroups found so far. This is called each time a task is finished
  ///     (used by method="depth_first_search" only).
  ///
  std::vector<Subset> const &all_subgroups(
      Index n_subtrees = 100, std::string method = "depth_first_search",
      std::function<void(Index, Index)> progress_callback = nullptr) const {
    if (!m_all_subgroups.has_value()) {
      if (method == "normal_subgroup") {
        if (!m_N_translations.has_value()) {
          throw std::runtime_error(
              "Subset::all_subgroups: method='normal_subgroup' requires "
              "N_translations to be set on the Subset.");
        }
        MakeAllSubgroupsFromNormalSubgroup maker(m_group, *m_N_translations);
        maker.run();

        m_all_subgroups_generators = std::vector<std::set<Index>>();
        m_all_subgroups = std::vector<Subset>();
        for (auto const &[indices, generators] : maker.subgroups) {
          m_all_subgroups_generators->push_back(generators);
          m_all_subgroups->emplace_back(m_group, indices);
        }
      } else {
        // method == "depth_first_search"
        if (progress_callback == nullptr) {
          progress_callback = DefaultProgressCallback(n_subtrees);
        }
        MakeAllSubgroupsFromGenerators x(m_group, m_indices);
        x.run(n_subtrees, progress_callback);

        m_all_subgroups_generators = std::vector<std::set<Index>>();
        m_all_subgroups = std::vector<Subset>();
        for (auto const &res : x.subgroups) {
          m_all_subgroups_generators->push_back(res.second);
          m_all_subgroups->emplace_back(m_group, res.first);
        }
      }
    }
    return *m_all_subgroups;
  }

  /// \brief Return generators for all subgroups
  ///
  /// Uses a depth-first search for combinations of subset elements to use
  /// as subgroup generators
  ///
  /// Notes:
  /// - This subset should be a group
  ///
  std::vector<std::set<Index>> const &all_subgroups_generators() const {
    if (!m_all_subgroups.has_value()) {
      throw std::runtime_error(
          "Error in Subset::all_subgroups_generators: all_subgroups() must be "
          "called first.");
    }
    return *m_all_subgroups_generators;
  }

  bool is_simple_group() const {
    if (!m_is_simple_group.has_value()) {
      if (!m_all_subgroups.has_value()) {
        throw std::runtime_error(
            "Error in Subset::is_simple_group: all_subgroups() must be called "
            "first.");
      }
      m_is_simple_group = this->_is_simple_group();
    }
    return *m_is_simple_group;
  }

 private:
  bool _is_group() const {
    // Check identity
    if (m_indices.count(0) == 0) {
      return false;
    }

    // Check closure
    for (auto i : m_indices) {
      for (auto j : m_indices) {
        Index product_index = m_group->mult(i, j);
        if (m_indices.count(product_index) == 0) {
          return false;
        }
      }
    }

    // Check inverses
    for (auto i : m_indices) {
      Index inv_index = m_group->inv(i);
      if (m_indices.count(inv_index) == 0) {
        return false;
      }
    }

    return true;
  }

  bool _is_abelian_group() const {
    if (!this->is_group()) {
      return false;
    }
    for (Index i : m_indices) {
      for (Index j : m_indices) {
        if (m_group->mult(i, j) != m_group->mult(j, i)) {
          return false;
        }
      }
    }
    return true;
  }

  /// \brief Check if this is a normal subgroup of the group
  bool _is_normal() const {
    // For each g in G and each n in N, check if g*n*g^-1 is in N
    for (auto g_index : m_group->head_group_index) {
      Index g_inv_index = m_group->inv(g_index);
      for (auto n_index : m_indices) {
        Index conjugate_index =
            m_group->mult(g_index, m_group->mult(n_index, g_inv_index));
        if (m_indices.count(conjugate_index) == 0) {
          return false;
        }
      }
    }
    return true;
  }

  bool _is_simple_group() const {
    if (!this->is_group()) {
      return false;
    }
    for (const auto &subgroup : *m_all_subgroups) {
      if (subgroup.indices().size() == 1) {
        continue;  // Skip the trivial subgroup
      }
      if (subgroup.indices().size() == this->indices().size()) {
        continue;  // Skip the full subset
      }
      if (subgroup.is_normal()) {
        return false;  // Found a non-trivial normal proper subgroup
      }
    }
    return true;
  }

  /// \brief Modify this subset to be the closure under group
  /// multiplication, checking every combination of elements until no new
  /// elements are added
  Subset _naive_close() const {
    std::set<Index> new_indices(m_indices.begin(), m_indices.end());
    std::vector<Index> velements(m_indices.begin(), m_indices.end());
    for (Index i = 0; i < velements.size(); i++) {
      for (Index j = 0; j < velements.size(); j++) {
        Index product_index = m_group->mult(velements[i], velements[j]);
        if (new_indices.insert(product_index).second) {
          velements.push_back(product_index);
        }
      }
    }
    return Subset(m_group, new_indices);
  }

  /// \brief Use the generators of this and `other` to extend this subset
  /// and close it under multiplication
  ///
  /// This approach does not use `_naive_close`. The method is:
  /// - Create the union of the elements in this and other.
  /// - Create the union of the the minimal generators for this and other.
  /// - Use a queue to iteratively multiply generators and the newly added
  ///   elements until no new elements are added.
  void _extend_and_close(Subset const &other);

  void _make_all_cyclic_subgroups() const {
    m_all_cyclic_subgroups = std::vector<Subset>();

    for (Index i : m_indices) {
      std::set<Index> indices;
      Index prod = i;
      while (indices.insert(prod).second) {
        prod = m_group->mult(i, prod);
      }
      m_all_cyclic_subgroups->emplace_back(m_group, indices);
    }
  }

  std::shared_ptr<GenericGroup const> const m_group;

  std::set<Index> const m_indices;

  /// \brief Size of the translation subgroup, if this group has the structure
  ///     G = T.F with T = {0, ..., N_translations-1}. Optional.
  std::optional<Index> m_N_translations;

  /// \brief Stores whether the subset is a group, if known
  mutable std::optional<bool> m_is_group;

  /// \brief Stores whether the subset is an abelian group, if known
  mutable std::optional<bool> m_is_abelian_group;

  /// \brief Stores whether the subset is normal (invariant to conjugation),
  /// if known
  mutable std::optional<bool> m_is_normal;

  /// \brief Stores whether the subset is a simple group, if known
  mutable std::optional<bool> m_is_simple_group;

  /// \brief A vector of all cyclic subgroups generated by elements of
  /// this subset, if known
  mutable std::optional<std::vector<Subset>> m_all_cyclic_subgroups;

  /// \brief A vector of generators for the unique cyclic subgroups.
  ///
  /// - m_cyclic_generators[i] generates m_cyclic_subgroups[i]
  mutable std::optional<std::vector<Index>> m_cyclic_generators;

  /// \brief The unique cyclic subgroups generated by the generators
  ///
  /// - Does include cyclic subgroups that are a subgroup of another
  ///   cyclic subgroup (i.e. the cyclic subgroup of a 4-fold rotation is
  ///   included, and the cyclic subgroup of the associated 2-fold rotation
  ///   is included separately).
  mutable std::optional<std::vector<Subset>> m_cyclic_subgroups;

  /// \brief A vector of generators for the maximal cyclic subgroups, if
  /// known.
  ///
  /// - All subgroup elements are included in the cyclic subgroups generated
  /// by these generators
  /// - m_maximal_cyclic_generators[i] generates
  /// m_maximal_cyclic_subgroups[i]
  mutable std::optional<std::vector<Index>> m_maximal_cyclic_generators;

  /// \brief The maximal cyclic subgroups generated by the generators
  ///
  /// - Does not include cyclic subgroups that are a subgroup of another
  ///   cyclic subgroup (i.e. the cyclic subgroup of a 4-fold rotation is
  ///   included, but the cyclic subgroup of the associated 2-fold rotation
  ///   is not included separately).
  mutable std::optional<std::vector<Subset>> m_maximal_cyclic_subgroups;

  /// \brief A minimal set of generators
  ///
  /// - This may not be the minimum set of generators
  mutable std::optional<std::set<Index>> m_minimal_generators;

  /// \brief Generators for `m_all_subgroups`
  mutable std::optional<std::vector<std::set<Index>>>
      m_all_subgroups_generators;

  /// \brief A vector of all subgroups of this subset
  mutable std::optional<std::vector<Subset>> m_all_subgroups;

  /// \brief The left cosets of this subset
  mutable std::optional<std::vector<Subset>> m_left_cosets;

  /// \brief The right cosets of this subset
  mutable std::optional<std::vector<Subset>> m_right_cosets;
};

/// \brief Given a vector of subgroups, return their orbits
SubgroupOrbitVec to_vector_of_orbit_vec(std::vector<Subset> const &subgroups);

/// \brief Convert subgroup orbits from vectors to sets
GroupIndicesOrbitSet to_set_of_orbits_sets(
    SubgroupOrbitVec const &vector_of_orbit_vec);

}  // namespace group
}  // namespace CASM
// --- Implementation ---

#include <numeric>

namespace CASM {
namespace group {

namespace subgroups_impl {

typedef std::set<Index> CosetIndices;

/// \brief Helper function to make left or right cosets
///
/// \param group The group to find cosets of
/// \param subgroup_indices The indices of the subgroup to find cosets of
/// \param prod A function that takes a group element index and a subgroup
///     element index and returns the product index. The signature of `prod` is
///     `Index prod(Index group_element_index, Index subgroup_element_index)`.
///
template <typename ProductFunction>
inline std::set<CosetIndices> _make_cosets(
    GenericGroup const &group, SubgroupIndices const &subgroup_indices,
    ProductFunction prod) {
  std::set<CosetIndices> left_cosets;

  // each group element is only included in one coset
  std::vector<bool> check(group.size(), false);
  Index product_index;
  Index group_element_index = 0;
  while (group_element_index < group.size()) {
    if (check[group_element_index]) {
      ++group_element_index;
      continue;
    }
    CosetIndices left_coset;
    for (auto subgroup_element_index : subgroup_indices) {
      product_index = prod(group_element_index, subgroup_element_index);
      left_coset.insert(product_index);
      check[product_index] = true;
    }
    left_cosets.insert(left_coset);
    ++group_element_index;
  }
  return left_cosets;
}

/// \brief Return the unique left cosets of a subgroup
///
/// - If subgroup B of group G contains elements: (E, B1, B2, …, Bg),
/// the "left coset” of X is (X*E, X*B1, X*B2, …, X*Bg),
/// where X is an element of G.
/// - A coset need not be a subgroup.
/// - If X is an element of B, then the coset will be a subgroup of B.
/// - Two left cosets of a given subgroup either contain exactly the same
/// elements, or have no elements in common.
inline std::set<CosetIndices> _make_left_cosets(
    GenericGroup const &group, SubgroupIndices const &subgroup_indices) {
  return _make_cosets(
      group, subgroup_indices,
      [&](Index group_element_index, Index subgroup_element_index) {
        return group.mult(group_element_index, subgroup_element_index);
      });
}

/// \brief Return the unique right cosets of a subgroup
///
/// - If subgroup B of group G contains elements: (E, B1, B2, …, Bg),
/// the "right coset” of X is (E*X, B1*X, B2*X, …, Bg*X),
/// where X is an element of G.
/// - A coset need not be a subgroup.
/// - If X is an element of B, then the coset will be a subgroup of B.
/// - Two right cosets of a given subgroup either contain exactly the same
/// elements, or have no elements in common.
inline std::set<CosetIndices> _make_right_cosets(
    GenericGroup const &group, SubgroupIndices const &subgroup_indices) {
  return _make_cosets(
      group, subgroup_indices,
      [&](Index group_element_index, Index subgroup_element_index) {
        return group.mult(subgroup_element_index, group_element_index);
      });
}

inline SubgroupOrbit _make_subgroup_orbit(GenericGroup const &group,
                                          SubgroupIndices const &subgroup) {
  SubgroupOrbit orbit;
  std::set<subgroups_impl::CosetIndices> left_cosets =
      subgroups_impl::_make_left_cosets(group, subgroup);
  for (auto const &coset : left_cosets) {
    Index X_index = *coset.begin();
    Index X_inv_index = group.inv(X_index);
    SubgroupIndices equiv_subgroup;
    for (auto const &A_index : subgroup) {
      equiv_subgroup.insert(
          group.mult(X_index, group.mult(A_index, X_inv_index)));
    }
    orbit.insert(equiv_subgroup);
  }
  return orbit;
}

inline std::function<bool(SubgroupIndices const &)> _make_subgroup_count(
    std::set<SubgroupOrbit> const &subgroups) {
  return [&](SubgroupIndices const &subgroup) {
    for (auto const &orbit : subgroups) {
      if (orbit.count(subgroup)) {
        return true;
      }
    }
    return false;
  };
}

inline std::function<void(SubgroupIndices &)> _make_close_subgroup(
    GenericGroup const &group) {
  return [&](SubgroupIndices &subgroup) {
    std::vector<Index> vgroup(subgroup.begin(), subgroup.end());
    for (Index i = 0; i < vgroup.size(); i++) {
      for (Index j = 0; j < vgroup.size(); j++) {
        Index product_index = group.mult(vgroup[i], vgroup[j]);
        if (subgroup.insert(product_index).second) {
          vgroup.push_back(product_index);
        }
      }
    }
  };
}

}  // namespace subgroups_impl

/// \brief Return all cyclic subgroups
///
/// - A cyclic subgroup, A, is the subgroup generated by repeated
/// multiplication of a single element, i.e. A={a, a^2, a^3, ..., a^k}, where
/// a^k=E, the identity element
/// - An equivalent subgroup is {X*a*X^-1, X*(a^2)*X^-1, X*(a^3)*X^-1, ...,
/// E}, where X is an element in a coset of A
/// - The orbit of a cyclic subgroup is all distinct equivalent subgroups
///
/// \param group The group to find cyclic subgroups of
/// \returns A set of orbits of cyclic subgroups
///
inline std::set<SubgroupOrbit> make_cyclic_subgroups(
    GenericGroup const &group) {
  using namespace subgroups_impl;

  std::set<SubgroupOrbit> cyclic_subgroups;
  Index group_element_index = 0;
  while (group_element_index < group.size()) {
    // Make cyclic subgroup of element `group_element_index`
    SubgroupIndices cyclic_subgroup;
    cyclic_subgroup.insert(group_element_index);
    Index product_index = group_element_index;
    while (product_index != 0) {
      product_index = group.mult(group_element_index, product_index);
      cyclic_subgroup.insert(product_index);
    }

    // Make orbit of subgroups equivalent to `cyclic_subgroup` && Insert orbit
    cyclic_subgroups.insert(_make_subgroup_orbit(group, cyclic_subgroup));

    ++group_element_index;
  }
  return cyclic_subgroups;
}

/// \brief Return all subgroups
///
/// Method:
/// - Start with m_subgroups = m_small_subgroups, then add new subgroups by
/// finding the closure of a union of a large_group and a small_group.
/// - If the the new large_group is unique, add it as a large_group.
/// - Repeat for all (large_group, small_group) pairs, until no new
/// m_subgroups are found.
///
/// Note:
/// - This is probably not the fastest algorithm, but it is complete
///
/// \param group The group to find subgroups of
/// \returns A set of orbits of all subgroups
///
inline std::set<SubgroupOrbit> make_all_subgroups(GenericGroup const &group) {
  using namespace subgroups_impl;
  std::set<SubgroupOrbit> small_subgroups = make_cyclic_subgroups(group);
  std::set<SubgroupOrbit> all_subgroups = small_subgroups;

  // functor to close incomplete subgroup
  auto _close_subgroup = _make_close_subgroup(group);

  // functor to find if any orbit contains a particular subgroup
  auto _subgroup_count = _make_subgroup_count(all_subgroups);

  auto all_subgroups_it = all_subgroups.begin();
  while (all_subgroups_it != all_subgroups.end()) {
    for (auto const &small_subgroups_orbit : small_subgroups) {
      for (auto const &small_subgroups_equiv : small_subgroups_orbit) {
        // Combine an existing subgroup and a small (cyclic) subgroup
        SubgroupIndices subgroup = *(all_subgroups_it->begin());
        Index init_size = subgroup.size();
        subgroup.insert(small_subgroups_equiv.begin(),
                        small_subgroups_equiv.end());
        if (subgroup.size() == init_size) continue;

        // Find group closure
        _close_subgroup(subgroup);

        // If subgroup already exists in all_subgroups, continue
        if (_subgroup_count(subgroup)) continue;

        // Else, make orbit and insert
        all_subgroups.insert(_make_subgroup_orbit(group, subgroup));
      }
    }
    ++all_subgroups_it;
  }
  return all_subgroups;
}

/// A functor which makes all cyclic subgroups at construction and then
/// returns them when called:
class MakeCyclicSubgroups {
 public:
  MakeCyclicSubgroups(std::shared_ptr<GenericGroup const> group)
      : m_group(group),
        m_subgroups_constructed(false),
        m_cyclic_subgroups(std::make_shared<std::set<SubgroupOrbit>>()) {}

  std::set<SubgroupOrbit> operator()() const {
    if (!m_subgroups_constructed) {
      *m_cyclic_subgroups = make_cyclic_subgroups(*m_group);
      m_subgroups_constructed = true;
    }
    return *m_cyclic_subgroups;
  }

 private:
  std::shared_ptr<GenericGroup const> m_group;

  mutable bool m_subgroups_constructed;

  mutable std::shared_ptr<std::set<SubgroupOrbit>> m_cyclic_subgroups;
};

//// A functor which makes all subgroups at construction and then
/// returns them when called:
class MakeAllSubgroups {
 public:
  MakeAllSubgroups(std::shared_ptr<GenericGroup const> group)
      : m_group(group),
        m_subgroups_constructed(std::make_shared<bool>(false)),
        m_all_subgroups(std::make_shared<std::set<SubgroupOrbit>>()) {}

  std::set<SubgroupOrbit> operator()() const {
    if (*m_subgroups_constructed == false) {
      std::set<Index> indices;
      for (Index i = 0; i < m_group->size(); ++i) {
        indices.insert(i);
      }

      MakeAllSubgroupsFromGenerators x(m_group, indices);
      x.run(100, [](Index, Index) {});

      for (auto const &pair : x.subgroups) {
        m_all_subgroups->insert(
            subgroups_impl::_make_subgroup_orbit(*x.group, pair.first));
      }

      *m_subgroups_constructed = true;
    }
    return *m_all_subgroups;
  }

 private:
  std::shared_ptr<GenericGroup const> m_group;

  mutable std::shared_ptr<bool> m_subgroups_constructed;

  mutable std::shared_ptr<std::set<SubgroupOrbit>> m_all_subgroups;
};

/// \brief Make the invariant subgroup for each orbit element, as
///     indices of group elements
///
/// \param equivalence_map The indices equivalence_map[i] are the
///     indices of the group elements which map orbit element 0 onto
///     orbit element i
/// \param group The group used to generate the orbit
///
/// \returns invariant_subgroups, The indices invariant_subgroups[i] are
///     the indices of the group elements which leave orbit element i
///     invariant.
inline std::vector<SubgroupIndices> make_invariant_subgroups(
    std::vector<std::vector<Index>> const &equivalence_map,
    GenericGroup const &group) {
  /// The first row of equivalence_map is the invariant subgroup
  /// of the first element in the orbit
  ///
  /// Invariant subgroups of subsequent orbit elements are constructed
  /// using:
  ///   inv_subgrp(i) = eq_map(i,0) * eq_map(0,j) * inverse(eq_map(i,0)), for
  ///   all j
  ///
  /// inverse(eq_map(i,0)): transforms orbit element i back to orbit element 0
  /// eq_map(0,j): invariant transformation of orbit element 0
  /// eq_map(i,0): transforms orbit element 0 to orbit element i
  ///
  std::vector<SubgroupIndices> invariant_subgroups;
  if (!equivalence_map.size()) {
    return invariant_subgroups;
  }

  // first row of equivalence map is an invariant subgroup
  {
    SubgroupIndices subgroup;
    for (Index e_0j : equivalence_map[0]) {
      subgroup.insert(e_0j);
    }
    invariant_subgroups.emplace_back(std::move(subgroup));
  }

  // first column can be used to generate others
  for (Index i = 1; i < equivalence_map.size(); ++i) {
    if (!equivalence_map[i].size()) {
      throw std::runtime_error(
          "Error in make_invariant_subgroups: failed due to empty row in "
          "equivalence_map");
    }
    SubgroupIndices subgroup;
    Index e_i0 = equivalence_map[i][0];
    for (Index e_0j : equivalence_map[0]) {
      subgroup.insert(group.mult(e_i0, group.mult(e_0j, group.inv(e_i0))));
    }
    invariant_subgroups.emplace_back(std::move(subgroup));
  }
  return invariant_subgroups;
}

/// \brief Run the multi-threaded subgroup finding algorithm
///
/// \param n_subtrees The number of subtrees to divide the search tree into.
/// \param progress_callback A callback function which takes two Index
///     arguments: the number of finished subtrees and the total number of
///     subgroups found so far. This is called each time a task is finished.
inline void MakeAllSubgroupsFromGenerators::run(
    Index n_subtrees, std::function<void(Index, Index)> progress_callback) {
  // Perform a threaded depth first search for combinations of elements to
  // use as subgroup generators

  // Example:
  // - brackets indicate the set of generators for a subgroup
  // - don't need to check {identity}
  //
  // {1} -> {1,2} -> {1,2,3}  (stop if full group generated)
  //              -> {1,2,4} -> {1,2,4,5} (stop...)
  //              -> {1,2,5} (stop...)
  //     -> {1,3} (stop...)
  //     -> {1,4} (stop...)
  //     -> {1,5} (stop...)
  //     -> {1,6} (stop...)
  // {2} -> {2,3} (stop...)
  // {3} -> {3,4} -> ...
  //

  if (progress_callback == nullptr) {
    progress_callback = DefaultProgressCallback(n_subtrees);
  }

  // -- Find cyclic subgroup generators --
  std::set<std::set<Index>> all_cyclic_subgroups;
  for (Index i : indices) {
    if (i == 0) {
      // Skip identity element
      continue;
    }
    std::set<Index> subgroup;
    Index prod = i;
    while (subgroup.insert(prod).second) {
      prod = group->mult(i, prod);
    }
    auto res = all_cyclic_subgroups.emplace(subgroup);
    if (res.second) {
      // New cyclic subgroup found, add generator
      candidate_generators.insert(i);
    }
  }

  // If no candidate generators, the only subgroup is the trivial one.
  if (candidate_generators.empty()) {
    subgroups.emplace(std::set<Index>{0}, std::set<Index>{0});
    progress_callback(n_subtrees, subgroups_size());
    return;
  }

  std::set<Task> unfinished;

  std::vector<member_iterator> iters;
  std::vector<member_iterator> end_iters;

  Index subtree = 0;
  Index max_k = 10;

  // To prepare for the first task:
  end_iters = subtree_position(max_k, subtree, n_subtrees);

  // Track unfinished tasks and last task assigned:
  // std::set<Task> assigned_tasks;
  // Task last_task;
  Index n_finished_tasks = 0;

  auto producer = [&]() -> std::optional<Task> {
    if (end_iters.size() == 0) {
      return std::nullopt;
    }
    iters = end_iters;  // from previous task
    end_iters = subtree_position(max_k, subtree + 1, n_subtrees);

    ++subtree;
    return Task{iters, end_iters};
  };

  // Worker functions run in parallel threads at the same time as the
  // producer and merger threads run in the controller thread. They must
  // synchronize access to shared state (especially `subgroups`). They
  // should not read/write assigned_tasks.
  auto worker = [&](Task task, Index worker_id) -> Result {
    Index failed_insertions = 0;
    Index successful_insertions = 0;

    if (task.begin.size() == 0) {
      throw std::runtime_error(
          "Error in MakeAllSubgroupsFromGenerators worker: cannot "
          "start task from empty iters (signals end)");
    }

    // Local nodes (subgroups) for this worker
    std::vector<node_type> local_nodes;

    // Current position in the tree for this worker
    std::vector<member_iterator> local_iters = task.begin;

    initialize_nodes(local_iters, local_nodes);

    Index stop_check_count = 0;
    while (true) {
      // Check for termination:
      if (!less_than(local_iters, task.end)) {
        return Result{task};
      }
      if (stop_check_count == 1000) {
        if (stop_requested()) {
          return Result{task};
        }
        stop_check_count = 0;
      }
      stop_check_count++;

      make_next_node(local_iters, local_nodes);

      bool successful_insert = insert_local_node(local_iters, local_nodes);

      if (successful_insert) {
        successful_insertions++;
      } else {
        failed_insertions++;
      }
      advance_if_invalid(local_iters, local_nodes);
    }
  };

  // In threaded_pipeline, producer and merger run in the same controller
  // thread, so they can both act on assigned_tasks without synchronization.
  auto merger = [&](Result const &result) {
    n_finished_tasks += 1;
    progress_callback(n_finished_tasks, subgroups_size());
    // assigned_tasks.erase(result.task);
  };

  // // Serial version for debugging:
  // while (true) {
  //   std::optional<Task> task = producer();
  //   if (!task.has_value()) {
  //     break;
  //   }
  //   auto result = worker(*task, 0);
  //   merger(result);
  // }

  std::optional<Index> task_queue_max_size = max_threads() + 10;
  std::optional<Index> result_queue_max_size = std::nullopt;
  progress_callback(0, 0);
  threaded_pipeline(producer, worker, merger, task_queue_max_size,
                    result_queue_max_size);
  progress_callback(n_subtrees, subgroups_size());

  if (stop_requested()) {
    throw std::runtime_error(
        "MakeAllSubgroupsFromGenerators: Did not complete. Stop "
        "requested.");
  }

  // Finally, always add the trivial subgroup
  subgroups.emplace(std::set<Index>{0}, std::set<Index>{0});
}

/// \brief Brute-force enumeration of all sections ξ: K → T/S satisfying the
///     group extension cocycle equation
///
/// A section σ_ξ: K → G is defined by choosing, for each k ∈ K, an element
/// of G in coset k with translation part ξ(k) ∈ T/S:
///
///   σ_ξ(k) has G-index  k * N_T + rep_TS[ξ(k)]
///
/// For σ_ξ to define a subgroup H = {σ_ξ(k) * s : k ∈ K, s ∈ S}, ξ must
/// satisfy the cocycle equation for all k1, k2 ∈ K (in T/S, written
/// additively):
///
///   ξ(k1 * k2) = ξ(k1) + act(k1, ξ(k2)) + c_K(k1, k2)
///
/// where:
/// - act(k, ts) = act_on_TS[k][ts] is the K-action on T/S
/// - c_K(k1, k2) = c_K_TS[k1][k2] is the 2-cocycle (always 0 for symmorphic)
///
/// Algorithm: enumerate all ξ on generators of K (BFS from identity),
/// extend to all K via the recurrence, check all |K|² equations.
///
/// Complexity: O(|T/S|^r × |K|²) where r = number of generators of K.
///
/// \param K_indices Elements of K (as indices into F_group)
/// \param K_gen Generators of K (as indices into F_group)
/// \param F_group The quotient group F = G/T
/// \param TS Quotient group data for T/S
/// \param act_on_TS act_on_TS[f][ts]: action of F-element f on T/S-element ts
/// \param c_K_TS c_K_TS[f1][f2]: 2-cocycle φ(f1,f2) as a T/S index
///
/// \returns Vector of valid sections; each xi[k] gives the T/S index for k ∈ K
///     (size = F_group.size(); entries for k ∉ K are meaningless).
inline std::vector<std::vector<Index>> find_all_sections_brute_force(
    std::set<Index> const &K_indices, std::set<Index> const &K_gen,
    GenericGroup const &F_group, QuotientGroupData const &TS,
    std::vector<std::vector<Index>> const &act_on_TS,
    std::vector<std::vector<Index>> const &c_K_TS) {
  Index N_TS = TS.group->size();
  std::vector<Index> gen_vec(K_gen.begin(), K_gen.end());
  Index r = static_cast<Index>(gen_vec.size());

  // BFS ordering of K elements (vector used as queue).
  // For i >= 1: bfs_order[i] = bfs_order[bfs_parent_node[i-1]] *
  // gen[bfs_parent_gen[i-1]]
  std::vector<Index> bfs_order;
  std::vector<Index> bfs_parent_node;
  std::vector<Index> bfs_parent_gen;
  std::vector<bool> visited(F_group.size(), false);
  bfs_order.push_back(0);  // identity
  visited[0] = true;
  for (Index qi = 0; qi < static_cast<Index>(bfs_order.size()); ++qi) {
    Index prev = bfs_order[qi];
    for (Index gi = 0; gi < r; ++gi) {
      Index k = F_group.mult(prev, gen_vec[gi]);
      if (!visited[k]) {
        visited[k] = true;
        bfs_order.push_back(k);
        bfs_parent_node.push_back(prev);
        bfs_parent_gen.push_back(gi);
      }
    }
  }
  Index K_size = static_cast<Index>(bfs_order.size());

  std::vector<std::vector<Index>> results;
  std::vector<Index> xi_on_gens(r, 0);  // ξ values on generators, odometer

  while (true) {
    // Extend ξ to all K via the BFS recurrence:
    //   ξ(prev * g) = ξ(prev) +_{T/S} act(prev, ξ(g)) +_{T/S} c_K(prev, g)
    std::vector<Index> xi(F_group.size(), 0);
    for (Index i = 1; i < K_size; ++i) {
      Index k = bfs_order[i];
      Index prev = bfs_parent_node[i - 1];
      Index gi = bfs_parent_gen[i - 1];
      xi[k] = TS.group->mult(
          TS.group->mult(xi[prev], act_on_TS[prev][xi_on_gens[gi]]),
          c_K_TS[prev][gen_vec[gi]]);
    }

    // Check all |K|² cocycle equations
    bool valid = true;
    for (Index k1 : K_indices) {
      for (Index k2 : K_indices) {
        Index rhs = TS.group->mult(
            TS.group->mult(xi[k1], act_on_TS[k1][xi[k2]]), c_K_TS[k1][k2]);
        if (xi[F_group.mult(k1, k2)] != rhs) {
          valid = false;
          break;
        }
      }
      if (!valid) break;
    }
    if (valid) results.push_back(xi);

    // Advance odometer
    bool done = true;
    for (Index i = r - 1; i >= 0; --i) {
      if (++xi_on_gens[i] < N_TS) {
        done = false;
        break;
      }
      xi_on_gens[i] = 0;
    }
    if (done) break;
  }

  return results;
}

/// \brief Run the subgroup finding algorithm using the normal translation
///     subgroup structure
inline void MakeAllSubgroupsFromNormalSubgroup::run(Index n_subtrees_F,
                                                    Index n_subtrees_T) {
  Index N_G = group->size();
  Index N_T = N_translations;
  Index N_F = N_G / N_T;

  // --- Build T as a standalone head group ---
  // T's multiplication table is the restriction of G's to indices {0..N_T-1},
  // which is valid because T is closed under multiplication.
  MultiplicationTable T_mult(N_T, std::vector<Index>(N_T));
  for (Index i = 0; i < N_T; ++i)
    for (Index j = 0; j < N_T; ++j) T_mult[i][j] = group->mult(i, j);
  auto T_group = std::make_shared<GenericGroup>(T_mult);

  // --- Find all subgroups of T ---
  MakeAllSubgroupsFromGenerators T_finder(T_group);
  T_finder.run(n_subtrees_T, [](Index, Index) {});

  // --- Build F = G/T as a standalone head group ---
  // The quotient multiplication: f1 *_F f2 = coset_of(rep(f1) *_G rep(f2)),
  // where rep(f) = f * N_T and coset_of(g) = g / N_T.
  // This is well-defined because T is normal in G.
  MultiplicationTable F_mult(N_F, std::vector<Index>(N_F));
  for (Index f1 = 0; f1 < N_F; ++f1)
    for (Index f2 = 0; f2 < N_F; ++f2)
      F_mult[f1][f2] = group->mult(f1 * N_T, f2 * N_T) / N_T;
  auto F_group = std::make_shared<GenericGroup>(F_mult);

  // --- Find all subgroups of F ---
  MakeAllSubgroupsFromGenerators F_finder(F_group);
  F_finder.run(n_subtrees_F, [](Index, Index) {});

  // --- Precompute conjugation action of F on T ---
  // conj_action[f][t] = index in T of rep(f) * t * rep(f)^{-1}
  // where rep(f) = f * N_T is the coset representative of f in G.
  // The result is guaranteed to be in T since T is normal in G.
  std::vector<std::vector<Index>> conj_action(N_F, std::vector<Index>(N_T));
  for (Index f = 0; f < N_F; ++f) {
    Index rep_f = f * N_T;
    Index inv_rep_f = group->inv(rep_f);
    for (Index t = 0; t < N_T; ++t)
      conj_action[f][t] = group->mult(rep_f, group->mult(t, inv_rep_f));
  }

  // --- Enumerate (K, S) pairs and find all subgroups of G ---
  //
  // For each subgroup K of F and K-invariant subgroup S of T, we enumerate
  // all valid sections ξ: K → T/S via find_all_sections_brute_force.
  // Each valid section defines a distinct subgroup H_ξ of G with H_ξ ∩ T = S
  // and H_ξ / S ≅ K, generated by {k*N_T + rep_TS[ξ(k)] : k ∈ K_gen} ∪ S_gen.
  //
  // act_on_TS and c_K_TS depend on T/S (i.e., on S), so they are recomputed
  // for each S in the outer loop.
  for (auto const &[S_indices, S_gen] : T_finder.subgroups) {
    // Compute T/S quotient group and coset map
    QuotientGroupData TS = make_quotient_group_data(*T_group, S_indices);
    Index N_TS = TS.group->size();

    // act_on_TS[f][ts] = T/S coset of conj_action[f][rep_TS[ts]]
    std::vector<std::vector<Index>> act_on_TS(N_F, std::vector<Index>(N_TS));
    for (Index f = 0; f < N_F; ++f)
      for (Index ts = 0; ts < N_TS; ++ts)
        act_on_TS[f][ts] = TS.coset_of[conj_action[f][TS.rep[ts]]];

    // c_K_TS[f1][f2] = T/S coset of the 2-cocycle φ(f1,f2) = rep(f1)*rep(f2)
    // mod T
    // (= 0 for all f1,f2 in the symmorphic case)
    std::vector<std::vector<Index>> c_K_TS(N_F, std::vector<Index>(N_F));
    for (Index f1 = 0; f1 < N_F; ++f1)
      for (Index f2 = 0; f2 < N_F; ++f2)
        c_K_TS[f1][f2] = TS.coset_of[group->mult(f1 * N_T, f2 * N_T) % N_T];

    for (auto const &[K_indices, K_gen] : F_finder.subgroups) {
      // Check K-invariance of S: for all k in K_generators and s in S,
      // conj(k, s) must be in S. Checking generators of K suffices since
      // if generators normalize S, the group they generate does too.
      bool k_invariant = true;
      for (Index f : K_gen) {
        for (Index t : S_indices) {
          if (!S_indices.count(conj_action[f][t])) {
            k_invariant = false;
            break;
          }
        }
        if (!k_invariant) break;
      }
      if (!k_invariant) continue;

      // Enumerate all valid sections ξ: K → T/S and build a subgroup for each
      auto sections = find_all_sections_brute_force(K_indices, K_gen, *F_group,
                                                    TS, act_on_TS, c_K_TS);
      for (auto const &xi : sections) {
        // σ_ξ(k) has G-index k*N_T + rep_TS[ξ(k)]; use K generators as seeds
        std::set<Index> H_generators;
        for (Index f : K_gen) H_generators.insert(f * N_T + TS.rep[xi[f]]);
        for (Index t : S_gen) H_generators.insert(t);

        MakeSubgroupFromGenerators H_maker(*group, H_generators);
        subgroups.emplace(H_maker.indices, H_maker.generators);
      }
    }
  }

  // Always add the trivial subgroup
  subgroups.emplace(std::set<Index>{0}, std::set<Index>{0});
}

/// \brief Given a vector of subgroups, return their orbits
inline SubgroupOrbitVec to_vector_of_orbit_vec(
    std::vector<Subset> const &subgroups) {
  GroupIndicesOrbitSet set_of_orbit_sets;
  for (auto const &subset : subgroups) {
    set_of_orbit_sets.insert(subgroups_impl::_make_subgroup_orbit(
        *subset.group(), subset.indices()));
  }
  SubgroupOrbitVec vector_of_orbit_vec;
  for (auto const &orbit_set : set_of_orbit_sets) {
    std::vector<std::vector<Index>> orbit_vec;
    for (auto const &subgroup : orbit_set) {
      std::vector<Index> subgroup_vec(subgroup.begin(), subgroup.end());
      orbit_vec.push_back(subgroup_vec);
    }
    vector_of_orbit_vec.push_back(orbit_vec);
  }
  return vector_of_orbit_vec;
}

/// \brief Convert subgroup orbits from vectors to sets
inline GroupIndicesOrbitSet to_set_of_orbits_sets(
    SubgroupOrbitVec const &vector_of_orbit_vec) {
  GroupIndicesOrbitSet set_of_orbit_sets;
  for (auto const &orbit_vec : vector_of_orbit_vec) {
    std::set<std::set<Index>> orbit_set;
    for (auto const &subgroup : orbit_vec) {
      std::set<Index> subgroup_set(subgroup.begin(), subgroup.end());
      orbit_set.insert(subgroup_set);
    }
    set_of_orbit_sets.insert(orbit_set);
  }
  return set_of_orbit_sets;
}

inline std::vector<Subset> const &Subset::left_cosets() const {
  if (!m_left_cosets.has_value()) {
    if (!this->is_group()) {
      throw std::runtime_error(
          "Error in CASM::group::Subset left_cosets: `subgroup` is not a "
          "group.");
    }
    std::set<std::set<Index>> cosets =
        subgroups_impl::_make_left_cosets(*m_group, m_indices);

    m_left_cosets = std::vector<Subset>();
    for (auto const &coset : cosets) {
      m_left_cosets->emplace_back(m_group, coset);
    }
  }
  return *m_left_cosets;
}

inline std::vector<Subset> const &Subset::right_cosets() const {
  if (!m_left_cosets.has_value()) {
    if (!this->is_group()) {
      throw std::runtime_error(
          "Error in CASM::group::Subset left_cosets: `subgroup` is not a "
          "group.");
    }
    std::set<std::set<Index>> cosets =
        subgroups_impl::_make_right_cosets(*m_group, m_indices);

    m_right_cosets = std::vector<Subset>();
    for (auto const &coset : cosets) {
      m_right_cosets->emplace_back(m_group, coset);
    }
  }
  return *m_right_cosets;
}

}  // namespace group
}  // namespace CASM

#endif
