#include "casm/configuration/Prim.hh"
#include "casm/configuration/PrimSymInfo.hh"
#include "casm/configuration/Supercell.hh"
#include "casm/configuration/SupercellSymOp.hh"
#include "casm/configuration/group/subgroups.hh"
#include "gtest/gtest.h"
#include "teststructures.hh"

// debug
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"

using namespace CASM;

namespace {

// Build a GenericGroup from a multiplication table given as a lambda
std::shared_ptr<group::GenericGroup> make_generic_group(
    Index n, std::function<Index(Index, Index)> mult_f) {
  group::MultiplicationTable table(n, std::vector<Index>(n));
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) table[i][j] = mult_f(i, j);
  return std::make_shared<group::GenericGroup>(table);
}

}  // namespace

namespace cyclic_subgroups_test {

Index multiply_f(Index i, Index j) { return (i + j) % 10; }

bool equal_to_f(Index i, Index j) { return i == j; }

std::function<bool(group::SubgroupIndices const &)> make_any_count(
    std::set<group::SubgroupOrbit> const &cyclic_subgroups) {
  return [&](group::SubgroupIndices const &subgroup) {
    for (auto const &orbit : cyclic_subgroups) {
      if (orbit.count(subgroup)) {
        return true;
      }
    }
    return false;
  };
}

void print_subgroups(std::set<group::SubgroupOrbit> const &subgroups) {
  jsonParser json;
  to_json(subgroups, json);
  std::cout << "--- subgroups ---" << std::endl;
  std::cout << json << std::endl << std::endl;

  for (auto const &orbit : subgroups) {
    std::cout << "EXPECT_EQ(any_count({";
    for (auto i : *orbit.begin()) {
      std::cout << i << ", ";
    }
    std::cout << "}), 1);" << std::endl;
  }
}

}  // namespace cyclic_subgroups_test

TEST(CyclicSubgroupsTest, Test1) {
  using namespace group;
  using namespace cyclic_subgroups_test;
  std::vector<Index> elements;
  for (Index i = 0; i < 10; ++i) {
    elements.push_back(i);
  }
  Group<Index> group = make_group(elements, multiply_f, equal_to_f);
  std::set<SubgroupOrbit> cyclic_subgroups = make_cyclic_subgroups(group);

  EXPECT_EQ(cyclic_subgroups.size(), 4);
  auto any_count = make_any_count(cyclic_subgroups);
  // print_subgroups(cyclic_subgroups);
  EXPECT_EQ(any_count({0}), 1);
  EXPECT_EQ(any_count({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), 1);
  EXPECT_EQ(any_count({0, 2, 4, 6, 8}), 1);
  EXPECT_EQ(any_count({0, 5}), 1);
}

TEST(CyclicSubgroupsTest, Test2) {
  using namespace group;
  using namespace cyclic_subgroups_test;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  std::set<SubgroupOrbit> cyclic_subgroups =
      make_cyclic_subgroups(*prim_sym_info.factor_group);

  EXPECT_EQ(cyclic_subgroups.size(), 10);
  auto any_count = make_any_count(cyclic_subgroups);
  // print_subgroups(cyclic_subgroups);
  EXPECT_EQ(any_count({0}), 1);
  EXPECT_EQ(any_count({0, 1, 4, 21}), 1);
  EXPECT_EQ(any_count({0, 7, 11}), 1);
  EXPECT_EQ(any_count({0, 7, 11, 33, 37, 47}), 1);
  EXPECT_EQ(any_count({0, 15}), 1);
  EXPECT_EQ(any_count({0, 21}), 1);
  EXPECT_EQ(any_count({0, 21, 41, 44}), 1);
  EXPECT_EQ(any_count({0, 24}), 1);
  EXPECT_EQ(any_count({0, 30}), 1);
  EXPECT_EQ(any_count({0, 47}), 1);
}

TEST(AllSubgroupsTest, Test1) {
  using namespace group;
  using namespace cyclic_subgroups_test;
  std::vector<Index> elements;
  for (Index i = 0; i < 10; ++i) {
    elements.push_back(i);
  }
  Group<Index> group = make_group(elements, multiply_f, equal_to_f);
  std::set<SubgroupOrbit> all_subgroups =
      make_all_subgroups_by_cyclic_join(group);

  EXPECT_EQ(all_subgroups.size(), 4);
  auto any_count = make_any_count(all_subgroups);
  // print_subgroups(all_subgroups);
  EXPECT_EQ(any_count({0}), 1);
  EXPECT_EQ(any_count({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), 1);
  EXPECT_EQ(any_count({0, 2, 4, 6, 8}), 1);
  EXPECT_EQ(any_count({0, 5}), 1);
}

TEST(AllSubgroupsTest, Test2) {
  using namespace group;
  using namespace cyclic_subgroups_test;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  std::set<SubgroupOrbit> all_subgroups =
      make_all_subgroups_by_cyclic_join(*prim_sym_info.factor_group);

  EXPECT_EQ(all_subgroups.size(), 33);
  auto any_count = make_any_count(all_subgroups);
  // print_subgroups(all_subgroups);
  EXPECT_EQ(any_count({
                0,
            }),
            1);
  EXPECT_EQ(any_count({
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            }),
            1);
  EXPECT_EQ(any_count({
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                1,
                4,
                17,
                20,
                21,
                22,
                23,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                1,
                4,
                17,
                20,
                21,
                22,
                23,
                26,
                29,
                30,
                31,
                32,
                41,
                44,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                1,
                4,
                21,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                1,
                4,
                21,
                26,
                29,
                31,
                32,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                1,
                4,
                21,
                30,
                41,
                44,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                21,
                22,
                23,
            }),
            1);
  EXPECT_EQ(any_count({
                0,  7,  8,  9,  10, 11, 12, 13, 14, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 41, 42, 43, 44, 45, 46,
            }),
            1);
  EXPECT_EQ(any_count({
                0,  7,  8,  9,  10, 11, 12, 13, 14, 21, 22, 23,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                7,
                11,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                7,
                11,
                17,
                18,
                19,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                7,
                11,
                17,
                18,
                19,
                26,
                27,
                28,
                33,
                37,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                7,
                11,
                26,
                27,
                28,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                7,
                11,
                33,
                37,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                15,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                15,
                19,
                22,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                15,
                19,
                22,
                24,
                28,
                31,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                15,
                19,
                22,
                30,
                32,
                42,
                45,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                15,
                24,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                15,
                28,
                31,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
                22,
                23,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
                22,
                23,
                24,
                28,
                42,
                45,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
                22,
                23,
                30,
                31,
                32,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
                26,
                29,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
                30,
                47,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
                31,
                32,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                21,
                41,
                44,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                24,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                30,
            }),
            1);
  EXPECT_EQ(any_count({
                0,
                47,
            }),
            1);
}

// Tests for MakeAllSubgroupsViaGroupExtension

/// Z_2 x Z_3 (order 6) with indexing g = f*2 + t (N_T=2, N_F=3)
TEST(MakeAllSubgroupsViaGroupExtensionTest, Z2xZ3) {
  using namespace group;
  // Multiplication: (f1,t1)*(f2,t2) = ((f1+f2)%3, (t1+t2)%2)
  auto g = make_generic_group(6, [](Index a, Index b) -> Index {
    Index f = (a / 2 + b / 2) % 3;
    Index t = (a % 2 + b % 2) % 2;
    return f * 2 + t;
  });

  MakeAllSubgroupsViaGroupExtension maker(g, /*N_T=*/2);
  maker.run();

  // Subgroups of Z_6: {0}, T={0,1}, Z_3={0,2,4}, G={0,1,2,3,4,5}
  EXPECT_EQ(maker.subgroups.size(), 4);

  std::set<std::set<Index>> found;
  for (auto const &[idx, gen] : maker.subgroups) found.insert(idx);

  EXPECT_EQ(found.count({0}), 1);
  EXPECT_EQ(found.count({0, 1}), 1);
  EXPECT_EQ(found.count({0, 2, 4}), 1);
  EXPECT_EQ(found.count({0, 1, 2, 3, 4, 5}), 1);
}

/// FCC prim factor group (order 48): verify MakeAllSubgroupsFromGenerators
/// with 1 subtree runs without crash
TEST(MakeAllSubgroupsViaGroupExtensionTest, FCCRefGenerators) {
  using namespace group;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  auto fg = prim_sym_info.factor_group;

  MakeAllSubgroupsFromGenerators ref_maker(fg);
  ref_maker.run(1, [](Index, Index) {});
  EXPECT_EQ(ref_maker.subgroups.size(), 98u);
}

/// FCC prim factor group (order 48) with N_T=1 — should find same 98 subgroups
/// as MakeAllSubgroupsFromGenerators
TEST(MakeAllSubgroupsViaGroupExtensionTest, FCCFactorGroupNT1) {
  using namespace group;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  auto fg = prim_sym_info.factor_group;

  // Method under test: N_T=1 means T={identity}, F=G
  MakeAllSubgroupsViaGroupExtension maker(fg, /*N_T=*/1);
  maker.run();

  std::set<std::set<Index>> from_normal;
  for (auto const &[idx, gen] : maker.subgroups) from_normal.insert(idx);
  EXPECT_EQ(from_normal.size(), 98u);
}

// Tests for SubgroupIteratorViaGroupExtension

/// Z_2 x Z_3 (order 6): iterator should yield same 4 subgroups as
/// MakeAllSubgroupsViaGroupExtension
TEST(SubgroupIteratorViaGroupExtensionTest, Z2xZ3) {
  using namespace group;

  auto mult = [](Index a, Index b) -> Index {
    Index f = (a / 2 + b / 2) % 3;
    Index t = (a % 2 + b % 2) % 2;
    return f * 2 + t;
  };
  auto inv = [&mult](Index a) -> Index {
    for (Index b = 0; b < 6; ++b)
      if (mult(a, b) == 0) return b;
    return -1;
  };

  // Collect subgroups from reference (MakeAllSubgroupsViaGroupExtension)
  auto g = make_generic_group(6, mult);
  MakeAllSubgroupsViaGroupExtension ref_maker(g, /*N_T=*/2);
  ref_maker.run();
  std::set<std::set<Index>> ref_subgroups;
  for (auto const &[idx, gen] : ref_maker.subgroups) ref_subgroups.insert(idx);

  // Collect subgroups from iterator — use both vector and set to detect
  // duplicates (a set would silently hide them)
  std::vector<std::set<Index>> iter_subgroups_vec;
  std::set<std::set<Index>> iter_subgroups;
  SubgroupIteratorViaGroupExtension it(6, /*N_T=*/2, mult, inv);
  while (!it.done()) {
    iter_subgroups_vec.push_back(it.value());
    iter_subgroups.insert(it.value());
    // Verify generators actually generate the subgroup
    MakeSubgroupFromGenerators check(*g, it.generators());
    EXPECT_EQ(check.indices, it.value());
    it.next();
  }

  EXPECT_EQ(iter_subgroups_vec.size(), iter_subgroups.size());  // no duplicates
  EXPECT_EQ(iter_subgroups, ref_subgroups);
  EXPECT_EQ(iter_subgroups.size(), 4u);
}

/// FCC prim factor group (order 48, N_T=1): iterator should yield same 98
/// subgroups as MakeAllSubgroupsViaGroupExtension
TEST(SubgroupIteratorViaGroupExtensionTest, FCCFactorGroupNT1) {
  using namespace group;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  auto fg = prim_sym_info.factor_group;

  // Reference
  MakeAllSubgroupsViaGroupExtension ref_maker(fg, /*N_T=*/1);
  ref_maker.run();
  std::set<std::set<Index>> ref_subgroups;
  for (auto const &[idx, gen] : ref_maker.subgroups) ref_subgroups.insert(idx);

  // Build functors from the factor group
  auto mult = [&fg](Index i, Index j) -> Index { return fg->mult(i, j); };
  auto inv = [&fg](Index i) -> Index { return fg->inv(i); };

  // Collect subgroups from iterator
  std::vector<std::set<Index>> iter_subgroups_vec;
  std::set<std::set<Index>> iter_subgroups_set;
  SubgroupIteratorViaGroupExtension it(fg->size(), /*N_T=*/1, mult, inv);
  while (!it.done()) {
    iter_subgroups_vec.push_back(it.value());
    iter_subgroups_set.insert(it.value());
    it.next();
  }

  EXPECT_EQ(iter_subgroups_vec.size(), ref_maker.subgroups.size());
  EXPECT_EQ(iter_subgroups_vec.size(), iter_subgroups_set.size());
  EXPECT_EQ(iter_subgroups_set, ref_subgroups);
  EXPECT_EQ(iter_subgroups_vec.size(), 98u);
}

/// FCC prim factor group: canonical_only=true should yield one subgroup per
/// orbit (total <= 98), and every canonical subgroup must be lex-max in orbit
TEST(SubgroupIteratorViaGroupExtensionTest, FCCFactorGroupCanonicalOnly) {
  using namespace group;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  auto fg = prim_sym_info.factor_group;
  Index N_G = fg->size();

  auto mult = [&fg](Index i, Index j) -> Index { return fg->mult(i, j); };
  auto inv = [&fg](Index i) -> Index { return fg->inv(i); };

  // Collect all canonical subgroups — use both vector and set to detect
  // duplicates
  std::vector<std::set<Index>> canonical_subgroups_vec;
  std::set<std::set<Index>> canonical_subgroups;
  SubgroupIteratorViaGroupExtension it(N_G, /*N_T=*/1, mult, inv,
                                       /*canonical_only=*/true);
  while (!it.done()) {
    std::set<Index> const &H = it.value();
    canonical_subgroups_vec.push_back(H);
    canonical_subgroups.insert(H);

    // Verify: no conjugate of H is greater under decreasing-sort lex order
    auto desc_gt = [](std::set<Index> const &a,
                      std::set<Index> const &b) -> bool {
      auto ai = a.rbegin(), bi = b.rbegin();
      for (; ai != a.rend() && bi != b.rend(); ++ai, ++bi) {
        if (*ai > *bi) return true;
        if (*ai < *bi) return false;
      }
      return false;
    };
    for (Index g = 0; g < N_G; ++g) {
      Index g_inv = fg->inv(g);
      std::set<Index> conj;
      for (Index h : H) conj.insert(fg->mult(g, fg->mult(h, g_inv)));
      EXPECT_FALSE(desc_gt(conj, H)) << "Non-canonical subgroup found";
    }
    it.next();
  }

  EXPECT_EQ(canonical_subgroups_vec.size(),
            canonical_subgroups.size());  // no duplicates

  // The number of canonical subgroups should equal the number of conjugacy
  // orbits of subgroups (at most 98 for FCC factor group)
  EXPECT_GT(canonical_subgroups.size(), 0u);
  EXPECT_LE(canonical_subgroups.size(), 98u);

  // Every non-canonical subgroup (from the full set) should have a canonical
  // representative in canonical_subgroups
  MakeAllSubgroupsViaGroupExtension ref_maker(fg, /*N_T=*/1);
  ref_maker.run();
  for (auto const &[H_idx, H_gen] : ref_maker.subgroups) {
    // Compute all conjugates of H_idx and check that at least one is canonical
    bool found_canonical = false;
    for (Index g = 0; g < N_G; ++g) {
      Index g_inv = fg->inv(g);
      std::set<Index> conj;
      for (Index h : H_idx) conj.insert(fg->mult(g, fg->mult(h, g_inv)));
      if (canonical_subgroups.count(conj)) {
        found_canonical = true;
        break;
      }
    }
    EXPECT_TRUE(found_canonical);
  }
}

/// FCC supercell, volume 2: iterator should yield the same subgroups as
/// MakeAllSubgroupsViaGroupExtension when both use functors built from
/// SupercellSymOp integer arithmetic.
TEST(SubgroupIteratorViaGroupExtensionTest, FCCSupercellV2) {
  using namespace group;

  auto prim = config::make_shared_prim(test::FCC_binary_prim());
  Eigen::Matrix3l T;
  T << 2, 0, 0, 0, 1, 0, 0, 0, 1;
  auto supercell = std::make_shared<config::Supercell const>(prim, T);
  config::SupercellSymOpFunctors f(supercell);

  EXPECT_EQ(f.N_T, 2);
  EXPECT_EQ(f.N_F(), 12);
  EXPECT_EQ(f.N_G, 24);

  // Reference: MakeAllSubgroupsViaGroupExtension with the full mult table
  auto g = make_generic_group(f.N_G, f.mult);
  MakeAllSubgroupsViaGroupExtension ref_maker(g, f.N_T);
  ref_maker.run();
  std::set<std::set<Index>> ref_subgroups;
  for (auto const &[idx, gen] : ref_maker.subgroups) ref_subgroups.insert(idx);

  // Iterator — use both vector and set to detect duplicates
  std::vector<std::set<Index>> iter_subgroups_vec;
  std::set<std::set<Index>> iter_subgroups;
  SubgroupIteratorViaGroupExtension it(f.N_G, f.N_T, f.mult, f.inv);
  while (!it.done()) {
    iter_subgroups_vec.push_back(it.value());
    iter_subgroups.insert(it.value());
    it.next();
  }

  EXPECT_EQ(iter_subgroups_vec.size(), iter_subgroups.size());  // no duplicates
  EXPECT_EQ(iter_subgroups, ref_subgroups);
}

/// BCC binary prim (BCC lattice, a=1) with T=2*I3 (N_T=8, N_F=48, N_G=384):
/// iterator (using SupercellSymOpFunctors) should yield the same subgroups as
/// MakeAllSubgroupsViaGroupExtension using the multiplication table from
/// make_symgroup (SymOp-based arithmetic).  This mirrors the Python
/// group_extension path (Subset built from generic_group()) and is the exact
/// case that shows a count mismatch in the Python timing test.
TEST(SubgroupIteratorViaGroupExtensionTest, BCCSupercellV8) {
  using namespace group;
  using namespace xtal;

  // BCC primitive lattice: a=1, lattice vectors as columns
  Eigen::Matrix3d lat;
  lat << -0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5;
  BasicStructure bcc_struc{Lattice{lat}};
  bcc_struc.set_title("BCC_binary");
  Molecule A = Molecule::make_atom("A");
  Molecule B = Molecule::make_atom("B");
  bcc_struc.push_back(
      Site(Coordinate(Eigen::Vector3d::Zero(), bcc_struc.lattice(), CART),
           std::vector<Molecule>{A, B}));
  bcc_struc.set_unique_names({{"A", "B"}});

  auto prim = config::make_shared_prim(bcc_struc);
  Eigen::Matrix3l T;
  T << 2, 0, 0, 0, 2, 0, 0, 0, 2;
  auto supercell = std::make_shared<config::Supercell const>(prim, T);
  config::SupercellSymOpFunctors f(supercell);

  EXPECT_EQ(f.N_T, 8);
  EXPECT_EQ(f.N_F(), 48);
  EXPECT_EQ(f.N_G, 384);

  // Collect all SupercellSymOp elements (same ordering as generic_group:
  // outer loop = factor group, inner loop = translations, k = f*N_T + t)
  std::vector<config::SupercellSymOp> symops;
  {
    auto it = config::SupercellSymOp::begin(supercell);
    auto end = config::SupercellSymOp::end(supercell);
    while (it != end) {
      symops.push_back(*it);
      ++it;
    }
  }
  EXPECT_EQ(static_cast<Index>(symops.size()), f.N_G);

  // Build SymGroup using SymOp-based arithmetic (mirrors Python generic_group
  // approach: multiplication table from element products).  SymGroup inherits
  // from GenericGroup so it can be passed directly.
  auto symgroup = config::make_symgroup(symops);

  // Reference: MakeAllSubgroupsViaGroupExtension on SymOp-based table
  MakeAllSubgroupsViaGroupExtension ref_maker(symgroup, f.N_T);
  ref_maker.run();
  std::set<std::set<Index>> ref_subgroups;
  for (auto const &[idx, gen] : ref_maker.subgroups) ref_subgroups.insert(idx);

  // Iterator using SupercellSymOpFunctors (integer arithmetic)
  std::vector<std::set<Index>> iter_subgroups_vec;
  std::set<std::set<Index>> iter_subgroups_set;
  SubgroupIteratorViaGroupExtension it(f.N_G, f.N_T, f.mult, f.inv);
  while (!it.done()) {
    iter_subgroups_vec.push_back(it.value());
    iter_subgroups_set.insert(it.value());
    it.next();
  }

  EXPECT_EQ(iter_subgroups_vec.size(), ref_maker.subgroups.size());
  EXPECT_EQ(iter_subgroups_vec.size(), iter_subgroups_set.size());
  EXPECT_EQ(iter_subgroups_set, ref_subgroups);
}

/// FCC supercell, 2x2x2 (volume 8, N_G=384): iterator should yield the same
/// subgroups as MakeAllSubgroupsViaGroupExtension.
TEST(SubgroupIteratorViaGroupExtensionTest, FCCSupercellV8) {
  using namespace group;

  auto prim = config::make_shared_prim(test::FCC_binary_prim());
  Eigen::Matrix3l T;
  T << 2, 0, 0, 0, 2, 0, 0, 0, 2;
  auto supercell = std::make_shared<config::Supercell const>(prim, T);
  config::SupercellSymOpFunctors f(supercell);

  EXPECT_EQ(f.N_T, 8);
  EXPECT_EQ(f.N_F(), 48);
  EXPECT_EQ(f.N_G, 384);

  // Reference: MakeAllSubgroupsViaGroupExtension with the full mult table
  auto g = make_generic_group(f.N_G, f.mult);
  MakeAllSubgroupsViaGroupExtension ref_maker(g, f.N_T);
  ref_maker.run();
  std::set<std::set<Index>> ref_subgroups;
  for (auto const &[idx, gen] : ref_maker.subgroups) ref_subgroups.insert(idx);

  // Iterator — use both vector and set to detect duplicates
  std::vector<std::set<Index>> iter_subgroups_vec;
  std::set<std::set<Index>> iter_subgroups;
  SubgroupIteratorViaGroupExtension it(f.N_G, f.N_T, f.mult, f.inv);
  while (!it.done()) {
    iter_subgroups_vec.push_back(it.value());
    iter_subgroups.insert(it.value());
    it.next();
  }

  EXPECT_EQ(iter_subgroups_vec.size(), iter_subgroups.size());  // no duplicates
  EXPECT_EQ(iter_subgroups.size(), ref_subgroups.size());
  EXPECT_EQ(iter_subgroups, ref_subgroups);
}

/// ZrO supercell, volume 2: verify canonical_only yields exactly one
/// representative per conjugacy orbit of subgroups.
TEST(SubgroupIteratorViaGroupExtensionTest, ZrOSupercellV2CanonicalOnly) {
  using namespace group;

  auto prim = config::make_shared_prim(test::ZrO_prim());
  Eigen::Matrix3l T;
  T << 2, 0, 0, 0, 1, 0, 0, 0, 1;
  auto supercell = std::make_shared<config::Supercell const>(prim, T);
  config::SupercellSymOpFunctors f(supercell);

  EXPECT_EQ(f.N_T, 2);
  EXPECT_EQ(f.N_F(), 8);
  EXPECT_EQ(f.N_G, 16);

  // Collect all subgroups via reference (full table)
  auto g = make_generic_group(f.N_G, f.mult);
  MakeAllSubgroupsViaGroupExtension ref_maker(g, f.N_T);
  ref_maker.run();

  // Compare two subgroups under decreasing-sort lex order
  auto desc_gt = [](std::set<Index> const &a,
                    std::set<Index> const &b) -> bool {
    auto ai = a.rbegin(), bi = b.rbegin();
    for (; ai != a.rend() && bi != b.rend(); ++ai, ++bi) {
      if (*ai > *bi) return true;
      if (*ai < *bi) return false;
    }
    return false;
  };

  // Compute conjugacy orbits: for each subgroup, find the canonical rep
  // (lex-max under decreasing-sort order)
  std::map<std::set<Index>, std::set<Index>> orbit_rep;
  for (auto const &[H_idx, H_gen] : ref_maker.subgroups) {
    std::set<Index> max_conj = H_idx;
    for (Index gi = 0; gi < f.N_G; ++gi) {
      Index g_inv = f.inv(gi);
      std::set<Index> conj;
      for (Index h : H_idx) conj.insert(f.mult(gi, f.mult(h, g_inv)));
      if (desc_gt(conj, max_conj)) max_conj = conj;
    }
    orbit_rep[H_idx] = max_conj;
  }

  // Collect the set of unique canonical representatives
  std::set<std::set<Index>> expected_canonicals;
  for (auto const &[H_idx, rep] : orbit_rep) expected_canonicals.insert(rep);

  // Collect what the iterator yields with canonical_only=true — use both
  // vector and set to detect duplicates
  std::vector<std::set<Index>> iter_canonicals_vec;
  std::set<std::set<Index>> iter_canonicals;
  SubgroupIteratorViaGroupExtension it(f.N_G, f.N_T, f.mult, f.inv,
                                       /*canonical_only=*/true);
  while (!it.done()) {
    std::set<Index> const &H = it.value();
    iter_canonicals_vec.push_back(H);
    iter_canonicals.insert(H);

    // Each yielded H must itself be canonical (no conjugate greater under
    // decreasing-sort lex order)
    for (Index gi = 0; gi < f.N_G; ++gi) {
      Index g_inv = f.inv(gi);
      std::set<Index> conj;
      for (Index h : H) conj.insert(f.mult(gi, f.mult(h, g_inv)));
      EXPECT_FALSE(desc_gt(conj, H))
          << "iterator yielded a non-canonical subgroup";
    }
    it.next();
  }

  EXPECT_EQ(iter_canonicals_vec.size(),
            iter_canonicals.size());  // no duplicates
  EXPECT_EQ(iter_canonicals, expected_canonicals);
}

/// BCC supercell, 2x2x2 (N_T=8, N_F=48, N_G=384): canonical_only=true should
/// yield exactly one representative per conjugacy orbit of subgroups, with no
/// duplicates.
TEST(SubgroupIteratorViaGroupExtensionTest, BCCSupercellV8CanonicalOnly) {
  using namespace group;
  using namespace xtal;

  // BCC primitive lattice: a=1, lattice vectors as columns
  Eigen::Matrix3d lat;
  lat << -0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5;
  BasicStructure bcc_struc{Lattice{lat}};
  bcc_struc.set_title("BCC_binary");
  Molecule A = Molecule::make_atom("A");
  Molecule B = Molecule::make_atom("B");
  bcc_struc.push_back(
      Site(Coordinate(Eigen::Vector3d::Zero(), bcc_struc.lattice(), CART),
           std::vector<Molecule>{A, B}));
  bcc_struc.set_unique_names({{"A", "B"}});

  auto prim = config::make_shared_prim(bcc_struc);
  Eigen::Matrix3l T;
  T << 2, 0, 0, 0, 2, 0, 0, 0, 2;
  auto supercell = std::make_shared<config::Supercell const>(prim, T);
  config::SupercellSymOpFunctors f(supercell);

  EXPECT_EQ(f.N_T, 8);
  EXPECT_EQ(f.N_F(), 48);
  EXPECT_EQ(f.N_G, 384);

  // Reference: all subgroups via MakeAllSubgroupsViaGroupExtension
  auto g = make_generic_group(f.N_G, f.mult);
  MakeAllSubgroupsViaGroupExtension ref_maker(g, f.N_T);
  ref_maker.run();

  // Compare subgroups under decreasing-sort lex order (matches _is_canonical)
  auto desc_gt = [](std::set<Index> const &a,
                    std::set<Index> const &b) -> bool {
    auto ai = a.rbegin(), bi = b.rbegin();
    for (; ai != a.rend() && bi != b.rend(); ++ai, ++bi) {
      if (*ai > *bi) return true;
      if (*ai < *bi) return false;
    }
    return false;
  };

  // Compute expected canonical representatives: lex-max under decreasing sort
  std::set<std::set<Index>> expected_canonicals;
  for (auto const &[H_idx, H_gen] : ref_maker.subgroups) {
    std::set<Index> max_conj = H_idx;
    for (Index gi = 0; gi < f.N_G; ++gi) {
      Index g_inv = f.inv(gi);
      std::set<Index> conj;
      for (Index h : H_idx) conj.insert(f.mult(gi, f.mult(h, g_inv)));
      if (desc_gt(conj, max_conj)) max_conj = conj;
    }
    expected_canonicals.insert(max_conj);
  }

  // Collect what the iterator yields with canonical_only=true — use both
  // vector and set to detect duplicates. Skip the per-subgroup conjugation
  // check (O(N_G × |H|) per subgroup) since N_G=384 and N_subgroups=2986
  // makes it very slow; the final EXPECT_EQ covers correctness.
  std::vector<std::set<Index>> iter_canonicals_vec;
  std::set<std::set<Index>> iter_canonicals;
  SubgroupIteratorViaGroupExtension it(f.N_G, f.N_T, f.mult, f.inv,
                                       /*canonical_only=*/true);
  while (!it.done()) {
    iter_canonicals_vec.push_back(it.value());
    iter_canonicals.insert(it.value());
    it.next();
  }

  EXPECT_EQ(iter_canonicals_vec.size(),
            iter_canonicals.size());  // no duplicates
  EXPECT_EQ(iter_canonicals, expected_canonicals);
}
