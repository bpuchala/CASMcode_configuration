#include "casm/configuration/PrimSymInfo.hh"
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
  std::set<SubgroupOrbit> all_subgroups = make_all_subgroups(group);

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
      make_all_subgroups(*prim_sym_info.factor_group);

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

// Tests for MakeAllSubgroupsFromNormalSubgroup

/// Z_2 x Z_3 (order 6) with indexing g = f*2 + t (N_T=2, N_F=3)
TEST(MakeAllSubgroupsFromNormalSubgroupTest, Z2xZ3) {
  using namespace group;
  // Multiplication: (f1,t1)*(f2,t2) = ((f1+f2)%3, (t1+t2)%2)
  auto g = make_generic_group(6, [](Index a, Index b) -> Index {
    Index f = (a / 2 + b / 2) % 3;
    Index t = (a % 2 + b % 2) % 2;
    return f * 2 + t;
  });

  MakeAllSubgroupsFromNormalSubgroup maker(g, /*N_T=*/2);
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
TEST(MakeAllSubgroupsFromNormalSubgroupTest, FCCRefGenerators) {
  using namespace group;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  auto fg = prim_sym_info.factor_group;

  MakeAllSubgroupsFromGenerators ref_maker(fg);
  ref_maker.run(1, [](Index, Index) {});
  EXPECT_EQ(ref_maker.subgroups.size(), 98u);
}

/// FCC prim factor group (order 48) with N_T=1 — should find same 98 subgroups
/// as MakeAllSubgroupsFromGenerators
TEST(MakeAllSubgroupsFromNormalSubgroupTest, FCCFactorGroupNT1) {
  using namespace group;
  config::PrimSymInfo prim_sym_info(test::FCC_binary_prim());
  auto fg = prim_sym_info.factor_group;

  // Method under test: N_T=1 means T={identity}, F=G
  MakeAllSubgroupsFromNormalSubgroup maker(fg, /*N_T=*/1);
  maker.run();

  std::set<std::set<Index>> from_normal;
  for (auto const &[idx, gen] : maker.subgroups) from_normal.insert(idx);
  EXPECT_EQ(from_normal.size(), 98u);
}
