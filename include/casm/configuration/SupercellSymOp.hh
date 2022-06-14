#ifndef CASM_config_SupercellSymOp
#define CASM_config_SupercellSymOp

#include <iterator>

#include "casm/configuration/definitions.hh"
#include "casm/misc/Comparisons.hh"

namespace CASM {
namespace config {

/// \brief Represents and allows iteration over symmetry operations consistent
/// with a given Supercell, combining pure factor group and pure translation
/// operations.
///
/// Notes:
/// - When permuting sites, the factor group operation permutation is applied
///   first, then the translation operation permutation
/// - When iterating over all operations the translation operations are
///   iterated in the inner loop and factor group operations iterated in the
///   outer loop
/// - Overall, the following sequence of permutations is replicated:
///
/// \code
/// Container before;
/// SupercellSymInfo sym_info = ...
/// for( f=0; f<sym_info.factor_group_permutations.size(); f++) {
///   Permutation const &factor_group_permute =
///       sym_info.factor_group_permutations[g];
///
///   for( t=0; t<sym_info.translation_permutations.size(); t++) {
///     Permutation const &trans_permute = sym_info.translation_permutations[t];
///     Container after = copy_apply(trans_permute,
///                           copy_apply(factor_group_permute, before));
///   }
/// }
/// \endcode
class SupercellSymOp : public Comparisons<CRTPBase<SupercellSymOp>> {
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = SupercellSymOp;
  using pointer = SupercellSymOp *;
  using reference = SupercellSymOp &;

  /// Default invalid SupercellSymOp, not equal to end iterator
  SupercellSymOp();

  /// Construct SupercellSymOp
  SupercellSymOp(std::shared_ptr<Supercell const> const &_supercell,
                 Index _factor_group_index, Index _translation_index);

  /// \brief Make supercell symop begin iterator
  static SupercellSymOp begin(
      std::shared_ptr<Supercell const> const &_supercell);

  /// \brief Make supercell symop end iterator
  static SupercellSymOp end(std::shared_ptr<Supercell const> const &_supercell);

  /// \brief Make translations supercell symop begin iterator
  static SupercellSymOp translation_begin(
      std::shared_ptr<Supercell const> const &_supercell);

  /// \brief Make translations supercell symop end iterator
  static SupercellSymOp translation_end(
      std::shared_ptr<Supercell const> const &_supercell);

  std::shared_ptr<Supercell const> const &supercell() const;

  Index factor_group_index() const;

  Index translation_index() const;

  /// \brief Returns the index of the site containing the site DoF values that
  ///     will be permuted onto site i
  Index permute_index(Index i) const;

  /// Returns a reference to this -- allows SupercellSymOp to be treated
  /// as an iterator to SupercellSymOp object
  SupercellSymOp const &operator*() const;

  /// Returns a pointer to this -- allows SupercellSymOp to be treated as
  /// an iterator to SupercellSymOp object
  SupercellSymOp const *operator->() const;

  /// \brief prefix ++SupercellSymOp
  SupercellSymOp &operator++();

  /// \brief postfix SupercellSymOp++
  SupercellSymOp operator++(int);

  /// \brief prefix --SupercellSymOp
  SupercellSymOp &operator--();

  /// \brief postfix SupercellSymOp--
  SupercellSymOp operator--(int);

  /// \brief Return the SymOp for the current operation
  SymOp to_symop() const;

  /// Returns the combination of factor group operation permutation and
  /// translation permutation
  Permutation combined_permute() const;

  /// \brief Returns the inverse supercell operation
  SupercellSymOp inverse() const;

  /// \brief Returns the supercell operation equivalent to applying first RHS
  /// and then *this
  SupercellSymOp operator*(SupercellSymOp const &RHS) const;

  /// \brief Less than comparison (used to implement operator<() and other
  /// standard comparisons via Comparisons)
  bool operator<(SupercellSymOp const &iter) const;

 private:
  friend Comparisons<CRTPBase<SupercellSymOp>>;

  /// \brief Equality comparison (used to implement operator==)
  bool eq_impl(const SupercellSymOp &iter) const;

  std::shared_ptr<Supercell const> m_supercell;
  Index m_factor_group_index;
  Index m_translation_index;
  Index m_N_translation;
};

/// \brief Return inverse SymOp
SymOp inverse(SymOp const &op);

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
/// ConfigDoFValues
ConfigDoFValues &apply(SupercellSymOp const &op, ConfigDoFValues &dof_values);

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
/// ConfigDoFValues
ConfigDoFValues copy_apply(SupercellSymOp const &op,
                           ConfigDoFValues dof_values);

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
///     xtal::UnitCellCoord
xtal::UnitCellCoord &apply(SupercellSymOp const &op,
                           xtal::UnitCellCoord &unitcellcoord);

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
///     xtal::UnitCellCoord
xtal::UnitCellCoord copy_apply(SupercellSymOp const &op,
                               xtal::UnitCellCoord unitcellcoord);

}  // namespace config
}  // namespace CASM

#endif
