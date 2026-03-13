#include "casm/configuration/SupercellSymOp.hh"

#include "casm/clexulator/ConfigDoFValues.hh"
#include "casm/clexulator/ConfigDoFValuesTools_impl.hh"
#include "casm/clexulator/DoFSpace.hh"
#include "casm/configuration/PrimSymInfo.hh"
#include "casm/configuration/Supercell.hh"
#include "casm/configuration/SupercellSymInfo.hh"
#include "casm/configuration/sym_info/definitions.hh"
#include "casm/configuration/sym_info/factor_group.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/crystallography/SymTypeComparator.hh"
#include "casm/global/threads.hh"

namespace CASM {
namespace config {

namespace {
xtal::SymOp make_point_op(xtal::SymOp const &fg_op) {
  return xtal::SymOp(get_matrix(fg_op), Eigen::Vector3d::Zero(),
                     get_time_reversal(fg_op));
};

}  // namespace

// --- Inline definitions ---

/// Default invalid SupercellSymOp, not equal to end iterator
SupercellSymOp::SupercellSymOp()
    : m_supercell_factor_group_index(),
      m_translation_index(),
      m_tmp_translation_index(-1) {}

/// Construct SupercellSymOp
///
/// \param _supercell Supercell
/// \param _supercell_factor_group_index Supercell factor group index
/// \param _translation_index Translation index, corresponding to
///     the translation of the origin to the unitcell with the
///     same linear index.
SupercellSymOp::SupercellSymOp(
    std::shared_ptr<Supercell const> const &_supercell,
    Index _supercell_factor_group_index, Index _translation_index)
    : m_supercell(_supercell),
      m_supercell_factor_group_index(_supercell_factor_group_index),
      m_supercell_factor_group_end_index(
          m_supercell->sym_info.factor_group_permutations.size()),
      m_translation_index(_translation_index),
      m_N_translation(m_supercell->superlattice.size()),
      m_tmp_translation_index(-1) {}

/// Construct SupercellSymOp
///
/// \param _supercell Supercell
/// \param _supercell_factor_group_index Supercell factor group index
/// \param _translation_frac A lattice translation, in fractional coordinates
SupercellSymOp::SupercellSymOp(
    std::shared_ptr<Supercell const> const &_supercell,
    Index _supercell_factor_group_index,
    xtal::UnitCell const &_translation_frac)
    : SupercellSymOp(_supercell, _supercell_factor_group_index,
                     _supercell->unitcell_index_converter(_translation_frac)) {}

/// Construct SupercellSymOp
///
/// \param _supercell Supercell
/// \param _supercell_factor_group_index Supercell factor group index
/// \param _translation_cart A lattice translation, in Cartesian coordinates
SupercellSymOp::SupercellSymOp(
    std::shared_ptr<Supercell const> const &_supercell,
    Index _supercell_factor_group_index,
    Eigen::Vector3d const &_translation_cart)
    : SupercellSymOp(
          _supercell, _supercell_factor_group_index,
          UnitCell::from_cartesian(_translation_cart,
                                   _supercell->superlattice.prim_lattice())) {}

/// \brief Make supercell symop begin iterator
SupercellSymOp SupercellSymOp::begin(
    std::shared_ptr<Supercell const> const &_supercell) {
  return SupercellSymOp(_supercell, 0, 0);
}

/// \brief Make supercell symop end iterator
SupercellSymOp SupercellSymOp::end(
    std::shared_ptr<Supercell const> const &_supercell) {
  return SupercellSymOp(
      _supercell, _supercell->sym_info.factor_group_permutations.size(), 0);
}

/// \brief Make translations supercell symop begin iterator
SupercellSymOp SupercellSymOp::translation_begin(
    std::shared_ptr<Supercell const> const &_supercell) {
  return SupercellSymOp(_supercell, 0, 0);
}

/// \brief Make translations supercell symop end iterator
SupercellSymOp SupercellSymOp::translation_end(
    std::shared_ptr<Supercell const> const &_supercell) {
  return SupercellSymOp(_supercell, 1, 0);
}

std::shared_ptr<Supercell const> const &SupercellSymOp::supercell() const {
  return m_supercell;
}

/// \brief Supercell factor group index
///
/// This is an index into:
/// - supercell()->sym_info.factor_group->element
/// - supercell()->sym_info.factor_group_permutations
Index SupercellSymOp::supercell_factor_group_index() const {
  return m_supercell_factor_group_index;
}

/// \brief Prim factor group index
///
/// This is an index into:
/// - supercell()->prim->sym_info.factor_group->element
Index SupercellSymOp::prim_factor_group_index() const {
  this->throw_invalid_if_end();
  return m_supercell->sym_info.factor_group
      ->head_group_index[m_supercell_factor_group_index];
}

/// \brief Lattice translation index
Index SupercellSymOp::translation_index() const { return m_translation_index; }

/// \brief Lattice translation in fractional coordinates of the prim lattice
/// vectors
xtal::UnitCell SupercellSymOp::translation_frac() const {
  return m_supercell->unitcell_index_converter(m_translation_index);
}

/// \brief Returns the index of the site containing the site DoF values that
///     will be permuted onto site i
///
/// Permutation of configuration site dof values occurs according to:
///     after[i] = before[permute_index(i)]
Index SupercellSymOp::permute_index(Index i) const {
  this->throw_invalid_if_end();
  SupercellSymInfo const &sym_info = m_supercell->sym_info;
  auto const &fg_perm =
      sym_info.factor_group_permutations[m_supercell_factor_group_index];
  auto const &trans_perm = this->translation_permute();
  return fg_perm[trans_perm[i]];
}

/// Returns a reference to this -- allows SupercellSymOp to be treated as an
/// iterator to SupercellSymOp object
SupercellSymOp const &SupercellSymOp::operator*() const { return *this; }

/// Returns a pointer to this -- allows SupercellSymOp to be treated as an
/// iterator to SupercellSymOp object
SupercellSymOp const *SupercellSymOp::operator->() const { return this; }

/// \brief prefix ++SupercellSymOp
SupercellSymOp &SupercellSymOp::operator++() {
  m_translation_index++;
  if (m_translation_index == m_N_translation) {
    m_translation_index = 0;
    m_supercell_factor_group_index++;
  }
  return *this;
}

/// \brief postfix SupercellSymOp++
SupercellSymOp SupercellSymOp::operator++(int) {
  SupercellSymOp cp(*this);
  ++cp;
  return cp;
}

/// \brief prefix --SupercellSymOp
SupercellSymOp &SupercellSymOp::operator--() {
  if (m_translation_index == 0) {
    m_supercell_factor_group_index--;
    m_translation_index = m_N_translation;
  }
  m_translation_index--;
  return *this;
}

/// \brief postfix SupercellSymOp--
SupercellSymOp SupercellSymOp::operator--(int) {
  SupercellSymOp cp(*this);
  --cp;
  return cp;
}

/// \brief Return the SymOp for the current operation
///
/// Defined by:
///
///   translation_op * factor_group_op
///
/// In other words, the symmetry operation equivalent to application of the
/// factor group operation, FOLLOWED BY application of the translation
/// operation;
SymOp SupercellSymOp::to_symop() const {
  this->throw_invalid_if_end();
  UnitCell translation_frac =
      this->m_supercell->unitcell_index_converter(this->m_translation_index);

  Eigen::Matrix3d const &prim_lat_column_mat =
      this->m_supercell->superlattice.prim_lattice().lat_column_mat();

  Eigen::Vector3d translation_cart =
      prim_lat_column_mat * translation_frac.cast<double>();

  SymOp const &fg_op = this->m_supercell->sym_info.factor_group
                           ->element[m_supercell_factor_group_index];

  return SymOp{fg_op.matrix, translation_cart + fg_op.translation,
               fg_op.is_time_reversal_active};
}

/// Returns the translation permutation. Reference not valid after increment.
sym_info::Permutation const &SupercellSymOp::translation_permute() const {
  this->throw_invalid_if_end();
  if (m_supercell->sym_info.translation_permutations.has_value()) {
    return (
        *m_supercell->sym_info.translation_permutations)[m_translation_index];
  }
  if (m_tmp_translation_index != m_translation_index) {
    m_tmp_translation_index = m_translation_index;
    m_tmp_translation_permute = make_translation_permutation(
        m_tmp_translation_index, this->m_supercell->unitcell_index_converter,
        this->m_supercell->unitcellcoord_index_converter);
  }
  return m_tmp_translation_permute;
}

/// Returns the combination of factor group operation permutation and
/// translation permutation
sym_info::Permutation SupercellSymOp::combined_permute() const {
  this->throw_invalid_if_end();
  SupercellSymInfo const &sym_info = m_supercell->sym_info;
  auto const &fg_permute =
      sym_info.factor_group_permutations[m_supercell_factor_group_index];
  auto const &trans_permute = translation_permute();
  return CASM::sym_info::combined_permute(fg_permute,      // first
                                          trans_permute);  // second
}

/// \brief Returns the inverse supercell operation
SupercellSymOp SupercellSymOp::inverse() const {
  this->throw_invalid_if_end();
  // Copy *this, then update m_supercell_factor_group_index and
  // m_translation_index
  SupercellSymOp inverse_op(*this);

  // Finding the inverse factor_group operation is straightforward
  SymGroup const &supercell_factor_group =
      *this->m_supercell->sym_info.factor_group;
  Index inverse_fg_index =
      supercell_factor_group
          .inverse_index[this->m_supercell_factor_group_index];
  inverse_op.m_supercell_factor_group_index = inverse_fg_index;

  // New translation: inverse of (f, t) is (f_inv, R_frac_inv * (-uc(t)))
  // where R_frac_inv is the integer fractional rotation matrix for f_inv.
  //
  // Derivation: (f, t) * (f_inv, t') = (identity, 0) requires
  //   action(f, t') + t = 0  =>  t' = R_frac_inv * (-uc(t))  [mod supercell]

  Index prim_fg_idx_inv =
      m_supercell->sym_info.factor_group->head_group_index[inverse_fg_index];
  Eigen::Matrix3l const &R_frac_inv =
      m_supercell->prim->sym_info.unitcellcoord_symgroup_rep[prim_fg_idx_inv]
          .point_matrix;

  UnitCell uc_t =
      m_supercell->unitcell_index_converter(this->m_translation_index);
  UnitCell inverse_uc{R_frac_inv * (-uc_t)};
  inverse_op.m_translation_index =
      m_supercell->unitcell_index_converter(inverse_uc);

  return inverse_op;
}

/// \brief Returns the supercell operation equivalent to applying first RHS
/// and then *this
SupercellSymOp SupercellSymOp::operator*(SupercellSymOp const &RHS) const {
  this->throw_invalid_if_end();
  RHS.throw_invalid_if_end();

  // Copy *this, then update m_supercell_factor_group_index and
  // m_translation_index
  SupercellSymOp product_op(*this);

  // Finding the factor_group product is straightforward
  SymGroup const &supercell_factor_group =
      *this->m_supercell->sym_info.factor_group;
  product_op.m_supercell_factor_group_index =
      supercell_factor_group
          .multiplication_table[this->m_supercell_factor_group_index]
                               [RHS.m_supercell_factor_group_index];

  // New translation: (f1,t1)*(f2,t2) = (f3, R_frac[f1]*uc(t2) + uc(t1))
  // where R_frac[f1] is the integer fractional rotation matrix for f1.
  //
  // Derivation: the prim translations tau cancel between the total SymOp
  // product and the product factor group element, leaving:
  //   translation_frac = R_frac[f1] * uc(t2) + uc(t1)  [mod supercell]

  Index prim_fg_idx =
      m_supercell->sym_info.factor_group
          ->head_group_index[this->m_supercell_factor_group_index];
  Eigen::Matrix3l const &R_frac =
      m_supercell->prim->sym_info.unitcellcoord_symgroup_rep[prim_fg_idx]
          .point_matrix;

  UnitCell uc_rhs =
      m_supercell->unitcell_index_converter(RHS.m_translation_index);
  UnitCell uc_lhs =
      m_supercell->unitcell_index_converter(this->m_translation_index);
  UnitCell product_uc{R_frac * uc_rhs + uc_lhs};
  product_op.m_translation_index =
      m_supercell->unitcell_index_converter(product_uc);

  return product_op;
}

/// \brief Less than comparison (used to implement operator<() and other
/// standard comparisons via Comparisons)
bool SupercellSymOp::operator<(SupercellSymOp const &RHS) const {
  if (this->m_supercell_factor_group_index ==
      RHS.m_supercell_factor_group_index) {
    return this->m_translation_index < RHS.m_translation_index;
  }
  return this->m_supercell_factor_group_index <
         RHS.m_supercell_factor_group_index;
}

/// \brief Equality comparison (used to implement operator==)
bool SupercellSymOp::eq_impl(SupercellSymOp const &RHS) const {
  if (m_supercell == RHS.m_supercell &&
      m_supercell_factor_group_index == RHS.m_supercell_factor_group_index &&
      m_translation_index == RHS.m_translation_index) {
    return true;
  }
  return false;
}

/// \brief Return inverse SymOp
SymOp inverse(SymOp const &op) {
  // x' = R * x + T
  // R.inv * x' = x * R.inv * T
  // R.inv * x' - R.inv * T = x

  // SymOp matrix is unitary, so inverse is equivalent to transpose.

  return SymOp{op.matrix.transpose(), -(op.matrix.transpose() * op.translation),
               op.is_time_reversal_active};
}

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
/// ConfigDoFValues
ConfigDoFValues &apply(SupercellSymOp const &op, ConfigDoFValues &dof_values) {
  op.throw_invalid_if_end();
  Supercell const &supercell = *op.supercell();
  Prim const &prim = *op.supercell()->prim;
  PrimSymInfo const &prim_sym_info = prim.sym_info;
  Index n_vol = supercell.superlattice.size();
  Index n_sublat = prim.basicstructure->basis().size();
  Index n_sites = n_vol * n_sublat;

  Index prim_fg_index = op.prim_factor_group_index();

  for (auto &dof : dof_values.global_dof_values) {
    Eigen::MatrixXd const &M =
        prim_sym_info.global_dof_symgroup_rep.at(dof.first)[prim_fg_index];
    dof.second = M * dof.second;
  }

  sym_info::Permutation combined_permute{op.combined_permute()};

  if (dof_values.occupation.size()) {
    // permute occupant indices (if anisotropic)
    Eigen::VectorXi tmp{dof_values.occupation};
    if (prim_sym_info.has_aniso_occs) {
      Index l = 0;
      for (Index b = 0; b < n_sublat; ++b) {
        for (Index n = 0; n < n_vol; ++n, ++l) {
          sym_info::Permutation const &occ_perm =
              prim_sym_info.occ_symgroup_rep[prim_fg_index][b];
          tmp[l] = occ_perm[tmp[l]];
        }
      }
    }

    // permute values amongst sites
    for (Index l = 0; l < n_sites; ++l) {
      dof_values.occupation[l] = tmp[combined_permute[l]];
    }
  }

  using clexulator::sublattice_block;
  for (auto &dof : dof_values.local_dof_values) {
    // vector of matrix, one per sublattice
    sym_info::LocalDoFSymOpRep const &local_dof_symop_rep =
        prim_sym_info.local_dof_symgroup_rep.at(dof.first)[prim_fg_index];

    // transform values on initial sites
    Eigen::MatrixXd const &init_value = dof.second;
    Eigen::MatrixXd tmp{init_value};
    for (Index b = 0; b < n_sublat; ++b) {
      Eigen::MatrixXd const &M = local_dof_symop_rep[b];
      Index dim = M.cols();
      if (dim == 0) continue;
      sublattice_block(tmp, b, n_vol).topRows(dim) =
          M * sublattice_block(init_value, b, n_vol).topRows(dim);
    }

    // permute values amongst sites
    for (Index l = 0; l < n_sites; ++l) {
      dof.second.col(l) = tmp.col(combined_permute[l]);
    }
  }

  return dof_values;
}

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
/// ConfigDoFValues
ConfigDoFValues copy_apply(SupercellSymOp const &op,
                           ConfigDoFValues dof_values) {
  apply(op, dof_values);
  return dof_values;
}

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
///     xtal::UnitCellCoord
xtal::UnitCellCoord &apply(SupercellSymOp const &op,
                           xtal::UnitCellCoord &unitcellcoord) {
  op.throw_invalid_if_end();
  UnitCell translation_frac =
      op.supercell()->unitcell_index_converter(op.translation_index());

  UnitCellCoordRep const &fg_op =
      op.supercell()
          ->prim->sym_info
          .unitcellcoord_symgroup_rep[op.prim_factor_group_index()];

  apply(fg_op, unitcellcoord);
  unitcellcoord += translation_frac;
  return unitcellcoord;
}

/// \brief Apply a symmetry operation specified by a SupercellSymOp to
///     xtal::UnitCellCoord
xtal::UnitCellCoord copy_apply(SupercellSymOp const &op,
                               xtal::UnitCellCoord unitcellcoord) {
  apply(op, unitcellcoord);
  return unitcellcoord;
}

/// \brief Make SupercellSymOp group rep for local property symmetry in a
/// supercell
///
/// \brief local_prim_subgroup A subgroup of prim->sym_info->factor_group.
///     Must be a local property group, in which each prim factor group
///     operation only appears once.
/// \brief supercell The supercell for which the supercell symgroup
///     group is being generated.
///
/// \returns supercell_symgroup_rep, the SupercellSymOp consistent with both
///     the supercell and the prim_subgroup
std::vector<SupercellSymOp> make_local_supercell_symgroup_rep(
    std::shared_ptr<SymGroup const> const &local_prim_subgroup,
    std::shared_ptr<Supercell const> const &supercell) {
  std::vector<SupercellSymOp> result;

  if (local_prim_subgroup->head_group !=
      supercell->sym_info.factor_group->head_group) {
    throw std::runtime_error(
        "Error in make_local_supercell_symgroup_rep: do not share the same "
        "prim factor group");
  }

  SymGroup const &local_group = *local_prim_subgroup;
  SymGroup const &prim_factor_group = *local_group.head_group;
  SymGroup const &supercell_factor_group = *supercell->sym_info.factor_group;

  std::map<Index, Index> prim_to_supercell_fg_index;
  Index supercell_fg_index = 0;
  for (Index prim_fg_index : supercell_factor_group.head_group_index) {
    prim_to_supercell_fg_index[prim_fg_index] = supercell_fg_index;
    ++supercell_fg_index;
  }

  // Iterate over local group operations
  for (Index i = 0; i < local_group.element.size(); ++i) {
    Index prim_fg_index = local_group.head_group_index[i];
    auto it = prim_to_supercell_fg_index.find(prim_fg_index);

    // If local group operation is in supercell factor group
    if (it != prim_to_supercell_fg_index.end()) {
      // Determine lattice translation and use it to construct SupercellSymOp
      Eigen::Vector3d translation_cart =
          local_group.element[i].translation -
          prim_factor_group.element[prim_fg_index].translation;
      result.emplace_back(supercell,
                          it->second,  // supercell_fg_index
                          translation_cart);
    }
  }

  return result;
}

/// \brief Make SymGroup from SupercellSymOp group rep for local property
///     symmetry in a supercell
std::shared_ptr<SymGroup const> make_local_symgroup(
    std::vector<SupercellSymOp> const &local_supercell_symgroup_rep,
    std::shared_ptr<Supercell const> const &supercell) {
  std::shared_ptr<SymGroup const> prim_factor_group =
      supercell->prim->sym_info.factor_group;
  std::shared_ptr<SymGroup const> supercell_factor_group =
      supercell->sym_info.factor_group;

  std::map<Index, xtal::SymOp> index_and_element;
  for (auto const &supercell_symop : local_supercell_symgroup_rep) {
    Index supercell_fg_index = supercell_symop.supercell_factor_group_index();
    Index prim_fg_index =
        supercell_factor_group->head_group_index[supercell_fg_index];
    auto result =
        index_and_element.emplace(prim_fg_index, supercell_symop.to_symop());
    if (!result.second) {
      throw std::runtime_error(
          "Error in config::make_local_symgroup: not a local property group, "
          "repeated prim factor group index.");
    }
  }

  std::vector<xtal::SymOp> element;
  std::set<Index> head_group_index;
  for (auto const &pair : index_and_element) {
    head_group_index.emplace(pair.first);
    element.push_back(pair.second);
  }

  return std::make_shared<SymGroup const>(prim_factor_group, element,
                                          head_group_index);
}

/// \brief Make the matrix representation of `group` that describes the
///     transformation of the specified DoF
///
/// \param group The group that is to be represented (this may be larger
///     than a crystallographic factor group)
/// \param key The type of DoF to be transformed. May be a global
///     continuous DoF, local continuous DoF, or "occ" for occupation DoF.
/// \param site_indices Set of site indices that define the subset of sites
///     where DoF will be transformed. This is ignored for global DoF,
///     and required for occupation / local DoF and will throw if it has
///     no value).
/// \param symgroup The resulting group as a SymGroup. For global DoF,
///     this is the point group with repeated elements are removed.
/// \param make_symgroup If true, `symgroup` is constructed, otherwise
///     it is unchanged.
///
/// \returns matrix_rep The matrix representation of `group` which transforms
///     the specified DoF. For global DoF repeated elements are removed (the
///     result is a point group representation) so the size of `matrix_rep`
///     may be less the size of `group`.
///
/// Notes:
/// - This checks if `key` is global or not (occuption/local) and
///   forwards arguments to `make_global_dof_matrix_rep` or
///   `make_local_dof_matrix_rep`.
std::vector<Eigen::MatrixXd> make_matrix_rep(
    std::vector<SupercellSymOp> const &group, DoFKey key,
    std::optional<std::set<Index>> site_indices,
    std::shared_ptr<SymGroup const> &symgroup, bool make_symgroup) {
  if (group.size() == 0) {
    throw std::runtime_error("Error in make_matrix_rep: group has size==0.");
  }
  if (AnisoValTraits(key).global()) {
    return make_global_dof_matrix_rep(group, key, symgroup, make_symgroup);
  } else {
    if (!site_indices.has_value()) {
      throw std::runtime_error(
          "Error in make_matrix_rep: site_indices has no value for occupation "
          "or local DoF");
    }
    return make_local_dof_matrix_rep(group, key, *site_indices, symgroup,
                                     make_symgroup);
  }
}

/// \brief Make the matrix representation of `group` that describes the
///     transformation of a particular global DoF
///
/// \param group The group that is to be represented (this may be larger than a
///     crystallographic factor group)
/// \param key The type of global DoF to be transformed.
/// \param symgroup The resulting group, which is a point group, as a SymGroup.
/// \param make_symgroup If true, `symgroup` is constructed, otherwise
///     it is unchanged.
///
/// \returns matrix_rep The matrix representation of `group` which transforms
///     the specified global DoF. Repeated elements are removed (the result
///     is a point group representation) so the size of `matrix_rep` may be
///     less the size of `group`.
///
std::vector<Eigen::MatrixXd> make_global_dof_matrix_rep(
    std::vector<SupercellSymOp> const &group, DoFKey key,
    std::shared_ptr<SymGroup const> &symgroup, bool make_symgroup) {
  if (group.size() == 0) {
    throw std::runtime_error(
        "Error in make_global_dof_matrix_rep: group has size==0.");
  }
  Supercell const &supercell = *group.begin()->supercell();
  Prim const &prim = *supercell.prim;
  auto const &factor_group_element = prim.sym_info.factor_group->element;
  xtal::Lattice const &prim_lattice = prim.basicstructure->lattice();
  double xtal_tol = prim_lattice.tol();

  // Get prim factor group indices for the point group of `group`
  std::set<Index> prim_factor_group_indices;
  xtal::SymOpCompare_f is_equal(xtal_tol);
  for (SupercellSymOp const &supercell_symop : group) {
    Index prim_fg_index = supercell_symop.prim_factor_group_index();
    xtal::SymOp point_op = make_point_op(factor_group_element[prim_fg_index]);
    auto is_same_point_op = [&](Index other_prim_fg_index) {
      return is_equal(point_op,
                      make_point_op(factor_group_element[other_prim_fg_index]));
    };
    auto it = std::find_if(prim_factor_group_indices.begin(),
                           prim_factor_group_indices.end(), is_same_point_op);
    if (it == prim_factor_group_indices.end()) {
      prim_factor_group_indices.insert(prim_fg_index);
    }
  }

  sym_info::GlobalDoFSymGroupRep global_dof_symgroup_rep =
      prim.sym_info.global_dof_symgroup_rep.at(key);

  std::vector<Eigen::MatrixXd> result;
  std::vector<xtal::SymOp> element;
  for (Index prim_factor_group_index : prim_factor_group_indices) {
    Eigen::MatrixXd M = global_dof_symgroup_rep.at(prim_factor_group_index);
    element.push_back(
        make_point_op(factor_group_element[prim_factor_group_index]));
    result.push_back(M);
  }

  if (make_symgroup) {
    std::multiplies<SymOp> multiply_f;
    xtal::SymOpPeriodicCompare_f equal_to_f(prim_lattice, xtal_tol);
    symgroup = std::make_shared<SymGroup const>(
        group::make_group(element, multiply_f, equal_to_f));
  }
  return result;
}

/// \brief Make the matrix representation of `group` that describes the
///     transformation of occupation DoF or a particular local DoF of
///     amongst a subset of supercell sites
///
/// \param group The group that is to be represented (this may be larger than a
///     crystallographic factor group)
/// \param key The type of local DoF to be transformed. May be a local
///     continuous DoF or "occ".
/// \param site_indices Set of site indices that define the subset of sites
///     where DoF will be transformed
/// \param symgroup The resulting group as a SymGroup.
/// \param make_symgroup If true, `symgroup` is constructed, otherwise
///     it is unchanged.
///
/// \returns matrix_rep The matrix representation of `group` which transforms
///     the specified occupation or local DoF.
///
std::vector<Eigen::MatrixXd> make_local_dof_matrix_rep(
    std::vector<SupercellSymOp> const &group, DoFKey key,
    std::set<Index> const &site_indices,
    std::shared_ptr<SymGroup const> &symgroup, bool make_symgroup) {
  if (group.size() == 0) {
    throw std::runtime_error(
        "Error in make_local_dof_matrix_rep: group has size==0.");
  }
  Supercell const &supercell = *group.begin()->supercell();
  Prim const &prim = *supercell.prim;
  xtal::Lattice const &prim_lattice = prim.basicstructure->lattice();
  double xtal_tol = prim_lattice.tol();

  // Usage:
  // \code
  // Eigen::MatrixXd const &M =
  //     local_dof_symgroup_rep.at(prim_fg_index).at(sublattice_index_before)
  // Eigen::MatrixXd sublattice_local_dof_values_after =
  //     M * sublattice_local_dof_values_before;
  // \endcode
  // Note:
  // - For each group element there is one matrix representation per sublattice
  // - Local DoF values transform using these symrep matrices *before*
  //   permuting among sites.

  sym_info::LocalDoFSymGroupRep local_dof_symgroup_rep =
      prim.sym_info.local_dof_symgroup_rep.at(key);
  if (local_dof_symgroup_rep.size() == 0) {
    throw std::runtime_error(
        "Error in make_collective_dof_matrix_rep: DoF symgroup rep has "
        "size==0.");
  }

  // make map of site_index -> beginning row in basis for that site
  // (number of rows per site == dof dimension on that site)
  std::map<Index, Index> site_index_to_basis_index;
  Index total_dim = 0;
  for (Index site_index : site_indices) {
    Index b = supercell.unitcellcoord_index_converter(site_index).sublattice();
    Index site_dof_dim = local_dof_symgroup_rep.at(0).at(b).cols();
    site_index_to_basis_index[site_index] = total_dim;
    total_dim += site_dof_dim;
  }

  // make matrix rep, by filling in blocks with site dof symreps
  // Eigen::MatrixXd trep(total_dim, total_dim);
  Index n = static_cast<Index>(group.size());
  std::vector<Eigen::MatrixXd> result;
  result.resize(n);
  std::vector<xtal::SymOp> element;
  if (make_symgroup) {
    element.resize(n, xtal::SymOp::identity());
  }
  // for (SupercellSymOp const &supercell_symop : group) {
  //   trep.setZero();
  //   for (Index site_index : site_indices) {
  //     // "to_site" (after applying symmetry) determines row of block
  //     // can't fail, because it was built from [begin, end)
  //     Index to_site_index = site_index;
  //     Index row = site_index_to_basis_index.find(to_site_index)->second;
  //
  //     // "from_site" (before applying symmetry) determines col of block
  //     // could fail, if mismatch between [begin, end) and group
  //     Index from_site_index = supercell_symop.permute_index(site_index);
  //     auto col_it = site_index_to_basis_index.find(from_site_index);
  //     if (col_it == site_index_to_basis_index.end()) {
  //       throw std::runtime_error(
  //           "Error in make_collective_dof_matrix_rep: Input group includes "
  //           "permutations "
  //           "between selected and unselected sites.");
  //     }
  //     Index col = col_it->second;
  //
  //     // "from_site" sublattice and factor group op index
  //     // are used to lookup the site dof rep matrix
  //     Index from_site_b =
  //         supercell.unitcellcoord_index_converter(from_site_index).sublattice();
  //     Index prim_factor_group_index =
  //     supercell_symop.prim_factor_group_index(); Eigen::MatrixXd U =
  //         local_dof_symgroup_rep.at(prim_factor_group_index).at(from_site_b);
  //
  //     // insert matrix as block in collective dof symrep
  //     trep.block(row, col, U.rows(), U.cols()) = U;
  //   }
  //   result.push_back(trep);
  //
  //   element.push_back(supercell_symop.to_symop());
  // }

  auto worker = [&](Index start, Index end, Index thread_id) {
    Eigen::MatrixXd trep(total_dim, total_dim);
    for (Index i = start; i < end; ++i) {
      SupercellSymOp const &supercell_symop = group[i];
      trep.setZero();
      for (Index site_index : site_indices) {
        // "to_site" (after applying symmetry) determines row of block
        // can't fail, because it was built from [begin, end)
        Index to_site_index = site_index;
        Index row = site_index_to_basis_index.find(to_site_index)->second;

        // "from_site" (before applying symmetry) determines col of block
        // could fail, if mismatch between [begin, end) and group
        Index from_site_index = supercell_symop.permute_index(site_index);
        auto col_it = site_index_to_basis_index.find(from_site_index);
        if (col_it == site_index_to_basis_index.end()) {
          throw std::runtime_error(
              "Error in make_collective_dof_matrix_rep: Input group "
              "includes "
              "permutations between selected and unselected sites.");
        }
        Index col = col_it->second;

        // "from_site" sublattice and factor group op index
        // are used to lookup the site dof rep matrix
        Index from_site_b =
            supercell.unitcellcoord_index_converter(from_site_index)
                .sublattice();
        Index prim_factor_group_index =
            supercell_symop.prim_factor_group_index();
        Eigen::MatrixXd U =
            local_dof_symgroup_rep.at(prim_factor_group_index).at(from_site_b);

        // insert matrix as block in collective dof symrep
        trep.block(row, col, U.rows(), U.cols()) = U;
      }
      result[i] = trep;

      if (make_symgroup) {
        element[i] = supercell_symop.to_symop();
      }
    }
  };

  threaded_run(n, worker);

  if (make_symgroup) {
    std::multiplies<SymOp> multiply_f;
    xtal::SymOpPeriodicCompare_f equal_to_f(
        supercell.superlattice.superlattice(), xtal_tol);
    symgroup = std::make_shared<SymGroup const>(
        group::make_group(element, multiply_f, equal_to_f));
  }

  return result;
}

namespace {

/// \brief Like make_local_dof_matrix_rep, but projects each matrix immediately
///     via `basis_inv * M * basis` to avoid storing all full-space matrices
///     simultaneously.
///
/// Peak memory is O(one full-space matrix per thread) rather than
/// O(group.size() full-space matrices).
std::vector<Eigen::MatrixXd> make_projected_local_dof_matrix_rep(
    std::vector<config::SupercellSymOp> const &group, DoFKey const &key,
    std::set<Index> const &site_indices,
    std::shared_ptr<config::SymGroup const> &symgroup, bool make_symgroup_flag,
    Eigen::MatrixXd const &basis, Eigen::MatrixXd const &basis_inv) {
  if (group.size() == 0) {
    throw std::runtime_error(
        "Error in make_projected_local_dof_matrix_rep: group has size==0.");
  }

  config::Supercell const &supercell = *group.begin()->supercell();
  config::Prim const &prim = *supercell.prim;
  xtal::Lattice const &prim_lattice = prim.basicstructure->lattice();
  double xtal_tol = prim_lattice.tol();

  sym_info::LocalDoFSymGroupRep const &local_dof_symgroup_rep =
      prim.sym_info.local_dof_symgroup_rep.at(key);
  if (local_dof_symgroup_rep.size() == 0) {
    throw std::runtime_error(
        "Error in make_projected_local_dof_matrix_rep: DoF symgroup rep has "
        "size==0.");
  }

  std::map<Index, Index> site_index_to_basis_index;
  Index total_dim = 0;
  for (Index site_index : site_indices) {
    Index b = supercell.unitcellcoord_index_converter(site_index).sublattice();
    Index site_dof_dim = local_dof_symgroup_rep.at(0).at(b).cols();
    site_index_to_basis_index[site_index] = total_dim;
    total_dim += site_dof_dim;
  }

  Index n = static_cast<Index>(group.size());
  std::vector<Eigen::MatrixXd> result(n);
  std::vector<xtal::SymOp> element;
  if (make_symgroup_flag) {
    element.resize(n, xtal::SymOp::identity());
  }

  auto worker = [&](Index start, Index end, Index thread_id) {
    // One full-space temporary per thread — not stored after projection
    Eigen::MatrixXd trep(total_dim, total_dim);
    Eigen::MatrixXd temp;
    for (Index i = start; i < end; ++i) {
      config::SupercellSymOp const &supercell_symop = group[i];
      trep.setZero();
      for (Index site_index : site_indices) {
        Index to_site_index = site_index;
        Index row = site_index_to_basis_index.find(to_site_index)->second;

        Index from_site_index = supercell_symop.permute_index(site_index);
        auto col_it = site_index_to_basis_index.find(from_site_index);
        if (col_it == site_index_to_basis_index.end()) {
          throw std::runtime_error(
              "Error in make_projected_local_dof_matrix_rep: Input group "
              "includes permutations between selected and unselected sites.");
        }
        Index col = col_it->second;

        Index from_site_b =
            supercell.unitcellcoord_index_converter(from_site_index)
                .sublattice();
        Index prim_factor_group_index =
            supercell_symop.prim_factor_group_index();
        Eigen::MatrixXd const &U =
            local_dof_symgroup_rep.at(prim_factor_group_index).at(from_site_b);
        trep.block(row, col, U.rows(), U.cols()) = U;
      }
      // Project to subspace immediately; trep is reused next iteration
      temp.noalias() = trep * basis;
      result[i].noalias() = basis_inv * temp;

      if (make_symgroup_flag) {
        element[i] = supercell_symop.to_symop();
      }
    }
  };

  threaded_run(n, worker);

  if (make_symgroup_flag) {
    std::multiplies<xtal::SymOp> multiply_f;
    xtal::SymOpPeriodicCompare_f equal_to_f(
        supercell.superlattice.superlattice(), xtal_tol);
    symgroup = std::make_shared<config::SymGroup const>(
        group::make_group(element, multiply_f, equal_to_f));
  }

  return result;
}

/// \brief Implementation for make_dof_space_rep and make_dof_space_symmetry
std::pair<std::vector<Eigen::MatrixXd>, std::shared_ptr<config::SymGroup const>>
make_dof_space_rep_impl(std::vector<config::SupercellSymOp> const &group,
                        clexulator::DoFSpace const &dof_space,
                        bool make_symgroup) {
  std::shared_ptr<config::SymGroup const> symgroup;

  Eigen::MatrixXd const &basis = dof_space.basis;
  double tol = 1e-10;
  bool project = !basis.isIdentity(tol);

  if (dof_space.is_global) {
    // Global DoF matrices are small (e.g. 6×6 for strain), build all at once
    std::vector<Eigen::MatrixXd> fullspace_rep =
        config::make_global_dof_matrix_rep(group, dof_space.dof_key, symgroup,
                                           make_symgroup);
    if (!project) {
      return std::make_pair(std::move(fullspace_rep), symgroup);
    }
    Eigen::MatrixXd const &basis_inv = dof_space.basis_inv;
    Index const n = static_cast<Index>(fullspace_rep.size());
    std::vector<Eigen::MatrixXd> dof_space_rep(n);
    auto worker = [&fullspace_rep, &dof_space_rep, &basis_inv, &basis](
                      Index start, Index end, Index thread_id) {
      Eigen::MatrixXd temp;
      for (Index i = start; i < end; ++i) {
        temp.noalias() = fullspace_rep[i] * basis;
        dof_space_rep[i].noalias() = basis_inv * temp;
      }
    };
    threaded_run(n, worker);
    return std::make_pair(std::move(dof_space_rep), symgroup);

  } else {
    if (!dof_space.sites.has_value()) {
      throw std::runtime_error(
          "Error in make_dof_space_rep with local DoF: no DoFSpace sites");
    }

    if (!project) {
      // Identity basis: return full-space rep directly
      std::vector<Eigen::MatrixXd> fullspace_rep =
          config::make_local_dof_matrix_rep(group, dof_space.dof_key,
                                            *dof_space.sites, symgroup,
                                            make_symgroup);
      return std::make_pair(std::move(fullspace_rep), symgroup);
    } else {
      // Non-identity basis: build and project one matrix at a time to avoid
      // storing all N full-space matrices simultaneously (peak memory
      // reduction from O(N * dim^2) to O(threads * dim^2)).
      std::vector<Eigen::MatrixXd> dof_space_rep =
          make_projected_local_dof_matrix_rep(
              group, dof_space.dof_key, *dof_space.sites, symgroup,
              make_symgroup, basis, dof_space.basis_inv);
      return std::make_pair(std::move(dof_space_rep), symgroup);
    }
  }
}

}  // namespace

/// \brief Make the matrix representation of `group` that describes the
///     transformation of values in the basis of the given DoFSpace
///
/// \param group The group that is to be represented (this may be larger than a
///     crystallographic factor group)
/// \param dof_space The DoFSpace. May be any type.
///
/// \returns matrix_rep The matrix representation of `group` which transforms
///     values in the basis of `dof_space`.
///
std::vector<Eigen::MatrixXd> make_dof_space_rep(
    std::vector<config::SupercellSymOp> const &group,
    clexulator::DoFSpace const &dof_space) {
  return make_dof_space_rep_impl(group, dof_space, false).first;
}

/// \brief Make the matrix representation of `group` that describes the
///     transformation of values in the basis of the given DoFSpace and make
///     the corresponding SymGroup
std::pair<std::vector<Eigen::MatrixXd>, std::shared_ptr<config::SymGroup const>>
make_dof_space_symmetry(std::vector<config::SupercellSymOp> const &group,
                        clexulator::DoFSpace const &dof_space) {
  return make_dof_space_rep_impl(group, dof_space, true);
}

/// \brief Make a SymGroup from a list of SupercellSymOp
///
/// \brief group The group as a list of SupercellSymOp
///
/// \returns symgroup The SymGroup corresponding to `group`. This is a head
/// group, it is not a subgroup of the prim factor group because it can
/// include unit cell translations.
std::shared_ptr<SymGroup const> make_symgroup(
    std::vector<SupercellSymOp> const &group) {
  if (group.size() == 0) {
    throw std::runtime_error("Error in make_symgroup: group has size==0.");
  }

  std::shared_ptr<Supercell const> supercell = group.begin()->supercell();

  std::vector<xtal::SymOp> element;
  for (auto const &supercell_symop : group) {
    element.push_back(supercell_symop.to_symop());
  }
  std::multiplies<SymOp> multiply_f;
  xtal::SymOpPeriodicCompare_f equal_to_f(
      supercell->superlattice.superlattice(),
      supercell->prim->basicstructure->lattice().tol());
  return std::make_shared<SymGroup const>(
      group::make_group(element, multiply_f, equal_to_f));
}

/// \brief Make a SymGroup from a list of SupercellSymOp
///
/// \brief group The group as a list of SupercellSymOp
/// \brief point_group If true, make a point group (set translations to zero
///     and remove repeated elements). Otherwise, keep all elements as provided.
///
/// \returns symgroup The SymGroup corresponding to `group`. This is a head
/// group, it is not a subgroup of the prim factor group because it can
/// include unit cell translations.
std::shared_ptr<SymGroup const> make_symgroup_v2(
    std::vector<SupercellSymOp> const &group, bool point_group) {
  if (!point_group) {
    return make_symgroup(group);
  }
  if (group.size() == 0) {
    throw std::runtime_error(
        "Error in make_symgroup with point_group=true: group has size==0.");
  }
  Supercell const &supercell = *group.begin()->supercell();
  Prim const &prim = *supercell.prim;
  auto const &factor_group_element = prim.sym_info.factor_group->element;
  xtal::Lattice const &prim_lattice = prim.basicstructure->lattice();
  double xtal_tol = prim_lattice.tol();

  // Get prim factor group indices for the point group of `group`
  std::set<Index> prim_factor_group_indices;
  std::vector<xtal::SymOp> point_ops;
  xtal::SymOpCompare_f is_equal(xtal_tol);
  for (SupercellSymOp const &supercell_symop : group) {
    Index prim_fg_index = supercell_symop.prim_factor_group_index();
    if (prim_factor_group_indices.count(prim_fg_index)) {
      continue;
    }

    xtal::SymOp point_op = make_point_op(factor_group_element[prim_fg_index]);
    auto it = std::find_if(point_ops.begin(), point_ops.end(),
                           [&](xtal::SymOp const &other_point_op) {
                             return is_equal(point_op, other_point_op);
                           });
    if (it == point_ops.end()) {
      point_ops.push_back(point_op);
    }
  }

  std::multiplies<SymOp> multiply_f;
  xtal::SymOpPeriodicCompare_f equal_to_f(prim_lattice, xtal_tol);
  return std::make_shared<SymGroup const>(
      group::make_group(point_ops, multiply_f, equal_to_f));
}

/// \brief Construct SupercellSymOpFunctors from a supercell
SupercellSymOpFunctors::SupercellSymOpFunctors(
    std::shared_ptr<Supercell const> const &supercell) {
  N_T = supercell->unitcell_index_converter.total_sites();
  Index n_f = static_cast<Index>(
      supercell->sym_info.factor_group->multiplication_table.size());
  N_G = n_f * N_T;

  Index n_t = N_T;
  mult = [supercell, n_t](Index i, Index j) -> Index {
    Index f1 = i / n_t;
    Index t1 = i % n_t;
    Index f2 = j / n_t;
    Index t2 = j % n_t;

    SymGroup const &fg = *supercell->sym_info.factor_group;
    Index f3 = fg.multiplication_table[f1][f2];

    Index prim_fg_idx = fg.head_group_index[f1];
    Eigen::Matrix3l const &R_frac =
        supercell->prim->sym_info.unitcellcoord_symgroup_rep[prim_fg_idx]
            .point_matrix;

    UnitCell uc1 = supercell->unitcell_index_converter(t1);
    UnitCell uc2 = supercell->unitcell_index_converter(t2);
    UnitCell uc3{R_frac * uc2 + uc1};
    Index t3 = supercell->unitcell_index_converter(uc3);

    return f3 * n_t + t3;
  };

  inv = [supercell, n_t](Index i) -> Index {
    Index f = i / n_t;
    Index t = i % n_t;

    SymGroup const &fg = *supercell->sym_info.factor_group;
    Index f_inv = fg.inverse_index[f];

    Index prim_fg_idx_inv = fg.head_group_index[f_inv];
    Eigen::Matrix3l const &R_frac_inv =
        supercell->prim->sym_info.unitcellcoord_symgroup_rep[prim_fg_idx_inv]
            .point_matrix;

    UnitCell uc_t = supercell->unitcell_index_converter(t);
    UnitCell uc_inv{R_frac_inv * (-uc_t)};
    Index t_inv = supercell->unitcell_index_converter(uc_inv);

    return f_inv * n_t + t_inv;
  };
}

}  // namespace config
}  // namespace CASM
