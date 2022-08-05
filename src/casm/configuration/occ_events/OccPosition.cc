#include "casm/configuration/occ_events/OccPosition.hh"

#include "casm/configuration/occ_events/OccEventRep.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/UnitCellCoordRep.hh"
#include "casm/misc/algorithm.hh"

namespace CASM {
namespace occ_events {

OccPosition::OccPosition(bool _is_in_resevoir, bool _is_atom,
                         xtal::UnitCellCoord const &_integral_site_coordinate,
                         Index _occupant_index, Index _atom_position_index)
    : is_in_resevoir(_is_in_resevoir),
      is_atom(_is_atom),
      integral_site_coordinate(_integral_site_coordinate),
      occupant_index(_occupant_index),
      atom_position_index(_atom_position_index) {}

/// \brief Translate the OccPosition by a UnitCell translation
OccPosition &OccPosition::operator+=(xtal::UnitCell trans) {
  this->integral_site_coordinate += trans;
  return *this;
}

/// Compare (integral_site_coordinate, occupant_index, atom_position_index)
bool OccPosition::operator<(OccPosition const &B) const {
  // sort molecules < atomic components < resevoir molecules
  if (this->is_in_resevoir != B.is_in_resevoir) {
    return !this->is_in_resevoir;
  }
  if (this->is_atom != B.is_atom) {
    return !this->is_atom;
  }

  // if both in resevoir:
  if (this->is_in_resevoir) {
    return this->occupant_index < B.occupant_index;
  }

  // if both molecules:
  if (!this->is_atom) {
    return std::make_tuple(this->integral_site_coordinate,
                           this->occupant_index) <
           std::make_tuple(B.integral_site_coordinate, B.occupant_index);
  }
  // if both atomic components of molecules:
  return std::make_tuple(this->integral_site_coordinate, this->occupant_index,
                         this->atom_position_index) <
         std::make_tuple(B.integral_site_coordinate, B.occupant_index,
                         B.atom_position_index);
}

/// \brief Apply SymOp to OccPosition
OccPosition &apply(OccEventRep const &rep, OccPosition &occ_position) {
  if (occ_position.is_in_resevoir) {
    return occ_position;
  } else {
    Index b = occ_position.integral_site_coordinate.sublattice();
    Index i = occ_position.occupant_index;
    occ_position.occupant_index = rep.occupant_rep[b][i];

    if (occ_position.is_atom) {
      Index p = occ_position.atom_position_index;
      occ_position.atom_position_index = rep.atom_position_rep[b][i][p];
    }

    apply(rep.unitcellcoord_rep, occ_position.integral_site_coordinate);
  }
  return occ_position;
}

/// \brief Apply SymOp to OccPosition
OccPosition copy_apply(OccEventRep const &rep, OccPosition occ_position) {
  apply(rep, occ_position);
  return occ_position;
}

}  // namespace occ_events
}  // namespace CASM
