from typing import Optional

import libcasm.clusterography
import libcasm.configuration
import libcasm.configuration.io as config_io
import libcasm.occ_events

from ._methods import (
    make_occevent_equivalents_generators,
)


def make_equivalents_info_dict(
    prim: libcasm.configuration.Prim,
    prototype: libcasm.clusterography.Cluster,
    phenomenal_clusters: list[libcasm.clusterography.Cluster],
    equivalent_generating_op_indices: list[int],
) -> dict:
    R"""Generate minimal equivalents_info.json data

    Returns
    -------
    equivalents_info_dict: dict
        The equivalents info provides the phenomenal cluster and local-cluster
        orbits for all symmetrically equivalent local-cluster expansions, and the
        indices of the factor group operations used to construct each equivalent
        local cluster expansion from the prototype local-cluster expansion. When
        there is an orientation to the local-cluster expansion this information
        allows generating the proper diffusion events, etc. from the prototype.

        A description of the generated cluster expansion basis set, as described
        `here <TODO>`_.
    """  # noqa

    # Equivalents info, prototype
    equivalents_info = {}

    # if self._phenomenal is None:
    #     return equivalents_info
    # if len(self.orbit_matrix_rep_builders) == 0:
    #     raise Exception(
    #         "Error in ClusterFunctionsBuilder.equivalents_info_dict: No orbits"
    #     )

    # Write prim factor group info
    equivalents_info[
        "factor_group"
    ] = config_io.symgroup_to_dict_with_group_classification(prim, prim.factor_group)

    # Write equivalents generating ops
    # (actually prim factor group indices of those ops and the
    #  translations can be figured out from the phenomenal cluster)
    equivalents_info["equivalent_generating_ops"] = equivalent_generating_op_indices

    # Write prototype orbits info
    tmp = {}
    tmp["phenomenal"] = prototype.to_dict(xtal_prim=prim.xtal_prim)
    tmp["prim"] = prim.to_dict()
    tmp["orbits"] = None
    equivalents_info["prototype"] = tmp

    # Write equivalent orbits info
    equivalents_info["equivalents"] = []
    for i_equiv, cluster in enumerate(phenomenal_clusters):
        tmp = {}
        tmp["phenomenal"] = cluster.to_dict(xtal_prim=prim.xtal_prim)
        tmp["prim"] = prim.to_dict()
        tmp["orbits"] = None
        equivalents_info["equivalents"].append(tmp)

    return equivalents_info


class OccEventPrimSymInfo:
    """Information about OccEvent that are equivalent with respect to a prim factor
    group"""

    def __init__(
        self,
        prim: libcasm.configuration.Prim,
        prototype_event: libcasm.occ_events.OccEvent,
        phenomenal_clusters: Optional[list[libcasm.clusterography.Cluster]] = None,
        equivalent_generating_op_indices: Optional[list[int]] = None,
    ):
        """

        Parameters
        ----------
        prim: libcasm.configuration.Prim
            The prim.
        prototype_event: libcasm.occ_events.OccEvent
            The prototype event. The underlying cluster `prototype_event.cluster` must
            be chosen from one of the equivalents that is generated by
            :func:`~libcasm.clusterography.make_periodic_orbit` using the prim factor
            group.
        phenomenal_clusters: Optional[list[libcasm.clusterography.Cluster]]
            The clusters at which the equivalent events are located.
            By default, the phenomenal clusters are generated from `event` and
            the prim factor group. May be provided to ensure consistency of
            `(unitcell_index, equivalent_index)` event positions with those used by a
            local clexulator.

            Note that these clusters are expected to be with the
            site order as transformed from the prototype cluster by the equivalents
            generating factor group operations, without sites being sorted after
            transformation.

        equivalent_generating_op_indices: Optional[list[int]]
            The indices of prim factor group operations that generate the equivalent
            events from the prototype event, up to a translation. By default, the
            indices are generated from `event` and the prim factor group. May be
            provided to ensure consistency of `(unitcell_index, equivalent_index)` event
            positions with those used by a local clexulator.
        """

        # generate phenomenal clusters and equivalent generating factor group operations
        # if not provided
        if phenomenal_clusters is None or equivalent_generating_op_indices is None:
            if (
                phenomenal_clusters is not None
                or equivalent_generating_op_indices is not None
            ):
                raise ValueError(
                    "The arguments `phenomenal_clusters` and "
                    "`equivalent_generating_op_indices` must be both provided or None"
                )

            ops, indices, site_reps = make_occevent_equivalents_generators(
                prototype_event=prototype_event,
                prim=prim,
            )

            equivalent_generating_op_indices = indices
            cluster = prototype_event.cluster()
            phenomenal_clusters = [rep * cluster for rep in site_reps]

        self.prim = prim
        """libcasm.configuration.Prim: The prim."""

        self.prototype_event = prototype_event
        """libcasm.occ_events.OccEvent: The prototype event."""

        self.phenomenal_clusters = phenomenal_clusters
        """list[libcasm.clusterography.Cluster]: The phenomenal clusters at which
        local-cluster basis functions about the symmetrically equivalent events are
        generated.
        
        Note that these clusters are expected to be as transformed from the prototype 
        cluster by the equivalent generating factor group operations, without sites 
        being sorted after transformation.
        """

        self.equivalent_generating_op_indices = equivalent_generating_op_indices
        """list[int]: The indices of prim factor group operations that generate
        the equivalent events from the first event, up to a translation."""

        ### Generate phenomenal OccEvent consistent with local clexulator ###
        occevent_symgroup_rep = libcasm.occ_events.make_occevent_symgroup_rep(
            prim.factor_group.elements, prim.xtal_prim
        )

        self.occevent_symgroup_rep = occevent_symgroup_rep
        """list[libcasm.occ_events.OccEvent]: Group representation for transforming
        OccEvent by prim factor group operations."""

        occevent_orbit = []
        translations = []
        for i, generating_op_index in enumerate(equivalent_generating_op_indices):
            tmp = (
                occevent_symgroup_rep[generating_op_index] * prototype_event
            ).standardize()
            trans = (
                phenomenal_clusters[i].sorted()[0].unitcell()
                - tmp.cluster().sorted()[0].unitcell()
            )
            translations.append(trans)
            occevent_orbit.append(tmp + trans)

        self.translations = translations
        """list[np.ndarray]: The translations, in fractional coordinates, applied after
        the equivalent generating factor group operations, that result in the
        equivalent events."""

        self.events = occevent_orbit
        """list[libcasm.occ_events.OccEvent]: The list of OccEvent that are equivalent
        with respect to the prim factor group, in the order specified by applying 
        `equivalent_generating_op_indices` to `prototype_event` and at the translations
        specified by the phenomenal clusters."""

        invariant_groups = []
        for event in self.events:
            invariant_groups.append(
                libcasm.occ_events.make_occevent_group(
                    occ_event=event,
                    group=prim.factor_group,
                    lattice=prim.xtal_prim.lattice(),
                    occevent_symgroup_rep=self.occevent_symgroup_rep,
                )
            )
        self.invariant_groups = invariant_groups
        """list[libcasm.sym_info.SymGroup]: The group `invariant_groups[i]` is the
        subgroup of the prim factor group that leaves `events[i]` invariant."""

    @staticmethod
    def from_data(
        prototype_event_data: dict,
        equivalents_info_data: dict,
        prim: libcasm.configuration.Prim,
        system: libcasm.occ_events.OccSystem,
    ):
        """
        Construct from the contents of "event.json" and "equivalents_info.json" files

        Parameters
        ----------
        prototype_event_data : dict
            The serialized prototype event data
        equivalents_info_data : dict
            The serialized equivalents info. The prototype and phenomenal clusters
            and equivalents generating op indices are needed; the local clusters are
            not needed.
        prim : libcasm.configuration.Prim
            The :class:`libcasm.configuration.Prim`
        system : libcasm.occ_events.OccSystem
            The :class:`libcasm.occ_events.OccSystem`

        Returns
        -------
        prim_sym_info : OccEventPrimSymInfo
            The OccEventPrimSymInfo object
        """
        prototype_event = libcasm.occ_events.OccEvent.from_dict(
            data=prototype_event_data,
            system=system,
        )
        (
            phenomenal_clusters,
            equivalent_generating_op_indices,
        ) = libcasm.clusterography.equivalents_info_from_dict(
            data=equivalents_info_data,
            xtal_prim=prim.xtal_prim,
        )
        return OccEventPrimSymInfo(
            prim=prim,
            prototype_event=prototype_event,
            phenomenal_clusters=phenomenal_clusters,
            equivalent_generating_op_indices=equivalent_generating_op_indices,
        )

    def to_data(
        self,
        system: libcasm.occ_events.OccSystem,
    ):
        """
        Serialize the event and equivalents info

        Parameters
        ----------
        system : libcasm.occ_events.OccSystem
            The :class:`libcasm.occ_events.OccSystem`

        Returns
        -------
        prototype_event_data : dict
            The serialized prototype event data
        equivalents_info_data : dict
            The serialized equivalents info. Only the prototype and phenomenal
            clusters and equivalents generating op indices are included.
        """
        prototype_event_data = self.prototype_event.to_dict(
            system=system,
        )
        equivalents_info_data = make_equivalents_info_dict(
            prim=self.prim,
            phenomenal_clusters=self.phenomenal_clusters,
            equivalent_generating_op_indices=self.equivalent_generating_op_indices,
        )

        return (prototype_event_data, equivalents_info_data)

    def coordinate(
        self,
        occ_event: libcasm.occ_events.OccEvent,
        supercell: libcasm.configuration.Supercell,
    ):
        """
        Determine the coordinates `(unitcell_index, equivalent_index)` of a OccEvent,
        with `equivalent_index` referring to `self.events`, in a given supercell.

        Parameters
        ----------

        occ_event : libcasm.occ_events.OccEvent
            Input OccEvent, to find the coordinates of
        supercell : libcasm.configuration.Supercell
            The supercell in which the OccEvent is located

        Returns
        -------
        (unitcell_index, equivalent_index) : tuple[int, int]
            The coordinates (unitcell_index, equivalent_index) of the
            input OccEvent, with `equivalent_index` referring to `self.events`.

        Raises
        ------
        Exception
            If no match can be found, indicating the input OccEvent is not equivalent
            to any of the OccEvent in `self.events` up to a translation.
        """
        return libcasm.occ_events.get_occevent_coordinate(
            occ_event=occ_event,
            phenomenal_occevent=self.events,
            unitcell_index_converter=supercell.unitcell_index_converter,
        )
