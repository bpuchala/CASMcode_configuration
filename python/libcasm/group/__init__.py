"""Group theory utilities"""

import time
import typing

import alive_progress

from ._get_subgroup_orbits import (
    get_all_subgroup_orbits,
    get_cyclic_subgroup_orbits,
)
from ._Group import (
    Group,
)
from ._group import (
    GenericGroup,
    Subset,
)


def _subset_all_subgroups(
    self,
    n_subtrees: int = 100,
    progress: typing.Union[str, typing.Callable] = "alive",
    method: str = "generator_search",
):
    """Return all subgroups of this subset

    Notes
    -----

    The "generator_search" method (deprecated alias: "generator_search"):

    1. Finds all cyclic subgroups and stores the elements which generate unique
       cyclic subgroups as the candidate generators.
    2. Performs a depth-first search over combinations of the candidate
       generators to find all subgroups.

    This is a multithreaded method. The combinations of candidate generators is
    searched like a tree. The tree can be split into subtrees so that the search can
    be performed in parallel across multiple threads.

    There are some synchronization costs that come from reading and writing a common
    set of unique subgroups and splitting the search tree. The number of subtrees can
    be controlled with the `n_subtrees` parameter to balance the parallelization and
    synchronization costs. The number of threads used may be controlled through
    :func:`libcasm.casmglobal.set_max_threads`. Generally it is preferable to have more
    subtrees than threads to avoid waiting on subtrees that take longer to search.

    The "alive" progress bar gives an estimated time remaining based on the time taken
    to search previous subtrees, so it may be useful to have enough subtrees to get a
    rough estimate of the time remaining. However, the estimate is usually longer than
    the actual time remaining, because the initial subtrees are likely to be among the
    longest to search.

    The "group_extension" method (deprecated alias: "normal_subgroup") exploits the
    group extension structure G = T.F, where T = {0,...,N_translations-1} is the
    normal translation subgroup and F = G/T is the quotient. It enumerates all valid
    sections ξ: K → T/S for each pair (K ≤ F, S ≤ T) with K normalizing S, and
    closes the corresponding generators in G. This finds all subgroups for both
    symmorphic and non-symmorphic groups, and is faster for large supercell groups.
    Requires ``N_translations`` to be set on the Subset.

    The "cyclic_join" method iteratively joins cyclic subgroups until all subgroups
    have been found. This is an older, simpler algorithm that is less efficient for
    large groups.

    Parameters
    ----------
    n_subtrees: int = 100
        The number of subtrees to divide the search tree into (used by
        method="generator_search" only).
    progress: Union[str, Callable] = "alive"
        Indicates the type of progress reporting to use (used by
        method="generator_search" only). The options are:

        - "alive" (default): a live progress bar is shown.
        - "plain": use the default C++ stdout progress reporting.
        - "none": no progress is reported.

        If a callable is provided, it is used as a callback function to report
        progress. A callback function must take two int arguments: the number of
        finished subtrees and the total number of subgroups found so far.
        This is called each time a subtree is finished.

        .. code-block:: python

            def progress_f(n_subtrees_completed: int, subgroups_size: int) -> None:

    method: str = "generator_search"
        Which algorithm to use:

        - "generator_search" (default): multithreaded depth-first search
          over generator combinations. General purpose. Deprecated alias:
          "generator_search".
        - "group_extension": exploits the G = T.F extension structure.
          Requires ``N_translations`` to be set on the Subset. Deprecated
          alias: "normal_subgroup".
        - "cyclic_join": iteratively joins cyclic subgroups (older, simpler
          algorithm; less efficient for large groups).

    Returns
    -------
    subgroups: list[Subset]
        The subgroups.
    """
    if method in ("group_extension", "normal_subgroup", "cyclic_join"):
        return Subset._all_subgroups(self, method=method)

    # method == "generator_search" or "generator_search"
    if method not in ("generator_search", "generator_search"):
        raise ValueError(
            f"all_subgroups: unknown method '{method}'. "
            "Valid options: 'generator_search', 'group_extension', 'cyclic_join'."
        )

    if Subset._has_all_subgroups(self):
        return Subset._all_subgroups(self)

    if progress == "alive":
        with alive_progress.alive_bar(
            n_subtrees,
            manual=True,
            monitor="{percent:.0%}",
            stats="(eta: {eta})",
            stats_end=False,
            length=30,
            force_tty=True,
        ) as bar:
            bar.text = "#Subgroups: 0"

            # The progress callback is called once before the first subtree is searched,
            # so we use a flag to avoid updating the alive-progress bar at that time.
            first = True

            def progress_callback(
                n_finished_tasks: int,
                subgroups_count: int,
            ) -> None:  # Update the dynamic text with the latest sum
                nonlocal first, n_subtrees

                bar.text = f"#Subgroups: {subgroups_count}"
                bar(n_finished_tasks / n_subtrees)

            subgroups = Subset._all_subgroups(
                self, n_subtrees, "generator_search", progress_callback
            )

    else:

        if progress == "plain":
            progress = None

        elif progress == "none":

            def _no_progress_callback(
                n_finished_tasks: int,
                subgroups_count: int,
            ) -> None:
                pass

            progress = _no_progress_callback

        elif not callable(progress):
            raise ValueError("progress must be 'alive', 'plain', 'none', or a callable")

        subgroups = Subset._all_subgroups(
            self, n_subtrees, "generator_search", progress
        )

        if progress is None:
            print("", flush=True)

    return subgroups


# Attach the helper as a method on the imported Subset class
Subset.all_subgroups = _subset_all_subgroups
