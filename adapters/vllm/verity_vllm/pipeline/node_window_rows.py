"""A rank's node window on the served path: its values (`served_values`), its commitment through `interface.Window`
(`commit_rank_window`), and the joined record's two fields read on our side: ``ranks`` (`rank_records`, each rank's window
by its H, `window_replay.window_digest`) and ``free`` (`check_free`, refused before the serve registers the record, as
round 1 refuses it). Not built: each body raises `NotImplementedError` naming the goal."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verity.commitments.periods import interface as IF

if TYPE_CHECKING:
    from verity_vllm.commit import opened as OC
    from verity_vllm.program.predict import node_window_join as NWJ
    from verity_vllm.program.predict import node_window_values as NWV

GOAL = "vllm/compiler-wires-period-pieces"


def served_values(rank: NWJ.Rank, committed: OC.OpenedReader, weights: Any) -> NWV.Values:
    """Rank ``rank.rank``'s node-window values: each unit's outputs from the rank's committed store where serving tapped
    them, else from its Definition on its inputs (`node_window_values.Check`). In a class window
    (`class_window.class_window`) a decode step past the request's stop is such a unit: its rows are its Definition's on
    its wired inputs, the class's, which serving did not run."""
    raise NotImplementedError(f"{GOAL}: no serve fills a rank's node window from its committed store yet")


def commit_rank_window(rank: NWJ.Rank, values: NWV.Values, *, run: str, law: Mapping[str, Any], cap: int,
                       registered: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Commit ``rank``'s node window as ``interface.Window(rank.circuit, run, law, cap, registered=registered)`` and
    write ``partition.json``, ``circuit.json`` and ``window.json`` under ``out``. The circuit is a request's window
    (`node_window_reads.rank_window`) or a class's (`class_window.class_window`), its boundary reads in it, so its H is
    the one the join's ``ranks`` names. ``run`` is the string `tp.serving_rows.run_of` gives, the one
    `tp.serving_rows.write` hands `serving_rows.after_commit`; the window states no ``context`` of its own. ``law`` is
    the partition file's sampling design (`window_sampling_design.sampling_design_of`). Returns
    `tp.serving_rows.write`'s keys, ``run``, ``population`` (its units over every template), ``partition_digest``,
    ``registration_digest`` and ``seconds``, and ``window``, its H (`window_replay.window_digest` of its record)."""
    raise NotImplementedError(f"{GOAL}: no serve commits a rank's node window yet")


def rank_records(joined: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> IF.Ranks:
    """The joined record's ``ranks``, ``{prefix, of: [{window, units, wires}]}`` (`interface.ranks_of`), refused
    (ValueError) unless ``records`` holds one record a rank, in rank order, rank r's H (`window_replay.window_digest`:
    its record less its run fields) its entry's ``window``, its units of each template its entry's ``units`` and its
    count of wires its entry's ``wires``. A rank's wires are its own: the joined circuit lists them first, rank by rank,
    and the join's wires after them, coalesced one reader rank at a time (`node_window_join.join`), none of them a
    rank's. ``records`` are the ranks' ``circuit.json`` (`commit_rank_window`); `tp.serving_rows.commit` calls this over
    them before it returns, so before the coin."""
    raise NotImplementedError(f"{GOAL}: the served path commits no rank window yet, so no rank record is read")


def check_free(joined: Mapping[str, Any]) -> None:
    """Refuse (ValueError) a joined record that names free rows (`interface.free_of`: each ``{reader, port, at, units,
    kind}``, with ``cut`` ``[c0, C]`` where it is some of its port's rows), naming each row's kind, before the serve
    registers it: round 1 refuses such a record. `tp.serving_rows.require` calls it on the request's ``joined``, before
    the engine starts."""
    raise NotImplementedError(f"{GOAL}: the served path registers no joined record yet")


__all__ = ["GOAL", "check_free", "commit_rank_window", "rank_records", "served_values"]
