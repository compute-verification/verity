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
    them, else from its Definition on its inputs (`node_window_values.Check`)."""
    raise NotImplementedError(f"{GOAL}: no serve fills a rank's node window from its committed store yet")


def commit_rank_window(rank: NWJ.Rank, values: NWV.Values, *, run: Mapping[str, Any], cap: int,
                       registered: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Commit ``rank``'s node window through `interface.Window` and write ``partition.json``, ``circuit.json`` and
    ``window.json`` under ``out``: the circuit `node_window_reads.rank_window` builds, its boundary reads in it, so its H
    is the one the join's ``ranks`` names. Returns its ``run``, ``partition_digest``, ``registration_digest`` and
    ``window`` (its H, `window_replay.window_digest` of its record)."""
    raise NotImplementedError(f"{GOAL}: no serve commits a rank's node window yet")


def rank_records(joined: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> IF.Ranks:
    """The joined record's ``ranks``, ``{prefix, of: [{window, units, wires}]}`` (`interface.ranks_of`), refused
    (ValueError) unless ``records`` holds one record a rank, in rank order, rank r's H (`window_replay.window_digest`: its
    record less its run fields) its entry's ``window``, its units of each template its entry's ``units`` and its count of
    wires its entry's ``wires``."""
    raise NotImplementedError(f"{GOAL}: the served path commits no rank window yet, so no rank record is read")


def check_free(joined: Mapping[str, Any]) -> None:
    """Refuse (ValueError) a joined record that names free rows (`interface.free_of`: each ``{reader, port, at, units,
    kind}``, with ``cut`` ``[c0, C]`` where it is some of its port's rows), naming each row's kind, before the serve
    registers it: round 1 refuses such a record."""
    raise NotImplementedError(f"{GOAL}: the served path registers no joined record yet")


__all__ = ["GOAL", "check_free", "commit_rank_window", "rank_records", "served_values"]
