"""A rank's node window on the served path: its values (`served_values`), its commitment through `interface.Window`
(`commit_rank_window`), and the joined record's two fields read on our side: ``ranks`` (`rank_records`, each rank's window
by its H, `window_replay.window_digest`) and ``free`` (`check_free`, refused before the serve registers the record, as
round 1 refuses it). Built: `commit_rank_window`; each other body raises `NotImplementedError` naming the goal."""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from verity.commitments.periods import interface as IF

from verity_vllm.pipeline import window_replay as WRP

if TYPE_CHECKING:
    from verity_vllm.commit import opened as OC

GOAL = "vllm/compiler-wires-period-pieces"
#: what `commit_rank_window` returns: `tp.serving_rows.write`'s keys, then the window's H
KEYS = ("run", "population", "partition_digest", "registration_digest", "seconds", "window")


class Rank(Protocol):
    """A rank's window as `node_window_join.Rank` gives it, read here for its number and its circuit."""

    @property
    def rank(self) -> int: ...

    @property
    def circuit(self) -> IF.Circuit: ...


class Rows(Protocol):
    """A rank window's units' rows, as `node_window_values.Check` gives them: ``inputs``, each input port's
    ``(n, words)`` u16 rows of template ``tid``'s units ``units`` in its parameters' order; ``outputs``, each output
    port's by name."""

    def inputs(self, tid: str, units: Sequence[int]) -> list[np.ndarray]: ...

    def outputs(self, tid: str, units: Sequence[int]) -> dict[str, np.ndarray]: ...


def served_values(rank: Rank, committed: OC.OpenedReader, weights: Any) -> Rows:
    """Rank ``rank.rank``'s node-window values: each unit's outputs from the rank's committed store where serving tapped
    them, else from its Definition on its inputs (`node_window_values.Check`). In a class window
    (`class_window.class_window`) a decode step past the request's stop is such a unit: its rows are its Definition's on
    its wired inputs, the class's, which serving did not run."""
    raise NotImplementedError(f"{GOAL}: no serve fills a rank's node window from its committed store yet")


def _held(circuit: IF.Circuit, tid: str, port: str, n: int) -> bool:
    """Whether some row of ``tid``'s input ``port`` over its ``n`` units is neither wired nor a registered read, so the
    period holds the port's rows (`interface.Window.record`): a boundary read's and an opening's rows among them."""
    k = int(circuit.cuts.get(tid, {}).get(port, 1))
    filled = sum(x.reader_units * x.cut[1] for x in (*circuit.wires, *IF.committed_reads(circuit.registered))
                 if (x.reader, x.port) == (tid, port))
    return filled != n * k


def commit_rank_window(rank: Rank, values: Rows, *, run: str, law: Mapping[str, Any], cap: int,
                       registered: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Commit ``rank``'s node window as ``interface.Window(rank.circuit, run, law, cap, registered=registered)`` and
    write ``partition.json``, ``circuit.json`` and ``window.json`` under ``out``. The circuit is a request's window
    (`node_window_reads.rank_window`) or a class's (`class_window.class_window`), its boundary reads in it, so its H is
    the one the join's ``ranks`` names. ``run`` is the string `tp.serving_rows.run_of` gives, the one
    `tp.serving_rows.write` hands `serving_rows.after_commit`; the window states no ``context`` of its own. ``law`` is
    the partition file's sampling design (`window_sampling_design.sampling_design_of`). ``values`` gives each unit's
    rows (`served_values`); a port every row of which is wired or a registered read is recorded by reference, every
    other in full. ``registered`` holds the committed value of each parameter its registered reads and values read
    (`expert_rows.commit_bnk`'s among them), and no other. Returns `KEYS`: `tp.serving_rows.write`'s, ``run``,
    ``population`` (its units over every template), ``partition_digest``, ``registration_digest`` and ``seconds``, and
    ``window``, its H (`window_replay.window_digest` of its record).

    Refused (ValueError), by name and before anything is written: a circuit with ``ingress`` (a rank window's prompt
    is a boundary read); a ``cap`` that is not a positive count; ``registered`` values missing a parameter it reads,
    naming one it doesn't register, or committed in another row shape than its reads read (`interface.Window`). A
    window whose tables' supplied values aren't in ``registered`` is refused by `interface.Window.close`, before it
    writes too. Its registration is a served window's, by the registrar `serving_rows.after_commit` registers under."""
    from verity_vllm.pipeline import serving_rows as SRW

    t0, c, what = time.perf_counter(), rank.circuit, f"rank {rank.rank}"
    if c.ingress:
        raise ValueError(f"{what}: ingress {[x.doc() for x in c.ingress]}: a rank window declares none; its "
                         "prompt's rows are a boundary read (`node_window_reads.boundary_reads`)")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"{what}: cap {cap!r}: a window is committed at the positive cap its joined record's "
                         "ranks were stated at (`node_window_join.ranks_of`)")
    reg = c.registered or IF.Registered({})
    want = {x.param for x in reg.reads} | {v for v, _n, _b in IF.program_values(reg)}
    if missing := sorted(want - set(registered)):
        raise ValueError(f"{what}: registered: no committed value of {missing}, which its reads read")
    if extra := sorted(set(registered) - set(reg.params)):
        raise ValueError(f"{what}: registered: {extra} are no parameters its circuit registers")
    try:
        win = IF.Window(c, run, dict(law), cap, registered=dict(registered))
    except ValueError as e:
        raise ValueError(f"{what}: registered: {e}") from None
    for (fn, n), tid in zip(c.templates, c.ids):
        lay, units = IF.layout(fn), range(n)
        ins, outs = values.inputs(tid, units), values.outputs(tid, units)
        win.record(fn, [_elements(a, w) if _held(c, tid, p, n) else None
                        for a, (p, _), w in zip(ins, lay.rows, lay.widths)],
                   [_elements(outs[p], w) for (p, _), w in zip(lay.outs, lay.widths[len(lay.rows):])])
    committed = win.close()
    d = Path(out)
    res = win.write(committed, d, {}, registrar=SRW.R)
    res["seconds"]["hook_s"] = round(time.perf_counter() - t0, 3)
    (d / "window.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (d / "partition.json").write_text(json.dumps(SRW.partition_doc(win.window, {}), indent=1) + "\n")
    window = WRP.window_digest(json.loads((d / "circuit.json").read_bytes()))
    return {**{k: res[k] for k in KEYS[:-1]}, "window": window}


def _elements(words: np.ndarray, width: int) -> np.ndarray:
    """``(n, words)`` u16 rows as ``width``-bit elements, the same little-endian bytes (`interface.words`' inverse)."""
    return np.ascontiguousarray(words, dtype="<u2").view(f"<u{width // 8}")


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


__all__ = ["GOAL", "KEYS", "Rows", "check_free", "commit_rank_window", "rank_records", "served_values"]
