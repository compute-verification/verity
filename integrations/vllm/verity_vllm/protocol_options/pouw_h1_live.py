"""H1's live tables (VLLM-85): the served path's side of `PouwLiveSlots_v1{T}` (`verity.core.protocols.pouw.circuit.live`),
the unit that makes each round's served rows m_s a proved value, one ``live@<s>`` a round.

* **The tables.** With the Commit (`pouw_h1_served.Served`'s ``tap``) a window registers its round tables as values of
  the live Program, under the window Program's digest as its other values (`Served._tree`):

  * round 0's request table ``<prefix>requests@0`` (`live.REQUESTS`): T rows, its entries' headers in admission order
    (index, round, first slot, cap and prompt length, each u32: the first 20 bytes of `RequestEntry.row`, as
    `live.header_words`), then `EMPTY_REQUEST`'s zero header. It is registered as the window opens, after round 0's
    admissions, and its domain binds the window's session, since its root, round 0's ``request_table``, is inside round
    0's request root and so inside the open root (`credit.requests_root`), as ``ids@0``'s is. The prompts' tokens are
    ``ids@0``'s, not its.
  * round s's slot table ``<prefix>slots@<s>`` (`live.SLOTS`): T rows, slot i's leaf (q + 1 for a row of request q, 0
    past the step's batch, `live.slot_words`), registered at round s's close under the open root and listed whole among
    round s's segments (`Served.segments`), so round s's root commits it before its coin (`credit._a_link`).

  Each is its round's table: round 0's ``request_table`` and round s's ``slots`` are their roots, in place of the hiding
  tables (`pouw_h1_requests.request_table`, `slot_table`). Their leaves open in ``live@<s>``'s proof, not one by one:
  `pouw_h1.H1Run.open_request` and `open_slot` refuse them (`Table`). Row j's salt is the run's key's at `nonce`.
* **m_s** is step s's served rows, the scheduler's: its batch's rows (`pouw_h1.H1Run.owners`). Under CUDA graphs a call's
  A rows are padded to the graph's size, past them (`pouw_h1.Call.pad`), and those rows are no slot of the step's.
* **The statements.** ``live@<s>`` is unit s of the live Program (`live.program` of T, the window's S and its prefix), on
  round s's slot table, the request headers and the public m_s: `statement` gives its rows and its ``--registered``
  file (`live.reads_file`), and `verdicts` holds each round's tables to the roots its round committed and its rows to
  `live.live_ok` at m_s.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity.core.protocols.pouw.circuit import live as L
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.commitments import Commitment

from verity_vllm.protocol_options import pouw_h1_requests as RQ

SLOTS, REQUESTS = L.SLOTS, L.REQUESTS


def nonce(base: int, name: str, s: int) -> int:
    """Row 0's salt nonce of a live table, from the window's live nonce ``base`` (`pouw_h1_served.LIVE_NONCE`):
    ``requests@0``'s is ``base``, ``slots@<s>``'s ``base + 1 + s``."""
    return base if name == L.REQUESTS else base + 1 + int(s)


def headers(entries: Sequence[RQ.RequestEntry]) -> list[tuple[int, ...]]:
    """Each entry's header: index, round, first slot, cap and prompt length (`RequestEntry.row`'s first five u32)."""
    return [(e.index, e.round, e.slot, e.max_tokens, len(e.prompt)) for e in entries]


def request_words(entries: Sequence[RQ.RequestEntry], rows: int) -> np.ndarray:
    """``requests@0``'s rows ``(T, 10)`` u16: the entries' headers in admission order, then zero headers."""
    if len(entries) > rows:
        raise ValueError(f"round 0 admits {len(entries)} requests, past its {rows} leaves")
    hs = headers(entries) + [(0,) * L.HEADER] * (rows - len(entries))
    return np.asarray(L.header_words(hs), np.uint16)


def slot_leaves(owners: Sequence[int | None], rows: int) -> list[int]:
    """Round s's slot leaves: q + 1 for a row of request q, in the step's batch order, then 0 (`EMPTY_SLOT`)."""
    got = [0 if o is None else int(o) + 1 for o in owners[:rows]]
    return got + [0] * (rows - len(got))


def slot_words(owners: Sequence[int | None], rows: int) -> np.ndarray:
    """``slots@<s>``'s rows ``(T, 2)`` u16."""
    return np.asarray(L.slot_words(slot_leaves(owners, rows)), np.uint16)


class Table:
    """A live table as `pouw_h1.H1Run` holds a round's request or slot table: its registered value (``registered``) and
    its ``commitment``, and no leaf opened one by one."""

    def __init__(self, registered: Any) -> None:
        self.registered = registered
        self.commitment = Commitment(registered.root, registered.leaves)

    def open(self, position: int) -> Any:
        raise ValueError(f"{self.registered.value} is a live table: its rows open in its round's live@<s> proof, not leaf "
                         f"by leaf (leaf {position})")


def value(served: Any, name: str, s: int = 0) -> str:
    return L.table(served.program.fn.bindings["PREFIX"], name, s)


def table(served: Any, run: Any, name: str, r: int, base: int, session: str) -> Table | None:
    """Round r's live table, registered now in ``served.live``: ``requests@0`` (``name`` `live.REQUESTS`, round 0 only)
    bound to ``session``, or ``slots@<r>`` (`live.SLOTS`) bound to the window's open root. None without the tap, once
    the window is refused, and before its Program and open root exist (a slot table)."""
    if served.tap is None or served.refused or served.program is None or (name == L.REQUESTS and r):
        return None
    if name == L.REQUESTS:
        words, window, bound = request_words(run.admitted.get(0, []), run.rows), None, session
    elif served.root is None:
        return None
    else:
        words, window, bound = slot_words(run.owners.get(r, []), run.rows), served.root, None
    v = value(served, name, r)
    reg = served.live[v] = served._tree(run, v, words, nonce(base, name, r), window, bound)
    return Table(reg)


def m_of(run: Any, s: int) -> int:
    """Step s's served rows m_s (`pouw_h1.H1Run.step_rows`)."""
    return run.step_rows()[s]


def statement(run: Any, s: int, m: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Round s's ``live@<s>``: its input rows, each port's u16 words (`live.statement_rows`' ports, one instance), and
    its ``--registered`` file (`live.reads_file`) under the window Program's digest, the open root and the session, at
    m_s (or a claimed ``m``)."""
    served, T = run.served, run.rows
    if s >= len(run.round_log):
        raise ValueError(f"round {s} hasn't closed, so its slot table isn't committed")
    roots = {v: r.root.hex() for v, r in served.live.items()}
    m = m_of(run, s) if m is None else m
    rows = {f"slot_{i}": w for i, w in enumerate(slot_words(run.owners.get(s, []), T))}
    rows |= {f"req_{i}": w for i, w in enumerate(request_words(run.admitted.get(0, []), T))}
    rows["m"] = np.asarray(L.slot_words([m])[0], np.uint16)
    doc = L.reads_file(served.program_digest, T, run.steps, s, roots, served.root, served.live[value(served, L.REQUESTS)].session,
                       m, served.program.fn.bindings["PREFIX"])
    return {p: np.asarray(w, np.uint16).reshape(1, -1) for p, w in rows.items()}, doc


def evaluated(run: Any, s: int, m: int | None = None) -> int:
    """``live@<s>``'s ok: `PouwLiveSlots_v1{T}` evaluated on its statement's rows (`statement`, at m_s or ``m``), each
    row's u16 words low bit first."""
    rows, _doc = statement(run, s, m)
    bits = {p: [(int(w) >> j) & 1 for w in r[0] for j in range(16)] for p, r in rows.items()}
    T = run.rows
    (ok,) = evaluate(bind(L.PouwLiveSlots, T=T), sum((bits[f"slot_{i}"] for i in range(T)), []),
                     sum((bits[f"req_{i}"] for i in range(T)), []), bits["m"])
    return int(ok)


def verdicts(run: Any) -> list[dict[str, Any]]:
    """Each round's ``live@<s>``: its slot table's root is the one round s's row commits (``slots``) and is listed whole
    there, round 0's request headers' is round 0's ``request_table``, and `live.live_ok` of its rows at m_s, as is its
    unit's circuit (`evaluated`); ``ok`` when all hold, else ``reason``. Empty without live tables."""
    served = run.served
    if served is None or not served.live or served.refused:
        return []
    req = served.live.get(value(served, L.REQUESTS))
    heads = headers(run.admitted.get(0, []))
    heads += [(0,) * L.HEADER] * (run.rows - len(heads))
    out = []
    for s, row in enumerate(run.round_log):
        slots, m = served.live.get(value(served, L.SLOTS, s)), m_of(run, s)
        leaves = slot_leaves(run.owners.get(s, []), run.rows)
        why = ("its slot table was never registered" if slots is None
               else "its slot table isn't the one round s's root commits" if slots.root.hex() != row["slots"]
               or slots.value not in row["a"]
               else "round 0's request headers aren't its request_table" if req is None
               or req.root.hex() != run.round_log[0].get("request_table")
               else f"live_ok is 0 at m = {m}" if not L.live_ok(leaves, heads, m)
               else f"its unit evaluates to 0 at m = {m}" if not evaluated(run, s) else None)
        out.append({"round": s, "m": m, "admitted": L.admitted(leaves, heads), "ok": why is None,
                    **({} if why is None else {"reason": f"live@{s}: {why}"})})
    return out


__all__ = ["REQUESTS", "SLOTS", "Table", "evaluated", "headers", "m_of", "nonce", "request_words", "slot_leaves", "slot_words",
           "statement", "table", "value", "verdicts"]
