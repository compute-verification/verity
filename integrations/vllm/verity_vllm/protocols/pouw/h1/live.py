"""H1's live tables: the served path's side of `PouwLiveSlots_v2{T}` (`verity.core.protocols.pouw.circuit.live`),
the proof block that makes each round's served rows m_s a proved value, one ``live@<s>`` a round, each request's cap bound to its
public max_tokens.

* **The tables.** With the Commit (`pouw_h1_served.Served`'s ``tap``) a window registers its round tables as values of
  the live Program, under the window Program's digest as its other values (`Served._tree`):

  * round 0's request table ``<prefix>requests@0`` (`live_reads.REQUESTS`): T rows, its entries' headers in admission order
    (index, round, first slot, cap and prompt length, each u32: the first 20 bytes of `RequestEntry.row`, as
    `live_reads.header_words`), then `EMPTY_REQUEST`'s zero header. It is registered as the window opens, after round 0's
    admissions, and its domain binds the window's session, since its root, round 0's ``request_table``, is inside round
    0's request root and so inside the open root (`credit.requests_root`), as ``ids@0``'s is. The prompts' tokens are
    ``ids@0``'s, not its. Its rows are ``row-seg/v1`` rows (`live_reads.REQUEST_SCHEMA`), the live statement's one row
    format whose commitment compresses before its last chunk.
  * round s's slot table ``<prefix>slots@<s>`` (`live_reads.SLOTS`): T rows, slot i's leaf (q + 1 for a row of request q, 0
    past the step's batch, `live_reads.slot_words`), registered at round s's close under the open root and listed whole among
    round s's segments (`Served.segments`), so round s's root commits it before its coin (`credit._a_link`).

  Each is its round's table: round 0's ``request_table`` and round s's ``slots`` are their roots, in place of the hiding
  tables (`pouw_h1_requests.request_table`, `slot_table`). Their leaves open in ``live@<s>``'s proof, not one by one:
  `pouw_h1.H1Run.open_request` and `open_slot` refuse them (`Table`). Row j's salt is the run's key's at `nonce`.
* **m_s** is step s's served rows, the scheduler's: its batch's rows (`pouw_h1.H1Run.owners`). Under CUDA graphs a call's
  A rows are padded to the graph's size, past them (`pouw_h1.Call.pad`), and those rows are no slot of the step's.
* **The statements.** ``live@<s>`` is proof block s of the live Program (`live.program_v2` of T, the window's S and its prefix),
  on round s's slot table, the request headers, the public m_s, the public round s and the public caps (`caps`: round 0's
  requests' max_tokens, the fixed length the period's open commits): `statement` gives its rows and its ``--registered``
  file (`live_reads.reads_file`), and `verdicts` holds each round's tables to the roots its round committed and its rows to
  `live.live_ok_v2` at m_s, s and the caps (each request named at most its prompt length's slots in the round that admitted
  it, one in each of its cap's later rounds, `live.takes`, and each live request's cap the public one).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw.circuit import live as L
from verity.core.protocols.pouw.circuit import live_reads as LR
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.commitments import Commitment

from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import requests as RQ

SLOTS, REQUESTS = LR.SLOTS, LR.REQUESTS


def nonce(base: int, name: str, s: int) -> int:
    """Row 0's salt nonce of a live table, from the window's live nonce ``base`` (`pouw_h1_served.LIVE_NONCE`):
    ``requests@0``'s is ``base``, ``slots@<s>``'s ``base + 1 + s``."""
    return base if name == LR.REQUESTS else base + 1 + int(s)


def headers(entries: Sequence[RQ.RequestEntry]) -> list[tuple[int, ...]]:
    """Each entry's header: index, round, first slot, cap and prompt length (`RequestEntry.row`'s first five u32)."""
    return [(e.index, e.round, e.slot, e.max_tokens, len(e.prompt)) for e in entries]


def request_words(entries: Sequence[RQ.RequestEntry], rows: int) -> np.ndarray:
    """``requests@0``'s rows ``(T, 10)`` u16: the entries' headers in admission order, then zero headers."""
    if len(entries) > rows:
        raise ValueError(f"round 0 admits {len(entries)} requests, past its {rows} leaves")
    hs = headers(entries) + [(0,) * LR.HEADER] * (rows - len(entries))
    return np.asarray(LR.header_words(hs), np.uint16)


def caps(entries: Sequence[RQ.RequestEntry]) -> list[int]:
    """The public max_tokens of round 0's requests in admission order, the fixed length the period's open commits
    (`credit.committed_length`) and so the live proof block's public caps (`live_reads.caps`)."""
    return [e.max_tokens for e in entries]


def slot_leaves(owners: Sequence[int | None], rows: int) -> list[int]:
    """Round s's slot leaves: q + 1 for a row of request q, in the step's batch order, then 0 (`EMPTY_SLOT`)."""
    got = [0 if o is None else int(o) + 1 for o in owners[:rows]]
    return got + [0] * (rows - len(got))


def slot_words(owners: Sequence[int | None], rows: int) -> np.ndarray:
    """``slots@<s>``'s rows ``(T, 2)`` u16."""
    return np.asarray(LR.slot_words(slot_leaves(owners, rows)), np.uint16)


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
    return LR.table(served.program.fn.bindings["PREFIX"], name, s)


def table(served: Any, run: Any, name: str, r: int, base: int, session: str) -> Table | None:
    """Round r's live table, registered now in ``served.live``: ``requests@0`` (``name`` `live_reads.REQUESTS`, round 0 only)
    bound to ``session``, or ``slots@<r>`` (`live_reads.SLOTS`) bound to the window's open root. None without the tap, once
    the window is refused, and before its Program and open root exist (a slot table)."""
    if served.tap is None or served.refused or served.program is None or (name == LR.REQUESTS and r):
        return None
    if name == LR.REQUESTS:
        words, window, bound = request_words(run.admitted.get(0, []), run.rows), None, session
    elif served.root is None:
        return None
    else:
        words, window, bound = slot_words(run.owners.get(r, []), run.rows), served.root, None
    v = value(served, name, r)
    if name == LR.REQUESTS:
        ys = NR.salts(run._key, nonce(base, name, r), len(words))
        reg = G.Registered.commit(served.program_digest, v, words.tolist(), [ys[192 * j:192 * (j + 1)] for j in range(len(words))],
                                  G.WORD_BITS, LR.REQUEST_SCHEMA, session=bound)
    else:
        reg = served._tree(run, v, words, nonce(base, name, r), window, bound)
    served.live[v] = reg
    return Table(reg)


def admission_gap(run: Any) -> str | None:
    """Why the window's ``live@<s>`` can't be stated: it reads round 0's request table every round, so a window that admits
    per round (`pouw_h1_requests.PER_ROUND`) isn't served yet; None for a round-0 window."""
    if run.admission == RQ.ROUND0:
        return None
    return (f"{run.entry}: live@<s> reads round 0's request table every round, and per-round admission (admission = "
            f"{run.admission!r}) isn't served yet: a request admitted after round 0 has no header there")


def m_of(run: Any, s: int) -> int:
    """Step s's served rows m_s (`pouw_h1.H1Run.step_rows`)."""
    return run.step_rows()[s]


def statement(run: Any, s: int, m: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Round s's ``live@<s>``: its input rows, each port's u16 words (`live_reads.statement_rows`' ports, one instance), and
    its ``--registered`` file (`live_reads.reads_file`) under the window Program's digest, the open root and the session, at
    m_s (or a claimed ``m``)."""
    served, T = run.served, run.rows
    if (gap := admission_gap(run)) is not None:
        raise ValueError(gap)
    if s >= len(run.round_log):
        raise ValueError(f"round {s} hasn't closed, so its slot table isn't committed")
    roots = {v: r.root.hex() for v, r in served.live.items()}
    m = m_of(run, s) if m is None else m
    rows = {f"slot_{i}": w for i, w in enumerate(slot_words(run.owners.get(s, []), T))}
    rows |= {f"req_{i}": w for i, w in enumerate(request_words(run.admitted.get(0, []), T))}
    rows["m"] = np.asarray(LR.slot_words([m])[0], np.uint16)
    rows["s"] = np.asarray(LR.slot_words([s])[0], np.uint16)
    mt = caps(run.admitted.get(0, []))
    rows |= {f"cap_{i}": w for i, w in enumerate(LR.slot_words(LR.caps(mt, T)))}
    doc = LR.reads_file(served.program_digest, T, run.steps, s, roots, served.root, served.live[value(served, LR.REQUESTS)].session,
                       m, served.program.fn.bindings["PREFIX"], mt)
    return {p: np.asarray(w, np.uint16).reshape(1, -1) for p, w in rows.items()}, doc


def evaluated(run: Any, s: int, m: int | None = None) -> int:
    """``live@<s>``'s ok: `PouwLiveSlots_v2{T}` evaluated on its statement's rows (`statement`, at m_s or ``m``), each
    row's u16 words low bit first."""
    rows, _doc = statement(run, s, m)
    bits = {p: [(int(w) >> j) & 1 for w in r[0] for j in range(16)] for p, r in rows.items()}
    T = run.rows
    (ok,) = evaluate(bind(L.PouwLiveSlotsV2, T=T), sum((bits[f"slot_{i}"] for i in range(T)), []),
                     sum((bits[f"req_{i}"] for i in range(T)), []), bits["m"], bits["s"],
                     sum((bits[f"cap_{i}"] for i in range(T)), []))
    return int(ok)


def verdicts(run: Any) -> list[dict[str, Any]]:
    """Each round's ``live@<s>``: its slot table's root is the one round s's row commits (``slots``) and is listed whole
    there, round 0's request headers' is round 0's ``request_table``, and `live.live_ok_v2` of its rows at m_s and the
    public caps, as is its proof block's circuit (`evaluated`); ``ok`` when all hold, else ``reason`` (every round's
    `admission_gap` of a per-round window). Empty without live tables."""
    served = run.served
    if served is None or not served.live or served.refused:
        return []
    if (gap := admission_gap(run)) is not None:
        return [{"round": s, "m": m_of(run, s), "ok": False, "reason": gap} for s in range(len(run.round_log))]
    req = served.live.get(value(served, LR.REQUESTS))
    heads = headers(run.admitted.get(0, []))
    heads += [(0,) * LR.HEADER] * (run.rows - len(heads))
    cap = LR.caps(caps(run.admitted.get(0, [])), run.rows)
    out = []
    for s, row in enumerate(run.round_log):
        slots, m = served.live.get(value(served, LR.SLOTS, s)), m_of(run, s)
        leaves = slot_leaves(run.owners.get(s, []), run.rows)
        why = ("its slot table was never registered" if slots is None
               else "its slot table isn't the one round s's root commits" if slots.root.hex() != row["slots"]
               or slots.value not in row["a"]
               else "round 0's request headers aren't its request_table" if req is None
               or req.root.hex() != run.round_log[0].get("request_table")
               else f"live_ok is 0 at m = {m}" if not L.live_ok_v2(leaves, heads, m, s, cap)
               else f"its unit evaluates to 0 at m = {m}" if not evaluated(run, s) else None)
        out.append({"round": s, "m": m, "admitted": L.admitted(leaves, heads), "ok": why is None,
                    **({} if why is None else {"reason": f"live@{s}: {why}"})})
    return out


def dump(run: Any, out: Path) -> dict[str, Any]:
    """The prover's ``live@<s>`` of each closed round, for `benchmarks/pouw/h1_prove/live.py`: ``out/window.json`` (public:
    T, S, the prefix, the window Program's digest, the open root, the session, the live tables' roots, each round's
    served rows m_s and round 0's requests' max_tokens, the caps) and ``out/<s>/rows.npz`` in `window.py open`'s form (``w_``, ``s_``, ``v_``, ``r_`` and ``p_`` of each
    port: its words, salt, registered value, position and path, one instance, proof block s), with ``out/<s>/opening.json``, each
    registered row's ``b || c`` and path, which is all of it the verifier sees. Every row is the prover's own: ``out`` is
    the prover's directory, never a run's ``out/``."""
    served, T = run.served, run.rows
    prefix, prog, window = served.program.fn.bindings["PREFIX"], served.program_digest, served.root
    req = served.live[value(served, LR.REQUESTS)]
    pub = {"T": T, "S": run.steps, "prefix": prefix, "program": prog, "window": window, "session": req.session,
           "roots": {v: r.root.hex() for v, r in served.live.items()},
           "served": [m_of(run, s) for s in range(len(run.round_log))], "max_tokens": caps(run.admitted.get(0, []))}
    cap = LR.caps_table(prog, prefix, window, T, pub["max_tokens"])[2]
    out.mkdir(parents=True, exist_ok=True)
    (out / "window.json").write_text(json.dumps(pub, indent=1))
    for s, m in enumerate(pub["served"]):
        rows, _doc = statement(run, s)
        reads = {f"slot_{i}": (served.live[value(served, LR.SLOTS, s)], i) for i in range(T)}
        reads |= {f"req_{i}": (req, i) for i in range(T)}
        reads |= {f"cap_{i}": (cap, i) for i in range(T)}
        reads |= {"m": (LR.m_table(prog, prefix, window, s, m)[2], 0), "s": (LR.round_table(prog, prefix, window, s)[2], 0), LR.OK_PORT: (LR.ok_table(prog, prefix, window, run.steps)[2], s)}
        rows[LR.OK_PORT] = np.asarray([CC.OK_ROW], np.uint16)
        z: dict[str, Any] = {"units": np.array([s])}
        for p, (reg, i) in reads.items():
            z |= {f"w_{p}": rows[p], f"s_{p}": np.frombuffer(reg.salts[i], np.uint8).reshape(1, -1),
                  f"v_{p}": np.array([reg.value]), f"r_{p}": np.array([i], np.int64), f"p_{p}": np.array([reg.path(i).hex()])}
        d = out / str(s)
        d.mkdir(exist_ok=True)
        np.savez(d / "rows.npz", **z)
        (d / "opening.json").write_text(json.dumps({p: {"value": reg.value, "index": i, "bc": reg.bcs[i].hex(),
                                                        "path": reg.path(i).hex()} for p, (reg, i) in reads.items()}))
    return {"dir": str(out), "rounds": len(pub["served"])}


__all__ = ["REQUESTS", "SLOTS", "Table", "admission_gap", "caps", "dump", "evaluated", "headers", "m_of", "nonce", "request_words",
           "slot_leaves", "slot_words", "statement", "table", "value", "verdicts"]
