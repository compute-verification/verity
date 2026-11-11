"""A warm request replayed from the Program that recomputes its prefix-cache hit (`program.predict.steps.compile_steps`'s
``recompute_hits``; on a batch-invariant target, the uncached request's Program).

Serving ran the request's first step on positions C.. only, the cache holding 0..C-1 from other requests. Its Program
computes every position, so its first step's rows below C are the replay's own, evaluated from its certified prompt and
the registered weights, and no read reaches the KV another request committed:

* `prefix_rows`: those rows, out of the request's sampled population, since no row of it served them. A row of the first
  step is one position where its op holds one row per position, as many as the step's embedding.
* `RecomputedPrefix`: the request's reads, answered as `CommittedStore.words` answers them: a position below C at its
  first step from its `Replay`, one at or past C from what serving committed, C rows earlier in the member (its first step
  embedded prompt[C:]). ``reads`` logs each read the replay answered, for the VU's record.
* `adopt`, `stores`, `replay_vu`: how `driver.sampled_replay` takes a warm request's Program in this form, builds its store
  and replays its VUs through it.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay.index import IDS_MEMBER, ProgramIndex
from verity_vllm.check.replay.request_replay import Replay


def positions(pi: ProgramIndex) -> dict[tuple[str, str], int]:
    """The (op_path, member) pairs of the first step that hold one row per position, each with its row width: as many rows
    as the step's embedding, its input ids among them."""
    rows: Counter = Counter()
    width: dict[tuple[str, str], int] = {}
    for r in pi.rows:
        d = pi.describe(r)
        if d is None or int(d["step"]) != 0:
            continue
        for op, member, w in [(d["op_path"], m[0], int(m[1])) for m in d["members"]] + \
                             [(c["op_path"], c["member"], int(c["width"])) for c in d.get("copies") or ()]:
            rows[(op, member)] += 1
            width[(op, member)] = w
    embed = pi.embed_op_path()
    n = max((k for (op, _m), k in rows.items() if op == embed), default=0)
    if not n:
        raise ValueError("the Program's first step has no embedding row: its positions are unknown")
    return {**{k: width[k] for k, c in rows.items() if c == n}, (embed, IDS_MEMBER): 1}


def prefix_rows(pi: ProgramIndex, C: int) -> frozenset[int]:
    """The Program rows that compute the first step's positions below ``C``, refused where that step computes none past them
    (a hit recomputed in steps of its own, class G's producer launches, which this replay doesn't address)."""
    per = positions(pi)
    out, past = set(), False
    for r in pi.rows:
        d = pi.describe(r)
        if d is None or int(d["step"]) != 0 or not d["members"] or not all((d["op_path"], m[0]) in per for m in d["members"]):
            continue
        if int(d["row"]) < C:
            out.add(int(r["i"]))
        else:
            past = True
    if not past:
        raise ValueError(f"the Program's first step computes no position past its hit's {C}: its hit is recomputed in steps of its "
                         "own, which the one-stage replay doesn't address")
    return frozenset(out)


class RecomputedPrefix:
    """Request ``rid``'s reads (`CommittedStore.words`) when its Program recomputes its hit of ``C`` positions: below C at
    its first step (engine step ``arrive``) from its `Replay` on ``prompt`` and ``weights``, the rest from ``store``, C rows
    earlier in a first-step member that holds a row per position. ``plan``: the replay is not run, its reads answered with
    zero words of their size (`evaluate.plan_vu` reads what the replay would read and evaluates nothing)."""

    def __init__(self, store: Any, pi: ProgramIndex, rid: str, C: int, prompt: Sequence[int], weights: Callable | None, *,
                 arrive: int = 0, plan: bool = False) -> None:
        if not 0 < int(C) < len(prompt):
            raise ValueError(f"request {rid}: a hit of {C} positions of its {len(prompt)}-token prompt (at least one, and short of it)")
        self.store, self.rid, self.C, self.arrive = store, str(rid), int(C), int(arrive)
        self.per = positions(pi)
        self.replay = None if plan else Replay(pi, rid, prompt, weights, arrive=arrive)
        self.reads: list[list[Any]] = []

    def words(self, rid: str, step: int, op_path: str, member: str, elo: int, ehi: int) -> tuple[np.ndarray | None, str]:
        w = self.per.get((op_path, member)) if str(rid) == self.rid and int(step) == self.arrive else None
        if w is None:
            return self.store.words(rid, step, op_path, member, elo, ehi)
        cut, parts, dtype = self.C * w, [], ""
        if elo < cut:
            hi = min(ehi, cut)
            got, dtype = (np.zeros(hi - elo, dtype=np.uint16), "") if self.replay is None else \
                self.replay.words(rid, step, op_path, member, elo, hi)
            if got is None:
                return None, f"the recomputed prefix of {rid}: {dtype}"
            self.reads.append([rid, int(step), op_path, member, [elo, hi]])
            parts.append(got)
        if ehi > cut:
            got, dtype = self.store.words(rid, step, op_path, member, max(elo, cut) - cut, ehi - cut)
            if got is None:
                return None, dtype
            parts.append(got)
        return (parts[0] if len(parts) == 1 else np.concatenate(parts)), dtype

    def take(self) -> list[list[Any]]:
        """The reads the replay answered since the last call."""
        out, self.reads = self.reads, []
        return out


def adopt(pi: ProgramIndex, rid: str, warm: Mapping[str, Any] | None) -> frozenset[int]:
    """Request ``rid``'s prefix rows where it is served a hit (``warm``: its `cached_prefixes` entry, {"cached": C, "prompt": ids})
    that its Program ``pi`` recomputes, its C and prompt recorded on ``pi.recomputed``; none for a cold request or a Program
    that reads a cached prefix."""
    C = int((warm or {}).get("cached") or 0)
    if not C or pi.kv_prefix is not None:
        return frozenset()
    if (warm or {}).get("prompt") is None:
        raise ValueError(f"{rid} is served {C} cached prompt rows and its Program recomputes them: the replay needs its certified "
                         "prompt (`cached_prefixes[rid]['prompt']`)")
    pi.recomputed[rid] = (C, [int(t) for t in warm["prompt"]])
    return prefix_rows(pi, C)


def stores(indices: Mapping[Any, Any], store: Any, weights: Callable | None, arrivals: Mapping[str, int], *,
           plan: bool = False) -> dict[str, RecomputedPrefix]:
    """Each adopted request's store over ``store`` (`adopt`)."""
    return {rid: RecomputedPrefix(store, pi, rid, *pi.recomputed[rid], weights, arrive=int(arrivals.get(rid, 0)), plan=plan)
            for rid, pi in indices.items() if isinstance(rid, str) and rid in getattr(pi, "recomputed", {})}


def replay_vu(own: Mapping[str, RecomputedPrefix], pi: ProgramIndex, store: Any, weights: Callable, vu: dict, *,
              arrive: int) -> dict[str, Any]:
    """`evaluate.replay_vu` through the VU's request's store where it has one, its record naming the reads its replay
    answered (``recomputed``)."""
    mine = own.get(vu["request_id"])
    rec = EV.replay_vu(pi, mine or store, weights, vu, arrive=arrive)
    if mine is not None:
        rec["recomputed"] = mine.take()
    return rec
