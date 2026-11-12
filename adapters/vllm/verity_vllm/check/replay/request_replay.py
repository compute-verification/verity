"""One request's values replayed on the CPU from its Program, evaluating only what a read reaches.

`Replay` answers `CommittedStore.words` for one request: its Program's rows are evaluated by the replay's row kernels
(`evaluate.evaluated`) on its replayed inputs, its certified prompt ids, the registered weights, and whatever else
``given`` supplies. A row is evaluated only when a read reaches it, so what it evaluates is its readers' cone and never
the request's batch-mates: a request's Program is its rows alone at its launch shapes. A request served from the prefix
cache is replayed from its Program compiled with its hit recomputed (`program.predict.steps.compile_steps`'s
``recompute_hits``), which prefills the hit from its own prompt; a Program that reads a cached prefix is refused. Its
``arrive`` is the engine step of its Program's first step, so where that Program runs its hit's chunks in steps ahead of
the request's own (class G, `RequestUnit.hit_steps`), it is the request's first served step less their number.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay.compare import _tp_member_alias
from verity_vllm.check.replay.index import IDS_MEMBER, ProgramIndex, node_runs
from verity_vllm.check.replay.opening import _ITEMSIZE, _NPDTYPE
from verity_vllm.query.manifest import format as RM

Given = Callable[[int, str, str, int, int], tuple[np.ndarray | None, str]]


class Replay:
    """Request ``rid``'s values replayed from its Program ``pi`` at engine step ``arrive``: ``prompt`` its certified ids,
    ``weights(field, lo, hi)`` the registered weights, ``given(step, op_path, member, lo, hi)`` (request-local step) any
    other input its rows read that no row of it computes (a sampler seed, a run value). `words` answers a read of
    ``(rid, engine step, op_path, member, [lo, hi))`` as `CommittedStore.words` does, evaluating the rows that produce
    it and, first, every row of theirs not yet evaluated, in Program order. ``seconds`` is the evaluation time by row
    family and ``rows`` the rows evaluated."""

    def __init__(self, pi: ProgramIndex, rid: str, prompt: Sequence[int], weights: Callable, *, arrive: int = 0,
                 given: Given | None = None) -> None:
        if pi.kv_prefix:
            raise ValueError(f"request {rid} reads a cached prefix another request computed: a span is replayed from its "
                             "Program with the hit recomputed (compile_steps(..., recompute_hits=True))")
        self.pi, self.rid, self.arrive, self.weights, self.given = pi, str(rid), int(arrive), weights, given
        self.ids = np.asarray(list(prompt), dtype=np.uint32)
        self.embed = pi.embed_op_path()
        self.at: dict[tuple[int, str, str], tuple[int, str, dict[int, int]]] = {}
        for r in pi.rows:
            d = pi.describe(r)
            if d is None:
                continue
            outs = [(d["op_path"], m[0], int(m[1]), str(m[2]), int(d["row"])) for m in d["members"]]
            outs += [(c["op_path"], c["member"], int(c["width"]), str(c["dtype"]), int(c["row"])) for c in d.get("copies") or ()]
            for op, member, width, dtype, row in outs:
                w, dt, by = self.at.setdefault((int(d["step"]), op, member), (width, dtype, {}))
                if (w, dt) != (width, dtype) or by.setdefault(row, int(r["i"])) != int(r["i"]):
                    raise ValueError(f"rows {by[row]} and {r['i']} both commit ({op}/{member}) row {row} at step {d['step']}, "
                                     f"or as {dt} x {w} and {dtype} x {width}")
        self.out: dict[int, dict[tuple[str, str], np.ndarray]] = {}
        self.busy: set[int] = set()
        self.seconds: dict[str, float] = {}
        self.rows = 0

    def words(self, rid: str, step: int, op_path: str, member: str, elo: int, ehi: int) -> tuple[np.ndarray | None, str]:
        if str(rid) != self.rid:
            return None, f"the replay holds request {self.rid}, not {rid}"
        s = int(step) - self.arrive
        if member == IDS_MEMBER and op_path == self.embed and s == 0:
            if not 0 <= elo <= ehi <= self.ids.size:
                return None, f"prompt ids [{elo}, {ehi}) of request {rid}'s {self.ids.size}"
            return self.ids[elo:ehi].copy(), "i32"
        at = self.at.get((s, op_path, member))
        if at is None:
            if self.given is not None:
                return self.given(s, op_path, member, elo, ehi)
            return None, f"no row of request {rid} commits ({op_path}/{member}) at its step {s}, and nothing gives it"
        width, dtype, by = at
        if ehi <= elo or elo < 0:
            return None, f"element slice [{elo}, {ehi}) of ({op_path}/{member})"
        first = elo // width
        parts = []
        for row in range(first, (ehi - 1) // width + 1):
            i = by.get(row)
            if i is None:
                return None, f"no row {row} of ({op_path}/{member}) at request {rid}'s step {s}"
            if i in self.busy:
                return None, (f"row {i} ({op_path}) reads its own output {member}: a replay computes it, it doesn't read it "
                              "as a witness")
            self._ensure(i)
            parts.append(self.out[i][(op_path, member)])
        return np.concatenate(parts)[elo - first * width:ehi - first * width], dtype

    def _ensure(self, i: int) -> None:
        """Row ``i`` and every row of its inputs' cone not yet evaluated, in Program order (a producer precedes its
        readers), so no evaluation reads an unevaluated row."""
        if i in self.out:
            return
        need, todo, seen = [], [i], set()
        while todo:
            j = todo.pop()
            if j in seen or j in self.out:
                continue
            seen.add(j)
            r = self.pi.by_i[j]
            d = self.pi.describe(r)
            if d is not None and (d["members"] or d.get("copies")):
                need.append(j)
            todo.extend(int(run[1]) for g in r.get("args") or () for run in node_runs(g))
        for j in sorted(need):
            self._evaluate(j)

    def _evaluate(self, i: int) -> None:
        r = self.pi.by_i[i]
        d = self.pi.describe(r)
        fam = d["family"]
        t0 = time.perf_counter()
        self.busy.add(i)
        try:
            got, why = EV.evaluated(self.pi, self, self.weights, r, self.rid, self.arrive, [])
        finally:
            self.busy.discard(i)
        if why is not None:
            raise LookupError(f"row {i} ({fam}, {d['op_path']}, step {d['step']}) of request {self.rid}: {why}")
        if len(got) > 1:
            got = EV.by_member(got, self.pi.member_of_port(i) or EV.plane_ports(d))
        if d.get("plane") and d["plane"] not in got and len(got) == 1:
            got = {d["plane"]: next(iter(got.values()))}
        one = next(iter(got.values())) if len({id(v) for v in got.values()}) == 1 else None
        outs: dict[tuple[str, str], np.ndarray] = {}
        for member, width, dtype in d["members"]:
            v = got.get(member)
            if v is None:
                v = _tp_member_alias(got, member)
            if v is None and (d.get("copies") and len(got) == 1 or len(d["members"]) == 1):
                v = one
            if v is None:
                raise LookupError(f"row {i} ({fam}): its relation gives no output {member!r} (it gives {sorted(got)})")
            got[member] = v
            outs[(d["op_path"], member)] = _words(v, str(dtype), int(width), fam, f"row {i} {member}")
        for c in d.get("copies") or ():
            outs[(c["op_path"], c["member"])] = _words(got[c["of"]], str(c["dtype"]), int(c["width"]), fam,
                                                       f"row {i} {c['member']}")
        self.out[i] = outs
        self.rows += 1
        self.seconds[fam] = self.seconds.get(fam, 0.0) + time.perf_counter() - t0


def _words(v: Any, dtype: str, width: int, fam: str, what: str) -> np.ndarray:
    """An evaluated output as the words a committed member of ``dtype`` holds (`CommittedStore.words`' widths: the
    sampled token's i64 as u64)."""
    if dtype in RM.LIFTED_DTYPES:
        raise LookupError(f"{what}: a lifted member ({dtype}) of the padded MoE construction is not replayed")
    a = np.asarray(v).reshape(-1)
    size = 8 if RM.is_token_select(fam) else _ITEMSIZE.get(dtype, 2)
    if a.size != width or a.dtype.kind not in "ui":
        raise LookupError(f"{what}: {a.size} {a.dtype} values, not {width} {dtype} words")
    return a.astype(_NPDTYPE[size])
