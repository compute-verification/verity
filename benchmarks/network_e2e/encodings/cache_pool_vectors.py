"""`CacheStep_v2` (`verity.core.protocols.certifier.cache_step_v2`) against vLLM's block pool as the replay models it
(`replay._KV`) over a whole trace, and its difftest vectors.

`drive` replays a trace with its scheduler's pool tapped (not a fork's, not a dry run), and makes the unit's call for
each pool operation, in order:
- `allocate_slots` of a request admitted with nothing computed: its lookup, taking the hit if the allocation succeeds;
- an allocation that succeeds: a compute of its new blocks and the units `cache_blocks` caches in it;
- `cache_blocks` after a step (async scheduling caches a decode's block a step late): a compute with no new blocks;
- `free_request` (a stop or a preemption): a free.
Each call's witness is `_KV`'s own: the hit, its evictable blocks, the new blocks, the blocks held and returned. The
threshold stamp is the pool's e-th oldest queued stamp, the one stamp the unit accepts. After each call `drive` holds
the unit's pool to `_KV`'s. The same hashes are cached, each with its blocks in the same order and held by the same
requests' slots. The hashless free count is the same, and so is the order of the cached free blocks in the free
queue. Each eviction removes the blocks of the hashes vLLM evicts. Beside each honest call it makes refused ones:
- a lookup's hit a unit either side, and the hit a cache that never evicts would give where that one is longer (a
  hit past the cache after eviction);
- a lookup's evictable count one either side;
- a compute's threshold stamp at the queue's (e − 1)-th, (e + 1)-th or last queued copy, or between the e-th and the
  next. With e = 0 it is the oldest queued stamp (a made-up eviction);
- a free's returns one either side.
`drive` also checks, from `_KV` alone, that vLLM evicts a hash's last block only when no descendant of it is cached
(`dropped_with_descendant`), the fact `cache_step_v2`'s N = NB − 1 rests on.

    python3 cache_pool_vectors.py tests/fixtures/async-prefix-cache.jsonl tests/fixtures/async-crush.json.gz \\
        [--out ../../../verity/core/protocols/certifier/tests/fixtures/cache_step_v2_vectors.json]
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as RP  # noqa: E402

from verity.core.protocols.certifier import cache_step_v2 as C2  # noqa: E402

HERE = Path(__file__).parent
FIXTURES = HERE / "tests" / "fixtures"
VECTORS = HERE.parents[2] / "verity" / "core" / "protocols" / "certifier" / "tests" / "fixtures" / "cache_step_v2_vectors.json"

#: each trace's slice for the certifier's vectors: its first calls, and the window (L) its requests fit in
SLICES = {"async-prefix-cache.jsonl": {"calls": 400, "L": 368}, "async-crush.json.gz": {"calls": 160, "L": 1232}}


@dataclass
class Input:
    """What the replay takes from a trace (`replay.replay`), and the trace's own rows to hold it to."""
    engine: RP.Engine
    requests: list[RP.RequestIO]
    before: list[int]
    rows: list[list[tuple[int, int, int]]]


def replay_input(path: Path) -> Input:
    """A step trace (`replay.read_any`), or a derived-format trace (`*.json.gz`, `derived.load_trace`'s), as replay input.
    A derived-format request is admitted before the first step to end after its arrival, or at the step it is first
    scheduled in, if that is sooner (an idle engine starts its next step on the arrival)."""
    if path.suffix != ".gz":
        tr = RP.read_any(path)
        return Input(tr.engine, tr.requests, tr.before_step, [list(s.rows) for s in tr.steps])
    t = json.loads(gzip.open(path).read())
    c = t["config"]
    eng = RP.Engine(max_num_seqs=c["max_num_seqs"], max_num_batched_tokens=c["max_num_batched_tokens"],
                    max_model_len=c["max_model_len"], num_gpu_blocks=c["num_gpu_blocks"], block_size=c["block_size"],
                    long_prefill_token_threshold=c["long_prefill_token_threshold"],
                    enable_prefix_caching=c["enable_prefix_caching"], async_scheduling=c["scheduler"] == "async",
                    watermark=c["watermark"])
    rows = [[(x["request"], x["start"], x["n"]) for x in s["rows"]] for s in t["steps"]]
    first: dict[int, int] = {}
    for k, st in enumerate(rows):
        for i, _, _ in st:
            first.setdefault(i, k)
    ends = [s["t_end_s"] for s in t["steps"]]
    reqs, before = [], []
    for i, r in enumerate(t["requests"]):
        out = r["output_ids"]
        reqs.append(RP.RequestIO(prompt_len=len(r["prompt_ids"]), max_tokens=r["max_tokens"], output_len=len(out),
                                 eos=r["finish"] == "eos", tokens=tuple(r["prompt_ids"]) + tuple(out)))
        before.append(min(bisect.bisect_left(ends, r["arrival_s"]) + 1, first[i]))
    return Input(eng, reqs, before, rows)


def statics_for(inp: Input, *, L: int | None = None, R: int = 16, D: int = 1, SW: int = 16) -> dict:
    e = inp.engine
    TW = max(max(max(r.tokens) for r in inp.requests).bit_length(), 1)
    RW = max((len(inp.requests) - 1).bit_length(), 1)
    return C2.statics(e.num_gpu_blocks - 1, e.block_size, TW, RW, L or e.max_model_len, R, D, e.num_gpu_blocks, SW)


def window(tokens: Sequence[int], u0: int, S) -> tuple[tuple[int, ...], ...]:
    """KU units of `tokens` from unit u0, zero past their end."""
    KU = C2.widths(S).KU
    ts = list(tokens[u0 * S.U:(u0 + KU) * S.U])
    ts += [0] * (KU * S.U - len(ts))
    return tuple(tuple(ts[k * S.U:(k + 1) * S.U]) for k in range(KU))


def _queued(pool: C2.Pool) -> list[int]:
    return sorted(c.stamp for x in pool.table if x for c in x.copies if not c.mask)


def threshold(pool: C2.Pool, new: int) -> int:
    """The threshold stamp a compute taking `new` blocks needs: the e-th oldest queued copy's, 0 with e = 0."""
    e = max(new - pool.hf, 0)
    return _queued(pool)[e - 1] if e else 0


def _fits(S, op: C2.CallInputs) -> bool:
    """Whether each of the op's fields fits its width (a variant that doesn't has no arguments)."""
    Z = C2.widths(S)
    w = {"slot": Z.SL, "r": S.RW, "K": Z.LW, "tau": S.SW, **dict.fromkeys(("wh", "ev", "nins", "new", "held", "ret"), Z.HW)}
    return all(0 <= getattr(op, f) < 1 << n for f, n in w.items())


@dataclass
class Call:
    """One call as `drive` made it: its kind, its op, the request's tokens it reads (unit u0 on, `known` of them) and the
    refused variants beside it."""
    kind: str
    op: C2.CallInputs
    u0: int
    known: int
    refused: list[tuple[str, C2.CallInputs]] = field(default_factory=list)


class _Driver:
    def __init__(self, S, inp: Input, record: int | None) -> None:
        self.S, self.inp, self.record = S, inp, record
        self.pool = C2.empty(S)
        self.slots: dict[int, int] = {}
        self.calls: list[Call] = []
        self.counts: Counter = Counter()
        self.ever: set[int] = set()
        self.alloc: dict[str, Any] | None = None
        self.reqs: dict[int, Any] = {}
        self.hashes: Any = None

    # the unit's calls

    def _slot(self, i: int) -> int:
        if i not in self.slots:
            free = sorted(set(range(self.S.R)) - set(self.slots.values()))
            if not free:
                raise AssertionError(f"request {i}: more than R = {self.S.R} requests hold blocks")
            self.slots[i] = free[0]
        return self.slots[i]

    def _call(self, kv, kind: str, op: C2.CallInputs, u0: int, known: int, refused: list[tuple[str, C2.CallInputs]]) -> C2.Out:
        S = self.S
        if not _fits(S, op):
            raise AssertionError(f"call {len(self.calls)} ({kind}, {op}): a field past its width")
        refused = [(w, b) for w, b in refused if _fits(S, b)]
        out = C2.reference(S, self.pool, op)
        if not out.ok:
            raise AssertionError(f"call {len(self.calls)} ({kind}, {op}): the unit refuses vLLM's own operation")
        for what, bad in refused:
            if C2.reference(S, self.pool, bad).ok:
                raise AssertionError(f"call {len(self.calls)} ({kind}): the unit accepts {what}: {bad}")
            self.counts[f"refused {what}"] += 1
        if self.record is None or len(self.calls) < self.record:
            self.calls.append(Call(kind, op, u0, known, refused))
        self.counts[kind] += 1
        self.pool = out.pool
        self.counts["entries_max"] = max(self.counts["entries_max"], sum(x is not None for x in out.pool.table))
        self.counts["copies_max"] = max(self.counts["copies_max"], max((len(x.copies) for x in out.pool.table if x), default=0))
        self.counts["clock"] = out.pool.clock
        return out

    def lookup(self, kv, r, wh: int, ev: int, take: bool) -> None:
        S = self.S
        cap = (r.num_tokens - 1) // S.U
        never = next((k for k, h in enumerate(r.block_hashes[:cap]) if h not in self.ever), min(cap, len(r.block_hashes)))
        op = C2.CallInputs(slot=self._slot(r.i) if take else self._free_slot(), r=r.i, admit=True, take=take, K=r.num_tokens,
                   wh=wh, ev=ev, units=window(r.io.tokens[:r.num_tokens], 0, S))
        refused = [("a hit one unit longer", replace(op, wh=wh + 1)), ("the evictable count plus one", replace(op, ev=ev + 1))]
        if wh:
            refused.append(("a hit one unit shorter", replace(op, wh=wh - 1)))
        if ev:
            refused.append(("the evictable count less one", replace(op, ev=ev - 1)))
        if never > wh:
            refused.append(("the hit of a cache that never evicts", replace(op, wh=never)))
            self.counts["hits shortened by eviction"] += 1
        self._call(kv, "take" if take else "lookup", op, 0, r.num_tokens, refused)

    def _free_slot(self) -> int:
        return min(set(range(self.S.R)) - set(self.slots.values()), default=0)

    def compute(self, kv, r, new: int, done: int, full: int, evicted: Counter) -> None:
        if not new and full == done:
            return
        S = self.S
        e = max(new - self.pool.hf, 0)
        q = _queued(self.pool)
        tau = q[e - 1] if e else 0
        op = C2.CallInputs(slot=self._slot(r.i), r=r.i, nins=full - done, new=new, tau=tau,
                   units=window(r.io.tokens[:r.num_tokens], done, S))
        refused = []
        if e:
            if q[-1] + 1 < 1 << S.SW:
                refused.append(("the threshold past every queued copy", replace(op, tau=q[-1] + 1)))
            if e > 1:
                refused.append(("the threshold one copy older", replace(op, tau=q[e - 2])))
            if e < len(q):
                refused.append(("the threshold one copy newer", replace(op, tau=q[e])))
                if q[e] > tau + 1:
                    refused.append(("the threshold between two copies", replace(op, tau=tau + 1)))
            self.counts["evicting computes"] += 1
            self.counts["evicted"] += e
        elif q:
            refused.append(("a made-up eviction", replace(op, tau=q[0] or 1)))
        hashes = self._hashes(kv, self.pool)
        gone = Counter(hashes[x] for x, ent in enumerate(self.pool.table) if ent for c in ent.copies
                       if e and not c.mask and c.stamp <= tau)
        self._call(kv, "compute", op, done, r.num_tokens, refused)
        if gone != evicted:
            raise AssertionError(f"call {len(self.calls)}: the unit evicts {dict(gone)}, vLLM {dict(evicted)}")

    def free(self, kv, r, held: int, ret: int) -> None:
        if r.i not in self.slots:
            if held:
                raise AssertionError(f"request {r.i} frees {held} blocks it never allocated")
            return
        op = C2.CallInputs(slot=self.slots[r.i], free=True, held=held, ret=ret)
        refused = [("the returns plus one", replace(op, ret=ret + 1))] + ([("the returns less one", replace(op, ret=ret - 1))]
                                                                        if ret else [])
        self._call(kv, "free", op, 0, 0, refused)
        del self.slots[r.i]

    # the pool against _KV's

    def _hashes(self, kv, pool: C2.Pool) -> list[int | None]:
        ids = self.hashes._ids
        out: list[int | None] = [None] * self.S.N

        def of(x: int) -> int:
            if out[x] is None:
                ent = pool.table[x]
                p = 0 if ent.parent == self.S.N else of(ent.parent)
                out[x] = ids[(p, ent.tokens)]
            return out[x]

        for x, ent in enumerate(pool.table):
            if ent:
                of(x)
        return out

    def check(self, kv) -> None:
        S, pool = self.S, self.pool
        hashes = self._hashes(kv, pool)
        holder: dict[int, set[int]] = {}
        for i, bs in kv.req_blocks.items():
            for b in bs:
                holder.setdefault(b, set()).add(i)
        slot_of = {i: s for i, s in self.slots.items()}
        mine = {}
        for x, ent in enumerate(pool.table):
            if ent:
                mine[hashes[x]] = [(c.mask, c.stamp) for c in ent.copies]
        theirs = {}
        for h, bs in kv.by_hash.items():
            theirs[h] = [sum(1 << slot_of[i] for i in holder.get(b, ())) for b in bs]
        if {h: [m for m, _ in v] for h, v in mine.items()} != theirs:
            raise AssertionError(f"call {len(self.calls)}: the pool's cached blocks and holders aren't _KV's")
        free_cached = [(kv.block_hash[b], list(kv.by_hash[kv.block_hash[b]]).index(b)) for b in kv.free.d
                       if kv.block_hash[b] is not None]
        queued = sorted((c.stamp, hashes[x], d) for x, ent in enumerate(pool.table) if ent
                        for d, c in enumerate(ent.copies) if not c.mask)
        if [(h, d) for _, h, d in queued] != free_cached:
            raise AssertionError(f"call {len(self.calls)}: the queued copies by stamp aren't _KV's free queue")
        if pool.hf != len(kv.free) - len(free_cached):
            raise AssertionError(f"call {len(self.calls)}: hf {pool.hf}, _KV's hashless free {len(kv.free) - len(free_cached)}")
        for i, s in self.slots.items():
            want = S.N if not kv.num_cached.get(i) else next(
                x for x, h in enumerate(hashes) if h == self.reqs[i].block_hashes[kv.num_cached[i] - 1])
            if pool.spfx[s] != want:
                raise AssertionError(f"call {len(self.calls)}: slot {s}'s walk entry {pool.spfx[s]}, want {want}")
        self.ever |= set(kv.by_hash)
        self.counts["checks"] += 1


@contextmanager
def _tapped(drv: _Driver):
    K = RP._KV
    names = ("allocate_slots", "cache_blocks", "free_request", "_new_blocks", "_pop_hash")
    orig = {n: getattr(K, n) for n in names}

    def allocate_slots(self, r, n, hit=(), *, full_sequence_must_fit=False, has_scheduled_reqs=True, dry=False):
        kw = {"full_sequence_must_fit": full_sequence_must_fit, "has_scheduled_reqs": has_scheduled_reqs, "dry": dry}
        if dry or self.mine is not None:
            return orig["allocate_slots"](self, r, n, hit, **kw)
        drv.reqs[r.i] = r
        look = r.status != RP._RUN and r.num_computed == 0
        ev = sum(1 for b in hit if self.ref[b] == 0)
        drv.alloc = {"new": 0, "evicted": Counter(), "cache": None, "popping": False}
        ok = orig["allocate_slots"](self, r, n, hit, **kw)
        a, drv.alloc = drv.alloc, None
        if look:
            drv.lookup(self, r, len(hit), ev, ok)
        if ok:
            done, full = a["cache"] or (self.num_cached.get(r.i, 0),) * 2
            drv.compute(self, r, a["new"], done, full, a["evicted"])
        drv.check(self)
        return ok

    def cache_blocks(self, r, num_tokens):
        if self.mine is not None:
            return orig["cache_blocks"](self, r, num_tokens)
        drv.reqs[r.i] = r
        done = self.num_cached.get(r.i, 0)
        orig["cache_blocks"](self, r, num_tokens)
        full = max(self.num_cached.get(r.i, 0), done)
        if drv.alloc is not None:
            drv.alloc["cache"] = (done, full)
        else:
            drv.compute(self, r, 0, done, full, Counter())
            drv.check(self)

    def free_request(self, r):
        if self.mine is not None:
            return orig["free_request"](self, r)
        bs = self.req_blocks.get(r.i, [])
        held, ret = len(bs), sum(1 for b in bs if self.ref[b] == 1)
        orig["free_request"](self, r)
        drv.free(self, r, held, ret)
        drv.check(self)

    def _new_blocks(self, k):
        if self.mine is None and drv.alloc is not None:
            drv.alloc["new"], drv.alloc["popping"] = k, True
        try:
            return orig["_new_blocks"](self, k)
        finally:
            if self.mine is None and drv.alloc is not None:
                drv.alloc["popping"] = False

    def _pop_hash(self, b):
        h = self.block_hash[b]
        if self.mine is None and drv.alloc is not None and drv.alloc["popping"] and h is not None:
            drv.alloc["evicted"][h] += 1
            if len(self.by_hash[h]) == 1:
                drv.counts["entries dropped"] += 1
                drv.counts["dropped_with_descendant"] += any(drv.descends(x, h) for x in self.by_hash if x != h)
        return orig["_pop_hash"](self, b)

    taps = {"allocate_slots": allocate_slots, "cache_blocks": cache_blocks, "free_request": free_request,
            "_new_blocks": _new_blocks, "_pop_hash": _pop_hash}
    for n in names:
        setattr(K, n, taps[n])
    try:
        yield
    finally:
        for n in names:
            setattr(K, n, orig[n])


def drive(inp: Input, st: dict, *, record: int | None = None) -> _Driver:
    """The unit over the whole trace, held to `_KV` (the module's docstring); the first `record` calls kept (all with
    None). Raises AssertionError where the two disagree."""
    S = SimpleNamespace(**st)
    drv = _Driver(S, inp, record)
    parent: dict[int, int] = {}

    def descends(x: int, h: int) -> bool:
        while x:
            x = parent.get(x, 0)
            if x == h:
                return True
        return False

    drv.descends = descends  # type: ignore[attr-defined]
    H = RP._Hashes
    intern = H.__call__

    def call(self, p, block):
        h = intern(self, p, block)
        parent[h] = p
        drv.hashes = self
        return h

    H.__call__ = call
    try:
        with _tapped(drv):
            rp = RP.replay(inp.engine, inp.requests, admit=inp.before)
    finally:
        H.__call__ = intern
    got = [list(s.rows) for s in rp.steps]
    if got != inp.rows:
        k = next((k for k, (a, b) in enumerate(zip(got, inp.rows)) if a != b), min(len(got), len(inp.rows)))
        raise AssertionError(f"the replay leaves the trace at step {k}")
    drv.counts["steps"] = len(got)
    return drv


# --- the certifier's vectors -------------------------------------------------------------------------------------------------

_FIELDS = ("slot", "r", "admit", "take", "free", "K", "wh", "ev", "nins", "new", "tau", "held", "ret")


def _op_row(op: C2.CallInputs) -> list[int]:
    return [int(getattr(op, f)) for f in _FIELDS]


def slice_of(name: str, calls: int, L: int) -> dict[str, Any]:
    """A trace's first `calls` calls as the certifier's vectors: the statics, the tokens they read, and each call as
    (kind, op fields, u0, known) with its refused variants' op fields."""
    inp = replay_input(FIXTURES / name)
    st = statics_for(inp, L=L)
    drv = drive(inp, st, record=calls)
    used: dict[int, int] = {}
    for c in drv.calls:
        used[c.op.r] = max(used.get(c.op.r, 0), c.known)
    return {"statics": st, "tokens": {str(i): list(inp.requests[i].tokens[:n]) for i, n in sorted(used.items())},
            "calls": [[c.kind, _op_row(c.op), c.u0, c.known, [[w, _op_row(b)] for w, b in c.refused]] for c in drv.calls]}


def ops_of(sl: dict[str, Any]) -> list[tuple[str, C2.CallInputs, list[tuple[str, C2.CallInputs]]]]:
    """A slice's calls as ops (the inverse of `slice_of`)."""
    S = SimpleNamespace(**sl["statics"])
    out = []
    for kind, row, u0, known, refused in sl["calls"]:
        f = dict(zip(_FIELDS, row))
        units = window(sl["tokens"][str(f["r"])][:known], u0, S) if known else ()
        mk = lambda r: C2.CallInputs(**{k: (bool(v) if k in ("admit", "take", "free") else v)  # noqa: E731
                                for k, v in zip(_FIELDS, r)}, units=units)
        out.append((kind, mk(row), [(w, mk(b)) for w, b in refused]))
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="write the certifier's vectors (the traces' SLICES) here")
    a = ap.parse_args(argv)
    for p in a.traces:
        inp = replay_input(p)
        drv = drive(inp, statics_for(inp, L=SLICES.get(p.name, {}).get("L"), D=4))
        print(p.name, json.dumps(dict(drv.counts), sort_keys=True))
    if a.out:
        doc = {"kind": "cache_step_v2_vectors", "slices": {p.name: slice_of(p.name, **SLICES[p.name]) for p in a.traces}}
        a.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
