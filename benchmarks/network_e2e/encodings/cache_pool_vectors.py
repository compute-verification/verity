"""`CacheStep_v2` (`verity.protocols.certifier.prover.cache_step_v2`) against vLLM's block pool as the replay models it
(`replay.BlockPool`) over a whole trace, and its difftest vectors.

`drive` replays a trace with its scheduler's pool tapped (not a fork's, not a dry run), and makes the unit's call for
each pool operation, in order:
- `allocate_slots` of a request admitted with nothing computed: its lookup, taking the hit if the allocation succeeds;
- an allocation that succeeds: a compute of its new blocks and the units `cache_blocks` caches in it;
- `cache_blocks` after a step (async scheduling caches a decode's block a step late): a compute with no new blocks;
- `free_request` (a stop or a preemption): a free.
Each call's witness is `BlockPool`'s own: the hit, its evictable blocks, the new blocks, the blocks held and returned. The
threshold stamp is the pool's e-th oldest queued stamp, the one stamp the unit accepts. After each call `drive` holds
the unit's pool to `BlockPool`'s. The same hashes are cached, each with its blocks in the same order and held by the same
requests' slots. The hashless free count is the same, and so is the order of the cached free blocks in the free
queue. Each eviction removes the blocks of the hashes vLLM evicts. Beside each honest call it makes refused ones:
- a lookup's hit a unit either side, and the hit a cache that never evicts would give where that one is longer (a
  hit past the cache after eviction);
- a lookup's evictable count one either side;
- a compute's threshold stamp at the queue's (e − 1)-th, (e + 1)-th or last queued copy, or between the e-th and the
  next. With e = 0 it is the oldest queued stamp (a made-up eviction);
- a free's returns one either side.
`drive` also checks, from `BlockPool` alone, that vLLM evicts a hash's last block only when no child of it is cached
(`dropped_with_descendant`; never at any drop means never a descendant), the fact `cache_step_v2`'s N = NB − 1 rests on.

    python3 cache_pool_vectors.py tests/fixtures/async-prefix-cache.jsonl tests/fixtures/async-crush.json.gz \\
        [--out ../../../verity/protocols/certifier/verifier/tests/fixtures/cache_step_v2_vectors.json]
    python3 cache_pool_vectors.py LABEL=CORPUS_TRACE_DIR ... --slots 256 --copies 64 --stamp-width 40 --sparse \\
        --arrivals first [--jobs N] [--json OUT]    # a served trace-corpus trace: R its max_num_seqs, SW its frees' clock
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as RP  # noqa: E402

from verity.protocols.certifier.prover import cache_step_v2 as C2  # noqa: E402

HERE = Path(__file__).parent
FIXTURES = HERE / "tests" / "fixtures"
VECTORS = HERE.parents[2] / "verity" / "protocols" / "certifier" / "verifier" / "tests" / "fixtures" / "cache_step_v2_vectors.json"

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
    def __init__(self, S, inp: Input, record: int | None, every: bool = True, progress=None) -> None:
        self.S, self.inp, self.record, self.every, self.progress = S, inp, record, every, progress
        self.ncalls, self.seen = 0, None
        self.pool = C2.empty(S)
        self.slots: dict[int, int] = {}
        self.calls: list[Call] = []
        self.counts: Counter = Counter()
        self.ever: set[int] = set()
        self.alloc: dict[str, Any] | None = None
        self.reqs: dict[int, Any] = {}
        self.hashes: Any = None
        self.returned: dict[int, set[int]] = {}     # the blocks each request's last preemption returned to the free queue

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
        self.ncalls += 1
        self.counts[kind] += 1
        if self.progress and self.ncalls % 10000 == 0:
            self.progress(self.ncalls, op.r)
        self.pool = out.pool
        self.counts["entries_max"] = max(self.counts["entries_max"], sum(x is not None for x in out.pool.table))
        self.counts["copies_max"] = max(self.counts["copies_max"], max((len(x.copies) for x in out.pool.table if x), default=0))
        self.counts["clock"] = out.pool.clock
        return out

    def lookup(self, kv, r, hit: Sequence[int], queued: set[int], take: bool) -> None:
        """Request r's lookup, its `hit` the blocks it hits and `queued` those of them free before its allocation."""
        S, wh, ev = self.S, len(hit), len(queued)
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
        if wh and self.counts["evicted"]:
            self.counts["hits past the first eviction"] += 1
        if wh == cap and (cap + 1) * S.U <= r.num_tokens and r.block_hashes[cap] in kv.by_hash:
            self.counts["hits stopped a token short with the next block cached"] += 1
        if any(len(kv.by_hash[h]) > 1 for h in r.block_hashes[:wh]):
            self.counts["hits on a hash two blocks hold"] += 1
        own = [b for b in hit if b in self.returned.get(r.i, ())]
        if own:
            self.counts["resumes hitting the blocks their preemption returned"] += 1
            self.counts["resumes taking the blocks their preemption returned"] += take
            if take:
                self.counts["returned blocks a resume took from the free queue"] += sum(b in queued for b in own)
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
            KU = C2.widths(S).KU
            freed = Counter(s // KU for s in q[:e])
            if max(freed.values()) > 1:
                self.counts["evictions of blocks one free queued together"] += 1
            if e < len(q) and q[e] // KU == tau // KU:
                self.counts["evictions splitting one free's blocks"] += 1
        elif q:
            refused.append(("a made-up eviction", replace(op, tau=q[0] or 1)))
        hashes = self._hashes(kv, self.pool)
        gone = Counter(hashes[x] for x, ent in enumerate(self.pool.table) if ent for c in ent.copies
                       if e and not c.mask and c.stamp <= tau)
        self._call(kv, "compute", op, done, r.num_tokens, refused)
        if gone != evicted:
            raise AssertionError(f"call {len(self.calls)}: the unit evicts {dict(gone)}, vLLM {dict(evicted)}")

    def free(self, kv, r, held: int, ret: int, preempted: bool = False) -> None:
        if r.i not in self.slots:
            if held:
                raise AssertionError(f"request {r.i} frees {held} blocks it never allocated")
            return
        if preempted:
            self.counts["preempted frees"] += 1
            self.counts["preempted frees keeping shared blocks"] += ret < held
        op = C2.CallInputs(slot=self.slots[r.i], free=True, held=held, ret=ret)
        refused = [("the returns plus one", replace(op, ret=ret + 1))] + ([("the returns less one", replace(op, ret=ret - 1))]
                                                                        if ret else [])
        self._call(kv, "free", op, 0, 0, refused)
        del self.slots[r.i]

    # the pool against BlockPool's

    def _hashes(self, kv, pool: C2.Pool) -> list[int | None]:
        ids, N = self.hashes._ids, self.S.N
        out: list[int | None] = [None] * N
        for x, ent in enumerate(pool.table):
            chain, y = [], x
            while ent and y != N and out[y] is None:
                chain.append(y)
                y = pool.table[y].parent
            p = 0 if y == N else out[y]
            for z in reversed(chain):
                out[z] = p = ids[(p, pool.table[z].tokens)]
        return out

    def check(self, kv) -> None:
        seen = (self.ncalls, len(kv.free), len(kv.by_hash), len(kv.req_blocks))
        if not self.every and seen == self.seen:
            return
        self.seen = seen
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
            raise AssertionError(f"call {len(self.calls)}: the pool's cached blocks and holders aren't BlockPool's")
        free_cached = [(kv.block_hash[b], list(kv.by_hash[kv.block_hash[b]]).index(b)) for b in kv.free.d
                       if kv.block_hash[b] is not None]
        queued = sorted((c.stamp, hashes[x], d) for x, ent in enumerate(pool.table) if ent
                        for d, c in enumerate(ent.copies) if not c.mask)
        if [(h, d) for _, h, d in queued] != free_cached:
            raise AssertionError(f"call {len(self.calls)}: the queued copies by stamp aren't BlockPool's free queue")
        if pool.hf != len(kv.free) - len(free_cached):
            raise AssertionError(f"call {len(self.calls)}: hf {pool.hf}, BlockPool's hashless free "
                                 f"{len(kv.free) - len(free_cached)}")
        entry_of: dict[int, int] = {}
        for x, h in enumerate(hashes):
            if h is not None:
                entry_of.setdefault(h, x)
        for i, s in self.slots.items():
            want = S.N if not kv.num_cached.get(i) else entry_of[self.reqs[i].block_hashes[kv.num_cached[i] - 1]]
            if pool.spfx[s] != want:
                raise AssertionError(f"call {len(self.calls)}: slot {s}'s walk entry {pool.spfx[s]}, want {want}")
        self.ever |= set(kv.by_hash)
        self.counts["checks"] += 1


@contextmanager
def _tapped(drv: _Driver):
    K = RP.BlockPool
    names = ("allocate_slots", "cache_blocks", "free_request", "_get_new_blocks", "_pop_hash")
    orig = {n: getattr(K, n) for n in names}

    def allocate_slots(self, r, n, hit=(), *, full_sequence_must_fit=False, has_scheduled_reqs=True, dry=False):
        kw = {"full_sequence_must_fit": full_sequence_must_fit, "has_scheduled_reqs": has_scheduled_reqs, "dry": dry}
        if dry or self.mine is not None:
            return orig["allocate_slots"](self, r, n, hit, **kw)
        drv.reqs[r.i] = r
        look = r.status != RP.RUNNING and r.num_computed == 0
        queued = {b for b in hit if self.ref[b] == 0}
        drv.alloc = {"new": 0, "evicted": Counter(), "cache": None, "popping": False}
        ok = orig["allocate_slots"](self, r, n, hit, **kw)
        a, drv.alloc = drv.alloc, None
        drv.counts["allocations a full pool refuses"] += not ok
        if look:
            drv.lookup(self, r, hit, queued, ok)
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
        if r.status == RP.RUNNING:
            drv.returned[r.i] = {b for b in bs if self.ref[b] == 1}
        orig["free_request"](self, r)
        drv.free(self, r, held, ret, preempted=r.status == RP.RUNNING)
        drv.check(self)

    def _get_new_blocks(self, k):
        if self.mine is None and drv.alloc is not None:
            drv.alloc["new"], drv.alloc["popping"] = k, True
        try:
            return orig["_get_new_blocks"](self, k)
        finally:
            if self.mine is None and drv.alloc is not None:
                drv.alloc["popping"] = False

    def _pop_hash(self, b):
        h = self.block_hash[b]
        if self.mine is None and drv.alloc is not None and drv.alloc["popping"] and h is not None:
            drv.alloc["evicted"][h] += 1
            if len(self.by_hash[h]) == 1:
                drv.counts["entries dropped"] += 1
                drv.counts["dropped_with_descendant"] += any(drv.parent.get(x) == h for x in self.by_hash if x != h)
        return orig["_pop_hash"](self, b)

    taps = {"allocate_slots": allocate_slots, "cache_blocks": cache_blocks, "free_request": free_request,
            "_get_new_blocks": _get_new_blocks, "_pop_hash": _pop_hash}
    for n in names:
        setattr(K, n, taps[n])
    try:
        yield
    finally:
        for n in names:
            setattr(K, n, orig[n])


def drive(inp: Input, st: dict, *, record: int | None = None, every: bool = True, progress=None,
          arrivals: str = "admit") -> _Driver:
    """The unit over the whole trace, held to `BlockPool` (the module's docstring); the first `record` calls kept (all with
    None). Raises AssertionError where the two disagree. With `every` the pools are compared after each of `BlockPool`'s
    operations; without it only after one that made a call or changed `BlockPool`'s counts (its free blocks, cached
    hashes or holding requests), which a trace of thousands of blocks and steps needs. `progress(calls, request)` is
    told every 10,000 calls. Requests arrive at their `before` steps (`arrivals="admit"`), or as the advice encoder
    replays a record (`"first"`: request i has arrived at step k once k reaches its first scheduled step), which follows
    an async trace where an empty schedule shares its step count with the next."""
    S = SimpleNamespace(**st)
    drv = _Driver(S, inp, record, every, progress)
    parent: dict[int, int] = {}
    drv.parent = parent  # type: ignore[attr-defined]
    H = RP.BlockHashes
    intern = H.__call__

    def call(self, p, block):
        h = intern(self, p, block)
        parent[h] = p
        drv.hashes = self
        return h

    H.__call__ = call
    try:
        with _tapped(drv):
            if arrivals == "first":
                first = [len(inp.rows)] * len(inp.requests)
                for k in reversed(range(len(inp.rows))):
                    for j, _, _ in inp.rows[k]:
                        first[j] = k
                rp = RP.replay(inp.engine, inp.requests, decide=lambda i, k, _: k >= first[i])
            else:
                rp = RP.replay(inp.engine, inp.requests, admit=inp.before)
    finally:
        H.__call__ = intern
    # a step's rows compare as a set (`replay.compare`'s): a served trace keeps Model Runner V2's batch order (sorted by
    # n), the replay the scheduler's
    got = [sorted(s.rows) for s in rp.steps]
    if got != [sorted(r) for r in inp.rows]:
        k = next((k for k, (a, b) in enumerate(zip(got, inp.rows)) if a != sorted(b)), min(len(got), len(inp.rows)))
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


def _drive_one(job: tuple[str, Path, dict[str, Any]]) -> dict[str, Any]:
    label, p, o = job
    t0 = time.perf_counter()
    row: dict[str, Any] = {"trace": label}
    try:
        inp = replay_input(p)
        e = inp.engine
        R = max(o["slots"], e.max_num_seqs) if o["slots"] else 16
        st = statics_for(inp, L=SLICES.get(p.name, {}).get("L"), R=R, D=o["copies"], SW=o["stamp_width"])
        row |= {"requests": len(inp.requests), "num_gpu_blocks": e.num_gpu_blocks, "block_size": e.block_size,
                "async": e.async_scheduling, "statics": st}
        def progress(calls: int, r: int) -> None:
            print(f"{label}: {calls} calls, request {r} of {len(inp.requests)}, {time.perf_counter() - t0:.0f} s", flush=True)

        row["counts"] = dict(drive(inp, st, every=not o["sparse"], progress=progress if o["sparse"] else None,
                                   arrivals=o["arrivals"]).counts)
    except (AssertionError, ValueError, RuntimeError, OSError) as ex:  # a disagreement, or a trace the replay can't read
        row["error"] = f"{type(ex).__name__}: {ex}"
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="step traces or trace-corpus trace dirs, each PATH or LABEL=PATH")
    ap.add_argument("--out", type=Path, help="write the certifier's vectors (the traces' SLICES) here")
    ap.add_argument("--slots", type=int, help="R: at least the engine's max_num_seqs (default 16, the vectors')")
    ap.add_argument("--copies", type=int, default=4, help="D, the copies an entry holds")
    ap.add_argument("--stamp-width", type=int, default=16, help="SW: it must hold the clock, KU a free")
    ap.add_argument("--sparse", action="store_true", help="compare the pools only where either changed (`drive`'s every)")
    ap.add_argument("--arrivals", choices=("admit", "first"), default="admit", help="`drive`'s arrivals")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", type=Path, help="write each trace's counts, statics and seconds here")
    a = ap.parse_args(argv)
    o = {"slots": a.slots, "copies": a.copies, "stamp_width": a.stamp_width, "sparse": a.sparse, "arrivals": a.arrivals}
    jobs = [(t.split("=", 1)[0], Path(t.split("=", 1)[1]), o) if "=" in t else (Path(t).name, Path(t), o) for t in a.traces]
    with ProcessPoolExecutor(a.jobs) if a.jobs > 1 else nullcontext() as ex:
        rows = []
        for row in (ex.map(_drive_one, jobs) if ex else map(_drive_one, jobs)):
            print(row["trace"], f"{row['seconds']} s", row.get("error") or json.dumps(row["counts"], sort_keys=True), flush=True)
            rows.append(row)
    if a.json:
        a.json.write_text(json.dumps({"kind": "cache_step_v2_drive", "rows": rows}, indent=1) + "\n")
    if any("error" in r for r in rows):
        raise SystemExit(1)
    if a.out:
        doc = {"kind": "cache_step_v2_vectors", "slices": {p.name: slice_of(p.name, **SLICES[p.name]) for _, p, _ in jobs}}
        a.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
