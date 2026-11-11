"""A prototype of `CacheStep_v3`'s reference: vLLM's block pool (the vLLM integration's `advice_code.pool.BlockPool`,
which the replay runs) as a machine over word-addressed memories, every read and write of which is one access under
offline memory checking.

Memories (address → value; an address starts at its initial value, a public function of NB):
- B[b], b < NB: block b's free-queue links (prev, next), its links among the blocks cached under its key (cprev, cnext),
  its holders (ref) and the request that computed it (owner, 1 + its index; 0 for none). B[0] is the null block, which
  vLLM never hands out; its (prev, next) are the free queue's (tail, head). Initially the queue is 1, …, NB − 1.
- K[b]: the key block b is cached under, 0 for none.
- M[key]: (first, last), the blocks cached under key in vLLM's order (`by_hash[h]`), a list threaded through B.
- RB[(r, k)]: request r's k-th block.

The step machine (ReplayStep_v1) holds each request's block count and cached count, and the free count, in its own
state. The pool returns it a lookup's hit and evictable count, and a free's returns.

The operations, each a fixed list of accesses (`SHAPE`; an access the operation doesn't need is a no-op one):
- lookup, one unit of a request's walk: M[key], then B[first] (its holders: evictable or not).
- take, one hit block: B[b] (holders + 1, out of the queue if it was free), the queue neighbours B[prev], B[next], and
  RB[(r, k)].
- alloc, one new block: B[0] (pop the head), B[b], B[next], RB[(r, k)], and its eviction: K[b], M[key], and the
  neighbours among its copies B[cprev], B[cnext].
- cache, one newly full block: RB[(r, k)], K[b], M[key] (append), B[last], B[b].
- free, one block of a freed request: RB[(r, k)], B[b], K[b] (which end of the queue), and the two blocks it goes
  between: after the run of hashless blocks this free has pushed to the front (vLLM's `reversed(first)` pushes), or
  before B[0] (the back).

Every access is a read-modify-write: it reads (a, v, t), t the time v was written, and writes (a, v', now). The log is
consistent iff, as multisets, reads ∪ final = writes ∪ initial, and every read's t is below its now (`check`); the
circuit checks the multiset equality as a fingerprint under a live coin (`fingerprint`).
"""
from __future__ import annotations

import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network_e2e" / "encodings"))
import replay as RP

from verity_vllm.pipeline.advice_code.schedule import PREEMPTED, WAITING

#: accesses per operation in the circuit's fixed shape
SHAPE = {"lookup": 2, "take": 4, "alloc": 8, "cache": 5, "free": 5}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Blk(NamedTuple):
    prev: int = 0
    next: int = 0
    cprev: int = 0
    cnext: int = 0
    ref: int = 0
    owner: int = 0


class PoolOp(NamedTuple):
    kind: str
    desc: dict
    base: int                   # 8·(o + 1) for the o-th operation: its position j accesses at base + j
    at: dict                    # position -> index in the log


class Ram:
    """Memories under offline memory checking: `access` logs (address, value read, its time, value written, now). Each
    access belongs to an operation (`begin`) and is at one of its positions, whose time is fixed: operation o's position
    j is at 8·(o + 1) + j, used or not (in the circuit, slot o's times are wiring). Without `keep`, only the current
    operation and no log are kept (a long load's memory), so there is nothing for `check`."""

    def __init__(self, initial: Callable[[tuple], Any], keep: bool = True) -> None:
        self.initial = initial
        self.keep = keep
        self.cells: dict[tuple, tuple[Any, int]] = {}
        self.log: list[tuple[tuple, Any, int, Any, int]] = []
        self.accesses = 0
        self.first: dict[tuple, Any] = {}
        self.ops: list[PoolOp] = []
        self.begun = 0
        self.touch: Callable[[tuple], None] | None = None

    def begin(self, kind: str, **desc: Any) -> None:
        self.begun += 1
        op = PoolOp(kind, desc, 8 * self.begun, {})
        if self.keep:
            self.ops.append(op)
        else:
            self.ops[:] = [op]

    def value(self, a: tuple) -> Any:
        """The value at a now, without an access: an operation's write may depend on a read it makes later (the circuit
        has every read as witness), so this is read only at an address the operation hasn't yet accessed."""
        c = self.cells.get(a)
        return c[0] if c else self.initial(a)

    def access(self, pos: int, a: tuple, f: Callable[[Any], Any] = lambda v: v) -> Any:
        op = self.ops[-1]
        assert pos < SHAPE[op.kind] and pos not in op.at, (op.kind, pos)
        now = op.base + pos
        if a in self.cells:
            v, t = self.cells[a]
        else:
            v, t = self.initial(a), 0
            self.first[a] = v
        assert t < now
        w = f(v)
        self.cells[a] = (w, now)
        op.at[pos] = self.accesses
        self.accesses += 1
        if self.touch:
            self.touch(a)
        if self.keep:
            self.log.append((a, v, t, w, now))
        return v

    def multisets(self) -> tuple[Counter, Counter]:
        """(reads ∪ final, writes ∪ initial) over the addresses accessed (an untouched one's initial and final agree)."""
        rs, ws = Counter(), Counter()
        for a, v, t, w, now in self.log:
            rs[(a, v, t)] += 1
            ws[(a, w, now)] += 1
        for a, v in self.first.items():
            ws[(a, v, 0)] += 1
            rs[(a, *self.cells[a])] += 1
        return rs, ws


def check(ram: Ram) -> bool:
    rs, ws = ram.multisets()
    return rs == ws and all(t < now for _, _, t, _, now in ram.log)


# --- the fingerprint, over GF(2^128) mod x^128 + x^7 + x^2 + x + 1 ------------------------------------------------------

_POLY = (1 << 128) | 0x87


def gf_mul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 128:
            a ^= _POLY
    return r


class Widths(NamedTuple):
    """Field widths: block ids (NW), requests (RW), a request's block index (KW), holders (REFW), times (TW), keys (HW).
    The defaults are the GLM-5.3 class's: NB ≤ 2^13, 2^20 requests a window, L/B ≤ 2^11, R ≤ 2^9, 2^25 operations."""
    NW: int = 13
    RW: int = 20
    KW: int = 11
    REFW: int = 9
    TW: int = 28
    HW: int = 128


GLM = Widths()
TAGS = {"B": 0, "K": 1, "M": 2, "RB": 3}


def fields(rec: tuple, S: Widths = GLM) -> list[tuple[int, int]]:
    """A record (address, value, time) as (value, width) fields, LSB first: the tag, the address, the value, the time."""
    a, v, t = rec
    head = [(TAGS[a[0]], 2)]
    if a[0] == "B":
        return head + [(a[1], S.NW), *((x, S.NW) for x in v[:4]), (v.ref, S.REFW), (v.owner, S.RW), (t, S.TW)]
    if a[0] == "K":
        return head + [(a[1], S.NW), (v, S.HW), (t, S.TW)]
    if a[0] == "M":
        return head + [(a[1], S.HW), (v[0], S.NW), (v[1], S.NW), (t, S.TW)]
    return head + [(a[1], S.RW), (a[2], S.KW), (v, S.NW), (t, S.TW)]


def pack(rec: tuple, S: Widths = GLM) -> list[int]:
    """A record's bits in 128-bit chunks."""
    x, s = 0, 0
    for val, w in fields(rec, S):
        assert 0 <= val < 1 << w, (rec, val, w)
        x |= val << s
        s += w
    return [(x >> (128 * i)) & ((1 << 128) - 1) for i in range(cdiv(s, 128))]


def element(rec: tuple, gamma: int, S: Widths = GLM) -> int:
    """Σ_i γ^i c_i over the record's chunks c_i."""
    e, g = 0, 1
    for c in pack(rec, S):
        e ^= gf_mul(g, c)
        g = gf_mul(g, gamma)
    return e


def fingerprint(ms: Counter, alpha: int, gamma: int, S: Widths = GLM) -> int:
    """∏ (α + `element`) over the records (addition is XOR)."""
    out = 1
    for rec, n in ms.items():
        e = element(rec, gamma, S)
        for _ in range(n):
            out = gf_mul(out, alpha ^ e)
    return out


# --- the pool -----------------------------------------------------------------------------------------------------------


class RamPool:
    """vLLM's block pool over `Ram` (the module's docstring), with `BlockPool`'s calls."""

    def __init__(self, B: int, L: int, caching: bool, watermark_blocks: int, NB: int, keep: bool = True) -> None:
        self.B, self.L, self.caching, self.watermark_blocks, self.NB = B, L, caching, watermark_blocks, NB
        self.ram = Ram(self._initial, keep)
        self.nf = NB - 1
        self.held: dict[int, int] = {}
        self.num_cached: dict[int, int] = {}
        self.looked: dict[int, tuple[list[int], int]] = {}
        self.ops: Counter = Counter()

    @classmethod
    def like(cls, kv: Any, keep: bool = True) -> RamPool:
        return cls(kv.B, kv.L, kv.caching, kv.watermark_blocks, len(kv.ref), keep)

    def _initial(self, a: tuple) -> Any:
        if a[0] == "B":
            b, NB = a[1], self.NB
            if b == 0:
                return Blk(NB - 1 if NB > 1 else 0, 1 if NB > 1 else 0)
            return Blk(b - 1, b + 1 if b + 1 < NB else 0)
        if a[0] == "M":
            return (0, 0)
        return 0

    def hit(self, r: Any) -> list[int]:
        if not self.caching:
            return []
        out, ev = [], 0
        for h in r.block_hashes[: (r.num_tokens - 1) // self.B]:
            self._begin("lookup", key=h)
            first, _ = self.ram.access(0, ("M", h))
            if not first:
                break
            ev += self.ram.access(1, ("B", first)).ref == 0
            out.append(first)
        self.looked[r.i] = (out, ev)
        return out

    def allocate_slots(self, r: Any, n: int, hit: Sequence[int] = (), *, full_sequence_must_fit: bool = False,
                       has_scheduled_reqs: bool = True) -> bool:
        hit = list(hit)
        ev = 0
        if hit:
            seen, ev = self.looked[r.i]
            assert seen == hit, (seen, hit)
        held = self.held.get(r.i, 0)
        total = min(r.num_computed + len(hit) * self.B, self.L)
        wm = self.watermark_blocks if has_scheduled_reqs and r.status in (WAITING, PREEMPTED) else 0

        def need(tokens: int) -> int:
            return max(cdiv(tokens, self.B) - len(hit) - held, 0) + ev

        if full_sequence_must_fit and need(min(r.num_tokens, self.L)) + wm > self.nf:
            return False
        need_slot = min(total + n, self.L)
        if need(need_slot) + wm > self.nf:
            return False
        for b in hit:
            self._take(r.i, held, b)
            held += 1
        if hit:
            self.num_cached[r.i] = held
        for _ in range(cdiv(need_slot, self.B) - held):
            self._alloc(r.i, held)
            held += 1
        self.held[r.i] = held
        if self.caching:
            self.cache_blocks(r, min(total + n, r.num_tokens))
        return True

    def _begin(self, kind: str, **desc: Any) -> None:
        self.ops[kind] += 1
        self.ram.begin(kind, **desc)

    def _take(self, i: int, k: int, b: int) -> None:
        self._begin("take", rid=i, k=k, b=b)
        old = self.ram.access(0, ("B", b),
                              lambda v: v._replace(ref=v.ref + 1, **({"prev": 0, "next": 0} if not v.ref else {})))
        if not old.ref:
            p, n = old.prev, old.next
            self.ram.access(1, ("B", p), lambda v: v._replace(next=n))
            self.ram.access(2, ("B", n), lambda v: v._replace(prev=p))
            self.nf -= 1
        self._set_rb(3, i, k, b)

    def _set_rb(self, pos: int, i: int, k: int, b: int) -> None:
        def put(v: int) -> int:
            assert v == 0, (i, k, v)
            return b
        self.ram.access(pos, ("RB", i, k), put)

    def _alloc(self, i: int, k: int) -> None:
        self._begin("alloc", rid=i, k=k)
        b = self.ram.value(("B", 0)).next
        assert b, "alloc from an empty free queue"
        n = self.ram.value(("B", b)).next
        self.ram.access(0, ("B", 0), lambda v: v._replace(next=n))
        old = self.ram.access(1, ("B", b), lambda v: Blk(ref=1, owner=i + 1))
        self.ram.access(2, ("B", n), lambda v: v._replace(prev=0))
        self.nf -= 1
        if self.caching:
            key = self.ram.access(3, ("K", b), lambda v: 0)
            if key:
                self.ops["evict"] += 1
                cp, cn = old.cprev, old.cnext
                self.ram.access(4, ("M", key), lambda v: (cn if v[0] == b else v[0], cp if v[1] == b else v[1]))
                if cp:
                    self.ram.access(5, ("B", cp), lambda v: v._replace(cnext=cn))
                if cn:
                    self.ram.access(6, ("B", cn), lambda v: v._replace(cprev=cp))
        self._set_rb(7, i, k, b)

    def cache_blocks(self, r: Any, num_tokens: int) -> None:
        """vLLM pops a block's old key before caching it under a new one; the blocks a request caches (past its cached
        count) were all allocated since, so have none, and the cache operation requires that."""
        done = self.num_cached.get(r.i, 0)
        full = num_tokens // self.B
        if done >= full:
            return
        for k in range(done, full):
            h = r.block_hashes[k]
            self._begin("cache", rid=r.i, k=k, key=h)
            b = self.ram.access(0, ("RB", r.i, k))

            def put(v: int) -> int:
                assert v == 0, f"request {r.i}: block {b} at {k} is already cached"
                return h
            self.ram.access(1, ("K", b), put)
            _, l = self.ram.access(2, ("M", h), lambda v: (v[0] or b, b))
            if l:
                self.ram.access(3, ("B", l), lambda v: v._replace(cnext=b))
            self.ram.access(4, ("B", b), lambda v: v._replace(cprev=l, cnext=0))
        self.num_cached[r.i] = full

    def free_request(self, r: Any) -> int:
        """Frees r's blocks, deepest first; returns how many it returned to the free queue. `cur` is the last block this
        free pushed to the front (0 for none): vLLM's front pushes, in `reversed(first)`, leave its hashless blocks at
        the queue's front in the order freed."""
        held = self.held.pop(r.i, 0)
        self.num_cached.pop(r.i, None)
        cur, ret = 0, 0
        for k in reversed(range(held)):
            self._begin("free", rid=r.i, k=k, cur=cur)
            b = self.ram.access(0, ("RB", r.i, k), lambda v: 0)
            if self.ram.value(("B", b)).ref > 1:
                self.ram.access(1, ("B", b), lambda v: v._replace(ref=v.ref - 1))
                continue
            key = self.ram.value(("K", b)) if self.caching else 0
            if not key:
                c, n = cur, self.ram.value(("B", cur)).next
                self.ram.access(1, ("B", b), lambda v: v._replace(ref=0, prev=c, next=n))
                if self.caching:
                    self.ram.access(2, ("K", b))
                self.ram.access(3, ("B", c), lambda v: v._replace(next=b))
                self.ram.access(4, ("B", n), lambda v: v._replace(prev=b))
                cur = b
            else:
                t = self.ram.value(("B", 0)).prev
                self.ram.access(1, ("B", b), lambda v: v._replace(ref=0, prev=t, next=0))
                self.ram.access(2, ("K", b))
                self.ram.access(3, ("B", t), lambda v: v._replace(next=b))
                self.ram.access(4, ("B", 0), lambda v: v._replace(prev=b))
            ret += 1
            self.nf += 1
        return ret

    # --- the state, read back for the difftest ---

    def queue(self) -> list[int]:
        out, b = [], self.ram.value(("B", 0)).next
        while b:
            out.append(b)
            b = self.ram.value(("B", b)).next
            assert len(out) < self.NB, "the free queue has a cycle"
        return out

    def copies(self, key: int) -> list[int]:
        out, b = [], self.ram.value(("M", key))[0]
        while b:
            out.append(b)
            b = self.ram.value(("B", b)).cnext
        return out

    def keys(self) -> set[int]:
        return {a[1] for a, (v, _) in self.ram.cells.items() if a[0] == "M" and v[0]}


def disagreement(kv: Any, pool: RamPool) -> str | None:
    """Where the RAM pool's state differs from `BlockPool`'s, or None."""
    if pool.queue() != list(kv.free.d):
        return "free queue"
    if pool.nf != len(kv.free):
        return "free count"
    for b in range(pool.NB):
        v = pool.ram.value(("B", b))
        if b and v.ref != kv.ref[b]:
            return f"ref of block {b}"
        if b and (v.owner - 1 if v.owner else None) != kv.owner[b]:
            return f"owner of block {b}"
        if pool.ram.value(("K", b)) != (kv.block_hash[b] or 0):
            return f"key of block {b}"
    if pool.keys() != set(kv.by_hash):
        return "cached keys"
    for h, bs in kv.by_hash.items():
        if pool.copies(h) != list(bs):
            return f"copies of key {h}"
    for i, bs in kv.req_blocks.items():
        if [pool.ram.value(("RB", i, k)) for k in range(pool.held.get(i, 0))] != bs:
            return f"blocks of request {i}"
    if any(n and i not in kv.req_blocks for i, n in pool.held.items()):
        return "a request holds blocks only here"
    if {i: n for i, n in pool.num_cached.items()} != {i: n for i, n in kv.num_cached.items()}:
        return "cached counts"
    return None


# --- the difftest: a replay with its pool mirrored onto a RamPool -------------------------------------------------------


class Mirror:
    """Mirrors every call on the replay's pool (not a fork's, not a dry run) onto a RamPool and holds the two to each
    other after each; counts each step's operations, calls, block keys first used (RK writes) and lookup units a memo
    or certificate would save. Within `window` (step indices), collects the sparse addresses touched (M, RB, and RK:
    each lookup unit and cache reads its block's key)."""

    def __init__(self, every: int = 1, keep: bool = True, window: range | None = None) -> None:
        self.pool: RamPool | None = None
        self.kv: Any = None
        self.depth = 0
        self.every = every
        self.keep = keep
        self.window = window
        self.touched: dict[str, set] = {"M": set(), "RB": set(), "RK": set()}
        self.keyed: dict[int, int] = {}
        self.calls = 0
        self.steps: list[Counter] = [Counter()]
        self.counts: Counter = Counter()
        self.last_walk: dict[int, int] = {}
        self.walked: int | None = None
        self.last_result: dict[int, tuple] = {}
        #: each request's last executed step (`run` sets it)
        self.last_step: dict[int, int] = {}

    def _pool(self, kv: Any) -> RamPool:
        if self.pool is None:
            self.pool, self.kv = RamPool.like(kv, self.keep), kv
            self.pool.ram.touch = self._touch
        assert kv is self.kv, "one pool per replay"
        return self.pool

    def _in_window(self) -> bool:
        return self.window is not None and len(self.steps) - 1 in self.window

    def _touch(self, a: tuple) -> None:
        if a[0] in self.touched and self._in_window():
            self.touched[a[0]].add(a[1:])

    def _keys(self, rid: int, ks: range) -> None:
        if ks.stop > self.keyed.get(rid, 0):
            self.steps[-1]["keys"] += ks.stop - self.keyed.get(rid, 0)
            self.keyed[rid] = ks.stop
        if self._in_window():
            self.touched["RK"].update((rid, k) for k in ks)

    def _check(self, kv: Any) -> None:
        self.calls += 1
        if self.calls % self.every == 0:
            where = disagreement(kv, self.pool)
            if where:
                raise AssertionError(f"call {self.calls}: the RAM pool leaves vLLM's at its {where}")

    def certify(self, p: RamPool, r: Any, n: int, hit: list[int], kw: dict) -> None:
        """Counts, for a lookup whose allocation is refused, whether a one-unit refusal certificate would do instead of
        the walk: x, the hit length of the request's last full walk (a function of the history, so the witness stays
        unique), still a miss (so the hit is at most x), and the blocks needed with at most x hit, not counting
        evictable ones, already past the free count. The walk's units are what it saves."""
        h = len(hit)
        units = min(h + 1, (r.num_tokens - 1) // p.B)
        x = self.last_walk.get(r.i)
        held = p.held.get(r.i, 0)
        wm = p.watermark_blocks if kw["has_scheduled_reqs"] and r.status in (WAITING, PREEMPTED) else 0
        total = min(r.num_computed + h * p.B, p.L)

        def need_lb(tokens: int) -> int:
            return max(cdiv(tokens, p.B) - x - held, 0)

        miss = x is not None and (x >= len(r.block_hashes) or not p.ram.value(("M", r.block_hashes[x]))[0])
        ok = (miss and x < (r.num_tokens - 1) // p.B
              and ((kw["full_sequence_must_fit"] and need_lb(min(r.num_tokens, p.L)) + wm > p.nf)
                   or need_lb(min(total + n, p.L)) + wm > p.nf))
        if ok:
            self.counts["certified refusals"] += 1
            self.counts["lookup units a certificate saves"] += max(units - 1, 0)
            self.steps[-1]["certificate saves"] += max(units - 1, 0)
        else:
            self.last_walk[r.i] = h

    def _tally(self, before: Counter) -> None:
        """Adds a call's operations to its step's; a call with none is one the step machine makes no pool call for."""
        d = self.pool.ops - before
        self.steps[-1].update(d)
        if any(d[k] for k in SHAPE):
            self.counts["calls"] += 1
            self.steps[-1]["calls"] += 1


@contextmanager
def mirrored(m: Mirror, K: type = RP.BlockPool, S: type = RP.Scheduler):
    names = ("hit", "allocate_slots", "cache_blocks", "free_request")
    orig = {n: getattr(K, n) for n in names}
    sched = S.schedule

    def hit(self, r):
        out = orig["hit"](self, r)
        if self.mine is None:
            p = m._pool(self)
            before = Counter(p.ops)
            got = p.hit(r)
            m._tally(before)
            if got != out:
                raise AssertionError(f"lookup of request {r.i}: {got}, vLLM {out}")
            m.counts["lookups"] += 1
            m.walked = r.i
            if p.caching:
                units = p.ops["lookup"] - before["lookup"]
                m._keys(r.i, range(units))
                res = (tuple(got), p.looked[r.i][1])
                if m.last_result.get(r.i) == res:
                    m.counts["repeated walks"] += 1
                    m.counts["lookup units in repeated walks"] += units
                    m.steps[-1]["repeated units"] += units
                m.last_result[r.i] = res
        return out

    def allocate_slots(self, r, n, hit=(), *, full_sequence_must_fit=False, has_scheduled_reqs=True, dry=False):
        kw = {"full_sequence_must_fit": full_sequence_must_fit, "has_scheduled_reqs": has_scheduled_reqs}
        if dry or self.mine is not None:
            return orig["allocate_slots"](self, r, n, hit, dry=dry, **kw)
        m.depth += 1
        try:
            ok = orig["allocate_slots"](self, r, n, hit, **kw)
        finally:
            m.depth -= 1
        p = m._pool(self)
        walked, m.walked = m.walked == r.i, None
        if walked and not ok:
            m.certify(p, r, n, hit, kw)
        before = Counter(p.ops)
        got = p.allocate_slots(r, n, hit, **kw)
        m._tally(before)
        if got != ok:
            raise AssertionError(f"allocate_slots of request {r.i}: {got}, vLLM {ok}")
        m.counts["allocations" if ok else "refused allocations"] += 1
        if hit and not ok:
            m.counts["refused after a lookup"] += 1
        m._check(self)
        return ok

    def cache_blocks(self, r, num_tokens):
        if self.mine is not None or m.depth:
            return orig["cache_blocks"](self, r, num_tokens)
        orig["cache_blocks"](self, r, num_tokens)
        p = m._pool(self)
        before = Counter(p.ops)
        done = p.num_cached.get(r.i, 0)
        p.cache_blocks(r, num_tokens)
        m._tally(before)
        m._keys(r.i, range(done, p.num_cached.get(r.i, done)))
        m._check(self)

    def free_request(self, r):
        if self.mine is not None:
            return orig["free_request"](self, r)
        ret = sum(1 for b in self.req_blocks.get(r.i, []) if self.ref[b] == 1)
        orig["free_request"](self, r)
        p = m._pool(self)
        before = Counter(p.ops)
        got = p.free_request(r)
        m._tally(before)
        if got != ret:
            raise AssertionError(f"free of request {r.i}: returns {got}, vLLM {ret}")
        m.counts["frees"] += 1
        m._check(self)

    def schedule(self, *a, **kw):
        m.steps.append(Counter())
        return sched(self, *a, **kw)

    taps = {"hit": hit, "allocate_slots": allocate_slots, "cache_blocks": cache_blocks, "free_request": free_request}
    for n in names:
        setattr(K, n, taps[n])
    S.schedule = schedule
    try:
        yield m
    finally:
        for n in names:
            setattr(K, n, orig[n])
        S.schedule = sched


def run(engine: RP.Engine, requests: Sequence[RP.RequestIO], *, every: int = 1, keep: bool = True,
        window: range | None = None, **how: Any) -> Mirror:
    """The replay (admitting as `how` says: `replay.replay`'s `admit` or `decide`) with its pool mirrored (`Mirror`);
    raises AssertionError where the two disagree, or (with `keep`) where the RAM log fails its memory check."""
    m = Mirror(every, keep, window)
    with mirrored(m):
        rp = RP.replay(engine, requests, **how)
    m.counts["steps"] = len(rp.steps)
    m.counts["tokens"] = sum(n for s in rp.steps for _, _, n in s.rows)
    m.last_step = {i: k for k, s in enumerate(rp.steps) for i, _, _ in s.rows}
    if m.pool is not None:
        where = disagreement(m.kv, m.pool)
        if where:
            raise AssertionError(f"at the end: the RAM pool leaves vLLM's at its {where}")
        if keep and not check(m.pool.ram):
            raise AssertionError("the RAM log fails its memory check")
    return m
