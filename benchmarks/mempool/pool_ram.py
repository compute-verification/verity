"""A prototype of `CacheStep_v3`'s reference: vLLM's block pool (`BlockPool`, as the replay's `_KV` and the vLLM
integration's `pool.py` model it) as a machine over word-addressed memories, every read and write of which is one access
under offline memory checking.

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
import replay as RP  # noqa: E402

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


class Ram:
    """Memories under offline memory checking: `access` logs (address, value read, its time, value written, now)."""

    def __init__(self, initial: Callable[[tuple], Any]) -> None:
        self.initial = initial
        self.cells: dict[tuple, tuple[Any, int]] = {}
        self.now = 0
        self.log: list[tuple[tuple, Any, int, Any, int]] = []
        self.first: dict[tuple, Any] = {}

    def value(self, a: tuple) -> Any:
        """The value at a now, without an access: an operation's write may depend on a read it makes later (the circuit
        has every read as witness), so this is read only at an address the operation hasn't yet accessed."""
        c = self.cells.get(a)
        return c[0] if c else self.initial(a)

    def access(self, a: tuple, f: Callable[[Any], Any] = lambda v: v) -> Any:
        self.now += 1
        if a in self.cells:
            v, t = self.cells[a]
        else:
            v, t = self.initial(a), 0
            self.first[a] = v
        w = f(v)
        self.cells[a] = (w, self.now)
        self.log.append((a, v, t, w, self.now))
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


#: widths of a record's fields, by memory (the GLM-5.3 class's: NB < 2^13, keys 128 bits, requests < 2^20, k < 2^11)
def _pack(rec: tuple, tw: int = 26) -> list[int]:
    (a, v, t) = rec
    tag = a[0]
    if tag == "B":
        fields = [(0, 3), (a[1], 13), *((x, 13) for x in v[:4]), (v.ref, 9), (v.owner, 20), (t, tw)]
    elif tag == "K":
        fields = [(1, 3), (a[1], 13), (v, 128), (t, tw)]
    elif tag == "M":
        fields = [(2, 3), (a[1], 128), (v[0], 13), (v[1], 13), (t, tw)]
    else:
        fields = [(3, 3), (a[1], 20), (a[2], 11), (v, 13), (t, tw)]
    x, s = 0, 0
    for val, w in fields:
        assert 0 <= val < 1 << w, (rec, val, w)
        x |= val << s
        s += w
    return [(x >> (128 * i)) & ((1 << 128) - 1) for i in range(cdiv(s, 128))]


def fingerprint(ms: Counter, alpha: int, gamma: int) -> int:
    """∏ (α + Σ_i γ^i c_i) over the records' 128-bit chunks c_i (addition is XOR)."""
    out = 1
    for rec, n in ms.items():
        chunks = _pack(rec)
        e, g = 0, 1
        for c in chunks:
            e ^= gf_mul(g, c)
            g = gf_mul(g, gamma)
        for _ in range(n):
            out = gf_mul(out, alpha ^ e)
    return out


# --- the pool -----------------------------------------------------------------------------------------------------------


class RamPool:
    """vLLM's block pool over `Ram` (the module's docstring), with `_KV`'s calls."""

    def __init__(self, B: int, L: int, caching: bool, watermark_blocks: int, NB: int) -> None:
        self.B, self.L, self.caching, self.watermark_blocks, self.NB = B, L, caching, watermark_blocks, NB
        self.ram = Ram(self._initial)
        self.nf = NB - 1
        self.held: dict[int, int] = {}
        self.num_cached: dict[int, int] = {}
        self.looked: dict[int, tuple[list[int], int]] = {}
        self.ops: Counter = Counter()

    @classmethod
    def like(cls, kv: Any) -> RamPool:
        return cls(kv.B, kv.L, kv.caching, kv.watermark_blocks, len(kv.ref))

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
            self.ops["lookup"] += 1
            first, _ = self.ram.access(("M", h))
            if not first:
                break
            ev += self.ram.access(("B", first)).ref == 0
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
        wm = self.watermark_blocks if has_scheduled_reqs and r.status in (RP._W, RP._P) else 0

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

    def _take(self, i: int, k: int, b: int) -> None:
        self.ops["take"] += 1
        old = self.ram.access(("B", b), lambda v: v._replace(ref=v.ref + 1, **({"prev": 0, "next": 0} if not v.ref else {})))
        if not old.ref:
            p, n = old.prev, old.next
            self.ram.access(("B", p), lambda v: v._replace(next=n))
            self.ram.access(("B", n), lambda v: v._replace(prev=p))
            self.nf -= 1
        self._set_rb(i, k, b)

    def _set_rb(self, i: int, k: int, b: int) -> None:
        def put(v: int) -> int:
            assert v == 0, (i, k, v)
            return b
        self.ram.access(("RB", i, k), put)

    def _alloc(self, i: int, k: int) -> None:
        self.ops["alloc"] += 1
        b = self.ram.value(("B", 0)).next
        assert b, "alloc from an empty free queue"
        n = self.ram.value(("B", b)).next
        self.ram.access(("B", 0), lambda v: v._replace(next=n))
        old = self.ram.access(("B", b), lambda v: Blk(ref=1, owner=i + 1))
        self.ram.access(("B", n), lambda v: v._replace(prev=0))
        self.nf -= 1
        if self.caching:
            key = self.ram.access(("K", b), lambda v: 0)
            if key:
                self.ops["evict"] += 1
                self._unlink(key, b, old.cprev, old.cnext)
        self._set_rb(i, k, b)

    def _unlink(self, key: int, b: int, cp: int, cn: int) -> None:
        f, l = self.ram.access(("M", key), lambda v: (cn if v[0] == b else v[0], cp if v[1] == b else v[1]))
        if cp:
            self.ram.access(("B", cp), lambda v: v._replace(cnext=cn))
        if cn:
            self.ram.access(("B", cn), lambda v: v._replace(cprev=cp))

    def cache_blocks(self, r: Any, num_tokens: int) -> None:
        done = self.num_cached.get(r.i, 0)
        full = num_tokens // self.B
        if done >= full:
            return
        for k in range(done, full):
            self.ops["cache"] += 1
            h = r.block_hashes[k]
            b = self.ram.access(("RB", r.i, k))
            old = self.ram.value(("K", b))
            if old:
                self.ops["recache"] += 1
                cur = self.ram.value(("B", b))
                self._unlink(old, b, cur.cprev, cur.cnext)
            self.ram.access(("K", b), lambda v: h)
            _, l = self.ram.access(("M", h), lambda v: (v[0] or b, b))
            if l:
                self.ram.access(("B", l), lambda v: v._replace(cnext=b))
            self.ram.access(("B", b), lambda v: v._replace(cprev=l, cnext=0))
        self.num_cached[r.i] = full

    def free_request(self, r: Any) -> int:
        """Frees r's blocks, deepest first; returns how many it returned to the free queue."""
        held = self.held.pop(r.i, 0)
        self.num_cached.pop(r.i, None)
        cur, ret = 0, 0
        for k in reversed(range(held)):
            self.ops["free"] += 1
            b = self.ram.access(("RB", r.i, k), lambda v: 0)
            if self.ram.value(("B", b)).ref > 1:
                self.ram.access(("B", b), lambda v: v._replace(ref=v.ref - 1))
                continue
            key = self.ram.value(("K", b)) if self.caching else 0
            if not key:
                c, n = cur, self.ram.value(("B", cur)).next
                self.ram.access(("B", b), lambda v: v._replace(ref=0, prev=c, next=n))
                if self.caching:
                    self.ram.access(("K", b))
                self.ram.access(("B", c), lambda v: v._replace(next=b))
                self.ram.access(("B", n), lambda v: v._replace(prev=b))
                cur = b
            else:
                t = self.ram.value(("B", 0)).prev
                self.ram.access(("B", b), lambda v: v._replace(ref=0, prev=t, next=0))
                self.ram.access(("K", b))
                self.ram.access(("B", t), lambda v: v._replace(next=b))
                self.ram.access(("B", 0), lambda v: v._replace(prev=b))
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
    """Where the RAM pool's state differs from `_KV`'s (or `BlockPool`'s), or None."""
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
    other after each; counts each step's operations."""

    def __init__(self, every: int = 1) -> None:
        self.pool: RamPool | None = None
        self.kv: Any = None
        self.depth = 0
        self.every = every
        self.calls = 0
        self.steps: list[Counter] = [Counter()]
        self.counts: Counter = Counter()

    def _pool(self, kv: Any) -> RamPool:
        if self.pool is None:
            self.pool, self.kv = RamPool.like(kv), kv
        assert kv is self.kv, "one pool per replay"
        return self.pool

    def _check(self, kv: Any) -> None:
        self.calls += 1
        if self.calls % self.every == 0:
            where = disagreement(kv, self.pool)
            if where:
                raise AssertionError(f"call {self.calls}: the RAM pool leaves vLLM's at its {where}")

    def _tally(self, before: Counter) -> None:
        d = self.pool.ops - before
        self.steps[-1].update(d)


@contextmanager
def mirrored(m: Mirror, K: type = RP._KV, S: type = RP._Sched):
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
        p.cache_blocks(r, num_tokens)
        m._tally(before)
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


def run(engine: RP.Engine, requests: Sequence[RP.RequestIO], *, every: int = 1, **how: Any) -> Mirror:
    """The replay (admitting as `how` says: `replay.replay`'s `admit` or `decide`) with its pool mirrored (`Mirror`);
    raises AssertionError where the two disagree, or where the RAM log fails its memory check."""
    m = Mirror(every)
    with mirrored(m):
        rp = RP.replay(engine, requests, **how)
    m.counts["steps"] = len(rp.steps)
    m.counts["tokens"] = sum(n for s in rp.steps for _, _, n in s.rows)
    if m.pool is not None:
        where = disagreement(m.kv, m.pool)
        if where:
            raise AssertionError(f"at the end: the RAM pool leaves vLLM's at its {where}")
        if not check(m.pool.ram):
            raise AssertionError("the RAM log fails its memory check")
    return m
