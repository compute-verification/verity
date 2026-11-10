"""vLLM's prefix-cache block pool as memory-checked RAM: a window's pool calls as one stream of access slots, each slot one
read-modify-write of one memory, under offline memory checking (`MemCheck_v1`, `verity.definitions.circuits.boolean.memcheck`).

    PoolSlot_v1{NB, RW, KW, REFW, TW}
        (carry, start: Value<1>, kind: Array<3, Value<1>>, r: Array<RW, Value<1>>, a: Array<CW, Value<1>>,
         b: Array<CW, Value<1>>, v: F, t: Array<TW, Value<1>>, alpha: F, gamma: F) -> (carry', Value<1>)
    PoolEntry_v1{NB, RW, KW, REFW, TW}
        (carry, en: Value<1>, tag: Array<3, Value<1>>, a: F, v0: F, v: F, t: Array<TW, Value<1>>, alpha: F, gamma: F)
        -> (carry', Value<1>)
    CacheStep_v3{NS, NE, NB, RW, KW, REFW, TW}
        (alpha: F, gamma: F, head: Array<NW, Value<1>>, tail: Array<NW, Value<1>>, start, kind, r, a, b, v, t (NS of each),
         l_en, l_tag, l_a, l_v0, l_v, l_t (NE of each)) -> (ok: Value<1>, head', tail', calls: Array<TW, Value<1>>, P: F)
        F = Array<128, Value<1>>, an element of GF(2^128); NW = bitlen(NB − 1), CW = KW + 1

`CacheStep_v2` (`cache_step_v2`) holds vLLM's pool as a trie in one table, which every call reads and writes whole, so
its price grows with the table. Here the pool is the five memories vLLM's `BlockPool` is made of, each call a fixed
sequence of accesses to them, and a window's calls run in one stream of NS slots that each perform one access: which
call runs where is private, so every slot can perform any call's access at any position. A window costs NS slots,
whatever the pool's size. `Stream` is the specification in Python; `benchmarks/mempool/stream.py` holds it to the
replay's pool (`pool_ram`, itself held to vLLM's `_KV`) after every call, with prefix caching on.

**The memories** (tag: address → value; a value's fields least significant first):
- B (0): b → (prev, next, cprev, cnext, ref, owner, stamp), b < NB: the free queue's links, the links among the blocks
  cached under one key, the holders, 1 + the request that allocated it, and the memo's stamp. The free queue's ends
  (head, tail) are the stream's own registers, so vLLM's null block 0, the queue's sentinel, is never accessed: a link
  0 is the sentinel.
- K (1): b → the key b is cached under, 0 for none.
- M (2): key → (first, last, stamp), the blocks cached under the key in vLLM's order (`by_hash`), a list threaded
  through B, and the memo's stamp.
- RB (3): (r, k) → request r's k-th block, 0 for none.
- RK (4): (r, k) → the key of request r's k-th block of tokens, read only.
B's and K's addresses are b, M's a key, RB's and RK's r | k << RW. A key is a block's 128-bit print under the window's
coin (`verity.definitions.commitments.gates.block_key`), which the caller makes each RK address's initial value.

**The calls** (`Call`), each one record on the bus to `ReplayStep_v1` (seq, kind, r, a, b, x, y), its results x and y:
- hit (1), r, a = the walk's limit, (num_tokens − 1) // B: `find_longest_cache_hit`, lookups k = 0, 1, … of RK[(r, k)],
  M[key] and, while cached, B[first], until a miss or the limit. x = the hit's blocks, y = those no request holds.
- take (2), r, a = k0, b = n: `allocate_slots`' touch of the hit blocks k0, …, k0 + n − 1: RK, M (the key's first
  copy, vLLM's), B[b] (one more holder, out of the free queue if it was in it), its queue neighbours, RB[(r, k)] := b.
- alloc (3), r, a = k0, b = n: n new blocks from the queue's head: B[b] := (ref 1, owner r + 1), the next block's
  prev, K[b] (its key, cleared) and, if it had one, its eviction: M[key] and its copies' neighbours; RB[(r, k)] := b.
- cache (4), r, a = k0, b = n: `cache_blocks` of blocks k0, …, k0 + n − 1: RB, RK, K[b] := key (it had none), M[key]
  (b appended), the old last copy's cnext, B[b]'s copy links.
- free (5), r, a = held: `free_request`, blocks held − 1, …, 0: RB[(r, k)] := 0, K[b], B[b] (one holder fewer). A block
  whose last holder this is returns (x counts them): a hashless one goes after the hashless blocks this call has
  returned to the queue's front (vLLM's `reversed(first)` pushes), a cached one to the queue's back.
An access a call doesn't need (a neighbour that is the sentinel, an eviction of a block with no key) is skipped, not
idled: each call's accesses are a function of the pool, so the stream's slots are too. A call with no operation (a = 0
for hit and free, b = 0 else) is in neither the stream nor the bus.

**The head-of-line memo.** vLLM's scheduler looks up the waiting queue's head again every step it can't admit it, most
often over a prefix nothing has changed. A hit that walks increments the stream's EW-bit walk epoch and stamps each M
and B cell it reads with it; at its end the stream keeps (r, a, x, y) and clears its dirty bit. Any access to a cell
stamped with the current epoch that changes its first copy (M) or whether it is held (B, ref = 0 or not) sets the dirty
bit. The walk's result reads nothing else (RK is read only, and a request number is one request's in a window), so a hit
of the kept r and a with the dirty bit clear returns the kept x and y in one slot with no access (`Slot.use` 0). A
stale stamp of an earlier epoch can only set the bit spuriously, costing a walk. The epoch, the bit and the kept call
are functions of the stream, so the memo adds no choice; a window starts with none kept.

**The stream.** Slot i is at time i + 1. A call starts in the slot after the last one ended, which carries its fields
(start = 1, kind, r, a, b) and performs its first access; every other slot's fields are zero. A slot between calls that
doesn't start one makes the stream idle for good: its fields and its read are zero and it accesses nothing. Each
access reads (tag, address, v, t), v the value at the address and t the time it was written (witness), and writes
(tag, address, w, now), w a function of the call, its earlier reads and v. The touched list (`MemEntry_v1`'s, of NE
entries) holds each address the stream touches, sorted by (tag, address), with its initial value v0 and its final value
and time. An entry the stream doesn't touch has v = v0 and t = 0, so a caller may list B's and K's addresses whole.
The initial values are the caller's to bind: the pool the previous window left, and RK's keys (`CompileSlice_v4`,
which carries the pool between windows); this unit takes them as given.

**ok = 0 refuses**: a slot's fields or read not zero where they must be; a call of no kind, of no operation, or whose
block indices pass 2^KW; a read that isn't consistent (`MemCheck_v1`: R ≠ W, or a read's t not below its now); a take
of an uncached key or past REFW holders; an alloc from an empty queue, of a held block or for r = 2^RW − 1; a cache of
an empty request slot or of a block that has a key; a free of an empty request slot or of an unheld block; a take or an
alloc into a request slot that holds a block; the stream ending inside a call. The valid witness is unique: the reads
are the memories' values, and every unused field is zero.

**The bus**: P = ∏ (α + e) over the calls' records, e the record's bits (seq ‖ kind ‖ r ‖ a ‖ b ‖ x ‖ y, least
significant first, one 128-bit chunk), seq numbering the stream's calls from 1. `ReplayStep_v1`'s side multiplies the
same records for its calls with an operation, so the two products agree iff the stream ran ReplayStep_v1's calls in its
order and returned its results, except with probability (calls)/2^128 over α. A key print that is 0 reads as no key:
probability 2^-128 a key over the print's coin.

**Statics** (public): NS stream slots (NS < 2^TW), NE list entries, NB blocks, RW a request number's width, KW a block
index's (k < 2^KW), REFW a holder count's, TW a time's. NS and NE come from a public ladder (`size_class`): a window
takes the least class its slots and its entries fit, with no refusal and no fallback. A record is `MemCheck_v1`'s at
(G, A, V, T) = (3, 128, 128, TW): 3 chunks at TW ≥ 26.

**Cost** (`size`): a slot is its access's check (`MemAccess_v1`: 6 products at 3 chunks), the bus product, and its
call's logic over every kind's positions, the memo's 169 ANDs at GLM-5.3's widths among them; an entry is
`MemEntry_v1`.
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, NamedTuple, Sequence

from verity.circuits import boolean as BL
from verity.definitions.circuits.boolean import memcheck as MC
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.circuits.boolean.gf128 import mul, mul_forms
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.types import Array, Tuple
from verity.protocols.certifier.verifier.head_check import _coll, _Consts, _refs, ands

__all__ = ["CacheStep", "Call", "KINDS", "Out", "PARTS", "PoolEnd", "PoolEntry", "PoolSlot", "Refused", "SLOT_STATICS",
           "STATICS", "Slot", "Stream", "TAGS", "arguments", "bput", "bsplit", "bus", "empty_pool", "evaluate", "rk",
           "size", "size_class", "statics", "unit", "widths"]

BIT = BL.BIT
F = Array(128, BIT)
STATICS = ("NS", "NE", "NB", "RW", "KW", "REFW", "TW")
SLOT_STATICS = ("NB", "RW", "KW", "REFW", "TW")
TAGS = {"B": 0, "K": 1, "M": 2, "RB": 3, "RK": 4}
KINDS = {"hit": 1, "take": 2, "alloc": 3, "cache": 4, "free": 5}
BF = ("prev", "next", "cprev", "cnext", "ref", "owner", "stamp")
#: positions an operation of each kind has (its accesses, before skipping)
SHAPE = {"hit": 3, "take": 6, "alloc": 7, "cache": 6, "free": 5}
G = 3
#: a walk epoch's width: the stamp a hit's walk leaves on each M and B cell it reads
EW = 4


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def widths(S) -> SimpleNamespace:
    for k in SLOT_STATICS + tuple(k for k in ("NS", "NE") if hasattr(S, k)):
        x = getattr(S, k)
        if not isinstance(x, int) or isinstance(x, bool) or x < 1:
            raise ValueError(f"CacheStep_v3's {k} is {x!r}, not a positive int")
    if S.NB < 2:
        raise ValueError(f"a pool of {S.NB} blocks: NB ≥ 2 (block 0 is vLLM's null block)")
    NW, CW = _bitlen(S.NB - 1), S.KW + 1
    BV = 4 * NW + S.REFW + S.RW + EW
    if BV > 128 or S.RW + S.KW > 128:
        raise ValueError(f"a block's fields ({BV} bits) or a request slot's address ({S.RW + S.KW}) pass 128 bits")
    if S.TW + 3 + S.RW + 4 * CW > 128:
        raise ValueError("a bus record passes 128 bits")
    if hasattr(S, "NS") and S.NS >= 1 << S.TW:
        raise ValueError(f"the stream's times 1..{S.NS} need more than TW = {S.TW} bits")
    off = {"prev": 0, "next": NW, "cprev": 2 * NW, "cnext": 3 * NW, "ref": 4 * NW, "owner": 4 * NW + S.REFW,
           "stamp": 4 * NW + S.REFW + S.RW}
    bw = {"prev": NW, "next": NW, "cprev": NW, "cnext": NW, "ref": S.REFW, "owner": S.RW, "stamp": EW}
    return SimpleNamespace(NW=NW, CW=CW, BV=BV, off=off, bw=bw, MST=2 * NW,
                           MS=SimpleNamespace(G=G, A=128, V=128, T=S.TW))


def rk(S, r: int, k: int) -> int:
    """RB's and RK's address of request r's block k."""
    return r | k << S.RW


def bsplit(S, v: int) -> dict[str, int]:
    Z = widths(S)
    return {f: (v >> Z.off[f]) & ((1 << Z.bw[f]) - 1) for f in BF}


def bput(S, d: dict[str, int]) -> int:
    Z = widths(S)
    x = 0
    for f in BF:
        if not 0 <= d.get(f, 0) < 1 << Z.bw[f]:
            raise ValueError(f"block field {f} = {d[f]} out of {Z.bw[f]} bits")
        x |= d.get(f, 0) << Z.off[f]
    return x


# --- the specification ---------------------------------------------------------------------------------------------------


class Call(NamedTuple):
    kind: str
    r: int
    a: int
    b: int = 0


class Slot(NamedTuple):
    """One slot: its call fields (zero but where a call starts) and its access, which a memoized hit's slot doesn't
    make (``use`` 0, every access field 0)."""
    start: int
    kind: int
    r: int
    a: int
    b: int
    tag: int
    addr: int
    v: int
    t: int
    w: int
    now: int
    use: int = 1


class Refused(ValueError):
    """A call the pool can't run (the circuit's ok = 0)."""


class Stream:
    """The pool's memories and the free queue's ends, and the stream of slots the calls so far ran in. `initial(tag,
    address)` is an address's value before the stream first touches it. Not ``strict``, a call the pool can't run is
    listed in `refusals` and runs on as the circuit runs it (counts wrap, a missing block is block 0), so a test can
    build a refused call whose accesses are otherwise consistent; a call's fields out of their ranges still raise."""

    def __init__(self, S, initial: Callable[[int, int], int], head: int, tail: int, strict: bool = True) -> None:
        self.S, self.Z = S, widths(S)
        self.initial = initial
        self.head, self.tail = head, tail
        self.strict = strict
        self.refusals: list[str] = []
        self.cells: dict[tuple[int, int], tuple[int, int]] = {}
        self.first: dict[tuple[int, int], int] = {}
        self.slots: list[Slot] = []
        self.records: list[tuple[int, ...]] = []
        self._fields: tuple[int, int, int, int] | None = None
        self.epoch, self.dirty = 0, 0
        self.memo: tuple[int, int, int, int] | None = None
        self.memos = self.rewalks = 0

    def _refuse(self, why: str) -> None:
        if self.strict:
            raise Refused(why)
        self.refusals.append(why)

    def value(self, tag: int, addr: int) -> int:
        c = self.cells.get((tag, addr))
        return c[0] if c else self.initial(tag, addr)

    def _acc(self, tag: int, addr: int, f: Callable[[int], int] = lambda v: v) -> int:
        key = (tag, addr)
        if key in self.cells:
            v, t = self.cells[key]
        else:
            v, t = self.initial(tag, addr), 0
            self.first[key] = v
        now = len(self.slots) + 1
        w = f(v)
        self.dirty |= self._dirties(tag, v, w)
        fields, self._fields = self._fields or (0, 0, 0, 0), None
        self.slots.append(Slot(int(any(fields)), *fields, tag, addr, v, t, w, now))
        self.cells[key] = (w, now)
        return v

    def _dirties(self, tag: int, v: int, w: int) -> int:
        """Whether an access that writes w over v changes what the last walk read: a cell stamped with its epoch whose
        first copy (M) or whose being held (B) changes."""
        Z, mask = self.Z, (1 << self.Z.NW) - 1
        if tag == TAGS["M"]:
            return int((v >> Z.MST) & ((1 << EW) - 1) == self.epoch and (v & mask) != (w & mask))
        if tag == TAGS["B"]:
            d, e = bsplit(self.S, v), bsplit(self.S, w)
            return int(d["stamp"] == self.epoch and (d["ref"] == 0) != (e["ref"] == 0))
        return 0

    def _b(self, b: int, **put) -> dict[str, int]:
        """B[b]'s access with the fields ``put`` written; its old fields."""
        S = self.S
        return bsplit(S, self._acc(TAGS["B"], b, lambda v: bput(S, {**bsplit(S, v), **put})))

    def call(self, c: Call) -> tuple[int, int]:
        """Runs ``c``; its results (x, y). Refused raises where the circuit's ok is 0."""
        S, Z = self.S, self.Z
        if c.kind not in KINDS:
            raise Refused(f"no call kind {c.kind!r}")
        n = c.a if c.kind in ("hit", "free") else c.b
        if c.kind in ("hit", "free") and c.b or not 0 <= c.r < 1 << S.RW or not 0 <= c.a < 1 << Z.CW \
                or not 0 <= c.b < 1 << Z.CW:
            raise Refused(f"{c}: fields out of their widths, or b given to a {c.kind}")
        if n == 0:
            return 0, 0
        if (c.a if c.kind in ("hit", "free") else c.a + c.b) > 1 << S.KW:
            raise Refused(f"{c}: block indices past 2^KW")
        self._fields = (KINDS[c.kind], c.r, c.a, c.b)
        x, y = getattr(self, "_" + c.kind)(c)
        self.records.append((len(self.records) + 1, KINDS[c.kind], c.r, c.a, c.b, x, y))
        return x, y

    def _hit(self, c: Call) -> tuple[int, int]:
        S, Z = self.S, self.Z
        if self.memo and self.memo[:2] == (c.r, c.a) and not self.dirty:
            got = self.memo[2:]
            if got != self.walk(c.r, c.a):
                raise AssertionError(f"{c}: the memo's {got}, the pool's {self.walk(c.r, c.a)}")
            fields, self._fields = self._fields, None
            self.slots.append(Slot(1, *fields, 0, 0, 0, 0, 0, len(self.slots) + 1, use=0))
            self.memos += 1
            return got
        self.rewalks += bool(self.memo and self.memo[:2] == (c.r, c.a))
        self.epoch, self.dirty = (self.epoch + 1) % (1 << EW), 0
        st = (1 << EW) - 1 << Z.MST
        h = ev = 0
        for k in range(c.a):
            key = self._acc(TAGS["RK"], rk(S, c.r, k))
            first = self._acc(TAGS["M"], key, lambda v: v & ~st | self.epoch << Z.MST) & ((1 << Z.NW) - 1)
            if not first:
                break
            ev += self._b(first, stamp=self.epoch)["ref"] == 0
            h += 1
        self.memo = (c.r, c.a, h, ev)
        return h, ev

    def walk(self, r: int, a: int) -> tuple[int, int]:
        """`find_longest_cache_hit` on the memories as they are, accessing nothing: (hit, held by no request)."""
        S, mask = self.S, (1 << self.Z.NW) - 1
        h = ev = 0
        for k in range(a):
            first = self.value(TAGS["M"], self.value(TAGS["RK"], rk(S, r, k))) & mask
            if not first:
                break
            ev += bsplit(S, self.value(TAGS["B"], first))["ref"] == 0
            h += 1
        return h, ev

    def _take(self, c: Call) -> tuple[int, int]:
        S, Z = self.S, self.Z
        for k in range(c.a, c.a + c.b):
            key = self._acc(TAGS["RK"], rk(S, c.r, k))
            b = self._acc(TAGS["M"], key) & ((1 << Z.NW) - 1)
            if not b:
                self._refuse(f"take of request {c.r}'s block {k}: its key isn't cached")
            d = bsplit(S, self.value(TAGS["B"], b))
            if d["ref"] == (1 << S.REFW) - 1:
                self._refuse(f"take of block {b}: its holders pass REFW bits")
            free = d["ref"] == 0
            self._b(b, ref=(d["ref"] + 1) % (1 << S.REFW), **({"prev": 0, "next": 0} if free else {}))
            p, n = d["prev"], d["next"]
            if free:
                if p:
                    self._b(p, next=n)
                else:
                    self.head = n
                if n:
                    self._b(n, prev=p)
                else:
                    self.tail = p
            self._rb_put(c.r, k, b)
        return 0, 0

    def _rb_put(self, r: int, k: int, b: int) -> None:
        if self.value(TAGS["RB"], rk(self.S, r, k)):
            self._refuse(f"request {r}'s block slot {k} already holds a block")
        self._acc(TAGS["RB"], rk(self.S, r, k), lambda v: b)

    def _alloc(self, c: Call) -> tuple[int, int]:
        S = self.S
        if c.r == (1 << S.RW) - 1:
            self._refuse(f"request {c.r}: its owner number r + 1 passes RW bits")
        for k in range(c.a, c.a + c.b):
            b = self.head
            if not b:
                self._refuse("alloc from an empty free queue")
            if bsplit(S, self.value(TAGS["B"], b))["ref"]:
                self._refuse(f"the queue's head {b} is held")
            d = self._b(b, prev=0, next=0, cprev=0, cnext=0, ref=1, owner=(c.r + 1) % (1 << S.RW))
            n, cp, cn = d["next"], d["cprev"], d["cnext"]
            self.head = n
            if n:
                self._b(n, prev=0)
            else:
                self.tail = 0
            key = self._acc(TAGS["K"], b, lambda v: 0)
            if key:
                NW = self.Z.NW
                mask = (1 << NW) - 1
                self._acc(TAGS["M"], key, lambda v: (cn if v & mask == b else v & mask)
                          | (cp if (v >> NW) & mask == b else (v >> NW) & mask) << NW | v >> 2 * NW << 2 * NW)
                if cp:
                    self._b(cp, cnext=cn)
                if cn:
                    self._b(cn, cprev=cp)
            self._rb_put(c.r, k, b)
        return 0, 0

    def _cache(self, c: Call) -> tuple[int, int]:
        S, Z = self.S, self.Z
        mask = (1 << Z.NW) - 1
        for k in range(c.a, c.a + c.b):
            b = self._acc(TAGS["RB"], rk(S, c.r, k))
            if not b:
                self._refuse(f"cache of request {c.r}'s block {k}: it holds none there")
            key = self._acc(TAGS["RK"], rk(S, c.r, k))
            if self.value(TAGS["K"], b):
                self._refuse(f"cache of block {b}: it has a key")
            self._acc(TAGS["K"], b, lambda v: key)
            m = self._acc(TAGS["M"], key, lambda v: (v & mask or b) | b << Z.NW | v >> 2 * Z.NW << 2 * Z.NW)
            last = (m >> Z.NW) & mask
            if last:
                self._b(last, cnext=b)
            self._b(b, cprev=last, cnext=0)
        return 0, 0

    def _free(self, c: Call) -> tuple[int, int]:
        S = self.S
        cur, cnx, ret = 0, self.head, 0
        for k in reversed(range(c.a)):
            b = self._acc(TAGS["RB"], rk(S, c.r, k), lambda v: 0)
            if not b:
                self._refuse(f"free of request {c.r}'s block {k}: it holds none there")
            key = self._acc(TAGS["K"], b)
            d = bsplit(S, self.value(TAGS["B"], b))
            if d["ref"] != 1:
                if not d["ref"]:
                    self._refuse(f"free of block {b}, which nothing holds")
                self._b(b, ref=(d["ref"] - 1) % (1 << S.REFW))
                continue
            ret += 1
            if not key:
                p, n = cur, cnx
                self._b(b, ref=0, prev=p, next=n)
                if not p:
                    self.head = b
                if not n:
                    self.tail = b
                cur = b
                if p:
                    self._b(p, next=b)
                if n:
                    self._b(n, prev=b)
            else:
                p = self.tail
                self._b(b, ref=0, prev=p, next=0)
                if not p:
                    self.head = b
                self.tail = b
                if not cnx:
                    cnx = b
                if p:
                    self._b(p, next=b)
        return ret, 0

    def entries(self) -> list[tuple[int, int, int, int, int]]:
        """The touched list: (tag, address, initial value, final value, final time), sorted."""
        return sorted((tag, a, v0, *self.cells[(tag, a)]) for (tag, a), v0 in self.first.items())

    def bus(self, alpha: int) -> int:
        return bus(self.S, self.records, alpha)

    def fingerprints(self, alpha: int, gamma: int) -> tuple[int, int]:
        """(R, W) of the stream's accesses and its touched list under (α, γ) (`MemCheck_v1`'s records)."""
        MS, R, W = self.Z.MS, 1, 1
        for s in self.slots:
            if not s.use:
                continue
            R = mul(R, alpha ^ MC.element(MS, gamma, s.tag, s.addr, s.v, s.t))
            W = mul(W, alpha ^ MC.element(MS, gamma, s.tag, s.addr, s.w, s.now))
        for tag, a, v0, v, t in self.entries():
            R = mul(R, alpha ^ MC.element(MS, gamma, tag, a, v, t))
            W = mul(W, alpha ^ MC.element(MS, gamma, tag, a, v0, 0))
        return R, W


def _record(S, Z, seq: int, kind: int, r: int, a: int, b: int, x: int, y: int) -> int:
    out, at = 0, 0
    for val, w in ((seq, S.TW), (kind, 3), (r, S.RW), (a, Z.CW), (b, Z.CW), (x, Z.CW), (y, Z.CW)):
        out |= val << at
        at += w
    return out


def bus(S, records: Sequence[tuple[int, ...]], alpha: int) -> int:
    """∏ (α + e) over the calls' records (seq, kind, r, a, b, x, y): the bus product either side computes."""
    Z, P = widths(S), 1
    for rec in records:
        P = mul(P, alpha ^ _record(S, Z, *rec))
    return P


def size_class(n: int, least: int = 64) -> int:
    """The public ladder NS and NE are drawn from: the least of 2^k and 3·2^(k−1) (k ≥ 1) that is at least n and
    ``least``. A window takes the class its slots (entries) fit, so the class is a function of the window, and it
    reveals at most log2 of the classes up to the pool's bound."""
    c = 2
    while c < max(n, least):
        c = c * 3 // 2 if c & (c - 1) == 0 else c // 3 * 4
    return c


def empty_pool(S, keys: dict[int, int]) -> Callable[[int, int], int]:
    """The empty pool's initial values: every block free, in order (the queue's head 1 and tail NB − 1), nothing cached
    or held; RK's keys are ``keys`` (by RK address)."""
    widths(S)

    def initial(tag: int, addr: int) -> int:
        if tag == TAGS["B"]:
            return bput(S, {"prev": addr - 1 if addr > 1 else 0, "next": addr + 1 if addr + 1 < S.NB else 0})
        return keys.get(addr, 0) if tag == TAGS["RK"] else 0

    return initial


# --- the circuit -----------------------------------------------------------------------------------------------------------


def _carry_layout(S) -> list[tuple[str, int]]:
    Z = widths(S)
    NW, CW = Z.NW, Z.CW
    return [("ok", 1), ("R", 128), ("W", 128), ("P", 128), ("now", S.TW), ("seq", S.TW), ("head", NW), ("tail", NW),
            ("busy", 1), ("idle", 1), ("kind", 3), ("r", S.RW), ("ca", CW), ("cb", CW), ("k", CW), ("left", CW),
            ("x", CW), ("y", CW), ("pos", 3), ("key", 128), ("blk", NW), ("p", NW), ("n", NW), ("cp", NW), ("cn", NW),
            ("flag", 1), ("cur", NW), ("cnx", NW), ("epoch", EW), ("dirty", 1), ("mv", 1), ("mr", S.RW), ("ma", CW),
            ("mx", CW), ("my", CW)]


def _size(layout) -> int:
    return sum(w for _, w in layout)


def _unpack(layout, bits) -> SimpleNamespace:
    out, at = {}, 0
    for name, w in layout:
        out[name] = list(bits[at:at + w])
        at += w
    return SimpleNamespace(**out)


def _pad(x, w):
    return (list(x) + [ZERO] * w)[:w]


def _nz(C, x):
    return or_reduce(C, x)


def _z(C, x):
    return or_reduce(C, x).inv()


def _eq(C, x, y):
    return _z(C, [p ^ q for p, q in zip(x, y)])


def _onehot(C, x, n):
    """x's value as n one-hot bits (x little-endian)."""
    out = []
    for i in range(n):
        out.append(and_reduce(C, [b if i >> j & 1 else b.inv() for j, b in enumerate(x)]))
    return out


def _sel(C, s, x):
    return [C.AND(s, b) for b in x]


def _xor(*xs):
    out = list(xs[0])
    for x in xs[1:]:
        out = [a ^ b for a, b in zip(out, x)]
    return out


def _any(C, sels):
    return or_reduce(C, list(sels))


def slot_forms(C, S, carry, start, kind, r, a, b, v, t, alpha, gamma) -> list[list]:
    """One slot (the module's docstring): its access, its call's progress, its checks."""
    Z = widths(S)
    NW, CW, RW, KW = Z.NW, Z.CW, S.RW, S.KW
    c = _unpack(_carry_layout(S), carry)
    start = start[0]
    zn = [ZERO] * NW
    between = C.AND(c.busy[0].inv(), c.idle[0].inv())
    s = C.AND(between, start)
    goidle = C.AND(between, start.inv())
    active = OR(C, c.busy[0], s)

    # the call's fields: the slot's at a start, else carried
    kin = _onehot(C, kind, 8)
    kind_e = mux(C, s, kind, c.kind)
    r_e, ca_e, cb_e = mux(C, s, r, c.r), mux(C, s, a, c.ca), mux(C, s, b, c.cb)
    # the memo: a hit of the last walk's request and limit, with nothing that walk read changed since, makes no access
    hs = C.AND(s, kin[1])
    memo = and_reduce(C, [hs, c.mv[0], _eq(C, r, c.mr), _eq(C, a, c.ma), c.dirty[0].inv()])
    walks = C.AND(hs, memo.inv())
    acc = C.AND(active, memo.inv())
    K = [C.AND(acc, x) for x in _onehot(C, kind_e, 6)[1:]]
    hit, take, alloc, cache, free = K
    a1 = sub(C, a, const_bits(1, CW), CW)[0]
    some = [kin[2], kin[3], kin[4]]
    k0 = _xor(_sel(C, _any(C, some), a), _sel(C, kin[5], a1))
    left0 = mux(C, OR(C, kin[1], kin[5]), a, b)
    k_e, left_e = mux(C, s, k0, c.k), mux(C, s, left0, c.left)
    x_e, y_e = _sel(C, s.inv(), c.x), _sel(C, s.inv(), c.y)
    pos_e = _sel(C, s.inv(), c.pos)
    cur_e, cnx_e = _sel(C, s.inv(), c.cur), mux(C, s, c.head, c.cnx)
    P = _onehot(C, pos_e, 7)
    sel = {(kd, j): C.AND(K[i], P[j]) for i, kd in enumerate(KINDS) for j in range(SHAPE[kd])}
    L, T_, A, Cc, Fr = ([sel[(kd, j)] for j in range(SHAPE[kd])] for kd in KINDS)

    # the read's fields
    vb = {f: v[Z.off[f]:Z.off[f] + Z.bw[f]] for f in BF}
    vlo, vhi = v[:NW], v[NW:2 * NW]
    ref0, ref1 = _z(C, vb["ref"]), _eq(C, vb["ref"], const_bits(1, S.REFW))
    hl = _z(C, c.key)                                   # the freed block has no key (read at F1)
    ret = C.AND(Fr[2], ref1)
    p_new = mux(C, hl, cur_e, c.tail)                   # where a returned block goes after

    # the address
    rka = _pad(r_e, RW) + _pad(k_e, KW)
    srcs = [((L[0], T_[0], T_[5], A[6], Cc[0], Cc[1], Fr[0]), rka), ((L[1], T_[1], A[3], Cc[3]), c.key),
            ((L[2], T_[2], A[2], Cc[2], Cc[5], Fr[1], Fr[2]), c.blk), ((A[0],), c.head),
            ((T_[3], Cc[4], Fr[3]), c.p), ((T_[4], A[1], Fr[4]), c.n), ((A[4],), c.cp), ((A[5],), c.cn)]
    addr = [ZERO] * 128
    for ss, val in srcs:
        addr = _xor(addr, _pad(_sel(C, _any(C, ss), val), 128))
    tags = {"RK": (L[0], T_[0], Cc[1]), "M": (L[1], T_[1], A[3], Cc[3]), "K": (A[2], Cc[2], Fr[1]),
            "RB": (T_[5], A[6], Cc[0], Fr[0])}
    tag = [ZERO] * G
    for name, ss in tags.items():
        for x in ss:
            tag = _xor(tag, [x if TAGS[name] >> j & 1 else ZERO for j in range(G)])

    # the write: w = v but for the fields a case puts
    delta = [ZERO] * 128

    def put(s_, lo, new):
        nonlocal delta
        new = list(new)
        d = _sel(C, s_, [x ^ y for x, y in zip(new, v[lo:lo + len(new)])])
        delta = delta[:lo] + _xor(delta[lo:lo + len(d)], d) + delta[lo + len(d):]

    def bput_(s_, f, new):
        put(s_, Z.off[f], _pad(new, Z.bw[f]))

    free_now = C.AND(T_[2], ref0)
    bput_(T_[2], "ref", add(C, vb["ref"], const_bits(1, S.REFW), width=S.REFW))
    bput_(free_now, "prev", zn)
    bput_(free_now, "next", zn)
    bput_(T_[3], "next", c.n)
    bput_(T_[4], "prev", c.p)
    put(OR(C, T_[5], A[6]), 0, c.blk)
    owner = add(C, r_e, const_bits(1, RW), width=RW)
    put(A[0], 0, _pad([], 4 * NW) + const_bits(1, S.REFW) + owner)
    bput_(A[1], "prev", zn)
    put(A[2], 0, [ZERO] * 128)
    put(C.AND(A[3], _eq(C, vlo, c.blk)), 0, c.cn)
    put(C.AND(A[3], _eq(C, vhi, c.blk)), NW, c.cp)
    bput_(A[4], "cnext", c.cn)
    bput_(A[5], "cprev", c.cp)
    put(Cc[2], 0, c.key)
    put(C.AND(Cc[3], _z(C, vlo)), 0, c.blk)
    put(Cc[3], NW, c.blk)
    bput_(Cc[4], "cnext", c.blk)
    bput_(Cc[5], "cprev", c.p)
    bput_(Cc[5], "cnext", zn)
    put(Fr[0], 0, zn)
    bput_(Fr[2], "ref", sub(C, vb["ref"], const_bits(1, S.REFW), S.REFW)[0])
    bput_(ret, "prev", p_new)
    bput_(ret, "next", _sel(C, hl, cnx_e))
    bput_(Fr[3], "next", c.blk)
    bput_(Fr[4], "prev", c.blk)
    put(L[1], Z.MST, c.epoch)
    bput_(L[2], "stamp", c.epoch)
    w = _xor(v, delta)

    # the dirty bit: a cell stamped with the last walk's epoch whose first copy (M) or whose being held (B) changes
    ro = Z.off["ref"]
    held_flips = ref0 ^ _z(C, w[ro:ro + S.REFW])
    so = Z.off["stamp"]
    dirties = OR(C, and_reduce(C, [_z(C, tag), _eq(C, v[so:so + EW], c.epoch), held_flips]),
                 and_reduce(C, [_eq(C, tag, const_bits(TAGS["M"], G)), _eq(C, v[Z.MST:Z.MST + EW], c.epoch),
                                _nz(C, delta[:NW])]))
    dirty = [C.AND(walks.inv(), OR(C, c.dirty[0], dirties))]
    epoch = mux(C, walks, add(C, c.epoch, const_bits(1, EW), width=EW), c.epoch)

    # the registers
    key = mux(C, _any(C, (L[0], T_[0], Cc[1], A[2], Fr[1])), v, c.key)
    blk = mux(C, _any(C, (L[1], T_[1], Cc[0], Fr[0])), vlo, mux(C, A[0], c.head, c.blk))
    p = mux(C, T_[2], vb["prev"], mux(C, Cc[3], vhi, mux(C, ret, p_new, c.p)))
    n = mux(C, OR(C, T_[2], A[0]), vb["next"], mux(C, C.AND(ret, hl), cnx_e, c.n))
    cp, cn = mux(C, A[0], vb["cprev"], c.cp), mux(C, A[0], vb["cnext"], c.cn)
    flag = mux(C, T_[2], [ref0], mux(C, Fr[2], [hl], c.flag))
    cur = mux(C, C.AND(ret, hl), c.blk, cur_e)
    cnx = mux(C, and_reduce(C, [ret, hl.inv(), _z(C, cnx_e)]), c.blk, cnx_e)
    p0, n0 = _z(C, vb["prev"]), _z(C, vb["next"])
    head = mux(C, C.AND(free_now, p0), vb["next"], c.head)
    head = mux(C, A[0], vb["next"], head)
    head = mux(C, C.AND(ret, _z(C, p_new)), c.blk, head)
    tail = mux(C, C.AND(free_now, n0), vb["prev"], c.tail)
    tail = mux(C, C.AND(A[0], n0), zn, tail)
    tail = mux(C, C.AND(ret, OR(C, hl.inv(), _z(C, cnx_e))), c.blk, tail)

    # the next position, and the operation's end
    first0 = _z(C, vlo)
    nz_n = _nz(C, c.n)
    steps = [  # (selector, next position, or None for the end)
        (L[0], 1), (C.AND(L[1], first0.inv()), 2), (Cc[0], 1), (Cc[1], 2), (Cc[2], 3), (Cc[4], 5),
        (T_[0], 1), (T_[1], 2), (T_[4], 5), (A[1], 2), (A[5], 6), (Fr[0], 1), (Fr[1], 2)]
    nxt = [ZERO] * 3
    for s_, j in steps:
        nxt = _xor(nxt, [s_ if j >> i & 1 else ZERO for i in range(3)])

    def to(s_, cond, j1, j0):
        """At s_: position j1 if cond else j0 (None ends the operation)."""
        nonlocal nxt
        for cnd, j in ((C.AND(s_, cond), j1), (C.AND(s_, cond.inv()), j0)):
            if j is not None:
                nxt = _xor(nxt, [cnd if j >> i & 1 else ZERO for i in range(3)])

    # take: T2 → T3 if free and p ≠ 0, T4 if free and n ≠ 0, else T5; T3 → T4 if n ≠ 0 else T5
    to(free_now, p0.inv(), 3, None)
    to(C.AND(free_now, p0), n0.inv(), 4, 5)
    to(C.AND(T_[2], ref0.inv()), ONE, 5, None)
    to(T_[3], nz_n, 4, 5)
    # alloc: A0 → A1 if n ≠ 0 else A2; A2 → A3 if keyed else A6; A3 → A4, A5 or A6; A4 → A5 or A6
    to(A[0], n0.inv(), 1, 2)
    to(A[2], _nz(C, v), 3, 6)
    nz_cp, nz_cn = _nz(C, c.cp), _nz(C, c.cn)
    to(A[3], nz_cp, 4, None)
    to(C.AND(A[3], nz_cp.inv()), nz_cn, 5, 6)
    to(A[4], nz_cn, 5, 6)
    # cache: C3 → C4 if it had a last copy else C5
    to(Cc[3], _nz(C, vhi), 4, 5)
    # free: F2 → F3 if the block returns after a block, F4 if hashless before one, else the end; F3 → F4 if hashless
    # before one
    nz_pn, nz_cnx = _nz(C, p_new), _nz(C, cnx_e)
    to(ret, nz_pn, 3, None)
    to(C.AND(ret, nz_pn.inv()), C.AND(hl, nz_cnx), 4, None)
    to(Fr[3], C.AND(c.flag[0], nz_n), 4, None)
    miss = C.AND(L[1], first0)
    end = _any(C, (miss, L[2], T_[5], A[6], Cc[5], C.AND(Fr[2], ret.inv()),
                   and_reduce(C, [ret, nz_pn.inv(), C.AND(hl, nz_cnx).inv()]),
                   C.AND(Fr[3], C.AND(c.flag[0], nz_n).inv()), Fr[4]))

    # the call's progress
    dec = C.AND(end, miss.inv())
    left = sub(C, left_e, _pad([dec], CW), CW)[0]
    k = add(C, k_e, [dec] + [C.AND(dec, free)] * (CW - 1), width=CW)
    x = add(C, x_e, _pad([OR(C, L[2], ret)], CW), width=CW)
    y = add(C, y_e, _pad([C.AND(L[2], ref0)], CW), width=CW)
    walked = C.AND(OR(C, miss, C.AND(dec, _z(C, left))), hit)
    mv, mr, ma = [OR(C, c.mv[0], walked)], mux(C, walked, r_e, c.mr), mux(C, walked, ca_e, c.ma)
    mx, my = mux(C, walked, x, c.mx), mux(C, walked, y, c.my)
    x, y = mux(C, memo, c.mx, x), mux(C, memo, c.my, y)
    call_end = _any(C, (miss, C.AND(dec, _z(C, left)), memo))
    busy = C.AND(active, call_end.inv())
    idle = OR(C, c.idle[0], goidle)
    seq = add(C, c.seq, _pad([call_end], S.TW), width=S.TW)
    rec = _pad(seq + _pad(kind_e, 3) + r_e + ca_e + cb_e + x + y, 128)
    factor = [o ^ C.AND(call_end, f ^ o) for f, o in zip(_xor(alpha, rec), [ONE] + [ZERO] * 127)]
    Pbus = mul_forms(C, c.P, factor)

    # the access's check, and the slot's
    R, W, ok = MC.access_forms(C, Z.MS, alpha, gamma, c.R, c.W, c.ok, [acc], tag, addr, v, t, w, c.now)
    lim = const_bits(1 << KW, CW + 1)
    span = add(C, a, b, width=CW + 1)
    over = sub(C, lim, mux(C, OR(C, kin[1], kin[5]), _pad(a, CW + 1), span), CW + 1)[1]
    bad = [
        C.AND(s.inv(), _nz(C, [start, *kind, *r, *a, *b])),
        C.AND(s, _any(C, (kin[0], kin[6], kin[7]))),
        C.AND(s, C.AND(_any(C, (kin[1], kin[5])), _nz(C, b))),
        C.AND(s, _z(C, mux(C, OR(C, kin[1], kin[5]), a, b))),
        C.AND(s, over),
        C.AND(T_[1], first0),
        C.AND(T_[2], and_reduce(C, vb["ref"])),
        C.AND(OR(C, T_[5], A[6]), _nz(C, vlo)),
        C.AND(A[0], OR(C, _z(C, c.head), ref0.inv())),
        C.AND(C.AND(s, kin[3]), and_reduce(C, r)),
        C.AND(OR(C, Cc[0], Fr[0]), first0),
        C.AND(Cc[2], _nz(C, v)),
        C.AND(Fr[2], ref0),
    ]
    ok = [C.AND(ok[0], _any(C, bad).inv())]
    now = add(C, c.now, const_bits(1, S.TW), width=S.TW)
    out = (ok + R + W + Pbus + now + seq + head + tail + [busy, idle] + _pad(kind_e, 3) + r_e + ca_e + cb_e + k + left
           + x + y + _pad(nxt, 3) + key + blk + p + n + cp + cn + flag + cur + cnx + epoch + dirty + mv + mr + ma + mx
           + my)
    assert len(out) == _size(_carry_layout(S))
    return [out, [active]]


def _sig_slot(S):
    Z = widths(S)
    carry = Array(_size(_carry_layout(S)), BIT)
    return ((("carry", carry), ("start", BIT), ("kind", Array(3, BIT)), ("r", Array(S.RW, BIT)), ("a", Array(Z.CW, BIT)),
             ("b", Array(Z.CW, BIT)), ("v", F), ("t", Array(S.TW, BIT)), ("alpha", F), ("gamma", F)), Tuple(carry, BIT))


def _entry_layout(S) -> list[tuple[str, int]]:
    return [("R", 128), ("W", 128), ("ok", 1), ("first", 1), ("pen", 1), ("ptag", G), ("pa", 128)]


def entry_forms(C, S, carry, en, tag, a, v0, v, t, alpha, gamma) -> list[list]:
    Z = widths(S)
    c = _unpack(_entry_layout(S), carry)
    R, W, ok = MC.entry_forms(C, Z.MS, alpha, gamma, c.R, c.W, c.ok, c.first, c.pen, c.ptag, c.pa, en, tag, a, v0, v, t)
    return [R + W + ok + [ZERO] + list(en) + list(tag) + list(a), list(en)]


def _sig_entry(S):
    widths(S)
    carry = Array(_size(_entry_layout(S)), BIT)
    return ((("carry", carry), ("en", BIT), ("tag", Array(G, BIT)), ("a", F), ("v0", F), ("v", F),
             ("t", Array(S.TW, BIT)), ("alpha", F), ("gamma", F)), Tuple(carry, BIT))


def _sig_end(S):
    Z = widths(S)
    return ((("slots", Array(_size(_carry_layout(S)), BIT)), ("entries", Array(_size(_entry_layout(S)), BIT))),
            Tuple(BIT, Array(Z.NW, BIT), Array(Z.NW, BIT), Array(S.TW, BIT), F))


def end_forms(C, S, slots, entries) -> list[list]:
    """The verdict: every check, R = W, and the stream not inside a call."""
    c, e = _unpack(_carry_layout(S), slots), _unpack(_entry_layout(S), entries)
    ok = and_reduce(C, [e.ok[0], c.busy[0].inv(), _eq(C, e.R, e.W)])
    return [[ok], c.head, c.tail, c.seq, c.P]


PoolSlot = TR.part(
    "PoolSlot", SLOT_STATICS, _sig_slot, slot_forms,
    "one slot of a window's stream over vLLM's block pool as memory-checked RAM: one access of the call it is in (or a "
    "new call's first), its call's progress and its bus record, its read's fingerprint factors")
PoolEntry = TR.part(
    "PoolEntry", SLOT_STATICS, _sig_entry, entry_forms,
    "one entry of the pool stream's touched list, sorted by (tag, address): its initial and final records "
    "(`MemEntry_v1`)")
PoolEnd = TR.part(
    "PoolEnd", SLOT_STATICS, _sig_end, end_forms,
    "the pool stream's verdict: every slot's and entry's check, R = W, and no call cut off; the queue's ends, the calls "
    "and the bus product")
PARTS = (PoolSlot, PoolEntry, PoolEnd)


def _in_fields(S) -> list[tuple[str, object]]:
    Z = widths(S)
    NS, NE = S.NS, S.NE
    return [("alpha", F), ("gamma", F), ("head", Array(Z.NW, BIT)), ("tail", Array(Z.NW, BIT)),
            ("start", Array(NS, BIT)), ("kind", Array(NS, Array(3, BIT))), ("r", Array(NS, Array(S.RW, BIT))),
            ("a", Array(NS, Array(Z.CW, BIT))), ("b", Array(NS, Array(Z.CW, BIT))), ("v", Array(NS, F)),
            ("t", Array(NS, Array(S.TW, BIT))), ("l_en", Array(NE, BIT)), ("l_tag", Array(NE, Array(G, BIT))),
            ("l_a", Array(NE, F)), ("l_v0", Array(NE, F)), ("l_v", Array(NE, F)), ("l_t", Array(NE, Array(S.TW, BIT)))]


def _out_type(S) -> Tuple:
    Z = widths(S)
    return Tuple(BIT, Array(Z.NW, BIT), Array(Z.NW, BIT), Array(S.TW, BIT), F)


def _sig(S):
    return tuple(_in_fields(S)), _out_type(S)


def _body(B, S, alpha, gamma, head, tail, start, kind, r, a, b, v, t, l_en, l_tag, l_a, l_v0, l_v, l_t):
    Kc = _Consts(B)
    st = {k: getattr(S, k) for k in SLOT_STATICS}
    slot_p, entry_p, end_p = (bind(d, **st) for d in PARTS)
    lay = _carry_layout(S)
    rest = _size(lay) - (1 + 3 * 128 + 2 * S.TW + 2 * widths(S).NW)
    init = (Kc.bits(1, 1) + Kc.bits(1, 128) * 3 + Kc.bits(1, S.TW) + Kc.bits(0, S.TW) + _refs(head) + _refs(tail)
            + Kc.bits(0, rest))
    carry_t = Array(_size(lay), BIT)
    done = _refs(B.scan(slot_p, _coll(carry_t, init), [start, kind, r, a, b, v, t], [alpha, gamma]))[:_size(lay)]
    fin = _unpack(lay, done)
    elay = _entry_layout(S)
    einit = fin.R + fin.W + fin.ok + Kc.bits(1, 1) + Kc.bits(0, 1 + G + 128)
    ecarry_t = Array(_size(elay), BIT)
    edone = _refs(B.scan(entry_p, _coll(ecarry_t, einit), [l_en, l_tag, l_a, l_v0, l_v, l_t], [alpha, gamma]))
    return B.call(end_p, _coll(carry_t, done), _coll(ecarry_t, edone[:_size(elay)]))


CacheStep = CompositeDefinition(
    "CacheStep", 3, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="a window's calls on vLLM's prefix-cache block pool, as one stream of NS memory-checked access slots and its "
        "touched list of NE addresses: 1 iff every call ran as the pool runs it (`cache_step_v3.Stream`), with the queue's "
        "ends after, the calls run and the bus product of their records")


def unit(**st):
    return bind(CacheStep, **st)


def statics(NS: int, NE: int, NB: int, RW: int, KW: int, REFW: int, TW: int) -> dict:
    return {"NS": NS, "NE": NE, "NB": NB, "RW": RW, "KW": KW, "REFW": REFW, "TW": TW}


# --- arguments and evaluation ----------------------------------------------------------------------------------------------


def _bits(x: int, w: int) -> list[int]:
    if not 0 <= x < 1 << w:
        raise ValueError(f"{x} doesn't fit {w} bits")
    return [(x >> j) & 1 for j in range(w)]


def arguments(S, alpha: int, gamma: int, head: int, tail: int, slots: Sequence[Slot],
              entries: Sequence[tuple[int, int, int, int, int]]) -> list[list[int]]:
    """CacheStep_v3's arguments, each a flat list of bits (one instance of `evaluate_bits`): the slots padded to NS with
    idle ones and the entries to NE with unused ones."""
    Z = widths(S)
    if len(slots) > S.NS or len(entries) > S.NE:
        raise ValueError(f"{len(slots)} slots or {len(entries)} entries past the class's {S.NS} and {S.NE}")
    sl = list(slots) + [Slot(*[0] * 11, use=0)] * (S.NS - len(slots))
    en = [(1, *e) for e in entries] + [(0, 0, 0, 0, 0, 0)] * (S.NE - len(entries))
    return [_bits(alpha, 128), _bits(gamma, 128), _bits(head, Z.NW), _bits(tail, Z.NW),
            [x.start for x in sl], [q for x in sl for q in _bits(x.kind, 3)], [q for x in sl for q in _bits(x.r, S.RW)],
            [q for x in sl for q in _bits(x.a, Z.CW)], [q for x in sl for q in _bits(x.b, Z.CW)],
            [q for x in sl for q in _bits(x.v, 128)], [q for x in sl for q in _bits(x.t, S.TW)],
            [e[0] for e in en], [q for e in en for q in _bits(e[1], G)], [q for e in en for q in _bits(e[2], 128)],
            [q for e in en for q in _bits(e[3], 128)], [q for e in en for q in _bits(e[4], 128)],
            [q for e in en for q in _bits(e[5], S.TW)]]


class Out(NamedTuple):
    ok: int
    head: int
    tail: int
    calls: int
    P: int


def _int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def evaluate(st_: dict, cases: Sequence[list[list[int]]]) -> list[Out]:
    """CacheStep_v3 at statics ``st_`` on each case's `arguments`, in one evaluation."""
    S = SimpleNamespace(**st_)
    Z = widths(S)
    cols = [list(x) for x in zip(*cases)]
    out = evaluate_bits(unit(**st_), *cols).reshape(len(cases), -1).tolist()
    res = []
    for row in out:
        at = [1, Z.NW, Z.NW, S.TW, 128]
        f, i = [], 0
        for w in at:
            f.append(_int(row[i:i + w]))
            i += w
        res.append(Out(*f))
    return res


# --- cost ------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def part_ands(key: tuple) -> dict[str, int]:
    st_ = dict(zip(SLOT_STATICS, key))
    return {d.name: ands(bind(d, **st_)) for d in PARTS}


def size(**st_) -> dict[str, int]:
    """CacheStep_v3's ANDs from its parts': NS slots, NE entries and the end."""
    p = part_ands(tuple(st_[k] for k in SLOT_STATICS))
    return {"slot": p["PoolSlot"], "entry": p["PoolEntry"], "end": p["PoolEnd"],
            "ands": st_["NS"] * p["PoolSlot"] + st_["NE"] * p["PoolEntry"] + p["PoolEnd"]}
