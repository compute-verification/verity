"""vLLM's prefix-cache block pool as a Boolean Definition: one call's pass over the pool, eviction and freeing included.

    CacheStep_v2{N, U, TW, RW, L, R, D, NB, SW} (pool, slot, r, admit, take, free, K, wh, ev, nins, new, tau, held, ret,
                                                units) -> (ok, pool', trail)

`CacheStep_v1` (`cache_step`) checks a hit against a cache that never evicts, so past the cumulative pool bound advice
v1 sends each hit's shortfall. This unit models what vLLM d9105ea8's `BlockPool` does to the cached blocks (the
replay's `_KV` is its line-for-line model): which requests hold each cached block, the free queue's order, eviction
and the blocks a leaving request returns. So the hit, the evictable count `ev` and the returns `ret` that
`ReplayStep_v1` reads are derived at every step, past the bound too. `reference` is the specification in Python. The
encodings' `test_cache_step_v2.py` holds it to `_KV` over whole traces.

**The pool** is a trie of whole hit units of U tokens (the block; U a power of 2) in a table of N entries. An entry is
a unit (its U token ids), its parent (ROOT = N for a request's first unit), its owner (the lowest-numbered request that
has computed it since the entry was made), its level (1 + its unit's index) and its copies: the blocks cached under
it, oldest first, at most D. vLLM's lookup takes a hash's oldest block, so the hit reads copy 0. A copy has a holder
mask (bit s: slot s holds it) and a stamp, its place in the free queue while no slot holds it. vLLM's free queue is the
hashless free blocks, then the cached free blocks by the time they were freed. The pool keeps the hashless count hf,
and a cached free block is a copy whose mask is 0 (*queued*). An entry is valid iff it has a copy. Beside the table,
the pool keeps each slot's walk entry (spfx, ROOT while it holds nothing) and the free queue's clock. The empty pool
has no entries, every slot at ROOT, hf = NB − 1 (block 0 is vLLM's null block) and clock 0.

**Slots.** A request holding blocks has a slot (s < R), which holds exactly one copy at each level on the path from
ROOT to its walk entry. A call keeps that invariant from the empty pool, so each free stamps distinct times, and no
call has to read a slot's path back.

**One call** is one of four kinds, by (admit, take, free):
- *a lookup* (1, 0, 0) or *a lookup that takes its hit* (1, 1, 0): `find_longest_cache_hit` for a request admitted with
  nothing computed. Its window is the request's first KU = ⌈L / U⌉ units, and its walk starts at ROOT. Unit j < wh is a
  hit unit: cached under the walk so far, and one the lookup may take, (j + 1)·U ≤ K − 1 for K the request's known
  tokens. Unit wh is not both, so wh is the longest cached prefix. `ev` must be the number of hit units whose copy 0
  no slot holds (`blocks_to_allocate`'s evictable blocks). A take is `allocate_slots`' touch: its slot must hold
  nothing, the slot's bit is set on each hit unit's copy 0, and its walk entry becomes the hit's.
- *a compute* (0, 0, 0): `allocate_slots`' new blocks and `cache_blocks`. The slot takes `new` blocks from the free
  queue's front: the hashless first, and past them e = new − hf evicted cached free blocks, the e oldest queued copies.
  Then the walk goes on from the slot's walk entry: unit j < nins is the request's next full unit, which the slot
  caches. A unit already cached gets a new copy (D copies refuse), and the request's number r becomes its owner if
  that is lower. One not cached is a new entry at the lowest invalid index. The slot holds the new copy either way.
- *a free* (0, 0, 1): `free_request`, `held` the blocks the request holds. Its copies that only it holds join the free
  queue's back, deepest first (vLLM frees a request's blocks in reverse), its other `held` − (its copies) blocks are
  hashless and join hf, and `ret` must be both together, the blocks whose last reference it drops. Its slot then
  holds nothing.
Any other kind refuses, as do new or nins ≠ 0 on a lookup or a free, and held or ret ≠ 0 off a free.

**The threshold hint.** Which copies an eviction removes is fixed by their stamps, and `tau` is the witness that names
them: with e > 0, exactly e queued copies have stamps ≤ tau and fewer than e have stamps < tau. Then tau is the e-th
oldest queued stamp, so the removed copies are the e oldest and the hint leaves no freedom. With e = 0, tau = 0. A
free at clock c stamps its copy at level ℓ c + KU − ℓ and moves the clock to c + KU, so stamps are distinct. An entry
left with no copy becomes invalid.

**N = NB − 1 entries hold every pool.** Each valid entry has a copy, and the copies are distinct blocks other than the
null block. That needs a copy's parent to be cached while it is: vLLM evicts an entry only after its cached
descendants. A held copy's slot holds its parent's copy too, and a queued copy was freed together with or before a
copy of its parent, so its stamp is the lower one. So eviction never orphans an entry, and the call refuses if it
would (a valid entry whose parent is invalid), rather than let a reused index give the orphan a new parent.

**ok = 0 refuses**: the above; a hit unit not cached or past the lookup's cap; a hit shorter than the cached prefix;
wh or nins past the window; more evicted than queued; a tau that isn't the threshold; an eviction that orphans an
entry; a new entry in a full table or past level KU; a free whose `held` is less than its slot's copies, whose `ret`
isn't its returns, or that overflows hf or the clock; a slot ≥ R.

**Statics** (all public): the size class N entries; U the hit unit; TW a token id's width; RW a request number's width;
L max_model_len; R slots; D copies an entry; NB num_gpu_blocks; SW a stamp's width. Widths: EW = bitlen(N) (an entry or
ROOT), KU = ⌈L / U⌉, HW = bitlen(KU), LW = bitlen(L), SL = bitlen(R − 1), BW = bitlen(NB). Every field is
little-endian. A slot's mask bit and its stamps' order are the only places the call reads a slot's number, and an
invalid entry's unit, parent, owner and level are never read (`decode` drops them).

**Composition** (`compile_slice_v2.CompileSlice_v2`): the calls follow the replay's `_KV` calls in order. `ReplayStep_v1`
reads each reached candidate's hit and evictable count from its lookup, and each victim's and leaving slot's returns
from its free. Each free's `held` and each compute's `new` are ReplayStep_v1's own (⌈comp / B⌉ and the row's need), and
its free count is hf plus the queued copies. The slice's pool rows carry the pool. `trail` and `sources` give a hit's
canonical sources as `cache_step` does.

**Cost** (`size`): a unit of the window costs about N·(2·(U·TW + EW) + 2·RW + HW + D·R) ANDs: a comparison against
every entry and a masked write to every entry. A call is KU units, the eviction (about N·(2^EW + D·(R + 2·SW))) and
the free (about N·D·(2·R + SW)).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits.boolean.forms import ONE, ZERO, add, and_reduce, compress_columns, const_bits, mux, or_reduce, sub
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier import cache_step as V1
from verity.core.protocols.certifier.cache_step import _bits, _eq, _lt, _size, _unit_bits, _unpack
from verity.core.protocols.certifier.head_check import BIT, _coll, _Consts, _part, _refs, ands

STATICS = ("N", "U", "TW", "RW", "L", "R", "D", "NB", "SW")


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def widths(S) -> SimpleNamespace:
    Z = V1.widths(S)
    if min(S.R, S.D, S.SW) < 1 or S.NB < 2:
        raise ValueError(f"statics {dict(S.__dict__)}: R, D, SW ≥ 1 and NB ≥ 2")
    if Z.KU >> S.SW:
        raise ValueError(f"stamps of {S.SW} bits can't hold a free's KU = {Z.KU} times")
    return SimpleNamespace(**Z.__dict__, SL=_bitlen(S.R - 1), BW=_bitlen(S.NB))


# --- the specification ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Copy:
    mask: int                   # the slots holding it, bit s for slot s
    stamp: int = 0              # its free-queue time, read while its mask is 0


@dataclass(frozen=True)
class Entry:
    parent: int                 # an entry, or ROOT = N
    tokens: tuple[int, ...]     # the unit's U token ids
    owner: int                  # the lowest-numbered request that has computed it since the entry was made
    level: int                  # 1 + its unit's index
    copies: tuple[Copy, ...]    # the blocks cached under it, oldest first


@dataclass(frozen=True)
class Pool:
    table: tuple[Entry | None, ...]     # N entries, None invalid
    spfx: tuple[int, ...]               # each slot's walk entry, ROOT while it holds nothing
    hf: int                             # the hashless free blocks
    clock: int


@dataclass(frozen=True)
class CallInputs:
    """One call's inputs beside the pool (the module's docstring)."""
    slot: int = 0
    r: int = 0
    units: tuple[tuple[int, ...], ...] = ()
    admit: bool = False
    take: bool = False
    free: bool = False
    K: int = 0
    wh: int = 0
    ev: int = 0
    nins: int = 0
    new: int = 0
    tau: int = 0
    held: int = 0
    ret: int = 0


@dataclass(frozen=True)
class Out:
    ok: bool
    pool: Pool | None = None
    trail: tuple[int, ...] = ()


def empty(S) -> Pool:
    return Pool((None,) * S.N, (S.N,) * S.R, S.NB - 1, 0)


def _in_class(S, pool: Pool, op: CallInputs) -> None:
    Z = widths(S)
    if len(pool.table) != S.N or len(pool.spfx) != S.R or len(op.units) > Z.KU:
        raise ValueError("the pool or the window isn't the size class's")
    if any(x is not None and not 1 <= len(x.copies) <= S.D for x in pool.table):
        raise ValueError(f"a valid entry has 1 to {S.D} copies")


def reference(S, pool: Pool, op: CallInputs) -> Out:
    """One call's pass over the pool (the module's docstring); `Out(False)` refuses."""
    _in_class(S, pool, op)
    Z = widths(S)
    N, KU, no = S.N, Z.KU, Out(False)
    if op.slot >= S.R or (op.take and not op.admit) or (op.admit and op.free):
        return no
    if (op.admit or op.free) and (op.new or op.nins) or not op.free and (op.held or op.ret):
        return no
    if op.wh > KU or op.nins > KU:
        return no
    table, spfx, bit = list(pool.table), list(pool.spfx), 1 << op.slot
    if op.take and spfx[op.slot] != N:
        return no

    e, hf = max(op.new - pool.hf, 0), max(pool.hf - op.new, 0)
    stamps = [c.stamp for x in table if x for c in x.copies if not c.mask]
    if e > len(stamps):
        return no
    if e:
        if sum(s <= op.tau for s in stamps) != e or sum(s < op.tau for s in stamps) >= e:
            return no
        for i, x in enumerate(table):
            if x:
                kept = tuple(c for c in x.copies if c.mask or c.stamp > op.tau)
                table[i] = replace(x, copies=kept) if kept else None
    elif op.tau:
        return no
    if any(x and x.parent != N and table[x.parent] is None for x in table):
        return no

    units = [tuple(u) for u in op.units] + [(0,) * S.U] * (KU - len(op.units))
    pfx = N if op.admit else spfx[op.slot]
    lvl = 0 if pfx == N or table[pfx] is None else table[pfx].level
    index = {(x.parent, x.tokens): i for i, x in enumerate(table) if x}
    trail, ev = [], 0
    for j, tok in enumerate(units):
        found = index.get((pfx, tok))
        look = op.admit and (j + 1) * S.U < op.K
        if j < op.wh:
            if not look or found is None:
                return no
            x = table[found]
            ev += not x.copies[0].mask
            if op.take:
                c0 = replace(x.copies[0], mask=x.copies[0].mask | bit)
                table[found] = replace(x, copies=(c0,) + x.copies[1:])
            pfx, lvl = found, lvl + 1
        elif j == op.wh and look and found is not None:
            return no
        elif j < op.nins:
            if found is not None:
                x = table[found]
                if len(x.copies) == S.D:
                    return no
                table[found] = replace(x, owner=min(x.owner, op.r), copies=x.copies + (Copy(bit),))
                pfx = found
            else:
                holes = [i for i, x in enumerate(table) if x is None]
                if not holes or lvl + 1 > KU:
                    return no
                table[holes[0]] = Entry(pfx, tok, op.r, lvl + 1, (Copy(bit),))
                index[(pfx, tok)] = holes[0]
                pfx = holes[0]
            lvl += 1
        trail.append(pfx)
    if ev != op.ev:
        return no

    clock = pool.clock
    if op.free:
        has = [c for x in table if x for c in x.copies if c.mask & bit]
        only = sum(c.mask == bit for c in has)
        if op.held < len(has) or op.ret != only + op.held - len(has):
            return no
        hf += op.held - len(has)
        if hf > S.NB - 1 or (clock + KU) >> S.SW:
            return no
        for i, x in enumerate(table):
            if x:
                table[i] = replace(x, copies=tuple(
                    Copy(c.mask & ~bit, clock + KU - x.level if c.mask == bit else c.stamp) for c in x.copies))
        clock += KU
        spfx[op.slot] = N
    elif not op.admit or op.take:
        spfx[op.slot] = pfx
    return Out(True, Pool(tuple(table), tuple(spfx), hf, clock), tuple(trail))


def sources(S, table: Sequence[Entry | None], trail: Sequence[int], wh: int) -> tuple[tuple[int, int], ...]:
    """A hit's canonical sources, (request, tokens) runs, from its call's trail and the table at the step's end."""
    runs: list[list[int]] = []
    for j in range(wh):
        a = table[trail[j]].owner
        if runs and runs[-1][0] == a:
            runs[-1][1] += S.U
        else:
            runs.append([a, S.U])
    return tuple((a, n) for a, n in runs)


# --- the circuit -----------------------------------------------------------------------------------------------------------


def _cut(xs, w: int) -> list[list]:
    return [list(xs[i * w:(i + 1) * w]) for i in range(len(xs) // w)]


def _flat(rows) -> list:
    return [b for row in rows for b in row]


def _pick(C, sel, rows) -> list:
    """The row a one-hot (or all-zero) `sel` selects (all-zero for none)."""
    out = [ZERO] * len(rows[0])
    for s, row in zip(sel, rows):
        out = [o ^ C.AND(s, x) for o, x in zip(out, row)]
    return out


def _encode(sel, w: int) -> list:
    """The index a one-hot `sel` selects, in w bits."""
    out = [ZERO] * w
    for e, s in enumerate(sel):
        out = [x ^ (s if e >> b & 1 else ZERO) for b, x in enumerate(out)]
    return out


def _xor(bits) -> object:
    out = ZERO
    for b in bits:
        out = out ^ b
    return out


def _index(C, idx, leaves) -> object:
    """leaves[idx], a mux tree over 2^len(idx) bits."""
    level = list(leaves)
    for b in idx:
        level = [mux(C, b, [level[2 * i + 1]], [level[2 * i]])[0] for i in range(len(level) // 2)]
    return level[0]


def _count(C, bits) -> list:
    return compress_columns(C, [list(bits)])


def _A(n: int) -> Array:
    return Array(n, BIT)


def _sig_evict(S):
    Z = widths(S)
    N, D, R = S.N, S.D, S.R
    return ((("parent", _A(N * Z.EW)), ("level", _A(N * Z.HW)), ("live", _A(N * D)), ("mask", _A(N * D * R)),
             ("stamp", _A(N * D * S.SW)), ("spfx", _A(R * Z.EW)), ("hf", _A(Z.BW)), ("slot", _A(Z.SL)),
             ("flags", _A(3)), ("new", _A(Z.HW)), ("tau", _A(S.SW))),
            Tuple(_A(N * D), _A(N * D * R), _A(N * D * S.SW), _A(Z.BW), _A(R), _A(Z.EW), _A(Z.HW), BIT))


def _build_evict(C, S, parent, level, live, mask, stamp, spfx, hf, slot, flags, new, tau):
    """The new blocks' eviction, the trie check after it, the slot's bit and the walk's start."""
    Z = widths(S)
    N, D, R, EW, HW, BW = S.N, S.D, S.R, Z.EW, Z.HW, Z.BW
    par, lev, msk, stp, sp = _cut(parent, EW), _cut(level, HW), _cut(mask, R), _cut(stamp, S.SW), _cut(spfx, EW)
    live, tau, admit, take = list(live), list(tau), flags[0], flags[1]
    W = max(HW, BW) + 1
    d1, short = sub(C, list(new), list(hf), W)
    e = [C.AND(x, short.inv()) for x in d1[:HW]]
    hf2 = [C.AND(x, short) for x in sub(C, list(hf), list(new), W)[0][:BW]]
    queued = [C.AND(live[k], or_reduce(C, msk[k]).inv()) for k in range(N * D)]
    qle = [C.AND(q, _lt(C, tau, stp[k]).inv()) for k, q in enumerate(queued)]
    qlt = [C.AND(q, _lt(C, stp[k], tau)) for k, q in enumerate(queued)]
    epos = or_reduce(C, e)
    bad = [_lt(C, _count(C, queued), e),
           C.AND(epos, C.AND(_eq(C, _count(C, qle), e), _lt(C, _count(C, qlt), e)).inv()),
           C.AND(epos.inv(), or_reduce(C, tau))]
    keep = [live[k] ^ C.AND(epos, qle[k]) for k in range(N * D)]
    live2, mask2, stamp2 = [], [], []
    for x in range(N):
        rank = [ONE] + [ZERO] * (D - 1)     # one-hot: the copies kept before copy d
        sel = [[ZERO] * D for _ in range(D)]
        for d in range(D):
            k = keep[x * D + d]
            for i in range(d + 1):
                sel[i][d] = C.AND(k, rank[i])
            rank = [mux(C, k, [rank[i - 1] if i else ZERO], [rank[i]])[0] for i in range(D)]
        for i in range(D):
            live2.append(_xor(sel[i]))
            mask2 += _pick(C, sel[i], msk[x * D:(x + 1) * D])
            stamp2 += _pick(C, sel[i], stp[x * D:(x + 1) * D])
    valid = [live2[x * D] for x in range(N)]
    vext = valid + [ONE] + [ZERO] * ((1 << EW) - N - 1)
    bad += [C.AND(valid[x], _index(C, par[x], vext).inv()) for x in range(N)]
    sb = [_eq(C, list(slot), const_bits(s, Z.SL)) for s in range(R)]
    bad.append(or_reduce(C, sb).inv())
    root = const_bits(N, EW)
    cur = _pick(C, sb, sp)
    bad.append(C.AND(take, _eq(C, cur, root).inv()))
    pfx0 = mux(C, admit, root, cur)
    lvl0 = _pick(C, [_eq(C, pfx0, const_bits(x, EW)) for x in range(N)], lev)
    return [live2, mask2, stamp2, hf2, sb, pfx0, lvl0, [or_reduce(C, bad).inv()]]


def _carry_layout(S) -> list[tuple[str, int]]:
    """The scan's carry: the table but its stamps, ok, the walk's entry and level, the unit's index and the evictable."""
    Z = widths(S)
    N, D = S.N, S.D
    return [("parent", N * Z.EW), ("tok", N * Z.UW), ("owner", N * S.RW), ("level", N * Z.HW), ("live", N * D),
            ("mask", N * D * S.R), ("ok", 1), ("pfx", Z.EW), ("lvl", Z.HW), ("j", Z.HW + 1), ("ev", Z.HW)]


def _sig_unit(S):
    Z = widths(S)
    carry = _A(_size(_carry_layout(S)))
    return (("carry", carry), ("unit", _A(Z.UW)), ("sb", _A(S.R)), ("r", _A(S.RW)), ("flags", _A(2)),
            ("K", _A(Z.LW)), ("wh", _A(Z.HW)), ("nins", _A(Z.HW))), Tuple(carry, _A(Z.EW))


def _build_unit(C, S, carry, unit, sb, r, flags, K, wh, nins):
    """Unit j of the window (j the carry's index): a hit unit, the unit after the hit, or a unit the slot caches."""
    Z = widths(S)
    N, D, R, EW, UW, HW = S.N, S.D, S.R, Z.EW, Z.UW, Z.HW
    st = _unpack(_carry_layout(S), carry)
    par, tok, own, lev = _cut(st["parent"], EW), _cut(st["tok"], UW), _cut(st["owner"], S.RW), _cut(st["level"], HW)
    live, msk = st["live"], _cut(st["mask"], R)
    admit, take = flags
    j, pfx, lvl, unit, sb, r = st["j"], st["pfx"], st["lvl"], list(unit), list(sb), list(r)
    valid = [live[x * D] for x in range(N)]
    hits = [C.AND(valid[x], and_reduce(C, [(a ^ b).inv() for a, b in zip(par[x] + tok[x], pfx + unit)]))
            for x in range(N)]
    found = or_reduce(C, hits)
    j1 = add(C, j, [ONE], width=HW + 1)
    look = C.AND(admit, _lt(C, [ZERO] * Z.LU + j1, K))
    hitu = _lt(C, j, wh)
    ins = C.AND(_lt(C, j, nins), hitu.inv())
    newe, upd = C.AND(ins, found.inv()), C.AND(ins, found)
    pre, holes = ONE, []
    for x in range(N):
        holes.append(C.AND(pre, valid[x].inv()))
        pre = C.AND(pre, valid[x])
    l1 = add(C, lvl, [ONE], width=HW + 1)
    bad = [C.AND(hitu, C.AND(look, found).inv()), and_reduce(C, [_eq(C, j, wh), look, found]), C.AND(newe, pre),
           C.AND(newe, _lt(C, const_bits(Z.KU, HW + 1), l1)),
           C.AND(upd, _xor(C.AND(hits[x], live[x * D + D - 1]) for x in range(N)))]
    unheld = _xor(C.AND(hits[x], or_reduce(C, msk[x * D]).inv()) for x in range(N))
    ev = add(C, st["ev"], [C.AND(hitu, unheld)], width=HW)
    tt = C.AND(hitu, take)
    out = {k: [] for k in ("parent", "tok", "owner", "level", "live", "mask")}
    for x in range(N):
        wr, wt, uh = C.AND(newe, holes[x]), C.AND(tt, hits[x]), C.AND(upd, hits[x])
        lower = C.AND(uh, _lt(C, r, own[x]))
        out["parent"] += mux(C, wr, pfx, par[x])
        out["tok"] += mux(C, wr, unit, tok[x])
        out["owner"] += mux(C, wr ^ lower, r, own[x])
        out["level"] += mux(C, wr, l1[:HW], lev[x])
        for d in range(D):
            k = x * D + d
            w = wr ^ wt if d == 0 else C.AND(uh, C.AND(live[k - 1], live[k].inv()))
            out["live"].append(live[k] ^ (wr if d == 0 else w))
            out["mask"] += [m ^ C.AND(w, s) for m, s in zip(msk[k], sb)]
    nxt = mux(C, C.AND(hitu, found) ^ upd, _encode(hits, EW), mux(C, newe, _encode(holes, EW), pfx))
    ok = C.AND(st["ok"][0], or_reduce(C, bad).inv())
    packed = [b for k in ("parent", "tok", "owner", "level", "live", "mask") for b in out[k]]
    packed += [ok] + nxt + mux(C, hitu ^ ins, l1[:HW], lvl) + j1 + ev
    return [packed, nxt]


def _sig_free(S):
    Z = widths(S)
    N, D, R = S.N, S.D, S.R
    return ((("level", _A(N * Z.HW)), ("mask", _A(N * D * R)), ("stamp", _A(N * D * S.SW)), ("spfx", _A(R * Z.EW)),
             ("pfx", _A(Z.EW)), ("hf", _A(Z.BW)), ("clock", _A(S.SW)), ("sb", _A(R)), ("flags", _A(3)),
             ("held", _A(Z.HW)), ("ret", _A(Z.HW))),
            Tuple(_A(N * D * R), _A(N * D * S.SW), _A(R * Z.EW), _A(Z.BW), _A(S.SW), BIT))


def _build_free(C, S, level, mask, stamp, spfx, pfx, hf, clock, sb, flags, held, ret, derive: bool = False):
    """A free's returns and stamps, and the slot's walk entry after the call. With `derive` `ret` is unread and the
    free's returns follow the outputs (HW + 1 bits)."""
    Z = widths(S)
    N, D, R, EW, HW, BW, SW = S.N, S.D, S.R, Z.EW, Z.HW, Z.BW, S.SW
    lev, msk, stp, sp = _cut(level, HW), _cut(mask, R), _cut(stamp, SW), _cut(spfx, EW)
    admit, take, free = flags
    fs = [C.AND(free, s) for s in sb]
    cleared = [[C.AND(m, f.inv()) for m, f in zip(row, fs)] for row in msk]
    has = [_xor(m ^ c for m, c in zip(row, cl)) for row, cl in zip(msk, cleared)]
    only = [C.AND(h, or_reduce(C, cl).inv()) for h, cl in zip(has, cleared)]
    nh, no = _count(C, has), _count(C, only)
    W = max(HW, len(nh)) + 1
    hl, under = sub(C, list(held), nh, W)
    t = add(C, list(clock), const_bits(Z.KU, SW), width=SW, carry_out=True)
    hf3 = add(C, list(hf), hl[:HW], width=max(BW, HW) + 1)
    back = add(C, no, hl[:HW], width=max(len(no), HW) + 1)
    bad = [under] + ([] if derive else [_eq(C, list(ret), back).inv()]) + [_lt(C, const_bits(S.NB - 1, len(hf3)), hf3),
                                                                           t[SW]]
    at = [sub(C, t[:SW], lev[x], SW)[0] for x in range(N)]
    stamp2 = [b for k in range(N * D) for b in mux(C, only[k], at[k // D], stp[k])]
    stay = C.AND(admit, take.inv())
    root = const_bits(S.N, EW)
    spfx2 = []
    for s in range(R):
        spfx2 += mux(C, fs[s], root, mux(C, and_reduce(C, [sb[s], free.inv(), stay.inv()]), pfx, sp[s]))
    return [_flat(cleared), stamp2, spfx2, mux(C, free, hf3[:BW], list(hf)), mux(C, free, t[:SW], list(clock)),
            [C.AND(free, or_reduce(C, bad)).inv()]] + ([back] if derive else [])


def _sig_end(S):
    Z = widths(S)
    H = _A(Z.HW)
    return (("oks", _A(3)), ("evd", H), ("ev", H), ("wh", H), ("nins", H), ("flags", _A(3)), ("new", H), ("held", H),
            ("ret", H)), BIT


def _build_end(C, S, oks, evd, ev, wh, nins, flags, new, held, ret):
    Z = widths(S)
    KU = const_bits(Z.KU, Z.HW + 1)
    admit, take, free = flags

    def zero(v):
        return or_reduce(C, v).inv()

    conds = list(oks) + [_eq(C, evd, ev), _lt(C, KU, wh).inv(), _lt(C, KU, nins).inv(), C.AND(take, admit.inv()).inv(),
                         C.AND(admit, free).inv(), C.AND(admit ^ free, C.AND(zero(new), zero(nins)).inv()).inv(),
                         C.AND(free.inv(), C.AND(zero(held), zero(ret)).inv()).inv()]
    return [and_reduce(C, conds)]


CachePoolEvict = _part("CachePoolEvict", STATICS, _sig_evict, _build_evict,
                       "a call's eviction from the block pool's free queue (the e oldest queued copies, by the threshold "
                       "hint), the trie check after it, the slot's bit and the walk's start")
CachePoolUnit = _part("CachePoolUnit", STATICS, _sig_unit, _build_unit,
                      "one unit of a call's window through the block pool's table: a hit unit (cached, within the "
                      "lookup's cap, taken by the slot), the unit after the hit (not both), or a unit the slot caches "
                      "(a new copy, or a new entry)")
CachePoolFree = _part("CachePoolFree", STATICS, _sig_free, _build_free,
                      "a free's returns (its only-held copies and its hashless blocks), its copies' stamps and the "
                      "clock, and the slot's walk entry after the call")
CachePoolEnd = _part("CachePoolEnd", STATICS, _sig_end, _build_end,
                     "the call's ok: its parts', the evictable count, the window's bounds and the call's kind")
PARTS = (CachePoolEvict, CachePoolUnit, CachePoolFree, CachePoolEnd)


def _in_fields(S) -> list[tuple[str, object]]:
    Z = widths(S)
    N, D, R = S.N, S.D, S.R
    H = Array(Z.HW, BIT)
    return [("parent", Array(N, Array(Z.EW, BIT))), ("tok", Array(N, Array(Z.UW, BIT))),
            ("owner", Array(N, Array(S.RW, BIT))), ("level", Array(N, H)), ("live", Array(N, Array(D, BIT))),
            ("mask", Array(N, Array(D, Array(R, BIT)))), ("stamp", Array(N, Array(D, Array(S.SW, BIT)))),
            ("spfx", Array(R, Array(Z.EW, BIT))), ("hf", Array(Z.BW, BIT)), ("clock", Array(S.SW, BIT)),
            ("slot", Array(Z.SL, BIT)), ("r", Array(S.RW, BIT)), ("admit", BIT), ("take", BIT), ("free", BIT),
            ("K", Array(Z.LW, BIT)), ("wh", H), ("ev", H), ("nins", H), ("new", H), ("tau", Array(S.SW, BIT)),
            ("held", H), ("ret", H), ("units", Array(Z.KU, Array(Z.UW, BIT)))]


def _out_type(S) -> Tuple:
    f = dict(_in_fields(S))
    Z = widths(S)
    return Tuple(BIT, *(f[k] for k in ("parent", "tok", "owner", "level", "live", "mask", "stamp", "spfx", "hf", "clock")),
                 Array(Z.KU, Array(Z.EW, BIT)))


def _sig(S):
    return tuple(_in_fields(S)), _out_type(S)


def _body(B, S, parent, tok, owner, level, live, mask, stamp, spfx, hf, clock, slot, r, admit, take, free, K, wh, ev,
          nins, new, tau, held, ret, units):
    Z = widths(S)
    N, D, R = S.N, S.D, S.R
    Kc = _Consts(B)
    st = {k: getattr(S, k) for k in STATICS}
    evict_p, unit_p, free_p, end_p = (bind(d, **st) for d in PARTS)

    def A(refs):
        return _coll(_A(len(refs)), refs)

    flags = A(_refs(admit) + _refs(take) + _refs(free))
    pre = _refs(B.call(evict_p, A(_refs(parent)), A(_refs(level)), A(_refs(live)), A(_refs(mask)), A(_refs(stamp)),
                       A(_refs(spfx)), hf, slot, flags, new, tau))
    pre = _unpack([("live", N * D), ("mask", N * D * R), ("stamp", N * D * S.SW), ("hf", Z.BW), ("sb", R),
                   ("pfx", Z.EW), ("lvl", Z.HW), ("ok", 1)], pre)
    lay = _carry_layout(S)
    init = (_refs(parent) + _refs(tok) + _refs(owner) + _refs(level) + pre["live"] + pre["mask"] + Kc.bits(1, 1)
            + pre["pfx"] + pre["lvl"] + Kc.bits(0, Z.HW + 1) + Kc.bits(0, Z.HW))
    sb = A(pre["sb"])
    done = _refs(B.scan(unit_p, _coll(_A(_size(lay)), init), [units], [sb, r, A(_refs(admit) + _refs(take)), K, wh,
                                                                        nins]))
    fin = _unpack(lay, done[:_size(lay)])
    post = _refs(B.call(free_p, A(fin["level"]), A(fin["mask"]), A(pre["stamp"]), A(_refs(spfx)), A(fin["pfx"]),
                        A(pre["hf"]), clock, sb, flags, held, ret))
    post = _unpack([("mask", N * D * R), ("stamp", N * D * S.SW), ("spfx", R * Z.EW), ("hf", Z.BW), ("clock", S.SW),
                    ("ok", 1)], post)
    ok = _refs(B.call(end_p, A(pre["ok"] + fin["ok"] + post["ok"]), A(fin["ev"]), ev, wh, nins, flags, new, held, ret))
    out = (ok + fin["parent"] + fin["tok"] + fin["owner"] + fin["level"] + fin["live"] + post["mask"] + post["stamp"]
           + post["spfx"] + post["hf"] + post["clock"] + done[_size(lay):])
    return _coll(_out_type(S), out)


CacheStep = CompositeDefinition(
    "CacheStep", 2, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="one call's pass over vLLM's prefix-cache block pool: a lookup's hit and evictable count derived from the pool, "
        "a compute's eviction (by the threshold hint) and cached units, and a free's returns "
        "(`cache_step_v2.reference`)")


def unit(**st):
    return bind(CacheStep, **st)


def statics(N: int, unit_tokens: int, token_width: int, request_width: int, max_model_len: int, slots: int, copies: int,
            num_gpu_blocks: int, stamp_width: int) -> dict:
    return {"N": N, "U": unit_tokens, "TW": token_width, "RW": request_width, "L": max_model_len, "R": slots,
            "D": copies, "NB": num_gpu_blocks, "SW": stamp_width}


# --- arguments and evaluation ----------------------------------------------------------------------------------------------


def arguments(S, pool: Pool, op: CallInputs) -> list[list[int]]:
    """The unit's arguments (each a flat list of bits, as `evaluate_bits` takes one instance)."""
    _in_class(S, pool, op)
    Z = widths(S)
    N, D, R = S.N, S.D, S.R
    none = Entry(0, (0,) * S.U, 0, 0, ())
    ents = [x or none for x in pool.table]
    copies = [list(x.copies) + [Copy(0, 0)] * (D - len(x.copies)) for x in ents]
    units = list(op.units) + [(0,) * S.U] * (Z.KU - len(op.units))
    H = Z.HW
    return [[b for x in ents for b in _bits(x.parent, Z.EW)], [b for x in ents for b in _unit_bits(S, x.tokens)],
            [b for x in ents for b in _bits(x.owner, S.RW)], [b for x in ents for b in _bits(x.level, H)],
            [int(d < len(x.copies)) for x in ents for d in range(D)],
            [b for cs in copies for c in cs for b in _bits(c.mask, R)],
            [b for cs in copies for c in cs for b in _bits(c.stamp, S.SW)],
            [b for p in pool.spfx for b in _bits(p, Z.EW)], _bits(pool.hf, Z.BW), _bits(pool.clock, S.SW),
            _bits(op.slot, Z.SL), _bits(op.r, S.RW), [int(op.admit)], [int(op.take)], [int(op.free)], _bits(op.K, Z.LW),
            _bits(op.wh, H), _bits(op.ev, H), _bits(op.nins, H), _bits(op.new, H), _bits(op.tau, S.SW),
            _bits(op.held, H), _bits(op.ret, H), [b for u in units for b in _unit_bits(S, u)]]


def decode(S, flat: Sequence[int]) -> Out:
    """The unit's output bits as an `Out` (the table's valid entries; an invalid entry's fields are never read)."""
    Z = widths(S)
    N, D, R, at = S.N, S.D, S.R, 0

    def take(n: int, w: int) -> list[int]:
        nonlocal at
        vals = [sum(int(flat[at + e * w + b]) << b for b in range(w)) for e in range(n)]
        at += n * w
        return vals

    ok, = take(1, 1)
    parent, toks, owner, level = take(N, Z.EW), take(N * S.U, S.TW), take(N, S.RW), take(N, Z.HW)
    live, mask, stamp = take(N * D, 1), take(N * D, R), take(N * D, S.SW)
    spfx, (hf,), (clock,), trail = take(R, Z.EW), take(1, Z.BW), take(1, S.SW), take(Z.KU, Z.EW)
    if not ok:
        return Out(False)
    table = []
    for x in range(N):
        n = next((d for d in range(D) if not live[x * D + d]), D)
        if any(live[x * D + n:(x + 1) * D]) or any(mask[x * D + n:(x + 1) * D]) or any(stamp[x * D + n:(x + 1) * D]):
            raise AssertionError(f"entry {x}: its live copies aren't a prefix, or a dead copy isn't zero")
        cs = tuple(Copy(mask[x * D + d], stamp[x * D + d]) for d in range(n))
        table.append(Entry(parent[x], tuple(toks[x * S.U:(x + 1) * S.U]), owner[x], level[x], cs) if n else None)
    return Out(True, Pool(tuple(table), tuple(spfx), hf, clock), tuple(trail))


def evaluate(st_: dict, cases: Sequence[tuple[Pool, CallInputs]]) -> list[Out]:
    """The unit at statics `st_` on each (pool, op), in one evaluation."""
    S = SimpleNamespace(**st_)
    args = [list(rows) for rows in zip(*(arguments(S, p, o) for p, o in cases))]
    out = evaluate_bits(unit(**st_), *args)
    return [decode(S, row) for row in out.reshape(len(cases), -1).tolist()]


# --- cost ------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def part_ands(key: tuple) -> dict[str, int]:
    st_ = dict(zip(STATICS, key))
    return {d.name: ands(bind(d, **st_)) for d in PARTS}


def size(**st_) -> dict[str, int]:
    """The unit's ANDs: its eviction, per unit of the window, its free and its end, and per call."""
    Z = widths(SimpleNamespace(**st_))
    p = part_ands(tuple(st_[k] for k in STATICS))
    return {"evict": p["CachePoolEvict"], "unit": p["CachePoolUnit"], "units": Z.KU, "free": p["CachePoolFree"],
            "end": p["CachePoolEnd"],
            "ands": p["CachePoolEvict"] + Z.KU * p["CachePoolUnit"] + p["CachePoolFree"] + p["CachePoolEnd"]}


__all__ = ["CachePoolEnd", "CachePoolEvict", "CachePoolFree", "CachePoolUnit", "CacheStep", "CallInputs", "Copy", "Entry", "Out",
           "PARTS", "Pool", "STATICS", "arguments", "decode", "empty", "evaluate", "reference", "size", "sources",
           "statics", "unit", "widths"]
