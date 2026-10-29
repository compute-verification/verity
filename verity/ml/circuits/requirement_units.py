"""The requirement's two proof blocks (I4): the plan unit, which holds a pass's plan to hold every block the registered
circuit reads at the pass's step (`Accounting.Requirement`'s `PlanMatches`), and the memory-map unit, which holds the
blocks a step writes to the circuit's values and every free, of a written block or a registered one, to its values' last
structural reader (`StoreMatches`' L1, L2 and L4), and each re-key to the lifetime cap. The plan unit is PlanMatch, or
PlanShard for a store past PlanMatch's 16-bit indices. Each is a Boolean Definition, one instance per step (PlanShard's
one per shard of the store), each returning one ok word: bit 0 the clause's verdict, bits 1-15 zero in gates.

    PlanMatch_v1{P, NR, V, G, TB, RPB}
        (count: U16, regmask: Array<NR / 16, U16>, kvmask: Array<V / 16, U16>, gmask: U16, lam: Array<2 V, U16>,
         plan: Array<P, U16>, tok: Array<G, U16>) -> U16
    PlanShard_v1{S, GAPS}
        (count: U16, size: U16, regmask: Array<S / 16, U16>, plan: Array<S, U16>) -> U16
    MemoryMapStep_v3{K, C, V, E, M, N, SLOTS, T}
        (t: U16, first: U16, count: U16, present: U16, pos: Array<K, U16>, ep: Array<K, U16>, last: Array<K, U16>,
         rfree: Array<⌈N / 16⌉, U16>, life: U16, lam: Array<2 V E, U16>, phi: Array<N + 2 M, U16>, pi: Array<T, U16>,
         val: Array<K, Array<VW / 16, U16>>, pay: Array<C, Array<2048, U16>>) -> U16
        U16 = Array<16, Value<1>>, a word's bits LSB first; VW = 32,768 / SLOTS

**The store.** Blocks are `requirement.BLOCK_BITS` (4,096 bytes), named by store index: the registered ones below N, then the
written ones from N in the order the memory map wrote them (memory-map/v1's fresh indices). A written block holds SLOTS values
of VW bits, value s at bits [s VW, (s + 1) VW) of its payload (bit t of a byte string is bit t & 7 of byte t >> 3), the bits
of an empty slot zero. A written value is named by its position in the run's written values (born step, then the circuit's
value order), and it sits in one block an epoch: epoch 0 from its production step, epoch e from its block's e-th re-key.
The layout row ``lam`` holds, for entry x = e V + i (position i's epoch e), its store index (word 2 x) and its slot (word
2 x + 1), zero for an epoch the value doesn't reach; the plan unit, bound at V E entries, reads it as it is. The map
row ``phi`` holds, for registered block j, the step that freed it (word j), and for written block b = j - N, the step that
wrote it (word N + 2 b) and the step that freed it (word N + 2 b + 1), `WORD - 1` for a block no step frees: a free at step f
takes effect after f's reads, as `Store.resident` and memory-map/v1 read it. The pass row ``pi`` holds, for each of the
session's T windows, the step of its last pass (the previous window's for a window without one, 0 before any).

**What is public** is the registered circuit's requirement (the census's schedule: each value's birth, its last structural
reader, which values each step reads; `requirement.Requirement`), the registered store's layout (which registered block
holds which registered value), each step's count of written blocks and so its first fresh index, each value's epochs
(the step each re-key moves it at), the session's windows T and lifetime cap L (`Params`), and each pass's plan length.
Each instance's public row (``count``, ``regmask``, ``kvmask``, ``gmask`` of the plan unit; ``t``, ``first``, ``count``,
``present``, ``pos``, ``ep``, ``last``, ``rfree``, ``life`` of the memory-map unit) is that data, which the verifier
holds the committed row to by its opening. The layout, the map, the passes' steps, the payloads, the plans and the
circuit's values are witness: committed rows, read only inside the proof, each bound to its one commitment by its commit
string (`b || c`). So is the stale bound, the store index below which every written block is past the cap: the unit
derives it from the pass row as a step, the cap, and holds a re-key's old block's write step in the map below it; no
input names it.

**The appends.** A step's written blocks are the remote audit's append for that step. A payload row's commit string is the
append leaf's (`hm96-sha512/row/v2` of the same 4,096 bytes under the same salt: the leaf hides ``d_j``), so an append's root
is a function of its instance's payload rows' ``b || c`` and its public ``first`` and ``count``. The chain over the steps'
appends is then the record's (remote_bandwidth's `Record.appends`): no dummy append, none out of the map's order, no re-key the
map doesn't write. That is checked on the commitments, outside the circuit, with no gate. An append's coin ``c_a`` isn't read
here: it keys the block's encoding, which the opening's decode checks against the same leaf (under zero knowledge, the
block-decode unit's payload-against-digest check, with ``c_a`` a public input beside the append's root and count).

**The value commitments' layout.** A memory-map instance's row ``val{k}`` (k present) is leaf ``pos[k]`` of the one-stage
value commitments, which hold the written values in position order, and its row ``pay{c}`` (c below ``count``) is block
``first + c``'s payload, its append's leaf at that store index. `map_value_leaves` gives both from the public row alone. So
a unit holding a block's payload (the block-decode unit) reaches the circuit's KV values by ``b || c``: its payload row is
the map instance's ``pay{c}``, which carries each ``val{k}`` in a slot the private layout names, and each ``val{k}`` is its
value leaf.

**PlanMatch**, at step t: the blocks t reads are the registered ones its structural reads hold (``regmask``, bit j block j
for j < NR), the block of each gathered table row (``gmask``'s bit g for each of the G gathers: the table's row ``tok[g]``,
a value of the circuit, at block TB + tok[g] / RPB, RPB rows a block, the table's blocks TB ... NR - 1), and the block of each
written value t reads (``kvmask``'s bit i for position i: block ``lam[2 i]``). The plan is ``plan[0 .. count)``, strictly
increasing, zero after; the unit's ok is that it is exactly that set, ``plan = readBlocks t``: every block t reads is
planned, which is `PlanMatches`' ``blocks (creditedReads t) ⊆ plan``, and no other is (a gather's unselected rows and a
paused request's pages are in no plan). ``count`` is at most P, and a gathered row is one of the table's.

**PlanShard**, at step t, for a store of registered blocks only (no step writes one) too large for one PlanMatch: the store's
N blocks are cut into public shards, shard k the ``size`` blocks from k S (`plan_shards`; S for every shard but the last), and
each pass has one instance per shard, its plan entries in the shard less k S. Its plan is ``plan[0 .. count)``, strictly
increasing below ``size``, zero after, missing at most GAPS of the shard's blocks (``size - count`` at most GAPS, so entry p
is one of blocks p .. p + GAPS), and holding every block ``regmask`` names (bit j block k S + j): the registered blocks the
step reads, a gathered table counted whole, since its rows are the circuit's values. So the shards' plans, in shard order,
are one strictly increasing plan of the store's blocks, and its instances' oks are `PlanMatches`' ``blocks (creditedReads
s) ⊆ plan``, the clause the Lean link reads; a planned block the step doesn't read (filler, a table's unselected rows) is
allowed, unlike PlanMatch's ``plan = readBlocks t``, so a planned block may carry no value's bits. Each instance's
``count`` is public, the pass's plan length their sum. Every word is below 2^16 at any N: no index past a shard's reaches
a unit.

**MemoryMapStep**, at step t: the step wrote blocks ``first .. first + count`` (at most C). Each value it produced that
a later step reads is present at epoch 0, and each value a re-key at t moves is present at its new epoch (``present``'s
bit k, its position ``pos[k]``, its epoch ``ep[k]`` and its last structural reader in that epoch ``last[k]``:
`Requirement.last`, a gather that may select it included, or the step of its next re-key), and its committed value is
``val[k]``. Its ok is that, for every present k: ``lam``'s entry ``ep[k] V + pos[k]`` puts it in a block the step wrote,
in a slot below SLOTS, no two in one slot, and that block's payload ``pay[j - first]`` carries ``val[k]`` there (L4,
Carries); and, for each block the step wrote: ``phi`` records it written at t (each write at its values' production step
or their re-key's), it holds at least one value (so no re-key carries a payload the map doesn't), its empty slots are
zero, and ``phi`` frees it at the last structural reader of its values, the greatest ``last[k]`` among them: an earlier
free drops a value still to be read (L2), a later one keeps a block no step reads (L1). And each registered block whose
values' last structural reader is t (``rfree``'s bit j, from the public schedule and the registered layout) is freed at
t, ``phi[j] = t``: each registered block's free is pinned by the one instance at its last structural reader. ``first``
is at least N, and ``first - N + count`` at most M.

**Re-keys at the cap.** A re-key at t moves a block's live values into a fresh block of t at the same slots, so its
payload is the old one's with any dead value's slot zeroed, and the old block's own instance frees it at t (its values'
last structural reader in the old epoch). For each value present at an epoch e ≥ 1, the unit reads its old block o from
entry ``(e - 1) V + pos[k]``, private, and holds it to the cap: o is a written block below ``first``, its slot is the
new one's, and its write step ``phi[N + 2 (o - N)]`` is below the cap. The cap is a step the unit computes from the pass
row and ``life`` (L): with W the windows whose last pass is at or before t (so W is the window of the first pass after
t), it is ``pi[W - L]``, and 0 if W < L or no pass follows t (W = T). A block's append joins the window of the first
pass after its write step (each writing instance pins the step, and the record's heads pin the append between the passes
around it: `remote_bandwidth`'s `RequirementLink.mapped`), so a write step below the cap is an age, counted from the
first root sent after the append, of at least L windows at the first pass after t, and o is below that pass's stale
bound (the blocks whose appends joined a window at least L before it): the cap is that bound in step form (`rekey_cap`).
It reads the passes' steps only through ``pi``, which is bound to the record's passes by its commitment.

**Cost.** PlanMatch: one 16-bit comparison of each plan entry with each block t may read, P (NR + G + V) of them, about
P (NR + G + V) 16 ANDs. PlanShard: each entry's order, range and tail checks and its count comparison, and each named
block's search over GAPS + 1 entries, about S (55 + 17 (GAPS + 1)) ANDs, 2,367,485 at S = 32,768 and GAPS = 0, its plan row
2^19 bits. MemoryMapStep: each present value against each slot of each written block, K C SLOTS VW ANDs, and the empty
slots' C SLOTS VW, then `lam` and `phi` read by one-hot selection, and a 16-bit comparison a registered block, 16 N; a
re-key's old block costs each present value one more selection over the layout's entries past epoch 0 and one over the
written blocks, about 32 (V (E - 1) + M) ANDs, and the cap a comparison, a count and a selection a window, about 80 T. A
statement adds each row's hm96 commitment.

`plan_match`, `plan_shard` and `memory_map` are the references, over plain ints; `plan_rows`, `plan_shard_rows` and
`memory_map_rows` name each instance's rows (C-Flock's `in_rows` params). `plan_inputs` and `map_inputs` are a step's arguments
from a `requirement.Requirement`, its `Layout` and `Store` (`layout_words`, `map_words` and `pass_words` the shared rows),
`shard_inputs` a shard's from a pass's plan and the blocks its step reads, and `row_words` an instance's rows from them.
"""
from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from . import boolean as BL
from .boolean import trace as TR
from .boolean.forms import ONE, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from .defs import CompositeDefinition
from .requirement import BLOCK_BITS, Layout, Requirement, Store
from .types import Array

__all__ = ["BLOCK_WORDS", "MemoryMapStep", "PlanMatch", "PlanShard", "U16", "layout_words", "map_inputs", "map_value_leaves",
           "map_words", "memory_map", "memory_map_rows", "pass_words", "plan_inputs", "plan_match", "plan_rows", "plan_shard",
           "plan_shard_rows", "plan_shards", "registered_frees", "rekey_cap", "row_words", "shard_inputs", "value_bits"]

U16 = Array(16, BL.BIT)
#: a block's payload in u16 words
BLOCK_WORDS = BLOCK_BITS // 16
#: a step or a store index, and the free of a block no step frees
WORD = 1 << 16


def value_bits(SLOTS: int) -> int:
    return BLOCK_BITS // SLOTS


def _pos_int(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"{name} is an int >= 1, not {v!r}")
    return v


def _check_plan(S) -> None:
    for k in ("P", "NR", "V", "G", "RPB"):
        _pos_int(k, getattr(S, k))
    if S.NR % 16 or S.V % 16:
        raise ValueError(f"NR and V are whole u16 words of mask bits, not {S.NR} and {S.V}")
    if S.G > 16 or S.P >= WORD or S.NR >= WORD or S.V >= WORD:
        raise ValueError(f"G is at most 16 and P, NR, V below 2^16, not G = {S.G}, P = {S.P}, NR = {S.NR}, V = {S.V}")
    if S.RPB & (S.RPB - 1):
        raise ValueError(f"RPB, the table's rows a block, is a power of two, not {S.RPB}")
    if isinstance(S.TB, bool) or not isinstance(S.TB, int) or not 0 <= S.TB < S.NR:
        raise ValueError(f"TB, the gathered table's first block, is a registered block below NR = {S.NR}, not {S.TB!r}")


def _check_shard(S) -> None:
    _pos_int("S", S.S)
    if S.S % 16 or S.S >= WORD:
        raise ValueError(f"S is whole u16 words of mask bits below 2^16, not {S.S}")
    if isinstance(S.GAPS, bool) or not isinstance(S.GAPS, int) or not 0 <= S.GAPS < S.S:
        raise ValueError(f"GAPS, the most blocks a shard's plan leaves out, is an int in [0, S), not {S.GAPS!r}")


def _check_map(S) -> None:
    for k in ("K", "C", "V", "E", "M", "N", "SLOTS", "T"):
        _pos_int(k, getattr(S, k))
    if S.K > 16 or S.SLOTS not in (1, 2, 4, 8) or S.N + S.M > WORD or S.V * S.E >= WORD or S.T >= WORD:
        raise ValueError(f"K is at most 16, SLOTS one of 1, 2, 4, 8, N + M at most 2^16 and V E and T below it, not K = {S.K}, "
                         f"SLOTS = {S.SLOTS}, N + M = {S.N + S.M}, V E = {S.V * S.E}, T = {S.T}")


# --- references ----------------------------------------------------------------------------------------------------------

def _bits(words: Sequence[int], n: int) -> list[int]:
    return [(words[i >> 4] >> (i & 15)) & 1 for i in range(n)]


def plan_match(S, count: int, regmask: Sequence[int], kvmask: Sequence[int], gmask: int, lam: Sequence[int],
               plan: Sequence[int], tok: Sequence[int]) -> int:
    """`PlanMatch`'s reference: 1 iff ``plan[0 .. count)`` is strictly increasing, zero after, and is exactly the blocks the
    step reads."""
    _check_plan(S)
    if count > S.P:
        return 0
    rows = [t >> (S.RPB.bit_length() - 1) for g, t in enumerate(tok) if (gmask >> g) & 1]
    if any(r >= S.NR - S.TB for r in rows):
        return 0
    need = {j for j, b in enumerate(_bits(regmask, S.NR)) if b} | {S.TB + r for r in rows}
    need |= {lam[2 * i] for i, b in enumerate(_bits(kvmask, S.V)) if b}
    head, tail = list(plan[:count]), list(plan[count:])
    return int(all(a < b for a, b in zip(head, head[1:])) and not any(tail) and set(head) == need)


def plan_shard(S, count: int, size: int, regmask: Sequence[int], plan: Sequence[int]) -> int:
    """`PlanShard`'s reference: 1 iff ``plan[0 .. count)`` is strictly increasing below ``size`` (at most S), zero after,
    leaves out at most GAPS of the shard's blocks, and holds every block ``regmask`` names."""
    _check_shard(S)
    if size > S.S or count > size or size - count > S.GAPS:
        return 0
    head, tail = list(plan[:count]), list(plan[count:])
    if any(a >= b for a, b in zip(head, head[1:])) or any(tail) or any(j >= size for j in head):
        return 0
    return int({j for j, b in enumerate(_bits(regmask, S.S)) if b} <= set(head))


def rekey_cap(S, pi: Sequence[int], life: int, t: int) -> int:
    """The cap at step ``t`` from the pass row ``pi`` and the lifetime cap ``life``: with W the windows whose last pass is at
    or before t, ``pi[W - life]``, 0 if W < life or W = T (no pass follows t)."""
    w = sum(p <= t for p in pi)
    return pi[w - life] if life <= w < S.T else 0


def memory_map(S, t: int, first: int, count: int, present: int, pos: Sequence[int], ep: Sequence[int], last: Sequence[int],
               rfree: Sequence[int], life: int, lam: Sequence[int], phi: Sequence[int], pi: Sequence[int],
               val: Sequence[int], pay: Sequence[int]) -> int:
    """`MemoryMapStep`'s reference over the values and payloads as ints (bit s VW + u of ``pay[c]`` is slot s's bit u): 1 iff
    every present value is in a slot of a block the step wrote that carries it, each such block is written at ``t``, holds a
    value, has its empty slots zero and is freed at its values' last structural reader, each registered block ``rfree``
    names is freed at ``t``, and each value present at an epoch past 0 comes from the same slot of a written block below
    ``first`` whose write step is below the cap (`rekey_cap`)."""
    _check_map(S)
    vw, cap = value_bits(S.SLOTS), rekey_cap(S, pi, life, t)
    if first < S.N or count > S.C or first - S.N + count > S.M:
        return 0
    if any(b and phi[j] != t for j, b in enumerate(_bits(rfree, S.N))):
        return 0
    held: dict[int, list[int]] = {c: [] for c in range(count)}
    slots: set[tuple[int, int]] = set()
    for k in range(S.K):
        if not (present >> k) & 1:
            continue
        if pos[k] >= S.V or ep[k] >= S.E:
            return 0
        x = ep[k] * S.V + pos[k]
        j, s = lam[2 * x], lam[2 * x + 1]
        if not first <= j < first + count or s >= S.SLOTS or (j, s) in slots:
            return 0
        if ep[k]:
            o, so = lam[2 * (x - S.V)], lam[2 * (x - S.V) + 1]
            if so != s or not S.N <= o < first or phi[S.N + 2 * (o - S.N)] >= cap:
                return 0
        slots.add((j, s))
        if (pay[j - first] >> (s * vw)) % (1 << vw) != val[k]:
            return 0
        held[j - first].append(last[k])
    for c in range(count):
        b = S.N + 2 * (first - S.N + c)
        if not held[c] or phi[b] != t or phi[b + 1] != max(held[c]):
            return 0
        if any((pay[c] >> (s * vw)) % (1 << vw) for s in range(S.SLOTS) if (first + c, s) not in slots):
            return 0
    return 1


# --- the circuits --------------------------------------------------------------------------------------------------------

def _eq(C, a, b):
    return and_reduce(C, [(x ^ y).inv() for x, y in zip(a, b, strict=True)])


def _eqc(C, a, k: int):
    return and_reduce(C, [x if (k >> i) & 1 else x.inv() for i, x in enumerate(a)]) if k < 1 << len(a) else ZERO


def _lt(C, a, b):
    return sub(C, a, b, len(a))[1]


def _ltc(C, a, k: int):
    return _lt(C, a, const_bits(k, len(a))) if k < 1 << len(a) else ONE


def _imp(C, a, b):
    """a -> b."""
    return C.AND(a, b.inv()).inv()


def _pick(C, onehot, words):
    """The word whose one-hot bit is set (zero if none)."""
    out = [ZERO] * len(words[0])
    for h, w in zip(onehot, words, strict=True):
        out = [o ^ C.AND(h, x) for o, x in zip(out, w)]
    return out


def _ok(C, oks) -> list:
    return [and_reduce(C, oks)] + [ZERO] * 15


def _plan_build(C, S, *e):
    nr, nv = S.NR // 16, S.V // 16
    count = e[0]
    regm = [b for w in e[1:1 + nr] for b in w]
    at = 1 + nr
    kvm = [b for w in e[at:at + nv] for b in w]
    at += nv
    gm = e[at][:S.G]
    at += 1
    lam = e[at:at + 2 * S.V]
    plan = e[at + 2 * S.V:at + 2 * S.V + S.P]
    tok = e[at + 2 * S.V + S.P:]
    blk = [lam[2 * i] for i in range(S.V)]
    oks = [_ltc(C, count, S.P + 1)]
    valid = [_ltc(C, count, p + 1).inv() for p in range(S.P)]
    for p in range(S.P - 1):
        oks.append(_imp(C, valid[p + 1], _lt(C, plan[p], plan[p + 1])))
    for p in range(S.P):
        oks.append(_imp(C, valid[p].inv(), or_reduce(C, plan[p]).inv()))
    r = S.RPB.bit_length() - 1
    rows = [list(t[r:]) + [ZERO] * r for t in tok]
    gblk = [add(C, row, const_bits(S.TB, 16), width=16) for row in rows]
    for g in range(S.G):
        oks.append(_imp(C, gm[g], _ltc(C, rows[g], S.NR - S.TB)))
    reg = [[_eqc(C, plan[p], j) for j in range(S.NR)] for p in range(S.P)]
    gat = [[_eq(C, plan[p], gblk[g]) for g in range(S.G)] for p in range(S.P)]
    kv = [[_eq(C, plan[p], blk[i]) for i in range(S.V)] for p in range(S.P)]
    for p in range(S.P):
        hit = [C.AND(regm[j], reg[p][j]) for j in range(S.NR)] + [C.AND(gm[g], gat[p][g]) for g in range(S.G)]
        hit += [C.AND(kvm[i], kv[p][i]) for i in range(S.V)]
        oks.append(_imp(C, valid[p], or_reduce(C, hit)))
    for j in range(S.NR):
        oks.append(_imp(C, regm[j], or_reduce(C, [C.AND(valid[p], reg[p][j]) for p in range(S.P)])))
    for g in range(S.G):
        oks.append(_imp(C, gm[g], or_reduce(C, [C.AND(valid[p], gat[p][g]) for p in range(S.P)])))
    for i in range(S.V):
        oks.append(_imp(C, kvm[i], or_reduce(C, [C.AND(valid[p], kv[p][i]) for p in range(S.P)])))
    return _ok(C, oks)


def _shard_build(C, S, *e):
    count, size = e[0], e[1]
    regm = [b for w in e[2:2 + S.S // 16] for b in w]
    plan = e[2 + S.S // 16:]
    oks = [_ltc(C, size, S.S + 1), _lt(C, size, count).inv(), _ltc(C, sub(C, size, count, 16)[0], S.GAPS + 1)]
    valid = [_ltc(C, count, p + 1).inv() for p in range(S.S)]
    for p in range(S.S - 1):
        oks.append(_imp(C, valid[p + 1], _lt(C, plan[p], plan[p + 1])))
    for p in range(S.S):
        oks.append(_imp(C, valid[p], _lt(C, plan[p], size)))
        oks.append(_imp(C, valid[p].inv(), or_reduce(C, plan[p]).inv()))
    # in order, below size and missing at most GAPS blocks, entry p is one of blocks p .. p + GAPS
    for j in range(S.S):
        held = [C.AND(valid[p], _eqc(C, plan[p], j)) for p in range(max(0, j - S.GAPS), j + 1)]
        oks.append(_imp(C, regm[j], or_reduce(C, held)))
    return _ok(C, oks)


def _map_build(C, S, *e):
    vw, vww, nx = value_bits(S.SLOTS), value_bits(S.SLOTS) // 16, S.V * S.E
    t, first, count, present = e[:4]
    pos, ep, last = e[4:4 + S.K], e[4 + S.K:4 + 2 * S.K], e[4 + 2 * S.K:4 + 3 * S.K]
    at = 4 + 3 * S.K
    rfree = [b for w in e[at:at + _rwords(S)] for b in w][:S.N]
    at += _rwords(S)
    life = e[at]
    at += 1
    lam, phi = e[at:at + 2 * nx], e[at + 2 * nx:at + 2 * nx + S.N + 2 * S.M]
    at += 2 * nx + S.N + 2 * S.M
    pi = e[at:at + S.T]
    at += S.T
    rphi, phi = phi[:S.N], phi[S.N:]
    val = [[b for w in e[at + vww * k:at + vww * (k + 1)] for b in w] for k in range(S.K)]
    at += vww * S.K
    pay = [[b for w in e[at + BLOCK_WORDS * c:at + BLOCK_WORDS * (c + 1)] for b in w] for c in range(S.C)]
    pres = present[:S.K]
    b0, borrow = sub(C, first, const_bits(S.N, 16), 16)
    oks = [borrow.inv(), _ltc(C, count, S.C + 1), _ltc(C, add(C, b0, count, width=17), S.M + 1)]
    oks += [_imp(C, rfree[j], _eq(C, rphi[j], t)) for j in range(S.N)]
    cvalid = [_ltc(C, count, c + 1).inv() for c in range(S.C)]
    firsts = [add(C, first, const_bits(c, 16), width=16) for c in range(S.C)]
    js, ss, sel, slot = [], [], [], []
    written, cap = phi[0::2], _cap(C, S, t, life, pi) if S.E > 1 else None
    for k in range(S.K):
        pv, pe = [_eqc(C, pos[k], i) for i in range(S.V)], [_eqc(C, ep[k], x) for x in range(S.E)]
        oh = [C.AND(pe[x], pv[i]) for x in range(S.E) for i in range(S.V)]
        oks.append(_imp(C, pres[k], or_reduce(C, oh)))
        j, s = _pick(C, oh, lam[0::2]), _pick(C, oh, lam[1::2])
        if S.E > 1:
            old = oh[S.V:]
            o, so = _pick(C, old, lam[0:2 * (nx - S.V):2]), _pick(C, old, lam[1:2 * (nx - S.V):2])
            moved = C.AND(pres[k], or_reduce(C, old))
            ob, below_n = sub(C, o, const_bits(S.N, 16), 16)
            wrote = _pick(C, [_eqc(C, ob, m) for m in range(S.M)], written)
            for clause in (_eq(C, so, s), below_n.inv(), _lt(C, o, first), _lt(C, wrote, cap)):
                oks.append(_imp(C, moved, clause))
        js.append(j)
        ss.append(s)
        sel.append([C.AND(_eq(C, j, firsts[c]), cvalid[c]) for c in range(S.C)])
        slot.append([_eqc(C, s, u) for u in range(S.SLOTS)])
        oks.append(_imp(C, pres[k], or_reduce(C, sel[k])))
        oks.append(_imp(C, pres[k], or_reduce(C, slot[k])))
    put = [[[C.AND(pres[k], C.AND(sel[k][c], slot[k][u])) for u in range(S.SLOTS)] for c in range(S.C)] for k in range(S.K)]
    for k in range(S.K):
        carried = [C.AND(put[k][c][u], _eq(C, pay[c][u * vw:(u + 1) * vw], val[k])) for c in range(S.C) for u in range(S.SLOTS)]
        oks.append(_imp(C, pres[k], or_reduce(C, carried)))
        for k2 in range(k + 1, S.K):
            both = C.AND(C.AND(pres[k], pres[k2]), C.AND(_eq(C, js[k], js[k2]), _eq(C, ss[k], ss[k2])))
            oks.append(both.inv())
    for c in range(S.C):
        oh = [_eqc(C, add(C, b0, const_bits(c, 16), width=16), m) for m in range(S.M)]
        wrote, freed = _pick(C, oh, phi[0::2]), _pick(C, oh, phi[1::2])
        holds = [C.AND(pres[k], sel[k][c]) for k in range(S.K)]
        oks.append(_imp(C, cvalid[c], or_reduce(C, holds)))
        oks.append(_imp(C, cvalid[c], _eq(C, wrote, t)))
        m = [ZERO] * 16
        for k in range(S.K):
            m = mux(C, C.AND(holds[k], _lt(C, m, last[k])), last[k], m)
        oks.append(_imp(C, cvalid[c], _eq(C, freed, m)))
        for u in range(S.SLOTS):
            empty = C.AND(cvalid[c], or_reduce(C, [put[k][c][u] for k in range(S.K)]).inv())
            oks.append(_imp(C, empty, or_reduce(C, pay[c][u * vw:(u + 1) * vw]).inv()))
    return _ok(C, oks)


def _cap(C, S, t, life, pi):
    """`rekey_cap` in gates."""
    w = [ZERO] * 16
    for p in pi:
        w = add(C, w, [_lt(C, t, p).inv()] + [ZERO] * 15, width=16)
    at, short = sub(C, w, life, 16)
    ok = C.AND(short.inv(), _ltc(C, w, S.T))
    return [C.AND(ok, b) for b in _pick(C, [_eqc(C, at, x) for x in range(S.T)], pi)]


def _plan_signature(S):
    _check_plan(S)
    return ((("count", U16), ("regmask", Array(S.NR // 16, U16)), ("kvmask", Array(S.V // 16, U16)), ("gmask", U16),
             ("lam", Array(2 * S.V, U16)), ("plan", Array(S.P, U16)), ("tok", Array(S.G, U16))), U16)


def _shard_signature(S):
    _check_shard(S)
    return (("count", U16), ("size", U16), ("regmask", Array(S.S // 16, U16)), ("plan", Array(S.S, U16))), U16


def _rwords(S) -> int:
    """``rfree``'s words: a bit a registered block."""
    return -(-S.N // 16)


def _map_signature(S):
    _check_map(S)
    return ((("t", U16), ("first", U16), ("count", U16), ("present", U16), ("pos", Array(S.K, U16)), ("ep", Array(S.K, U16)),
             ("last", Array(S.K, U16)), ("rfree", Array(_rwords(S), U16)), ("life", U16), ("lam", Array(2 * S.V * S.E, U16)),
             ("phi", Array(S.N + 2 * S.M, U16)), ("pi", Array(S.T, U16)),
             ("val", Array(S.K, Array(value_bits(S.SLOTS) // 16, U16))), ("pay", Array(S.C, Array(BLOCK_WORDS, U16)))), U16)


def _unit(name: str, version: int, statics: tuple[str, ...], signature, build, doc: str) -> CompositeDefinition:
    @lru_cache(maxsize=None)
    def traced(*key) -> TR.Traced:
        S = SimpleNamespace(**dict(zip(statics, key)))
        params, ret = signature(S)
        return TR.trace(lambda C, *e: build(C, S, *e), [t for _, t in params], ret)

    def body(B, S, *colls):
        params, ret = signature(S)
        return TR.emit(traced(*(getattr(S, k) for k in statics)), [t for _, t in params], ret, B, *colls)

    d = CompositeDefinition(name, version, statics, signature, body, conformance="exact-model-tested", doc=doc)
    d.traced = traced  # type: ignore[attr-defined]
    return d


PlanMatch = _unit(
    "PlanMatch", 1, ("P", "NR", "V", "G", "TB", "RPB"), _plan_signature, _plan_build,
    "1 iff plan[0 .. count) is strictly increasing, zero after, and exactly the blocks the step reads: the registered blocks "
    "regmask names, block TB + tok[g] / RPB of each gather gmask names, and block lam[2 i] of each written value kvmask names")
PlanShard = _unit(
    "PlanShard", 1, ("S", "GAPS"), _shard_signature, _shard_build,
    "1 iff plan[0 .. count) is strictly increasing below size (at most S), zero after, leaves out at most GAPS of the "
    "shard's size blocks, and holds every block regmask names: a shard of a pass's plan holds the registered blocks its "
    "step reads")
MemoryMapStep = _unit(
    "MemoryMapStep", 3, ("K", "C", "V", "E", "M", "N", "SLOTS", "T"), _map_signature, _map_build,
    "1 iff every present value k is in a slot of a block first .. first + count that carries val[k] there, no two in one "
    "slot, each such block is written at t, holds a value, has its empty slots zero and is freed at its values' last "
    "structural reader, each registered block rfree names is freed at t, and each value present at epoch ep[k] > 0 comes "
    "from the same slot of a written block below first whose write step is below the cap pi[W - life], W the windows "
    "whose last pass is at or before t (0 if W < life or W = T)")


# --- each instance's rows -------------------------------------------------------------------------------------------------

def plan_rows(S) -> list[tuple[str, int, list[str]]]:
    """PlanMatch's rows: (row, bits, params), the public row first."""
    return [("pub", 16 * (2 + S.NR // 16 + S.V // 16), ["count", "regmask", "kvmask", "gmask"]), ("lam", 32 * S.V, ["lam"]),
            ("plan", 16 * S.P, ["plan"])] + [(f"tok{g}", 16, [f"tok[{g}]"]) for g in range(S.G)]


def plan_shard_rows(S) -> list[tuple[str, int, list[str]]]:
    """PlanShard's rows: (row, bits, params), the public row first."""
    return [("pub", 32 + S.S, ["count", "size", "regmask"]), ("plan", 16 * S.S, ["plan"])]


def memory_map_rows(S) -> list[tuple[str, int, list[str]]]:
    """MemoryMapStep's rows: (row, bits, params), the public row first."""
    return ([("pub", 16 * (5 + 3 * S.K + _rwords(S)), ["t", "first", "count", "present", "pos", "ep", "last", "rfree", "life"]),
             ("lam", 32 * S.V * S.E, ["lam"]), ("phi", 16 * (S.N + 2 * S.M), ["phi"]), ("pi", 16 * S.T, ["pi"])]
            + [(f"val{k}", value_bits(S.SLOTS), [f"val[{k}]"]) for k in range(S.K)]
            + [(f"pay{c}", BLOCK_BITS, [f"pay[{c}]"]) for c in range(S.C)])


# --- a step's arguments from the requirement --------------------------------------------------------------------------------

def _words(x: int, n: int) -> list[int]:
    return [(x >> (16 * i)) & 0xFFFF for i in range(n)]


def _mask(bits, n: int) -> list[int]:
    out = [0] * (n // 16)
    for b in bits:
        out[b >> 4] |= 1 << (b & 15)
    return out


def layout_words(S, req: Requirement, layout: Layout, positions: Sequence[int]) -> list[int]:
    """The layout row at epoch 0: written value ``positions[i]``'s store index and slot at words 2 i and 2 i + 1, zero past
    them and at every later epoch (a re-key's entries, ``(e V + i)``'s words, are its caller's)."""
    vw, out = value_bits(S.SLOTS), [0] * (2 * S.V * S.E)
    if len(positions) > S.V:
        raise ValueError(f"{len(positions)} written values, more than V = {S.V}")
    for i, v in enumerate(positions):
        if req.width[v] != vw or layout.offset[v] % vw:
            raise ValueError(f"value {v}: a written value is {vw} bits at a slot's offset, not {req.width[v]} at {layout.offset[v]}")
        out[2 * i], out[2 * i + 1] = layout.block[v], layout.offset[v] // vw
    return out


def map_words(S, store: Store) -> list[int]:
    """The map row: registered block j's free step at word j, and written block N + b's write step and free step at words
    N + 2 b and N + 2 b + 1 (`WORD - 1` for a block no step frees), zero past the blocks written."""
    if store.registered != S.N:
        raise ValueError(f"the store has {store.registered} registered blocks, not N = {S.N}")
    out = [WORD - 1] * S.N + [0] * (2 * S.M)
    for s, js in store.written.items():
        for j in js:
            if not S.N <= j < S.N + S.M:
                raise ValueError(f"written block {j} is outside N .. N + M = {S.N} .. {S.N + S.M}")
            out[S.N + 2 * (j - S.N)], out[S.N + 2 * (j - S.N) + 1] = s, WORD - 1
    for s, js in store.freed.items():
        for j in js:
            out[j if j < S.N else S.N + 2 * (j - S.N) + 1] = s
    return out


def registered_frees(req: Requirement, registered_block: dict[int, int]) -> dict[int, int]:
    """Each registered block's free step from public data alone: the last structural reader of the values the registered layout
    puts in it (a block holding no value any step reads has none)."""
    out: dict[int, int] = {}
    for v, j in registered_block.items():
        if v in req.last:
            out[j] = max(out.get(j, -1), req.last[v])
    return out


def map_value_leaves(S, pub: dict) -> dict[str, tuple[str, int]]:
    """Where a MemoryMapStep instance's value and payload rows sit, from its public row alone (``pub``'s ``first``, ``count``,
    ``present`` and ``pos``, as `map_inputs` gives them): ``val{k}`` of each present k is ``("value", pos[k])``, leaf pos[k] of
    the one-stage value commitments (the written values in position order), and ``pay{c}`` of each c below ``count`` is
    ``("block", first + c)``, that store index's payload (its append's leaf)."""
    if pub["count"] > S.C or pub["present"] >> S.K or any(pub["pos"][k] >= S.V for k in range(S.K) if (pub["present"] >> k) & 1):
        raise ValueError(f"a public row of at most C = {S.C} blocks and K = {S.K} values below V = {S.V}, not count "
                         f"{pub['count']}, present {pub['present']:#x}, pos {list(pub['pos'])}")
    out = {f"val{k}": ("value", pub["pos"][k]) for k in range(S.K) if (pub["present"] >> k) & 1}
    out.update({f"pay{c}": ("block", pub["first"] + c) for c in range(pub["count"])})
    return out


def plan_inputs(S, req: Requirement, t: int, plan: Sequence[int], registered_block: dict[int, int],
                gathers: Sequence[int], gathered: frozenset[int], positions: Sequence[int], lam: Sequence[int],
                entries: dict[int, int] | None = None) -> dict:
    """PlanMatch's arguments at step ``t``: ``plan`` the pass's blocks; ``registered_block`` each registered value's block
    (public); ``gathers`` the table rows t gathers (``tok``, at most G) and ``gathered`` the registered values they are, which
    the structural reads (``regmask``) leave out; ``positions`` the written values in order; ``lam`` the layout row, bound
    with the plan's V the map's V E; ``entries`` the layout entry of each written value t reads, ``e V + i`` at its epoch e
    at t (default its position: epoch 0)."""
    reads = req.reads(t)
    at = {v: i for i, v in enumerate(positions)} | dict(entries or {})
    if len(plan) > S.P or len(gathers) > S.G:
        raise ValueError(f"step {t}: a plan of {len(plan)} and {len(gathers)} gathers, more than P = {S.P} or G = {S.G}")
    return {"count": len(plan), "regmask": _mask({registered_block[v] for v in reads & (req.registered - gathered)}, S.NR),
            "kvmask": _mask({at[v] for v in reads - req.registered}, S.V), "gmask": (1 << len(gathers)) - 1, "lam": list(lam),
            "plan": sorted(plan) + [0] * (S.P - len(plan)), "tok": list(gathers) + [0] * (S.G - len(gathers))}


def plan_shards(N: int, S: int) -> list[tuple[int, int]]:
    """The shards of a store of ``N`` registered blocks: (first block, size), S blocks each but the last."""
    return [(k, min(S, N - k)) for k in range(0, N, S)]


def shard_inputs(S, first: int, size: int, plan: Sequence[int], need: Sequence[int]) -> dict:
    """PlanShard's arguments for the shard of ``size`` blocks from ``first``: ``plan`` the pass's blocks and ``need`` the
    registered blocks its step reads, both sorted (a ``range`` is cut without listing it)."""
    lo, hi = bisect_left(plan, first), bisect_left(plan, first + size)
    if hi - lo > S.S:
        raise ValueError(f"the shard at {first} plans {hi - lo} blocks, more than S = {S.S}")
    mine = [j - first for j in plan[lo:hi]]
    return {"count": len(mine), "size": size, "plan": mine + [0] * (S.S - len(mine)),
            "regmask": _mask((j - first for j in need[bisect_left(need, first):bisect_left(need, first + size)]), S.S)}


def map_inputs(S, req: Requirement, store: Store, t: int, positions: Sequence[int], lam: Sequence[int],
               phi: Sequence[int], registered_block: dict[int, int], pi: Sequence[int], life: int) -> dict:
    """MemoryMapStep's arguments at step ``t``: the values t produces that a later step reads, at epoch 0, the blocks it
    wrote and their payloads (``store.payload``), the registered blocks whose values' last structural reader t is (``registered_block`` each
    registered value's block, public), the lifetime cap ``life``, the layout row ``lam``, the map row ``phi`` and the pass
    row ``pi`` (`pass_words`)."""
    vw = value_bits(S.SLOTS)
    rfree = _mask({j for j, f in registered_frees(req, registered_block).items() if f == t}, 16 * _rwords(S))
    made = [i for i, v in enumerate(positions) if req.born[v] == t and req.last.get(v, -1) > t]
    js = sorted(store.written.get(t, ()))
    first = js[0] if js else S.N + sum(len(b) for s, b in store.written.items() if s < t)
    if len(made) > S.K or len(js) > S.C or js != list(range(first, first + len(js))):
        raise ValueError(f"step {t}: {len(made)} values and blocks {js}: at most K = {S.K} values and C = {S.C} consecutive blocks")
    pad = S.K - len(made)
    return {"t": t, "first": first, "count": len(js), "present": (1 << len(made)) - 1, "pos": made + [0] * pad,
            "ep": [0] * S.K, "last": [req.last[positions[i]] for i in made] + [0] * pad, "rfree": rfree, "life": life,
            "lam": list(lam), "phi": list(phi), "pi": list(pi),
            "val": [req.values[positions[i]] % (1 << vw) for i in made] + [0] * pad,
            "pay": [store.payload.get(j, 0) for j in js] + [0] * (S.C - len(js))}


def pass_words(S, passes: Sequence[tuple[int, int]]) -> list[int]:
    """The pass row from the session's passes, (window, step) each: window w's word the step of its last pass, the previous
    window's for a window without one, 0 before any. A block written at step s joins the window of the first pass after s,
    so a write step below `rekey_cap` is an age, at the first pass after the cap's step, of at least L windows: the stale
    bound in step form."""
    last: dict[int, int] = {}
    for w, s in passes:
        if not 0 <= w < S.T or not 0 <= s < WORD:
            raise ValueError(f"a pass of window {w} at step {s}: a window below T = {S.T} and a step below 2^16")
        last[w] = max(last.get(w, s), s)
    out, prev = [], 0
    for w in range(S.T):
        prev = last.get(w, prev)
        out.append(prev)
    return out


def row_words(S, rows: Sequence[tuple[str, int, list[str]]], args: dict) -> dict[str, list[int]]:
    """Each row's u16 words from an instance's arguments (`plan_inputs`, `map_inputs`), its params' words in order."""
    vww = value_bits(S.SLOTS) // 16 if hasattr(S, "SLOTS") else 0

    def words(p: str) -> list[int]:
        name, _, at = p.partition("[")
        v = args[name] if not at else args[name][int(at[:-1])]
        if name == "val":
            return _words(v, vww)
        if name == "pay":
            return _words(v, BLOCK_WORDS)
        return list(v) if isinstance(v, list) else [v]

    out = {r: [w for p in ps for w in words(p)] for r, _bits, ps in rows}
    if any(16 * len(out[r]) != bits for r, bits, _ in rows):
        raise AssertionError("a row's words are not its bits")
    return out
