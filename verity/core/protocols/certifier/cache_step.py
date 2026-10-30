"""vLLM's prefix cache as the derived encoding derives it, as a Boolean Definition: one row's pass over the cache.

    CacheStep_v1{N, U, TW, RW, L} (table, pfx, r, admit, sent, K, wh, nins, units) -> (ok, table', pfx', trail)

`ReplayStep_v1` (`replay_step`) takes each admitting row's prefix-cache hit as witness that it only bounds, so a hit
the cache didn't give would be a channel nobody charges. The derived encoding's rules
(`benchmarks/network_e2e/encodings/derived.py`, `Prefixes`) fix the hit while the cumulative pool bound rules out
eviction, and fix the canonical sources always. Past the bound, advice v1 sends and charges each hit's shortfall from
the longest cached prefix. This Definition is those rules over a committed cache state, with
`reference` as their specification in Python. The encodings' `test_cache_step.py` difftests both against `Prefixes`
on the fixture traces.

**The cache** is a trie of whole hit units of U tokens (the block; U a power of 2). It is a table of at most N entries,
each a unit (its U token ids), its parent entry (ROOT = N for a request's first unit) and its owner: the lowest-numbered
request that has computed that whole unit, in one run from a unit boundary, after the same tokens. Entry e stands for
the token prefix on the path from ROOT to e, so a prefix is matched by comparing tokens directly. That is cheaper here
than `derived.unit_key`'s SHA-512, and as exact: no key can collide. The valid entries are a prefix of the table, and no
two have the same parent and tokens.

**One call** walks one row's window of KU = ⌈L / U⌉ units, which `units` holds. For an admitting row (`admit`: its
request's first row, or a resumed one) the window is the request's tokens from position 0, and its walk starts at ROOT.
For any other row, the window starts at the unit holding its start, and the walk starts at `pfx`, the entry of the last
unit its request's run completed. Unit j is:
- *a hit unit* when j < wh. It must be cached under the walk so far and be one the lookup may take,
  (j + 1)·U ≤ K − 1 (vLLM keeps a request's last known token to compute). K is the request's known tokens.
- *the unit after the hit*, j = wh, of an admitting row while the hit is derived (`sent` = 0). It must not be both
  cached and one the lookup may take. So wh is the longest cached prefix, the hit vLLM's lookup returns while nothing
  has been evicted, and any other is refused.
- *computed* when it isn't a hit unit and j < nins. The row completes it whole, and the request's number r becomes
  its owner if it has none or a higher one. A unit not cached is appended to the table, and a full table refuses.

With `sent` = 1 the call checks only that the hit is a cached prefix: an evicting cache never hits past what a cache
that never evicts holds. Advice v1 doesn't use that mode. Its lookups always derive the longest cached prefix, and past
the bound the hit is that prefix less v1's sent shortfall. The walk's entry after each unit is `trail`. A hit unit's canonical source is its trail entry's owner in the
table at the end of the engine step (`sources`). An owner can still drop later in the same step, when a lower-numbered
request completes the unit.

**ok = 0 refuses**: a hit unit not cached or past the lookup's cap; a derived hit shorter than the cached prefix; wh or
nins past the window; an append to a full table.

**Statics** (all public): the size class N entries; U the hit unit; TW a token id's width; RW a request number's width;
L max_model_len. Widths: EW = bitlen(N) (an entry or ROOT), KU = ⌈L / U⌉, HW = bitlen(KU), LW = bitlen(L). Every field
is little-endian.

**Composition**: the calls follow `ReplayStep_v1`'s scheduling order, the running rows first.
An admitting candidate's call (nins = 0) derives the longest cached prefix. The hit `ReplayStep_v1` reads is that
prefix less the candidate's shortfall (0 while the bound holds, advice v1's sent one past it). A second call, from
the hit's trail entry, computes the units the row completes. r is the row's I1 request number. The replay's state chain
(`replay_chain`) carries the table in its state row, and `ReplayStep_v1` carries each running slot's pfx.
N = num_gpu_blocks − 1 holds every unit computed while the bound holds. Past it, N is the session's units. The compile
slice, inside the bound, runs an engine step's calls by `build_scan_step` (`scan_step` its reference) on
ReplayStep_v1's wires and the requests' certified tokens, with the table and the slots' pfx in a cache row
(`cache_fields`) carried beside the state row from the empty one.

**Cost** (`size`): a unit costs about N·(2·(U·TW + EW) + 2·RW) ANDs. That is a comparison against every entry and a
masked write to every entry. A call costs KU units plus its range checks: 5,856 ANDs at (N, U, TW, RW, L) =
(8, 4, 8, 4, 32) (5,701 lowered), 18,658 at (8, 16, 17, 10, 64), 5.42M at the difftest's (128, 16, 17, 7, 1168), and
at smallkv's (511, 16, 17, 10, 2048) 300,049 a unit and 38.4M a call. A linear table is the size class of a slice. The
full session scales by offline memory checking: it sorts the session's unit operations by key and time, as
`advice_check` sorts its events, at O(log² E) a unit for E operations in all.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits.boolean.forms import ONE, ZERO, and_reduce, const_bits, mux, or_reduce, sub, add
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier.head_check import BIT, _coll, _Consts, _part, _refs, ands

STATICS = ("N", "U", "TW", "RW", "L")


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def widths(S) -> SimpleNamespace:
    if S.U < 1 or S.U & (S.U - 1):
        raise ValueError(f"hit unit {S.U}: a power of 2")
    if min(S.N, S.TW, S.RW, S.L) < 1:
        raise ValueError(f"statics {dict(S.__dict__)}: sizes ≥ 1")
    KU = -(-S.L // S.U)
    return SimpleNamespace(EW=_bitlen(S.N), KU=KU, HW=_bitlen(KU), LW=_bitlen(S.L), LU=S.U.bit_length() - 1,
                           UW=S.U * S.TW)


# --- the specification ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    parent: int                 # an entry, or ROOT = N
    tokens: tuple[int, ...]     # the unit's U token ids
    owner: int                  # the lowest-numbered request that has computed it


@dataclass(frozen=True)
class Row:
    """One call's inputs beside the table: the walk's start, the row's request and bounds, its window of units."""
    r: int
    units: tuple[tuple[int, ...], ...]
    admit: bool = False
    sent: bool = False
    K: int = 0
    wh: int = 0
    nins: int = 0
    pfx: int = 0


@dataclass(frozen=True)
class Out:
    ok: bool
    table: tuple[Entry, ...] = ()
    pfx: int = 0
    trail: tuple[int, ...] = ()


def reference(S, table: Sequence[Entry], row: Row) -> Out:
    """One row's pass over the cache (the module's docstring); `Out(False)` refuses."""
    Z = widths(S)
    if len(table) > S.N or len(row.units) > Z.KU:
        raise ValueError("the table or the window exceeds the size class")
    ents = list(table)
    units = list(row.units) + [(0,) * S.U] * (Z.KU - len(row.units))
    pfx = S.N if row.admit else row.pfx
    if row.wh > Z.KU or row.nins > Z.KU:
        return Out(False)
    index = {(x.parent, x.tokens): e for e, x in enumerate(ents)}
    trail = []
    for j, tok in enumerate(units):
        tok = tuple(tok)
        found = index.get((pfx, tok))
        look = row.admit and (j + 1) * S.U < row.K
        if j < row.wh:
            if not look or found is None:
                return Out(False)
            pfx = found
        else:
            if not row.sent and j == row.wh and look and found is not None:
                return Out(False)
            if j < row.nins:
                if found is not None:
                    if row.r < ents[found].owner:
                        ents[found] = replace(ents[found], owner=row.r)
                    pfx = found
                elif len(ents) == S.N:
                    return Out(False)
                else:
                    ents.append(Entry(pfx, tok, row.r))
                    index[(pfx, tok)] = len(ents) - 1
                    pfx = len(ents) - 1
        trail.append(pfx)
    return Out(True, tuple(ents), pfx, tuple(trail))


def sources(S, table: Sequence[Entry], trail: Sequence[int], wh: int) -> tuple[tuple[int, int], ...]:
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


def _pad(x, w):
    return (list(x) + [ZERO] * w)[:w]


def _eq(C, a, b):
    w = max(len(a), len(b))
    return and_reduce(C, [(x ^ y).inv() for x, y in zip(_pad(a, w), _pad(b, w))])


def _lt(C, a, b):
    w = max(len(a), len(b))
    return sub(C, _pad(a, w), _pad(b, w), w)[1]


def _sel(C, s, x):
    return [C.AND(s, b) for b in x]


def _carry_layout(S) -> list[tuple[str, int]]:
    """The scan's carry: the table, ok, the walk's entry and the unit's index."""
    Z = widths(S)
    N = S.N
    return [("valid", N), ("parent", N * Z.EW), ("tok", N * Z.UW), ("owner", N * S.RW), ("ok", 1), ("pfx", Z.EW),
            ("j", Z.HW + 1)]


def _size(layout) -> int:
    return sum(w for _, w in layout)


def _unpack(layout, bits) -> dict:
    out, at = {}, 0
    for name, w in layout:
        out[name] = list(bits[at:at + w])
        at += w
    return out


def _sig_unit(S):
    Z = widths(S)
    carry = Array(_size(_carry_layout(S)), BIT)
    return (("carry", carry), ("unit", Array(Z.UW, BIT)), ("r", Array(S.RW, BIT)), ("flags", Array(2, BIT)),
            ("K", Array(Z.LW, BIT)), ("wh", Array(Z.HW, BIT)), ("nins", Array(Z.HW, BIT))), \
        Tuple(carry, Array(Z.EW, BIT))


def _build_unit(C, S, carry, unit, r, flags, K, wh, nins):
    """Unit j of the window (j the carry's index): a hit unit, the unit after a derived hit, or a computed one."""
    Z = widths(S)
    N, EW, UW, RW, HW = S.N, Z.EW, Z.UW, S.RW, Z.HW
    st = _unpack(_carry_layout(S), carry)
    admit, sent = flags[0], flags[1]
    j = st["j"]
    first = _eq(C, j, [ZERO] * (HW + 1))
    pfx = mux(C, C.AND(first, admit), const_bits(N, EW), st["pfx"])
    valid = st["valid"]
    parent = [st["parent"][e * EW:(e + 1) * EW] for e in range(N)]
    tok = [st["tok"][e * UW:(e + 1) * UW] for e in range(N)]
    owner = [st["owner"][e * RW:(e + 1) * RW] for e in range(N)]
    hits = [C.AND(valid[e], and_reduce(C, [(x ^ y).inv() for x, y in zip(parent[e] + tok[e], pfx + list(unit))]))
            for e in range(N)]
    found = or_reduce(C, hits)
    found_id = [ZERO] * EW
    for e in range(N):
        found_id = [x ^ (hits[e] if e >> b & 1 else ZERO) for b, x in enumerate(found_id)]
    j1 = add(C, j, [ONE], width=HW + 1)
    look = C.AND(admit, _lt(C, [ZERO] * Z.LU + j1, K))
    hitu = _lt(C, j, wh)
    at_wh = _eq(C, j, wh)
    ins = C.AND(_lt(C, j, nins), hitu.inv())
    bad = [C.AND(hitu, C.AND(look, found).inv()),
           and_reduce(C, [sent.inv(), at_wh, look, found])]
    new = C.AND(ins, found.inv())
    slot = [C.AND(valid[e - 1] if e else ONE, valid[e].inv()) for e in range(N)]
    bad.append(C.AND(new, valid[N - 1]))
    new_id = [ZERO] * EW
    for e in range(N):
        new_id = [x ^ (slot[e] if e >> b & 1 else ZERO) for b, x in enumerate(new_id)]
    upd = C.AND(ins, found)
    out = {"valid": [], "parent": [], "tok": [], "owner": []}
    for e in range(N):
        wr = C.AND(new, slot[e])
        lower = C.AND(C.AND(upd, hits[e]), _lt(C, r, owner[e]))
        out["valid"].append(valid[e] ^ wr)
        out["parent"] += mux(C, wr, pfx, parent[e])
        out["tok"] += mux(C, wr, list(unit), tok[e])
        out["owner"] += mux(C, wr ^ lower, list(r), owner[e])
    nxt = mux(C, C.AND(hitu, found) ^ upd, found_id, mux(C, new, new_id, pfx))
    ok = C.AND(st["ok"][0], or_reduce(C, bad).inv())
    packed = out["valid"] + out["parent"] + out["tok"] + out["owner"] + [ok] + nxt + j1
    return [packed, nxt]


def _sig_end(S):
    Z = widths(S)
    return (("ok", BIT), ("wh", Array(Z.HW, BIT)), ("nins", Array(Z.HW, BIT))), BIT


def _build_end(C, S, ok, wh, nins):
    Z = widths(S)
    KU = const_bits(Z.KU, Z.HW + 1)
    return [and_reduce(C, [ok[0], _lt(C, KU, wh).inv(), _lt(C, KU, nins).inv()])]


CacheUnit = _part("CacheUnit", STATICS, _sig_unit, _build_unit,
                  "one unit of a row's window through the prefix cache's table: a hit unit (cached, within the lookup's "
                  "cap), the unit after a derived hit (not both), or a unit the row computes (owned, or appended)")
CacheStepEnd = _part("CacheStepEnd", STATICS, _sig_end, _build_end,
                     "the walk's ok, with the hit and the computed units within the window")
PARTS = (CacheUnit, CacheStepEnd)


def _in_fields(S) -> list[tuple[str, object]]:
    Z = widths(S)
    N = S.N
    return [("valid", Array(N, BIT)), ("parent", Array(N, Array(Z.EW, BIT))), ("tok", Array(N, Array(Z.UW, BIT))),
            ("owner", Array(N, Array(S.RW, BIT))), ("pfx", Array(Z.EW, BIT)), ("r", Array(S.RW, BIT)),
            ("admit", BIT), ("sent", BIT), ("K", Array(Z.LW, BIT)), ("wh", Array(Z.HW, BIT)),
            ("nins", Array(Z.HW, BIT)), ("units", Array(Z.KU, Array(Z.UW, BIT)))]


def _out_type(S) -> Tuple:
    Z = widths(S)
    N = S.N
    return Tuple(BIT, Array(N, BIT), Array(N, Array(Z.EW, BIT)), Array(N, Array(Z.UW, BIT)), Array(N, Array(S.RW, BIT)),
                 Array(Z.EW, BIT), Array(Z.KU, Array(Z.EW, BIT)))


def _sig(S):
    return tuple(_in_fields(S)), _out_type(S)


def _body(B, S, valid, parent, tok, owner, pfx, r, admit, sent, K, wh, nins, units):
    Z = widths(S)
    Kc = _Consts(B)
    st = {k: getattr(S, k) for k in STATICS}
    unit_p, end_p = bind(CacheUnit, **st), bind(CacheStepEnd, **st)
    lay = _carry_layout(S)
    init = _refs(valid) + _refs(parent) + _refs(tok) + _refs(owner) + Kc.bits(1, 1) + _refs(pfx) + Kc.bits(0, Z.HW + 1)
    carry_t = Array(_size(lay), BIT)
    flags = _coll(Array(2, BIT), _refs(admit) + _refs(sent))
    done = _refs(B.scan(unit_p, _coll(carry_t, init), [units], [r, flags, K, wh, nins]))
    fin = _unpack(lay, done[:_size(lay)])
    ok = _refs(B.call(end_p, _coll(BIT, fin["ok"]), wh, nins))
    out = ok + fin["valid"] + fin["parent"] + fin["tok"] + fin["owner"] + fin["pfx"] + done[_size(lay):]
    return _coll(_out_type(S), out)


CacheStep = CompositeDefinition(
    "CacheStep", 1, STATICS, _sig, _body, conformance="exact-model-tested",
    doc="one row's pass over the prefix cache's table (the derived encoding's rules): its hit checked against the "
        "cached prefix, derived while the pool bound holds, and the units it computes owned or appended "
        "(`cache_step.reference`)")


def unit(**st):
    return bind(CacheStep, **st)


def statics(N: int, unit_tokens: int, token_width: int, request_width: int, max_model_len: int) -> dict:
    return {"N": N, "U": unit_tokens, "TW": token_width, "RW": request_width, "L": max_model_len}


# --- arguments and evaluation ----------------------------------------------------------------------------------------------


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def _unit_bits(S, tok: Sequence[int]) -> list[int]:
    if len(tok) != S.U:
        raise ValueError(f"a unit of {len(tok)} tokens, not {S.U}")
    return [b for t in tok for b in _bits(t, S.TW)]


def arguments(S, table: Sequence[Entry], row: Row) -> list[list[int]]:
    """The unit's arguments (each a flat list of bits, as `evaluate_bits` takes one instance)."""
    Z = widths(S)
    if len(table) > S.N or len(row.units) > Z.KU:
        raise ValueError("the table or the window exceeds the size class")
    pad = [Entry(0, (0,) * S.U, 0)] * (S.N - len(table))
    ents = list(table) + pad
    units = list(row.units) + [(0,) * S.U] * (Z.KU - len(row.units))
    return [[1] * len(table) + [0] * len(pad), [b for x in ents for b in _bits(x.parent, Z.EW)],
            [b for x in ents for b in _unit_bits(S, x.tokens)], [b for x in ents for b in _bits(x.owner, S.RW)],
            _bits(row.pfx, Z.EW), _bits(row.r, S.RW), [int(row.admit)], [int(row.sent)], _bits(row.K, Z.LW),
            _bits(row.wh, Z.HW), _bits(row.nins, Z.HW), [b for u in units for b in _unit_bits(S, u)]]


def decode(S, flat: Sequence[int]) -> Out:
    """The unit's output bits as an `Out` (the table's valid entries)."""
    Z = widths(S)
    N, at = S.N, 0

    def take(n: int, w: int) -> list[int]:
        nonlocal at
        vals = [sum(int(flat[at + e * w + b]) << b for b in range(w)) for e in range(n)]
        at += n * w
        return vals

    ok, = take(1, 1)
    valid, parent = take(N, 1), take(N, Z.EW)
    toks = take(N * S.U, S.TW)
    owner, (pfx,), trail = take(N, S.RW), take(1, Z.EW), take(Z.KU, Z.EW)
    if not ok:
        return Out(False)
    table = tuple(Entry(parent[e], tuple(toks[e * S.U:(e + 1) * S.U]), owner[e]) for e in range(N) if valid[e])
    return Out(True, table, pfx, tuple(trail))


def evaluate(st_: dict, cases: Sequence[tuple[Sequence[Entry], Row]]) -> list[Out]:
    """The unit at statics `st_` on each (table, row), in one evaluation."""
    S = SimpleNamespace(**st_)
    args = [list(rows) for rows in zip(*(arguments(S, t, r) for t, r in cases))]
    out = evaluate_bits(unit(**st_), *args)
    return [decode(S, row) for row in out.reshape(len(cases), -1).tolist()]


# --- cost ------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def part_ands(key: tuple) -> dict[str, int]:
    st_ = dict(zip(STATICS, key))
    return {d.name: ands(bind(d, **st_)) for d in PARTS}


def size(**st_) -> dict[str, int]:
    """The unit's ANDs: per unit of the window, and per call (KU units and the range checks), from the parts' counts."""
    Z = widths(SimpleNamespace(**st_))
    p = part_ands(tuple(st_[k] for k in STATICS))
    return {"unit": p["CacheUnit"], "units": Z.KU, "end": p["CacheStepEnd"],
            "ands": Z.KU * p["CacheUnit"] + p["CacheStepEnd"]}


# --- the composition: an engine step's calls (the compile slice's) ---------------------------------------------------------


@dataclass(frozen=True)
class Cache:
    """The cache between engine steps: the table, and per running slot the walk's entry (its request's last complete
    unit, ROOT before the first; 0 for an empty slot)."""

    table: tuple[Entry, ...] = ()
    pfx: tuple[int, ...] = ()


def scan_widths(X) -> SimpleNamespace:
    """The calls' classes at an engine step, on the slice's statics X: ReplayStep_v1's R, W, A, the engine's B, T and
    IW, and CN entries, CTW a token id's width and CU the units of each arrival's tokens the slice holds. A lookup walks
    a request's first KL = min(CN + 1, CU) units from ROOT (no walk from ROOT matches more than CN, since every entry's
    parent is an earlier one); a compute walks the KC = ⌊(B − 1 + T) / B⌋ units a row of at most T tokens can complete."""
    KL, KC = min(X.CN + 1, X.CU), (X.B - 1 + X.T) // X.B
    look = SimpleNamespace(**statics(X.CN, X.B, X.CTW, X.IW, KL * X.B))
    comp = SimpleNamespace(**statics(X.CN, X.B, X.CTW, X.IW, KC * X.B))
    return SimpleNamespace(KL=KL, KC=KC, look=look, comp=comp, EW=_bitlen(X.CN), UW=X.B * X.CTW, LB=X.B.bit_length() - 1)


def cache_fields(X) -> list[tuple[str, int, int]]:
    """A cache row: the table (CN entries: valid, parent, the unit's tokens, owner) and the R running slots' entries."""
    Y = scan_widths(X)
    return [("valid", X.CN, 1), ("parent", X.CN, Y.EW), ("tok", X.CN, Y.UW), ("owner", X.CN, X.IW), ("pfx", X.R, Y.EW)]


def cache_width(X) -> int:
    return sum(n * w for _, n, w in cache_fields(X))


def encode_cache(X, c: Cache) -> list[int]:
    if len(c.table) > X.CN or len(c.pfx) > X.R:
        raise ValueError("the cache exceeds the size class")
    ents = list(c.table) + [Entry(0, (0,) * X.B, 0)] * (X.CN - len(c.table))
    vals = {"valid": [int(e < len(c.table)) for e in range(X.CN)], "parent": [x.parent for x in ents],
            "tok": [sum(t << (X.CTW * k) for k, t in enumerate(x.tokens)) for x in ents], "owner": [x.owner for x in ents],
            "pfx": list(c.pfx) + [0] * (X.R - len(c.pfx))}
    return [b for name, _n, w in cache_fields(X) for v in vals[name] for b in _bits(v, w)]


def decode_cache(X, bits: Sequence[int], running: int) -> Cache:
    """A cache row as `Cache`, its first `running` slots' entries; ValueError unless the valid entries are a prefix and
    every other entry and slot is zero."""
    at, vals = 0, {}
    for name, n, w in cache_fields(X):
        vals[name] = [sum(int(bits[at + e * w + b]) << b for b in range(w)) for e in range(n)]
        at += n * w
    m = sum(vals["valid"])
    if vals["valid"] != [1] * m + [0] * (X.CN - m) or any(vals[k][e] for k in ("parent", "tok", "owner")
                                                        for e in range(m, X.CN)) or any(vals["pfx"][running:]):
        raise ValueError("a cache row whose valid entries aren't a prefix, or with an unused entry or slot not zero")
    mask = (1 << X.CTW) - 1
    table = tuple(Entry(vals["parent"][e], tuple((vals["tok"][e] >> (X.CTW * k)) & mask for k in range(X.B)),
                        vals["owner"][e]) for e in range(m))
    return Cache(table, tuple(vals["pfx"][:running]))


def units_of(X, tokens: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """A request's first CU units, zero past its tokens."""
    t = (list(tokens) + [0] * (X.CU * X.B))[:X.CU * X.B]
    return tuple(tuple(t[k * X.B:(k + 1) * X.B]) for k in range(X.CU))


def candidates(X, st, inp) -> list[tuple[int, int] | None]:
    """ReplayStep_v1's candidates (W waiting slots, then A arrivals; when the engine is quiet the first arrival is
    waiting slot 0) as (request, known tokens P + out), None for an empty one."""
    arr = [(a.id, a.P) for a in inp.arrivals]
    wait = [(r.id, r.P + r.out) for r in st.waiting]
    if not st.running and not st.waiting and arr:
        wait, arr = arr[:1], arr[1:]
    return (wait + [None] * X.W)[:X.W] + (arr + [None] * X.A)[:X.A]


def longest(S, table: Sequence[Entry], units: Sequence[Sequence[int]], K: int) -> tuple[int, int]:
    """The longest cached prefix a lookup may take (each unit (j + 1)·U < K), in units, and the walk's entry after it."""
    index = {(x.parent, x.tokens): e for e, x in enumerate(table)}
    pfx, j = S.N, 0
    while j < len(units) and (j + 1) * S.U < K and (pfx, tuple(units[j])) in index:
        pfx, j = index[(pfx, tuple(units[j]))], j + 1
    return j, pfx


def scan_step(X, cache: Cache, st, inp, out, tokens: dict[int, Sequence[int]], *, derive: bool = False):
    """The cache through one engine step: ReplayStep_v1's state `st`, inputs `inp` and output `out` (its rows at their
    slots), the requests' certified `tokens`. Slot by slot, an admitted row (at a slot no request ran in) first looks
    its candidate's hit up from ROOT on its first KL units at the table so far, and every row computes the units it
    completes from its walk's entry; then the first candidate left unadmitted looks its hit up at the step's last table.
    The running slots' next entries follow ReplayStep_v1's next state by request.

    With `derive`, each lookup finds the `longest` prefix (the hits a prover runs, `hits`) and the row computes from
    the walk's entry after the row's own hit, where the two agree; else it checks `inp.hit` by `reference`, and None
    refuses: a lookup or a compute that `reference` refuses, a hit not inside the lookup's
    window, a row whose request isn't among the candidates or `tokens`, or a row reaching past CU units."""
    Y = scan_widths(X)
    table, pfx = tuple(cache.table), list(cache.pfx) + [0] * X.R
    cands = candidates(X, st, inp)
    rows = {r[0]: r[1:] for r in out.rows}
    hits, admitted, after = [0] * (X.W + X.A), set(), {}

    def lookup(c: int, units) -> int | None:
        rid, K = cands[c]
        h = inp.hit[c] if c < len(inp.hit) else 0
        if derive:
            hits[c] = longest(Y.look, table, units[:Y.KL], K)[0]
            return longest(Y.look, table, units[:min(h, hits[c])], K)[1] if hits[c] < Y.KL else None
        got = reference(Y.look, table, Row(rid, units[:Y.KL], admit=True, K=K, wh=h)) if h < Y.KL else Out(False)
        return got.pfx if got.ok else None

    for p in range(X.R):
        if p not in rows:
            if p < len(st.running):
                after[st.running[p].id] = pfx[p]
            continue
        rid, _kind, start, n = rows[p]
        if rid not in tokens:
            return None
        units = units_of(X, tokens[rid])
        w = pfx[p]
        if p >= len(st.running):
            c = next((c for c, x in enumerate(cands) if x is not None and x[0] == rid), None)
            w = None if c is None else lookup(c, units)
            if w is None:
                return None
            admitted.add(rid)
        u0, end = start >> Y.LB, (start + n) >> Y.LB
        if end > X.CU:
            return None
        got = reference(Y.comp, table, Row(rid, (units + ((0,) * X.B,) * Y.KC)[u0:u0 + Y.KC], pfx=w, nins=end - u0))
        if not got.ok:
            return None
        table, after[rid] = got.table, got.pfx
    rest = [c for c, x in enumerate(cands) if x is not None and x[0] not in admitted]
    if rest and (cands[rest[0]][0] not in tokens or lookup(rest[0], units_of(X, tokens[cands[rest[0]][0]])) is None):
        return None
    if any(r.id not in after for r in out.state.running):
        return None
    return SimpleNamespace(cache=Cache(table, tuple(after[r.id] for r in out.state.running)), hits=tuple(hits))


def _first(C, m):
    """The first set bit of `m`, one-hot."""
    out, seen = [], ZERO
    for x in m:
        out.append(C.AND(x, seen.inv()))
        seen = x ^ seen ^ C.AND(x, seen)
    return out


def _pick(C, sel, items):
    """The item a one-hot `sel` picks (zero for none): an XOR of the masked items, free beyond the masks."""
    acc = [ZERO] * len(items[0])
    for s, x in zip(sel, items):
        acc = [a ^ b for a, b in zip(acc, _sel(C, s, x))]
    return acc


def _clamp(C, x, w):
    """x saturated to w bits (all ones past them)."""
    hi = or_reduce(C, x[w:])
    return [b ^ hi ^ C.AND(b, hi) for b in _pad(x, w)]


def _call(C, S, table, pfx, r, flags, K, wh, nins, units):
    """One CacheStep_v1 call traced inline, its units unrolled (their index a constant): (ok, table', pfx')."""
    Z = widths(S)
    carry = list(table) + [ONE] + list(pfx) + const_bits(0, Z.HW + 1)
    for u in units:
        carry = _build_unit(C, S, carry, u, r, flags, K, wh, nins)[0]
    fin = _unpack(_carry_layout(S), carry)
    ok = _build_end(C, S, fin["ok"], wh, nins)[0]
    return ok, fin["valid"] + fin["parent"] + fin["tok"] + fin["owner"], fin["pfx"]


def build_scan_step(C, X, cache, st, fa, forced, out, hit, cert):
    """`scan_step` as a circuit: (ok, the next cache row). `cache` is the cache row's bits; `st`, `fa` and `out` are
    ReplayStep_v1's state, arrival and output fields (each a list of elements); `forced` its quiet step's arrival bit;
    `hit` the W + A hits it reads; `cert` the certified arrivals' (valid, id) and tokens: {"valid", "id", "tok"}, tok[k]
    the CU units of slot k. The table moves through the calls; the lookups (nins = 0) write nothing."""
    Y = scan_widths(X)
    Zl, Zc = widths(Y.look), widths(Y.comp)
    LW = len(fa["arr_P"][0])
    cur = {}
    at = 0
    for name, n, w in cache_fields(X):
        cur[name] = [cache[at + e * w:at + (e + 1) * w] for e in range(n)]
        at += n * w
    table = [x for name in ("valid", "parent", "tok", "owner") for e in cur[name] for x in e]

    def arrival(a):
        if a >= X.A:
            return ZERO, [ZERO] * X.IW, [ZERO] * (LW + 1)
        return fa["arr_valid"][a][0], fa["arr_id"][a], _pad(fa["arr_P"][a], LW + 1)

    cands = []
    for c in range(X.W):
        v, i, k = st["wait_valid"][c][0], st["wait_id"][c], add(C, st["wait_P"][c], st["wait_out"][c], width=LW + 1)
        if c == 0:
            av, ai, ak = arrival(0)
            v, i, k = v ^ forced, mux(C, forced, ai, i), mux(C, forced, ak, k)
        cands.append((v, i, k))
    for a in range(X.A):
        (v0, i0, k0), (v1, i1, k1) = arrival(a), arrival(a + 1)
        cands.append((mux(C, forced, [v1], [v0])[0], mux(C, forced, i1, i0), mux(C, forced, k1, k0)))

    def tokens_of(rid):
        m = [C.AND(cert["valid"][k], _eq(C, cert["id"][k], rid)) for k in range(len(cert["tok"]))]
        sel = _first(C, m)
        return or_reduce(C, m), [_pick(C, sel, [t[u] for t in cert["tok"]]) for u in range(X.CU)]

    def look(sel, rid, units):
        h, K = _pick(C, sel, hit), _pick(C, sel, [k for _v, _i, k in cands])
        wh = h[:Zl.HW]
        inside = C.AND(or_reduce(C, h[Zl.HW:]).inv(), _lt(C, wh, const_bits(Y.KL, Zl.HW + 1)))
        ok, _t, w = _call(C, Y.look, table, [ZERO] * Y.EW, rid, [ONE, ZERO], _clamp(C, K, Zl.LW), wh, [ZERO] * Zl.HW,
                          units[:Y.KL])
        return C.AND(inside, ok), w

    oks, adm, after = [], [], []
    for p in range(X.R):
        rv, rid, run_v = out["row_valid"][p][0], out["row_id"][p], st["run_valid"][p][0]
        start, n = out["row_start"][p], out["row_n"][p]
        a = C.AND(rv, run_v.inv())
        m = [C.AND(v, _eq(C, i, rid)) for v, i, _k in cands]
        found, units = tokens_of(rid)
        l_ok, l_pfx = look(_first(C, m), rid, units)
        oks.append(C.AND(a, C.AND(or_reduce(C, m), l_ok).inv()).inv())
        u0 = start[Y.LB:]
        end = add(C, start, n, width=LW + 1)[Y.LB:]
        d = sub(C, end, u0, len(end))[0]
        nins = [C.AND(rv, b) for b in d[:Zc.HW]]
        fits = and_reduce(C, [found, or_reduce(C, d[Zc.HW:]).inv(), _lt(C, const_bits(X.CU, len(end)), end).inv()])
        oks.append(C.AND(rv, fits.inv()).inv())
        w = mux(C, a, l_pfx, cur["pfx"][p])
        c_units = _front(C, units, u0[:_bitlen(X.CU)], Y.KC)
        c_ok, table, c_pfx = _call(C, Y.comp, table, w, rid, [ZERO, ZERO], [ZERO] * Zc.LW, [ZERO] * Zc.HW, nins, c_units)
        oks.append(c_ok)
        adm.append((a, rid))
        after.append((rv ^ run_v ^ C.AND(rv, run_v), mux(C, rv, rid, st["run_id"][p]), mux(C, rv, c_pfx, cur["pfx"][p])))

    left = [C.AND(v, or_reduce(C, [C.AND(x, _eq(C, i, rid)) for x, rid in adm]).inv()) for v, i, _k in cands]
    sel = _first(C, left)
    rid = _pick(C, sel, [i for _v, i, _k in cands])
    found, units = tokens_of(rid)
    l_ok, _w = look(sel, rid, units)
    oks.append(C.AND(or_reduce(C, left), C.AND(found, l_ok).inv()).inv())

    nxt = []
    for q in range(X.R):
        qv, qid = out["run_valid"][q][0], out["run_id"][q]
        m = [C.AND(v, _eq(C, i, qid)) for v, i, _w in after]
        oks.append(C.AND(qv, or_reduce(C, m).inv()).inv())
        nxt += _sel(C, qv, _pick(C, _first(C, m), [w for _v, _i, w in after]))
    return and_reduce(C, oks), table + nxt


def _front(C, xs, amt, n):
    """xs[amt:amt + n], zero entries past the end (`replay_chain._front`'s shifter; importing that module would make
    this one read the sampled-proofs package)."""
    zero = [ZERO] * len(xs[0])
    cur = [list(e) for e in xs]
    for j in reversed(range(len(amt))):
        d = 1 << j
        cur = [mux(C, amt[j], cur[k + d] if k + d < len(cur) else zero, cur[k]) for k in range(min(n + d - 1, len(cur)))]
    return (cur + [zero] * n)[:n]


__all__ = ["Cache", "CacheStep", "CacheStepEnd", "CacheUnit", "Entry", "Out", "PARTS", "Row", "STATICS", "arguments",
           "build_scan_step", "cache_fields", "cache_width", "candidates", "decode", "decode_cache", "encode_cache",
           "evaluate", "longest", "reference", "scan_step", "scan_widths", "size", "sources", "statics", "unit",
           "units_of", "widths"]
