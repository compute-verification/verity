"""The census of a Program (`verity/census` v1): what its gates are, by class, per Definition, per step and in total,
counted from the IR alone (Definitions and primitive ids, never kernels).

A gate's class is its primitive's identity, from the pinned table `KINDS` (and the gather family `GATHER_ID`):

- **matmul steps** (`Mma`): an FP tensor-core step (``fp``), or an exact s8 IMMA step (``s8-imma``, `BlackwellS8ImmaDot32_v1`),
  each with its k products;
- **roundings**: narrowing casts (``cast``), special functions and table reads (``special``), FP32 adds, the add of an FMA and
  every other FP32 rounding (``arith``), integer and bit operations (``integer``);
- **comparisons and selections** (``compare_select``), among them |x| (`F32Fabs_v1`, a comparison with 0 and a select);
- **gathers**: a data-dependent read of one of V words, with its fan-in V;
- **products**: scalar products, by the provenance of their two operands (below), and the k products of each matmul step.

A primitive the table doesn't list is **unclassified**: the census counts it by id and the property (`verity.ml.circuits.stacking`)
refuses it, so a gate nobody classified is never free. Exact value embeddings and negation (`Bf16ToF32_v1`, `E4m3ToF32_v1`,
`F32Neg_v1`) are free, and so is every primitive without parameters (constants and inputs: structure). This differs from
`cut.WIRING`, which also frees `F32Fabs_v1` and `F32BitsShl23_v1` as bit rearrangements: on values they are |x| and
x mod 2^9 (scaled), nonlinear, so the census charges them as a comparison and an integer operation.

Provenance. Every collection is marked constant, registered (a registered input: fixed before any seed), selected (a
registered word read through a gather whose index is a run value: an embedding row, an MoE expert) or run (anything that
depends on a run input). A primitive's output is the largest of its operands', a selected operand counting as run (a selected
word stays selected only through a gather), and a `scan`'s carry is the largest of all its arguments'. A scalar class is
counted only for gates with an operand that is selected or run; a matmul step's products are counted when an operand
(``a`` or ``b``) is, by `product_class`; every matmul step is counted in ``mma``, whatever its operands. These are the rules of
the stacking campaign's counter (``benchmarks/stacking_budgets/count_budgets.py``), so the classes agree with its numbers on the same
Programs, apart from the charges above.

Exactness. Each s8 IMMA step computes ``D = C + sum_k a_k * b_k`` and clamps D to s32 (``.satfinite``). An interval pass over the
Program bounds every step's D from its operands' ranges: a constant is its value, an IMMA output its D, and every other value
any word of its width (an s8 operand -128..127, an accumulator all of s32). A step is proven exact when D's bound lies in
s32; every other step is ``unproven`` and listed in ``may_overflow`` by its Definition and node.

Steps. A Program's Calls (its root's non-input nodes) are split into steps at the Call indices `census(program, steps)`
names (one step if none). Per step the census gives the classes of its Calls and its memory: the input bits it reads first
(run and registered), the dynamic bits it writes that a later step reads, by the distance in steps to their last read
(``lifetimes``), and the output bits it writes.

The table is fixed and versioned: changing an entry is a new `VERSION`, and so a new `digest()`.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .codec import canonical_json
from .defs import Body, PrimitiveDefinition
from .intervals import Intervals, norm, ref_intervals, restrict, total
from .program import RUN as RUN_BINDING
from .program import Program, input_id, is_input
from .refs import Affine, Concat, Explicit, Refs, Strided
from .types import Array, Record, Type

TAG = "verity/census"
VERSION = 1
SCHEMA = f"{TAG}/v{VERSION}"

CONSTANT, REGISTERED, SELECTED, RUN = 0, 1, 2, 3

S8 = (-(1 << 7), (1 << 7) - 1)
S32 = (-(1 << 31), (1 << 31) - 1)

#: a constant primitive's id (`PROTOCOL.md` §9)
CONST_ID = re.compile(r"^Const([1-9][0-9]*)\[0x([0-9a-f]+)\]_v1$")
#: the gathers: (index: i32, column: Array<V, word>) -> column[index]
GATHER_ID = re.compile(r"^Gather(?:Bf16|F32)x([1-9][0-9]*)_v1$")

ROUNDINGS = ("cast", "special", "arith", "integer")
PRODUCTS = ("rr", "rsel", "rs")


@dataclass(frozen=True)
class Mma:
    """A matmul step: ``kind`` ``fp`` or ``s8-imma``, k products a step, parameters (acc, a, b, ...). An s8 IMMA step's
    operands lie in ``operand`` and its result is clamped to ``acc``."""

    kind: str
    k: int
    operand: tuple[int, int] | None = None
    acc: tuple[int, int] | None = None


@dataclass(frozen=True)
class Kind:
    """What one gate of a primitive is: its roundings by class, comparisons and selections, scalar products (pairs of
    parameter indices), a division's rounding (``divisor``: special when that parameter is selected or run, else arith), a
    gather's fan-in, or a matmul step."""

    cast: int = 0
    special: int = 0
    arith: int = 0
    integer: int = 0
    compare: int = 0
    products: tuple[tuple[int, int], ...] = ()
    divisor: int | None = None
    gather: int = 0
    mma: Mma | None = None


_FREE = Kind()
_CAST, _SPECIAL, _ARITH, _INTEGER, _COMPARE = Kind(cast=1), Kind(special=1), Kind(arith=1), Kind(integer=1), Kind(compare=1)
_PRODUCT = Kind(arith=1, products=((0, 1),))
_ACT_MUL = Kind(special=1, cast=2, products=((0, 1),))


def _v(names: str, *versions: int) -> list[str]:
    return [f"{n}_v{v}" for n in names.split() for v in versions]


KINDS: dict[str, Kind] = {
    **dict.fromkeys(_v("Bf16ToF32 E4m3ToF32 F32Neg", 1), _FREE),
    **dict.fromkeys(_v("F2fpBf16 F32ToBf16Rn F32ToE4m3Sat F32ToE2m1Sat Bf16Add Bf16AddF2fp", 1), _CAST),
    **dict.fromkeys(_v("MufuEx2Ftz MufuRcpFtz MufuSqrtFtz MufuTanh RsqrtApprox DivFullRcp Fa2InvSum TanhF32Rn", 1), _SPECIAL),
    **dict.fromkeys(_v("F32Max F32GtStrict F32Fmaxf F32Fminf Bf16GtStrict F32Eq I32Le I32Eq F32IsFinite GuardNegInfZero F32Sat "
                       "SelectBf16 SelectF32 SelectI32 F32Fabs", 1), _COMPARE),
    **dict.fromkeys(_v("DivFullScaleA", 1, 2), _COMPARE),
    **dict.fromkeys(_v("I32Add BitAnd BitNot BitOr F32BitsShl23", 1), _INTEGER),
    **dict.fromkeys(_v("F32Add F32AddFtz F32Sub F32SubFtz", 1, 2), _ARITH),
    **dict.fromkeys(_v("F32Mul F32MulFtz F32Fma F32FmaFtz F32FmaSubFtz F32FmaRm", 1, 2), _PRODUCT),
    **dict.fromkeys(_v("F32Div", 1, 2), Kind(products=((0, 1),), divisor=1)),
    **dict.fromkeys(_v("Bf16MulBf16 Bf16MulF32", 1), Kind(cast=1, products=((0, 1),))),
    **dict.fromkeys(_v("SiluMulBf16", 1, 2) + _v("GeluTanhMulBf16", 1), _ACT_MUL),
    "RopeOut_v1": Kind(arith=1, cast=1, products=((0, 2), (1, 3))),
    "RopeOutAdd_v1": Kind(arith=1, cast=1, products=((1, 2), (0, 3))),
    **dict.fromkeys(_v("AmpereBF16TcDot16", 2) + _v("HopperBF16WgmmaDot16", 1), Kind(mma=Mma("fp", 16))),
    **dict.fromkeys(_v("HopperE4m3QgmmaDot32 HopperE4m3MmaV2Dot32 BlackwellE4m3QmmaDot32 BlackwellE5m2QmmaDot32", 1),
                    Kind(mma=Mma("fp", 32))),
    **dict.fromkeys(_v("BlackwellNvf4OmmaDot64 BlackwellMxf4OmmaDot64", 1), Kind(mma=Mma("fp", 64))),
    "BlackwellS8ImmaDot32_v1": Kind(mma=Mma("s8-imma", 32, S8, S32)),
}


def kind(fid: str) -> Kind | None:
    """The class of a primitive's gate by its id, or None when it is unclassified."""
    k = KINDS.get(fid)
    if k is not None:
        return k
    m = GATHER_ID.fullmatch(fid)
    return Kind(gather=int(m[1])) if m else None


def product_class(a: int, b: int) -> str:
    """``rr`` (two run or selected values, but not one of each), ``rsel`` (a run value and a selected one) or ``rs`` (a run
    or selected value and a constant or registered one)."""
    if min(a, b) >= SELECTED:
        return "rsel" if {a, b} == {RUN, SELECTED} else "rr"
    return "rs"


def spec() -> dict:
    """The table as canonical data: what `digest` hashes."""
    return {"format": TAG, "version": VERSION, "gather": GATHER_ID.pattern,
            "kinds": {fid: {k: v for k, v in asdict(kd).items() if v not in (0, (), None)} for fid, kd in sorted(KINDS.items())}}


def digest() -> str:
    """``SHA-512(TAG || "/v" || version || 0 || canon(spec))``: the table version a census names and a statement can pin."""
    return hashlib.sha512(f"{TAG}/v{VERSION}".encode() + b"\0" + canonical_json(_plain(spec()))).hexdigest()


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


# ---- provenance and the classes ----------------------------------------------------------------------------------------

Raw = dict  # {(primitive id, operand provenance): gates}


def _colls(refs: Refs) -> set[tuple[str, int]]:
    if isinstance(refs, (Affine, Strided)):
        return {(refs.space, refs.idx)} if refs.count else set()
    if isinstance(refs, Concat):
        return set().union(*(_colls(p) for p in refs.parts))
    if isinstance(refs, Explicit):
        return {(s, i) for s, i, _leaf in refs.items}
    return {(s, i) for s, i, _leaf in refs}


class _Provenance:
    def __init__(self) -> None:
        self.memo: dict[tuple[int, tuple[int, ...]], tuple[Raw, int]] = {}
        self.keep: list[Any] = []

    def prim(self, fn: PrimitiveDefinition, ts: tuple[int, ...]) -> tuple[Raw, int]:
        if GATHER_ID.fullmatch(fn.id) and ts and ts[0] == RUN and max(ts[1:], default=CONSTANT) <= REGISTERED:
            return {(fn.id, ts): 1}, SELECTED
        t = max(ts, default=CONSTANT)
        return {(fn.id, ts): 1}, RUN if t >= SELECTED else t

    def call(self, fn: Any, ts: tuple[int, ...]) -> tuple[Raw, int]:
        if isinstance(fn, PrimitiveDefinition):
            return self.prim(fn, ts)
        key = (id(fn), ts)
        hit = self.memo.get(key)
        if hit is None:
            self.keep.append(fn)
            out: Counter = Counter()
            node_t: list[int] = []
            for _k, nd, ats, sub, ot in self.walk(fn.body, ts, node_t):
                m = nd.n if nd.form in ("batch", "scan") else 1
                for k, v in sub.items():
                    out[k] += m * v
            hit = self.memo[key] = (dict(out), _taint(fn.body.ret, ts, node_t))
        return hit

    def walk(self, body: Body, ts: tuple[int, ...], node_t: list[int]):
        """Each node of `body` with its arguments' provenance, its one call's census and its output's provenance."""
        for k, nd in enumerate(body.nodes):
            ats = [_taint(a.refs, ts, node_t) for a in nd.args]
            if nd.form == "scan":
                ats[0] = max(ats)
            sub, ot = self.call(nd.fn, tuple(ats))
            ot = max(ot, ats[0]) if nd.form == "scan" else ot
            node_t.append(ot)
            yield k, nd, tuple(ats), sub, ot


def _taint(refs: Refs, ts: Sequence[int], node_t: Sequence[int]) -> int:
    t = CONSTANT
    for s, i in _colls(refs):
        t = max(t, ts[i] if s == "p" else node_t[i])
    return t


def _blank() -> dict:
    return {"macs": {"exact_s8": 0, "other": 0}, "mma": {}, "roundings": dict.fromkeys(ROUNDINGS, 0), "compare_select": 0,
            "gathers": 0, "gather_fanin": 0, "products": dict.fromkeys(PRODUCTS, 0), "mma_acc": dict.fromkeys(PRODUCTS, 0),
            "unclassified": {}, "primitives": {}}


def classify(raw: Raw, unproven: dict[str, int] | None = None) -> dict:
    """The classes (the census's ``counts`` form) of a primitive census ``{(id, operand provenance): gates}``, with the s8 IMMA
    steps the interval pass did not prove exact (``{id: steps}``)."""
    c = _blank()
    r, p, acc = c["roundings"], c["products"], c["mma_acc"]
    prims: Counter = Counter()
    uncl: Counter = Counter()
    for (fid, ts), n in raw.items():
        if not ts:
            continue
        runtime = max(ts) >= SELECTED
        if runtime:
            prims[fid] += n
        kd = kind(fid)
        if kd is None:
            uncl[fid] += n
            continue
        if kd.mma is not None:
            m = c["mma"].setdefault(fid, {"kind": kd.mma.kind, "k": kd.mma.k, "steps": 0, "exact": 0})
            m["steps"] += n
            if max(ts[1], ts[2]) >= SELECTED:
                cls = product_class(ts[1], ts[2])
                p[cls] += kd.mma.k * n
                acc[cls] += n
            continue
        if not runtime:
            continue
        for name in ROUNDINGS:
            r[name] += getattr(kd, name) * n
        c["compare_select"] += kd.compare * n
        for i, j in kd.products:
            p[product_class(ts[i], ts[j])] += n
        if kd.divisor is not None:
            r["special" if ts[kd.divisor] >= SELECTED else "arith"] += n
        if kd.gather:
            c["gathers"] += n
            c["gather_fanin"] += kd.gather * n
    for fid, m in c["mma"].items():
        if m["kind"] == "s8-imma":
            m["exact"] = m["steps"] - min(m["steps"], (unproven or {}).get(fid, 0))
        c["macs"]["exact_s8"] += m["k"] * m["exact"]
        c["macs"]["other"] += m["k"] * (m["steps"] - m["exact"])
    c["mma"] = dict(sorted(c["mma"].items()))
    c["unclassified"] = dict(sorted(uncl.items()))
    c["primitives"] = dict(sorted(prims.items()))
    return c


def _add_raw(acc: Counter, raw: Raw, m: int = 1) -> None:
    for k, v in raw.items():
        acc[k] += m * v


# ---- the interval pass: s8 IMMA exactness ------------------------------------------------------------------------------

Range = tuple[int, int] | None  # a value's signed range at its width; None: any word of its width
Segs = tuple[tuple[int, int, Range], ...]  # a collection's leaves [lo, hi) and their range, covering it
_EMPTY = (1, 0)  # the union's identity: an empty range
_EXACT_ITERATIONS = 1 << 16  # a scan's carry is iterated exactly this many times, then widened to any word


def _union(a: Range, b: Range) -> Range:
    if a is None or b is None:
        return None
    if a[0] > a[1]:
        return b
    if b[0] > b[1]:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def _signed(w: int, v: int) -> int:
    return v - (1 << w) if v >> (w - 1) else v


class _Intervals:
    def __init__(self) -> None:
        self.memo: dict[tuple[int, tuple[Range, ...]], tuple[Segs, Counter]] = {}
        self.has: dict[int, bool] = {}
        self.keep: list[Any] = []
        self.sites: dict[tuple[str, int], tuple[Range, Range]] = {}

    def has_imma(self, fn: Any) -> bool:
        if isinstance(fn, PrimitiveDefinition):
            k = kind(fn.id)
            return k is not None and k.mma is not None and k.mma.kind == "s8-imma"
        hit = self.has.get(id(fn))
        if hit is None:
            self.keep.append(fn)
            self.has[id(fn)] = False  # (a Definition never calls itself)
            hit = self.has[id(fn)] = any(self.has_imma(nd.fn) for nd in fn.body.nodes)
        return hit

    def prim(self, fn: PrimitiveDefinition, args: Sequence[Range], where: tuple[str, int]) -> tuple[Range, int]:
        """(output range, 1 if it is an s8 IMMA step not proven exact else 0)."""
        m = CONST_ID.fullmatch(fn.id)
        if m:
            v = _signed(int(m[1]), int(m[2], 16))
            return (v, v), 0
        kd = kind(fn.id)
        if kd is None or kd.mma is None or kd.mma.kind != "s8-imma":
            return None, 0
        mm = kd.mma
        c = _clip(args[0], mm.acc)
        a, b = _clip(args[1], mm.operand), _clip(args[2], mm.operand)
        corners = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
        d = (c[0] + mm.k * min(corners), c[1] + mm.k * max(corners))
        if mm.acc[0] <= d[0] and d[1] <= mm.acc[1]:
            return d, 0
        prev = self.sites.get(where)
        self.sites[where] = (c, d) if prev is None else (_union(prev[0], c), _union(prev[1], d))
        return (max(d[0], mm.acc[0]), min(d[1], mm.acc[1])), 1

    def call(self, fn: Any, args: tuple[Range, ...], where: tuple[str, int]) -> tuple[Segs, Counter]:
        if isinstance(fn, PrimitiveDefinition):
            r, u = self.prim(fn, args, where)
            return ((0, 1, r),), Counter({fn.id: u}) if u else Counter()
        key = (id(fn), args)
        hit = self.memo.get(key)
        if hit is None:
            self.keep.append(fn)
            segs: list[Segs] = []
            out: Counter = Counter()
            for _k, _nd, s, u in self.walk(fn.body, args, fn.id, segs):
                out.update(u)
            hit = self.memo[key] = (_read_segs(fn.body.ret, args, segs), out)
        return hit

    def walk(self, body: Body, params: Sequence[Range], def_id: str, segs: list[Segs]):
        """Each node of `body` with its output's segments and the IMMA steps of its gates not proven exact."""
        for k, nd in enumerate(body.nodes):
            ars = [_read(a.refs, params, segs) for a in nd.args]
            if nd.form in ("primitive", "call"):
                s, u = self.call(nd.fn, tuple(ars), (def_id, k))
            elif nd.form == "batch":
                s1, u1 = self.call(nd.fn, tuple(ars), (def_id, k))
                s = ((0, nd.out.leaves, _seg_union(s1)),)
                u = Counter({i: nd.n * v for i, v in u1.items()})
            else:
                s, u = self.scan(nd, ars, def_id, k)
            segs.append(s)
            yield k, nd, s, u

    def scan(self, nd: Any, ars: list[Range], def_id: str, k: int) -> tuple[Segs, Counter]:
        carry_t = nd.out.fields[0]
        cl = carry_t.leaves
        carry, rest = ars[0], tuple(ars[1:])
        ys: Range = _EMPTY
        u: Counter = Counter()
        for i in range(nd.n):
            if i == _EXACT_ITERATIONS:
                carry = None
            s, ui = self.call(nd.fn, (carry,) + rest, (def_id, k))
            nxt = _union(carry, _seg_union(s, 0, cl))
            ys = _union(ys, _seg_union(s, cl, None))
            if nxt == carry:
                u.update({j: (nd.n - i) * v for j, v in ui.items()})
                break
            u.update(ui)
            carry = nxt
        return ((0, cl, carry), (cl, nd.out.leaves, ys)), u


def _clip(r: Range, bound: tuple[int, int]) -> tuple[int, int]:
    if r is None or r[0] > r[1]:
        return bound
    return (max(r[0], bound[0]), min(r[1], bound[1]))


def _seg_union(segs: Segs, lo: int = 0, hi: int | None = None) -> Range:
    out: Range = _EMPTY
    for a, b, r in segs:
        if b > lo and (hi is None or a < hi):
            out = _union(out, r)
    return out


def _read(refs: Refs, params: Sequence[Range], segs: Sequence[Segs]) -> Range:
    ivs: dict[tuple[str, int], list] = {}
    ref_intervals(refs, ivs)
    out: Range = _EMPTY
    for (s, i), iv in ivs.items():
        if s == "p":
            out = _union(out, params[i])
            continue
        for lo, hi in norm(iv):
            out = _union(out, _seg_union(segs[i], lo, hi))
    return out


_MAX_SEGS = 256  # a return with more segments is one segment, their union


def _read_segs(refs: Refs, params: Sequence[Range], segs: Sequence[Segs]) -> Segs:
    """The segments of a body's return, in leaf order: an affine run of a node's leaves keeps that node's segments."""
    out: list[tuple[int, int, Range]] = []
    pos = 0
    for part in (refs.parts if isinstance(refs, Concat) else [refs]):
        n = part.count
        if n and isinstance(part, Affine) and part.space == "n":
            for a, b, r in segs[part.idx]:
                lo, hi = max(a, part.base), min(b, part.base + n)
                if lo < hi:
                    out.append((pos + lo - part.base, pos + hi - part.base, r))
        elif n:
            out.append((pos, pos + n, _read(part, params, segs)))
        pos += n
    merged: list[tuple[int, int, Range]] = []
    for a, b, r in out:
        if merged and merged[-1][2] == r and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b, r)
        else:
            merged.append((a, b, r))
    if len(merged) > _MAX_SEGS:
        return ((0, pos, _seg_union(tuple(merged))),)
    return tuple(merged)


# ---- the census ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Census:
    """A Program's census: ``total``, ``definitions`` (each Definition the root calls: its Calls and their classes), ``steps``
    (each step's first Call, its Calls' classes and its memory), ``memory`` (the whole Program's) and ``may_overflow`` (the
    s8 IMMA steps not proven exact, by Definition and node, with their accumulator's and result's bounds)."""

    program: str
    total: dict
    definitions: dict
    steps: tuple
    memory: dict
    may_overflow: tuple

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "classes": digest(), "program": self.program, "total": self.total,
                "definitions": self.definitions, "steps": list(self.steps), "memory": self.memory,
                "may_overflow": list(self.may_overflow)}


def count(fn: Any, provenance: Sequence[int], ranges: Sequence[Range] | None = None) -> dict:
    """The classes of one call of `fn` (a Definition or a primitive) whose parameters have `provenance` (`CONSTANT`,
    `REGISTERED`, `SELECTED` or `RUN` each) and value ranges `ranges` (None: any word)."""
    ts = tuple(provenance)
    raw, _t = _Provenance().call(fn, ts)
    iv = _Intervals()
    unproven: Counter = Counter()
    if iv.has_imma(fn):
        _s, unproven = iv.call(fn, tuple(ranges) if ranges is not None else (None,) * len(ts), ("", 0))
    return classify(raw, unproven)


def census(program: Program, steps: Sequence[int] | None = None) -> Census:
    """The census of `program`. `steps`: the index, among the root's Calls, of each step's first Call, ascending from 0 (None:
    one step)."""
    body = program.root.body
    calls = [k for k, nd in enumerate(body.nodes) if not is_input(nd.fn)]
    starts = [0] if steps is None else list(steps)
    if not starts or starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])) or (calls and starts[-1] >= len(calls)):
        raise ValueError(f"steps start at Call 0 and ascend strictly below {len(calls)} Calls, not {starts}")
    step_of: dict[int, int] = {}
    for s, (a, b) in enumerate(zip(starts, starts[1:] + [len(calls)])):
        for k in calls[a:b]:
            step_of[k] = s
    nsteps = len(starts)

    prov = _Provenance()
    node_t: list[int] = []
    raws: dict[int, Raw] = {}
    root_ts: tuple[int, ...] = ()
    for k, nd in enumerate(body.nodes):
        if k not in step_of:
            when = input_id(nd.fn.id)[0]  # type: ignore[index]
            node_t.append(RUN if when == RUN_BINDING else REGISTERED)
            continue
        ats = [_taint(a.refs, root_ts, node_t) for a in nd.args]
        if nd.form == "scan":
            ats[0] = max(ats)
        sub, ot = prov.call(nd.fn, tuple(ats))
        node_t.append(max(ot, ats[0]) if nd.form == "scan" else ot)
        m = nd.n if nd.form in ("batch", "scan") else 1
        raws[k] = {key: m * v for key, v in sub.items()}

    iv = _Intervals()
    unproven: dict[int, Counter] = {}
    if any(iv.has_imma(body.nodes[k].fn) for k in calls):
        segs: list[Segs] = []
        for k, nd, _s, u in iv.walk(body, (), program.root.id, segs):
            if k in step_of:
                unproven[k] = u

    def classes(ks: Sequence[int]) -> dict:
        raw: Counter = Counter()
        unp: Counter = Counter()
        for k in ks:
            _add_raw(raw, raws[k])
            unp.update(unproven.get(k, {}))
        return classify(dict(raw), unp)

    by_def: dict[str, list[int]] = {}
    for k in calls:
        by_def.setdefault(body.nodes[k].fn.id, []).append(k)
    definitions = {fid: {"calls": len(ks), **classes(ks)} for fid, ks in sorted(by_def.items())}
    mem = _memory(body, calls, step_of, nsteps)
    step_list = []
    for s, a in enumerate(starts):
        ks = [k for k in calls if step_of[k] == s]
        step_list.append({"first_call": a, "calls": len(ks), **classes(ks), "memory": mem[s]})
    sites = tuple({"definition": d, "node": n, "acc": list(c), "sum": list(r)} for (d, n), (c, r) in sorted(iv.sites.items()))
    return Census(program.fn.id, classes(calls), definitions, tuple(step_list), _memory_total(mem), sites)


# ---- memory --------------------------------------------------------------------------------------------------------------


def _minus(a: Intervals, b: Intervals) -> Intervals:
    out: list[tuple[int, int]] = []
    j = 0
    for lo, hi in a:
        while j < len(b) and b[j][1] <= lo:
            j += 1
        pos, i = lo, j
        while i < len(b) and b[i][0] < hi:
            if b[i][0] > pos:
                out.append((pos, b[i][0]))
            pos = max(pos, b[i][1])
            i += 1
        if pos < hi:
            out.append((pos, hi))
    return tuple(out)


def _bits(t: Type, iv: Intervals) -> int:
    """The bits of the leaves `iv` of a collection of type `t`."""
    if not iv:
        return 0
    w = t.homogeneous_width()
    if w is not None:
        return w * total(iv)
    if isinstance(t, Array):
        e = t.elem.leaves
        out = 0
        for lo, hi in iv:
            for i in range(lo // e, (hi - 1) // e + 1):
                out += _bits(t.elem, restrict(((lo, hi),), i * e, (i + 1) * e))
        return out
    fields = [ft for _n, ft in t.fields] if isinstance(t, Record) else list(t.fields)  # type: ignore[attr-defined]
    out, pos = 0, 0
    for ft in fields:
        out += _bits(ft, restrict(iv, pos, pos + ft.leaves))
        pos += ft.leaves
    return out


def _passes(by_step: dict[int, list], ascending: bool) -> list[tuple[int, Intervals]]:
    """[(step, the leaves whose first (ascending) or last read is that step)]."""
    out, seen = [], ()
    for t in sorted(by_step, reverse=not ascending):
        iv = norm(by_step[t])
        new = _minus(iv, seen)
        if new:
            out.append((t, new))
        seen = norm(seen + iv)
    return out


def _memory(body: Body, calls: Sequence[int], step_of: dict[int, int], nsteps: int) -> list[dict]:
    reads: dict[int, dict[int, list]] = {}
    for k in calls:
        out: dict[tuple[str, int], list] = {}
        for a in body.nodes[k].args:
            ref_intervals(a.refs, out)
        for (_s, j), iv in out.items():
            reads.setdefault(j, {}).setdefault(step_of[k], []).extend(iv)
    outs: dict[tuple[str, int], list] = {}
    ref_intervals(body.ret, outs)
    mem = [{"inputs": {"run_bits": 0, "registered_bits": 0}, "written_bits": 0, "lifetimes": Counter(), "held_input_bits": 0,
            "input_lifetimes": Counter(), "output_bits": 0} for _ in range(nsteps)]
    for j, by_step in reads.items():
        nd = body.nodes[j]
        if j in step_of:
            births, written, life = [(step_of[j], ((0, nd.out.leaves),))], "written_bits", "lifetimes"
        else:
            when = "run_bits" if input_id(nd.fn.id)[0] == RUN_BINDING else "registered_bits"  # type: ignore[index]
            births = _passes(by_step, True)
            for t, iv in births:
                mem[t]["inputs"][when] += _bits(nd.out, iv)
            if when != "run_bits":
                continue
            written, life = "held_input_bits", "input_lifetimes"
        lasts = _passes(by_step, False)
        for tb, ivb in births:
            for tl, ivl in lasts:
                if tl > tb and (n := _bits(nd.out, _isect(ivb, ivl))):
                    mem[tb][written] += n
                    mem[tb][life][tl - tb] += n
    for (_s, j), iv in outs.items():
        if j in step_of:
            mem[step_of[j]]["output_bits"] += _bits(body.nodes[j].out, norm(iv))
    for m in mem:
        for life in ("lifetimes", "input_lifetimes"):
            m[life] = {str(d): n for d, n in sorted(m[life].items())}
    return mem


def _isect(a: Intervals, b: Intervals) -> Intervals:
    return _minus(a, _minus(a, b))


def _memory_total(mem: list[dict]) -> dict:
    out: dict[str, Any] = {"inputs": {w: sum(m["inputs"][w] for m in mem) for w in ("run_bits", "registered_bits")}}
    longest = 0
    for written, life in (("written_bits", "lifetimes"), ("held_input_bits", "input_lifetimes")):
        acc: Counter = Counter()
        for m in mem:
            acc.update({int(d): n for d, n in m[life].items()})
        out[written] = sum(m[written] for m in mem)
        out[life] = {str(d): n for d, n in sorted(acc.items())}
        longest = max(longest, max(acc, default=0))
    out["max_lifetime"] = longest
    out["output_bits"] = sum(m["output_bits"] for m in mem)
    return out


__all__ = ["CONSTANT", "CONST_ID", "Census", "GATHER_ID", "KINDS", "Kind", "Mma", "PRODUCTS", "REGISTERED", "ROUNDINGS", "RUN",
           "S32", "S8", "SCHEMA", "SELECTED", "TAG", "VERSION", "census", "classify", "count", "digest", "kind", "product_class",
           "spec"]
