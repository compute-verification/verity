"""The stacking property (`verity/stacking-property` v1): P(c, R_in, L) on a Program, read with its census (`census`,
verity/census/v1: its steps, and each root value's producer, bits and readers).

A carrier is an audited block whose contents were chosen after the seed and hold another block's encoding. Such an encoding
reaches the circuit's values only as input bytes or through an encode inside the circuit, and an encode needs carries: a
floor, a comparison, a selection, a wrap or a clamp. P bounds both, and how long a value is held.

The free set is closed. A gate is free only when it is

- an s8 IMMA core step (``family``: `BlackwellS8ImmaDot32_v1`, ``D = C + sum_k a_k * b_k``) whose window the interval pass
  proves: D's bound lies in s32, from its operands' ranges as their producers give them, so it never clamps; or
- an integer affine map with constant coefficients (``affine``: `I32Add_v1`, `I32Sub_v1`, and `I32Mul_v1` with a constant
  operand) that the interval pass proves never wraps.

Every other primitive the table `CLASSES` lists is budgeted, one unit per output value (a primitive returns one value), in one
of three classes: ``roundings`` (FP arithmetic, casts, special functions, value embeddings and negation, integer floors, mods
and products, bit operations, and an affine map that may wrap), ``compare_select`` (comparisons, selections, |x|, clamps) and
``gathers`` (a data-dependent read of one of V words, counted in fresh index bits: ceil(log2 V) for each distinct index a body
reads, so one routing index read by many gathers counts once). Only gates with an operand that depends on a run input are
budgeted: a gate of constants and registered words computes the same function before the seed. A primitive the table doesn't
list is refused, and so are the FP tensor-core steps (``fp-mma``) and the integer MMA steps outside the s8 core
(``integer-mma``).

Ranges. A constant is its value, a proven core step its D, a proven affine map its exact result, and every other value any
word of its width (an s8 operand -128..127, an accumulator all of s32): ranges are derived from producers, never declared. So an
accumulator from a run input, a split-K combine that may wrap, or a chain long enough to clamp is never proven. A `scan`'s carry
is iterated to a fixpoint, exactly up to 2^16 iterations and then widened to any word.

A word is read at its own width. A reference to a leaf of another width than the reading position's type declares (an s32
word read back as s8 lanes: a limb split no gate computes) is a ``reinterpretation``, refused wherever it occurs, an argument
or a body's return.

P(c, R_in, L) holds when, under the audit's windows (`Window`: a range of the census's steps, the bytes the audit certified
as read in it and its duration):

1. ``unlisted``, ``fp-mma``, ``integer-mma``: no gate is refused;
2. ``reinterpretation``: no word is read at another width (the sites are listed);
3. ``s8-window``: every core step is proven in its window (the sites that aren't are listed);
4. ``uncovered``: every step lies in a window;
5. ``roundings``, ``compare_select``, ``gathers``: in each window, each class's units are at most c[class] per certified byte
   read;
6. ``inputs``: in each window, the run-input bytes the Program first reads there are at most R_in (bytes per second, 100 KB/s
   by default) times its duration;
7. ``lifetime``: no dynamic value (computed, or a run input) is held more than L steps. A computed value is born in its
   producer's step, a run input in the step before the first that reads it (or has it among a selection's candidates), and
   it is held until the last such step. L None (the default) leaves lifetimes to the audit's re-keying, and the verdict
   reports the longest.

The sink, coin and limb-split types of the s8 family's companions (the receipt fold, the noise derivation, `LimbSplit`) are
circuits' Build types; until the table lists those Definitions they are refused or budgeted gate by gate, never free. Bits
become bytes exactly (bits / 8), and every comparison is exact (`fractions.Fraction`).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterator, Mapping, Sequence

from .census import SCHEMA as CENSUS, step_of
from .codec import canonical_json
from .defs import Body, CompositeDefinition, PrimitiveDefinition, bind
from .intervals import norm, ref_intervals
from .program import RUN, input_id, is_input
from .refs import Affine, Concat, Explicit, Refs, Strided
from .types import Array, Tuple, Type

TAG = "verity/stacking-property"
VERSION = 1
SCHEMA = f"{TAG}/v{VERSION}"

BUDGETED = ("roundings", "compare_select", "gathers")
REFUSED = ("unlisted", "fp-mma", "integer-mma", "reinterpretation")

S8 = (-(1 << 7), (1 << 7) - 1)
S32 = (-(1 << 31), (1 << 31) - 1)
#: R_in's default: bytes per second
R_IN = 100_000

#: a constant primitive's id (`PROTOCOL.md` §9)
CONST_ID = re.compile(r"^Const([1-9][0-9]*)\[0x([0-9a-f]+)\]_v1$")
#: the gathers: (index: i32, column: Array<V, word>) -> column[index]
GATHER_ID = re.compile(r"^Gather(?:Bf16|F32)x([1-9][0-9]*)_v1$")


@dataclass(frozen=True)
class Step:
    """An s8 IMMA core step's shape: k products a step, parameters (acc, a, b), a and b in ``a`` and ``b``, D in s32."""

    k: int
    a: tuple[int, int]
    b: tuple[int, int]


FAMILY: dict[str, Step] = {"BlackwellS8ImmaDot32_v1": Step(32, S8, S8)}
AFFINE = ("I32Add_v1", "I32Sub_v1", "I32Mul_v1")


def _v(names: str, *versions: int) -> list[str]:
    return [f"{n}_v{v}" for n in names.split() for v in versions]


CLASSES: dict[str, str] = {
    **dict.fromkeys(FAMILY, "family"),
    **dict.fromkeys(AFFINE, "affine"),
    **dict.fromkeys(_v("Bf16ToF32 E4m3ToF32 F32Neg F2fpBf16 F32ToBf16Rn F32ToE4m3Sat F32ToE2m1Sat Bf16Add Bf16AddF2fp "
                       "MufuEx2Ftz MufuRcpFtz MufuSqrtFtz MufuTanh RsqrtApprox DivFullRcp Fa2InvSum TanhF32Rn "
                       "Bf16MulBf16 Bf16MulF32 GeluTanhMulBf16 RopeOut RopeOutAdd "
                       "BitAnd BitNot BitOr F32BitsShl23 U32And U32Xor I32Div I32Rem "
                       "F32ToU32RzByte NcpRneClampI7 Ue8m0CeilByte", 1)
                    + _v("F32Add F32AddFtz F32Sub F32SubFtz F32Mul F32MulFtz F32Fma F32FmaFtz F32FmaSubFtz F32FmaRm F32Div "
                         "SiluMulBf16", 1, 2), "roundings"),
    **dict.fromkeys(_v("F32Max F32GtStrict F32Fmaxf F32Fminf Bf16GtStrict F32Eq I32Le I32Eq F32IsFinite GuardNegInfZero F32Sat "
                       "SelectBf16 SelectF32 SelectI32 F32Fabs U32Ge PouwSqGe", 1) + _v("DivFullScaleA", 1, 2), "compare_select"),
    **dict.fromkeys(_v("AmpereBF16TcDot16", 2) + _v("HopperBF16WgmmaDot16", 1)
                    + _v("HopperE4m3QgmmaDot32 HopperE4m3MmaV2Dot32 BlackwellE4m3QmmaDot32 BlackwellE5m2QmmaDot32 "
                         "BlackwellNvf4OmmaDot64 BlackwellMxf4OmmaDot64", 1), "fp-mma"),
    **dict.fromkeys(_v("PouwDot16U8S8 NcpDot16I7", 1), "integer-mma"),
}


def class_of(fid: str) -> str | None:
    """A primitive's class by its id (`CLASSES`, and ``gathers`` for `GATHER_ID`), or None when it is unlisted."""
    c = CLASSES.get(fid)
    if c is None and GATHER_ID.fullmatch(fid):
        return "gathers"
    return c


def spec() -> dict:
    """The table as canonical data: what `digest` hashes."""
    return {"format": TAG, "version": VERSION, "budgeted": list(BUDGETED), "refused": list(REFUSED),
            "classes": dict(sorted(CLASSES.items())), "gather": GATHER_ID.pattern,
            "family": {f: {"k": s.k, "a": list(s.a), "b": list(s.b), "acc": list(S32)} for f, s in sorted(FAMILY.items())}}


def digest() -> str:
    """``SHA-512(TAG || "/v" || version || 0 || canon(spec))``: the table a verdict names and a statement can pin."""
    return hashlib.sha512(f"{TAG}/v{VERSION}".encode() + b"\0" + canonical_json(spec())).hexdigest()


# ---- the pass: run dependence, ranges and classes ------------------------------------------------------------------------

Range = tuple[int, int] | None  # a value's signed range at its width; None: any word of its width
Segs = tuple[tuple[int, int, Range], ...]  # a collection's leaves [lo, hi) and their range, covering it
_EMPTY = (1, 0)  # the union's identity: an empty range
_EXACT_ITERATIONS = 1 << 16
_MAX_SEGS = 256  # a return with more segments is one segment, their union


@dataclass
class Tally:
    """What gates computed: budgeted units by class, refused gates by (code, primitive), proven core steps, their products
    and proven affine maps, and the core steps not proven by primitive. ``index``: the gather index bits read from the
    Definition's parameters, by index leaves (refs over its parameters), which its caller counts (`add` leaves them out)."""

    units: Counter = field(default_factory=Counter)
    refused: Counter = field(default_factory=Counter)
    family: int = 0
    macs: int = 0
    affine: int = 0
    unproven: Counter = field(default_factory=Counter)
    index: dict = field(default_factory=dict)

    def add(self, o: "Tally", m: int = 1) -> None:
        for mine, theirs in ((self.units, o.units), (self.refused, o.refused), (self.unproven, o.unproven)):
            for key, n in theirs.items():
                mine[key] += m * n
        self.family += m * o.family
        self.macs += m * o.macs
        self.affine += m * o.affine


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


def _clip(r: Range, bound: tuple[int, int]) -> tuple[int, int]:
    if r is None or r[0] > r[1]:
        return bound
    return (max(r[0], bound[0]), min(r[1], bound[1]))


def _within(r: tuple[int, int], bound: tuple[int, int]) -> bool:
    return bound[0] <= r[0] and r[1] <= bound[1]


def _point(r: Range) -> int | None:
    return r[0] if r is not None and r[0] == r[1] else None


def _affine(fid: str, a: Range, b: Range) -> tuple[int, int] | None:
    """An affine map's exact result range over s32 operands, or None when it isn't affine (a product of two non-constants)."""
    x, y = _clip(a, S32), _clip(b, S32)
    if fid == "I32Add_v1":
        return (x[0] + y[0], x[1] + y[1])
    if fid == "I32Sub_v1":
        return (x[0] - y[1], x[1] - y[0])
    k = _point(a) if _point(a) is not None else _point(b)
    if k is None:
        return None
    o = y if _point(a) is not None else x
    return (min(k * o[0], k * o[1]), max(k * o[0], k * o[1]))


def _colls(refs: Refs) -> set[tuple[str, int]]:
    if isinstance(refs, (Affine, Strided)):
        return {(refs.space, refs.idx)} if refs.count else set()
    if isinstance(refs, Concat):
        return set().union(*(_colls(p) for p in refs.parts))
    if isinstance(refs, Explicit):
        return {(s, i) for s, i, _leaf in refs.items}
    return {(s, i) for s, i, _leaf in refs}


def _dyn(refs: Refs, params: Sequence[bool], nodes: Sequence[bool]) -> bool:
    return any(params[i] if s == "p" else nodes[i] for s, i in _colls(refs))


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


def _key(refs: Refs) -> Any:
    return tuple(refs) if refs.count <= 64 else canonical_json(refs.to_json()).decode()


def _width(t: Type, lo: int, hi: int) -> int | None:
    """The one width of `t`'s leaves [lo, hi), or None when they have two (structurally, never leaf by leaf)."""
    w = t.homogeneous_width()
    if w is not None:
        return w
    if isinstance(t, Array):
        L = t.elem.leaves
        a, b = lo // L, (hi - 1) // L
        if a == b:
            return _width(t.elem, lo - a * L, hi - a * L)
        ws = {_width(t.elem, lo - a * L, L), _width(t.elem, 0, hi - b * L)} | ({None} if b - a > 1 else set())
    else:
        ws, pos = set(), 0
        for f in (t.fields if isinstance(t, Tuple) else tuple(f for _name, f in t.fields)):  # type: ignore[attr-defined]
            n = f.leaves
            if pos < hi and pos + n > lo:
                ws.add(_width(f, max(lo, pos) - pos, min(hi, pos + n) - pos))
            pos += n
    return ws.pop() if len(ws) == 1 else None


def _span(t: Type, pos: int, ct: Type, base: int, n: int) -> bool:
    a, b = _width(t, pos, pos + n), _width(ct, base, base + n)
    if a is not None or b is not None:
        return a == b
    h = n // 2
    return _span(t, pos, ct, base, h) and _span(t, pos + h, ct, base + h, n - h)


def _parts(refs: Refs) -> Iterator[Refs]:
    if isinstance(refs, Concat):
        for p in refs.parts:
            yield from _parts(p)
    elif refs.count:
        yield refs


def _at_width(t: Type, refs: Refs, types: Callable[[str, int], Type]) -> bool:
    """Whether each of `refs` names a leaf of the width `t` declares at its position (`types`: a collection's type)."""
    pos = 0
    for part in _parts(refs):
        n = part.count
        if isinstance(part, Affine):
            ok = _span(t, pos, types(part.space, part.idx), part.base, n)
        else:
            w, ivs = _width(t, pos, pos + n), {}  # type: ignore[var-annotated]
            ref_intervals(part, ivs)
            ok = w is not None and all(_width(types(s, i), lo, hi) == w for (s, i), iv in ivs.items() for lo, hi in norm(iv))
            ok = ok or all(_width(t, pos + k, pos + k + 1) == _width(types(s, i), j, j + 1) for k, (s, i, j) in enumerate(part))
        if not ok:
            return False
        pos += n
    return True


def _reinterpretations(body: Body) -> list[int | None]:
    """The nodes of `body` with an argument read at another width, and None when its return is."""
    def types(s: str, i: int) -> Type:
        return body.params[i][1] if s == "p" else body.nodes[i].out

    out: list[int | None] = [k for k, nd in enumerate(body.nodes) if not all(_at_width(a.type, a.refs, types) for a in nd.args)]
    return out + ([] if _at_width(body.ret_type, body.ret, types) else [None])


class _Pass:
    def __init__(self) -> None:
        self.memo: dict[tuple, tuple[Segs, bool, Tally]] = {}
        self.keep: list[Any] = []
        self.sites: dict[tuple[str, int], tuple[Range, Range]] = {}
        self.widths: dict[int, frozenset[int | None]] = {}

    def prim(self, fn: PrimitiveDefinition, dyn: Sequence[bool], rng: Sequence[Range], where: tuple[str, int]) -> tuple[Range, Tally]:
        t = Tally()
        if not fn.params:
            m = CONST_ID.fullmatch(fn.id)
            v = _signed(int(m[1]), int(m[2], 16)) if m else None
            return (None if v is None else (v, v)), t
        d = any(dyn)
        cls = class_of(fn.id)
        if cls is None or cls in REFUSED:
            t.refused[(cls or "unlisted", fn.id)] += 1
            return None, t
        if cls == "family":
            st = FAMILY[fn.id]
            c, a, b = _clip(rng[0], S32), _clip(rng[1], st.a), _clip(rng[2], st.b)
            corners = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
            s = (c[0] + st.k * min(corners), c[1] + st.k * max(corners))
            if _within(s, S32):
                t.family, t.macs = 1, st.k
                return s, t
            prev = self.sites.get(where)
            self.sites[where] = (c, s) if prev is None else (_union(prev[0], c), _union(prev[1], s))
            t.unproven[fn.id] += 1
            return (max(s[0], S32[0]), min(s[1], S32[1])), t
        if cls == "affine":
            r = _affine(fn.id, rng[0], rng[1])
            if r is not None and _within(r, S32):
                t.affine = 1
                return r, t
            cls = "roundings"
        if d and cls != "gathers":
            t.units[cls] += 1
        return None, t

    def call(self, fn: Any, dyn: tuple[bool, ...], rng: tuple[Range, ...], where: tuple[str, int]) -> tuple[Segs, bool, Tally]:
        if isinstance(fn, PrimitiveDefinition):
            r, t = self.prim(fn, dyn, rng, where)
            return ((0, 1, r),), any(dyn), t
        key = (id(fn), dyn, rng)
        hit = self.memo.get(key)
        if hit is None:
            self.keep.append(fn)
            segs: list[Segs] = []
            dyns: list[bool] = []
            out = Tally()
            up: dict = {}
            for _k, t in self.walk(fn.body, dyn, rng, fn.id, segs, dyns, up):
                out.add(t)
            out.index = up
            hit = self.memo[key] = (_read_segs(fn.body.ret, rng, segs), _dyn(fn.body.ret, dyn, dyns), out)
        return hit

    def walk(self, body: Body, pdyn: Sequence[bool], prng: Sequence[Range], def_id: str, segs: list[Segs],
             dyns: list[bool], up: dict) -> Iterator[tuple[int, Tally]]:
        """Each node of `body` with the tally of its gates (a batch's and a scan's whole), filling its output's segments and
        run dependence, and a reinterpreting return as its last node's. Gather index bits: an index computed in this body
        counts once, in the tally of the first node that reads it; one read from the parameters goes in `up` for the caller;
        a batch member's own index (a mapped one) counts once a member."""
        bad = self.widths.get(id(body))
        if bad is None:
            self.keep.append(body)
            bad = self.widths[id(body)] = frozenset(_reinterpretations(body))
        fresh: dict[Any, int] = {}

        def note(refs: Refs, bits: int, t: Tally) -> None:
            key = _key(refs)
            if any(s == "p" for s, _i in _colls(refs)):
                up[key] = (refs, max(bits, up.get(key, (refs, 0))[1]))
            elif bits > fresh.get(key, 0):
                t.units["gathers"] += bits - fresh.get(key, 0)
                fresh[key] = bits

        for k, nd in enumerate(body.nodes):
            if is_input(nd.fn):
                segs.append(((0, nd.out.leaves, None),))
                dyns.append(input_id(nd.fn.id)[0] == RUN)  # type: ignore[index]
                continue
            ad = [_dyn(a.refs, pdyn, dyns) for a in nd.args]
            ar = [_read(a.refs, prng, segs) for a in nd.args]
            if nd.form == "scan":
                s, od, t = self.scan(nd, ad, ar, def_id, k)
            else:
                s, od, t0 = self.call(nd.fn, tuple(ad), tuple(ar), (def_id, k))
                batch = nd.form == "batch"
                if batch:
                    s = ((0, nd.out.leaves, _seg_union(s)),)
                t = Tally()
                t.add(t0, nd.n if batch else 1)
                for refs, bits in t0.index.values():
                    leaves = list(refs)
                    if batch and any(nd.axes[i] is not None for _s, i, _j in leaves):
                        t.units["gathers"] += nd.n * bits
                    else:
                        note(Explicit(tuple(nd.args[i].refs[j] for _s, i, j in leaves)), bits, t)
                m = GATHER_ID.fullmatch(nd.fn.id) if isinstance(nd.fn, PrimitiveDefinition) else None
                if m and ad[0]:
                    bits = (int(m[1]) - 1).bit_length()
                    if batch and nd.axes[0] is not None:
                        t.units["gathers"] += nd.n * bits
                    else:
                        note(nd.args[0].refs, bits, t)
            if k in bad:
                t.refused[("reinterpretation", def_id)] += 1
            segs.append(s)
            dyns.append(od)
            yield k, t
        if None in bad:
            t = Tally()
            t.refused[("reinterpretation", def_id)] += 1
            yield len(body.nodes) - 1, t

    def scan(self, nd: Any, ad: list[bool], ar: list[Range], def_id: str, k: int) -> tuple[Segs, bool, Tally]:
        cl = nd.out.fields[0].leaves
        d = tuple([any(ad)] + ad[1:])
        carry, rest = ar[0], tuple(ar[1:])
        ys: Range = _EMPTY
        out = Tally()
        for i in range(nd.n):
            if i == _EXACT_ITERATIONS:
                carry = None
            s, _od, t = self.call(nd.fn, d, (carry,) + rest, (def_id, k))
            nxt = _union(carry, _seg_union(s, 0, cl))
            ys = _union(ys, _seg_union(s, cl, None))
            reps = nd.n - i if nxt == carry else 1
            out.add(t, reps)
            if t.index:
                out.units["gathers"] += reps * sum(bits for _refs, bits in t.index.values())  # an iteration's index counts anew
            if nxt == carry:
                break
            carry = nxt
        return ((0, cl, carry), (cl, nd.out.leaves, ys)), d[0], out


# ---- the analysis and the check -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    """What P reads of a Program: per step (the census's) its budgeted units by class and the run-input bits it first reads;
    over the Program its refusals (gates by primitive, reinterpretations by the Definition whose body reads at another
    width), proven core steps with their products and proven affine maps, the core steps not proven with their sites, and the
    longest a dynamic value is held."""

    steps: tuple[dict, ...]
    refused: dict
    family: int
    macs: int
    affine: int
    unproven: dict
    sites: tuple[dict, ...]
    max_lifetime: int

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "classes": digest(), "steps": list(self.steps), "refused": self.refused, "family": self.family,
                "macs": self.macs, "affine": self.affine, "unproven": self.unproven, "sites": list(self.sites),
                "max_lifetime": self.max_lifetime}


def count(fn: Any, dyn: Sequence[bool], ranges: Sequence[Range] | None = None) -> Tally:
    """The tally of one call of `fn` (a Definition or a primitive) whose parameters depend on a run input as `dyn` says, with
    value ranges `ranges` (None: any word)."""
    if isinstance(fn, CompositeDefinition):
        fn = bind(fn)
    rng = tuple(ranges) if ranges is not None else (None,) * len(dyn)
    p = _Pass()
    if isinstance(fn, PrimitiveDefinition):
        return p.call(fn, tuple(dyn), rng, (fn.id, 0))[2]
    out = Tally()
    up: dict = {}
    for _k, t in p.walk(fn.body, tuple(dyn), rng, fn.id, [], [], up):
        out.add(t)
    if up:
        out.units["gathers"] += sum(bits for _refs, bits in up.values())
    return out


def analyze(program: Any, census: Mapping) -> Analysis:
    """P's reading of `program` with its census (verity/census/v1 of this Program: its steps, and its root values' producers,
    bits, readers and selections). A census of another schema or of another root is refused (ValueError)."""
    nodes = program.root.body.nodes
    if census.get("schema") != CENSUS or len(census["nodes"]["callee"]) != len(nodes) or census["inputs"] != sum(
            1 for nd in nodes if is_input(nd.fn)):
        raise ValueError(f"a {CENSUS} of this Program's root ({len(nodes)} nodes), not {census.get('schema')!r} with "
                         f"{len(census.get('nodes', {}).get('callee', []))}")
    S = len(census["steps"]["ends"])
    step = step_of(census)
    units = [Counter() for _ in range(S)]
    whole = Tally()
    p = _Pass()
    for k, t in p.walk(program.root.body, (), (), program.root.id, [], [], {}):
        whole.add(t)
        for cls, n in t.units.items():
            units[step(k)][cls] += n
    I = census["inputs"]
    binding = [census["classes"][census["callees"][i]]["binding"] for i in census["nodes"]["callee"][:I]]
    pools: dict[int, set[int]] = {}
    for k, vs in zip(census["selects"]["call"], census["selects"]["values"]):
        for v in vs:
            pools.setdefault(v, set()).add(step(k))
    first = [0] * S
    longest = 0
    vals = census["values"]
    for v, (j, bits, reads) in enumerate(zip(vals["producer"], vals["bits"], vals["reads"])):
        touched = {step(k) for k in reads} | pools.get(v, set())
        if j >= I:
            birth = step(j)
        elif binding[j] != "run" or not touched:
            continue
        else:
            birth = min(touched) - 1
            first[birth + 1] += bits
        longest = max(longest, max(touched, default=birth) - birth)
    steps = tuple({"units": {c: units[s][c] for c in BUDGETED}, "input_bits": first[s]} for s in range(S))
    refused: dict[str, dict[str, int]] = {}
    for (code, fid), n in sorted(whole.refused.items()):
        refused.setdefault(code, {})[fid] = n
    sites = tuple({"definition": d, "node": n, "acc": list(c), "sum": list(r)} for (d, n), (c, r) in sorted(p.sites.items()))
    return Analysis(steps, refused, whole.family, whole.macs, whole.affine, dict(sorted(whole.unproven.items())), sites, longest)


@dataclass(frozen=True)
class Window:
    """A window: a range of the census's steps, the bytes the audit certified as read in it, and its duration in seconds."""

    steps: range
    reads: int
    seconds: Any


@dataclass(frozen=True)
class Verdict:
    """Whether P holds, why not (``reasons``: each a ``code`` with its numbers) and the numbers it read."""

    ok: bool
    reasons: tuple[dict, ...]
    numbers: dict

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "classes": digest(), "ok": self.ok, "reasons": list(self.reasons), "numbers": self.numbers}


def check(program: Any, census: Mapping, *, c: Mapping[str, Any], windows: Sequence[Window], R_in: Any = R_IN,
          L: int | None = None) -> Verdict:
    """P(c, R_in, L) on `program` and its census under the audit's `windows`. c maps each budgeted class to its units per
    certified byte read, R_in is bytes per second, L is steps (None: left to re-keying). A window outside the census's steps,
    or a c without every budgeted class, is refused (ValueError)."""
    a = analyze(program, census)
    return judge(a, c=c, windows=windows, R_in=R_in, L=L)


def judge(a: Analysis, *, c: Mapping[str, Any], windows: Sequence[Window], R_in: Any = R_IN, L: int | None = None) -> Verdict:
    """P(c, R_in, L) on an `Analysis` (as `check`)."""
    if set(c) != set(BUDGETED):
        raise ValueError(f"c gives units per certified byte for exactly {list(BUDGETED)}, not {sorted(c)}")
    cq = {k: Fraction(c[k]) for k in BUDGETED}
    R = Fraction(R_in)
    S = len(a.steps)
    for w in windows:
        if (not isinstance(w.steps, range) or w.steps.step != 1 or not 0 <= w.steps.start < w.steps.stop <= S or w.reads < 0
                or Fraction(w.seconds) <= 0):
            raise ValueError(f"a window is a non-empty range of the census's {S} steps with reads >= 0 and seconds > 0, not {w}")
    reasons: list[dict] = [{"code": code, "definitions" if code == "reinterpretation" else "primitives": a.refused[code]}
                           for code in REFUSED if code in a.refused]
    if a.unproven:
        reasons.append({"code": "s8-window", "steps": a.unproven, "sites": list(a.sites)})
    covered = set().union(*(set(w.steps) for w in windows)) if windows else set()
    if len(covered) < S:
        reasons.append({"code": "uncovered", "steps": sorted(set(range(S)) - covered)})
    rows = []
    for w in windows:
        u = {k: sum(a.steps[s]["units"][k] for s in w.steps) for k in BUDGETED}
        inb, sec = Fraction(sum(a.steps[s]["input_bits"] for s in w.steps), 8), Fraction(w.seconds)
        row = {"steps": [w.steps.start, w.steps.stop], "reads": w.reads, "seconds": str(sec), "units": u, "input_bytes": str(inb)}
        rows.append(row)
        for k in BUDGETED:
            if u[k] > cq[k] * w.reads:
                reasons.append({"code": k, **row, "c": str(cq[k])})
        if inb > R * sec:
            reasons.append({"code": "inputs", **row, "R_in": str(R)})
    if L is not None and a.max_lifetime > L:
        reasons.append({"code": "lifetime", "max_lifetime": a.max_lifetime, "L": L})
    total = {k: sum(s["units"][k] for s in a.steps) for k in BUDGETED}
    numbers = {"c": {k: str(v) for k, v in cq.items()}, "R_in": str(R), "L": L, "windows": rows, "units": total,
               "family_steps": a.family, "macs": a.macs, "affine": a.affine, "max_lifetime": a.max_lifetime}
    return Verdict(not reasons, tuple(reasons), numbers)


__all__ = ["AFFINE", "Analysis", "BUDGETED", "CLASSES", "CONST_ID", "FAMILY", "GATHER_ID", "REFUSED", "R_IN", "S32", "S8",
           "SCHEMA", "Step", "TAG", "Tally", "VERSION", "Verdict", "Window", "analyze", "check", "class_of", "count", "digest",
           "judge", "spec"]
