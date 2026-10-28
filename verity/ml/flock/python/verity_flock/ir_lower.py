"""C-Flock's generic IR lowering: an IR Definition becomes a Flock binary circuit by composing :mod:`fp` pieces.

:func:`lower` walks the Definition's gates in canonical order, as ``verity.ml.circuits.evaluate`` does, and maps each primitive to
its piece in :data:`PIECES` (a primitive without one is an error naming it). Constant gates, and gates whose operands are
all constant, are evaluated by the primitive's own evaluator and become constant wires. An activation of a Boolean
Definition that the walk holds whole is lowered by the Definition's builder instead of its gates (:func:`lower_gates`).

:func:`units` splits a template instance into identical units read off its IR graph: the connected components of the gate
graph (with the input leaves) must all compute the same function of their inputs; the unit is one component, and the
arrangement lists which input and output leaves of the instance each unit takes. :func:`netlist` lays a unit circuit out as
``flock-ir-unit/v2`` (see its docstring), the format the block statement ``verity/flock-ir-block/v1`` reads.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property, lru_cache, partial, reduce
from typing import Callable, Collection, Sequence

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.trace import _flat_out
from verity.ml.circuits.layout import Region, param_leaf, resolve

from . import fp
from . import tail_pieces
from .fp import ceil_pow2
from .gf2 import Circuit, L, const_bits

FORMAT = "flock-ir-unit/v2"
WORD = 128
#: the netlist's rows per joined piece, and the characters a digest encodes at a time
NETLIST_ROWS = 1 << 16
HASH_CHUNK = 1 << 24
NAN_TORCH = 0x7FC0
NAN_F2FP = 0x7FFF


# --- pieces per primitive ----------------------------------------------------------------------------------------------

def _rope_out(C, x, y, c, s):
    t = fp.f32_mul(C, fp.bf16_to_f32(y), fp.bf16_to_f32(s))
    return fp.f32_to_bf16(C, fp.f32_fma(C, fp.bf16_to_f32(x), fp.bf16_to_f32(c), fp.f32_neg(t)), NAN_TORCH)


def _rope_out_add(C, x, y, c, s):
    t = fp.f32_mul(C, fp.bf16_to_f32(x), fp.bf16_to_f32(s))
    return fp.f32_to_bf16(C, fp.f32_fma(C, fp.bf16_to_f32(y), fp.bf16_to_f32(c), t), NAN_TORCH)


@lru_cache(maxsize=None)
def silu_table() -> tuple[int, ...]:
    """``bf16_rn(silu_f32(f32(g)))`` for every bf16 word g: the first half of ``SiluMulBf16``, from the primitive's own helpers."""
    from verity.ml.definitions.registry import prims as P

    return tuple(P.f32_to_bf16_rn_bits(P._silu_f32_bits(P.bf16_to_f32_bits(g)), P.BF16_NAN_TORCH) for g in range(1 << 16))


def _silu_mul(C, g, u):
    s = fp.lookup(C, list(g), silu_table(), 16)
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(s), fp.bf16_to_f32(u)), NAN_TORCH)


def _bf16_add(nan):
    return lambda C, a, b: fp.f32_to_bf16(C, fp.f32_add(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)), nan)


def _f32_div(C, a, b, **nan):
    """``F32Div_v1`` (``ir_nan=True``) or ``_v2`` (``nan_word=F32_NAN_PTX``): a multiply by the exact reciprocal for a constant
    power-of-two divisor, else the divider."""
    if all(w.is_const() for w in b):
        k = sum(w.c << i for i, w in enumerate(b))
        e, m = (k >> 23) & 0xFF, k & 0x7FFFFF
        if not m and 1 <= e <= 253:
            return fp.f32_mul(C, a, fp.f32_const((k & 0x80000000) | ((254 - e) << 23)), **nan)
    return fp.f32_div(C, a, b, **nan)


#: primitive id -> piece(C, *operand words) -> output word
PIECES: dict[str, Callable] = {
    "Bf16ToF32_v1": lambda C, a: fp.bf16_to_f32(a),
    "F2fpBf16_v1": lambda C, a: fp.f32_to_bf16(C, a, NAN_F2FP),
    "F32ToBf16Rn_v1": lambda C, a: fp.f32_to_bf16(C, a, NAN_TORCH),
    "F32Add_v1": partial(fp.f32_add, ir_nan=True),
    "F32Mul_v1": partial(fp.f32_mul, ir_nan=True),
    "F32Fma_v1": fp.f32_fma,
    "F32Div_v1": partial(_f32_div, ir_nan=True),
    "F32Add_v2": partial(fp.f32_add, nan_word=fp.F32_NAN_PTX),
    "F32Mul_v2": partial(fp.f32_mul, nan_word=fp.F32_NAN_PTX),
    "F32Fma_v2": partial(fp.f32_fma, nan_word=fp.F32_NAN_PTX),
    "F32Div_v2": partial(_f32_div, nan_word=fp.F32_NAN_PTX),
    "Bf16Add_v1": _bf16_add(NAN_TORCH),
    "Bf16AddF2fp_v1": _bf16_add(NAN_F2FP),
    "RopeOut_v1": _rope_out,
    "RopeOutAdd_v1": _rope_out_add,
    "SiluMulBf16_v1": _silu_mul,
    "AmpereBF16TcDot16_v1": lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:])),
    # verity.ml.circuits.silicon.models.HOPPER_BF16_WGMMA_K16: one group of 16, 26-bit adder, exponent floor -133 (the same total rules)
    "HopperBF16WgmmaDot16_v1": lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:]), groups=(16,), W=26, F=-133),
}
PIECES.update(tail_pieces.PIECES)
PIECES["AmpereBF16TcDot16_v2"] = PIECES["AmpereBF16TcDot16_v1"]  # core's id (verity.ml.circuits.ops.prims) for the same total function
# the Boolean core's gates (verity.ml.circuits.boolean): an n-input And is n - 1 ANDs, Xor and Not are free
PIECES["Not_v1"] = lambda C, a: [a[0].inv()]
GATE_FAMILIES = {r"And(?:[2-9]|[1-9][0-9]+)_v1": lambda C, *x: [fp.and_reduce(C, [b[0] for b in x])],
                 r"Xor(?:[2-9]|[1-9][0-9]+)_v1": lambda C, *x: [reduce(lambda s, b: s ^ b[0], x[1:], x[0][0])]}


@lru_cache(maxsize=None)
def _boolean_ops() -> None:
    import importlib
    import pkgutil

    from verity.ml.circuits.ops import boolean as OB

    for m in pkgutil.iter_modules(OB.__path__):
        importlib.import_module(f"{OB.__name__}.{m.name}")


def _boolean_build(key):
    """The ``build`` of a registered Boolean Definition (``verity.ml.circuits.boolean.trace.definition``) whose ``word`` is the
    primitive ``key``: a `forms` builder on the operand words, as a piece is. So the word primitive lowers by the builder that
    lowers an activation of the Definition itself (`lower_gates`), commits included. Every module of
    `verity.ml.circuits.ops.boolean` is imported first: a Program of word primitives (`ops.prims`' s8 IMMA steps, `ops.s8`'s
    limb steps) imports none of the modules that hold their Boolean versions, and without them the registry has no build."""
    from verity.ml.circuits.defs import REGISTRY

    _boolean_ops()
    for d in list(REGISTRY.defs.values()):
        w, build = getattr(d, "word", None), getattr(d, "build", None)
        if callable(build) and getattr(w, "id", None) == key:
            return build
    return None


class _Pieces(dict):
    """:data:`PIECES`, which also resolves a primitive whose id carries a width (``tail_pieces.FAMILIES``: ``BitAtx{W}_v1``)
    or an arity (:data:`GATE_FAMILIES`), or that a registered Boolean Definition computes (`_boolean_build`); ``wrap`` (the Boolean export's tracer) wraps a piece resolved later like the rest."""

    wrap = None

    def _resolve(self, key):
        import re

        for pat, fn in (*tail_pieces.FAMILIES.items(), *GATE_FAMILIES.items()):
            if isinstance(key, str) and re.fullmatch(pat, key):
                self[key] = self.wrap(key, fn) if self.wrap else fn
                return self[key]
        fn = _boolean_build(key)
        if fn is not None:
            self[key] = self.wrap(key, fn) if self.wrap else fn
            return self[key]
        return None

    def __missing__(self, key):
        fn = self._resolve(key)
        if fn is None:
            raise KeyError(key)
        return fn

    def __contains__(self, key):
        return dict.__contains__(self, key) or self._resolve(key) is not None

    def get(self, key, default=None):
        return self[key] if key in self else default


PIECES = _Pieces(PIECES)


class NoPiece(NotImplementedError):
    """An IR primitive that C-Flock's lowering has no piece for."""


# --- the walk ----------------------------------------------------------------------------------------------------------

def _width(prim) -> int:
    return prim.ret.w


def _is_const(word) -> bool:
    return all(w.is_const() for w in word)


def _value(word) -> int:
    return sum(w.c << i for i, w in enumerate(word))


def lower_gates(C: Circuit, gates, wires: dict[int, list], pieces: dict[str, Callable] | None = None,
                reads: Collection[int] = ()) -> dict[int, list]:
    """Lower ``gates`` (``verity.ml.circuits.layout.Gate``s in canonical order) into C, reading operand words from ``wires`` (gate index
    -> word) and writing each gate's output word there; ``pieces`` (default :data:`PIECES`) maps primitive ids to pieces.

    An activation of a Boolean Definition (`verity.ml.circuits.boolean.trace.definition`'s, whose Definition carries its
    ``build``) is lowered at its first gate by the builder, over its parameter leaves' wires and with an AND memo of its own,
    as its trace had: so C has the builder's commits (`forms.Circuit.commit`'s rows, which its gates drop) and its trace's
    ANDs, and only its returned gates get words. One that ``gates`` hold in part (a unit's split or cut, a dead gate dropped),
    whose parameter leaves are not all in ``wires``, or with a gate other than its returns in ``reads`` (gate indices the
    caller reads besides the gates' own operands) is lowered gate by gate."""
    pieces = PIECES if pieces is None else pieces
    region = isinstance(gates, Region)
    if region:
        i, end = gates.lo, gates.hi
        held = lambda lo, hi: gates.lo <= lo and hi <= gates.hi  # noqa: E731
    else:
        gates = list(gates)
        i, end = 0, len(gates)
        have = {g.index for g in gates}
        held = lambda lo, hi: all(j in have for j in range(lo, hi))  # noqa: E731
    reads = sorted(reads)
    whole: dict[int, bool] = {}
    while i < end:
        g = gates.circuit.gate(i) if region else gates[i]
        act = _boolean_activation(g)
        if act is not None:
            s, build = act
            lo, hi = s.offset, s.offset + s.spec.gates
            if lo not in whole:
                rets = {resolve(s, r) for r in s.spec.body.ret}
                whole[lo] = held(lo, hi) and all(j in rets for j in reads[bisect.bisect_left(reads, lo):bisect.bisect_left(reads, hi)])
            if whole[lo]:
                whole[lo] = _lower_activation(C, s, build, wires, pieces)
            if whole[lo]:
                i += hi - lo
                continue
        prim = g.prim
        fam = getattr(prim, "family", None)
        i += 1
        if fam == "Input":
            if g.index not in wires:
                raise KeyError(f"input gate {g.index} has no wires")
            continue
        ops = [wires[o] for o in g.operands]
        if fam == "Const" or all(_is_const(o) for o in ops):
            wires[g.index] = const_bits(prim.evaluate(*[_value(o) for o in ops]), _width(prim))
            continue
        piece = pieces.get(prim.id)
        if piece is None:
            raise NoPiece(f"C-Flock has no circuit piece for the IR primitive {prim.id}")
        out = piece(C, *ops)
        if len(out) != _width(prim):
            raise AssertionError(f"piece for {prim.id} returned {len(out)} bits, the primitive returns {_width(prim)}")
        wires[g.index] = out
    return wires


def _boolean_activation(g):
    """``(scope, build)``: the activation of a Boolean Definition that the gate ``g``'s scope chain reaches, and its builder on
    the activation's elements; or None. A Definition with statics takes them first (``build(C, S, *elements)``, as
    `verity.ml.definitions.registry.sampling_boolean.traced_family` traces it)."""
    s = getattr(g, "scope", None)
    while s is not None:
        d = getattr(s.spec, "defn", s.spec)
        build = getattr(d, "build", None)
        if callable(build):
            if getattr(d, "statics", ()):
                return s, partial(lambda S, C, *e: build(C, S, *e), s.spec.S())
            return s, build
        s = s.parent
    return None


def _lower_activation(C: Circuit, s, build: Callable, wires: dict[int, list], pieces) -> bool:
    """Run ``build`` on C for the activation ``s`` (`lower_gates`), setting its returned gates' words; False, building nothing,
    when a parameter leaf has no wire. Its operands are its parameters' elements (`boolean.elements`, one bit a leaf), as
    `trace` gives them."""
    params, ret = s.spec.signature()
    leaves = [[param_leaf(s, pi, j) for j in range(t.leaves)] for pi, (_, t) in enumerate(params)]
    if any(o not in wires for ls in leaves for o in ls):
        return False
    els = []
    for (_, t), ls in zip(params, leaves):
        at = 0
        for w in BL.elements(t):
            els.append([wires[o][0] for o in ls[at:at + w]])
            at += w
    wrap = getattr(pieces, "wrap", None)
    fn = wrap(s.spec.id, build) if wrap else build
    memo = C.cse
    C.cse = {} if memo is None else memo
    try:
        outs = _flat_out(fn(C, *els), ret)
    finally:
        C.cse = memo
    lo, hi = s.offset, s.offset + s.spec.gates
    for r, f in zip(s.spec.body.ret, outs):
        j = resolve(s, r)
        if lo <= j < hi:
            wires.setdefault(j, [f])
    return True


def standalone(target):
    from verity.ml.circuits.evaluation.reference import resolve, standalone_call

    return standalone_call(resolve(target))


def lower(C: Circuit, target, inputs: Sequence[list]) -> list[list]:
    """The output words of ``target`` (a primitive or a bound composite) on the input words ``inputs`` (one per input leaf)."""
    from verity.ml.circuits.evaluation.reference import resolve
    from verity.ml.circuits.defs import PrimitiveDefinition

    t = resolve(target)
    if isinstance(t, PrimitiveDefinition):
        wires = lower_gates(C, [_PrimGate(t, len(inputs))], {-(i + 1): list(w) for i, w in enumerate(inputs)})
        return [wires[0]]
    _prog, call = standalone(t)
    wires = dict(zip(call.args, inputs))
    lower_gates(C, call, wires)
    return [wires[r] for r in call.returns]


@dataclass(frozen=True)
class _PrimGate:
    prim: object
    n: int
    index: int = 0

    @property
    def operands(self):
        return [-(i + 1) for i in range(self.n)]


# --- units, read off the IR graph ------------------------------------------------------------------------------------------

@dataclass
class Units:
    """A template instance as identical units plus an optional native tail.

    ``in_src[u]`` / ``out_src[u]`` give unit u's ports in order: ``("leaf", p)`` is input leaf p of the instance, ``("ret", p)``
    output leaf p, ``("cut", k)`` the k-th cut gate (``cut[k]``, a gate index). Cut gates are the words where the instance is
    split: a unit that computes one outputs it, a unit that reads one takes it as an input. ``tail`` lists the gates of the
    components that read no input leaf (they compute cut words from cut words); the verifier evaluates them itself with the
    IR evaluator. ``gates`` is unit 0's gate list and ``signature`` the shape every unit shares."""

    target: object
    in_widths: list[int]
    out_widths: list[int]
    in_src: list[list[tuple]]
    out_src: list[list[tuple]]
    signature: tuple = field(repr=False)
    gates: list = field(repr=False)
    arg_gates: list[int] = field(repr=False)
    ret_gates: list[int] = field(repr=False)
    cut: list[int] = field(default_factory=list)
    tail: list = field(default_factory=list, repr=False)
    #: ``(cut word, input leaf)`` for input leaves the units read as public words (the verifier knows them)
    public: list = field(default_factory=list)
    #: per output leaf, the tail gate that computes it (the verifier's own), when the units return none
    tail_rets: list = field(default_factory=list)
    builder: Callable | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.in_src)

    def circuit(self) -> Circuit:
        """The unit circuit: inputs ``in0 .. in{k-1}``, outputs ``out0 ..`` (one word per port)."""
        if self.builder is not None:
            return self.builder()
        C = Circuit(cse=True)
        ins = [C.input(f"in{j}", w) for j, w in enumerate(self.in_widths)]
        wires = dict(zip(self.arg_gates, ins))
        lower_gates(C, self.gates, wires, reads=self.ret_gates)
        for j, g in enumerate(self.ret_gates):
            C.output(f"out{j}", wires[g])
        return C


def live_gates(call) -> list:
    """The call's gates in canonical order that some output depends on."""
    by = {g.index: g for g in call}
    live: set[int] = set()
    stack = list(call.returns)
    while stack:
        v = stack.pop()
        if v in live or v not in by:
            continue
        live.add(v)
        stack.extend(by[v].operands)
    return [g for g in call if g.index in live]


def reduction_cut(width: int) -> Callable:
    """A cut for a row template with one reduction: the row scalar (the live gate read by the most gates) and, in its cone,
    every gate whose input-leaf cone has at most ``width`` leaves while a consumer in the cone has more (the reduction's
    boundary words, e.g. one CUDA warp's aggregate)."""
    def cut(gates, args, _rets):
        by = {g.index: g for g in gates}
        consumers: dict[int, list[int]] = {}
        for g in gates:
            for o in g.operands:
                consumers.setdefault(o, []).append(g.index)
        scalar = max(by, key=lambda i: (len(consumers.get(i, ())), -i))
        leaf_bit = {a: 1 << i for i, a in enumerate(args)}
        cone_leaves: dict[int, int] = {}
        for g in gates:
            m = 0
            for o in g.operands:
                m |= leaf_bit.get(o, 0) | cone_leaves.get(o, 0)
            cone_leaves[g.index] = m
        cone: set[int] = set()
        stack = [scalar]
        while stack:
            v = stack.pop()
            if v in cone or v not in by:
                continue
            cone.add(v)
            stack.extend(by[v].operands)
        size = {g: cone_leaves[g].bit_count() for g in cone}
        return {scalar} | {g for g in cone if 0 < size[g] <= width and any(c in cone and size[c] > width for c in consumers.get(g, ()))}
    return cut


def units(target, cut: Callable | None = None, group: int = 1) -> Units:
    """Split ``target``'s one-call Program (its live gates) into the connected components of its gate graph, input leaves
    included and edges out of cut gates removed (``cut(gates, args, returns)`` returns the cut gate indices; default none).
    ``group`` consecutive components that read input leaves form one unit. Every unit must have the same signature
    (primitive ids and operand wiring relative to the unit, cut words as ports), else ValueError; the components that read
    no input leaf form the native tail."""
    _prog, call = standalone(target)
    args = list(call.args)
    rets = list(call.returns)
    gates = live_gates(call)
    K = sorted(cut(gates, args, rets)) if cut else []
    kset = set(K)
    kpos = {g: k for k, g in enumerate(K)}
    arg_pos = {g: i for i, g in enumerate(args)}
    ret_pos: dict[int, list[int]] = {}
    for i, g in enumerate(rets):
        ret_pos.setdefault(g, []).append(i)
    parent: dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in gates:
        find(g.index)
        for o in g.operands:
            if o not in kset:
                parent[find(o)] = find(g.index)
    comps: dict[int, list] = {}
    for g in gates:
        comps.setdefault(find(g.index), []).append(g)
    groups = sorted(comps.values(), key=lambda gs: gs[0].index)
    if group > 1:
        reading = [gs for gs in groups if any(o in arg_pos for g in gs for o in g.operands)]
        rest = [gs for gs in groups if not any(o in arg_pos for g in gs for o in g.operands)]
        if len(reading) % group:
            raise ValueError(f"{len(reading)} components do not group by {group}")
        groups = [sorted([g for gs in reading[i:i + group] for g in gs], key=lambda g: g.index) for i in range(0, len(reading), group)] + rest
    sigs, ins, outs, unit_groups, tail = [], [], [], [], []
    for gs in groups:
        local = {g.index: k for k, g in enumerate(gs)}
        leaves = sorted({o for g in gs for o in g.operands if o in arg_pos}, key=arg_pos.__getitem__)
        if not leaves:
            tail.extend(gs)
            continue
        cuts_in = sorted({o for g in gs for o in g.operands if o in kset and o not in local})
        port = {o: j for j, o in enumerate(leaves + cuts_in)}
        sig = tuple((g.prim.id, tuple(("g", local[o]) if o in local else ("in", port[o]) for o in g.operands)) for g in gs)
        r = sorted((p, local[g.index]) for g in gs for p in ret_pos.get(g.index, []))
        kout = [g.index for g in gs if g.index in kset]
        sigs.append((sig, tuple(k for _, k in r), tuple(local[g] for g in kout)))
        ins.append([("leaf", arg_pos[o]) for o in leaves] + [("cut", kpos[o]) for o in cuts_in])
        outs.append([("ret", p) for p, _ in r] + [("cut", kpos[g]) for g in kout])
        unit_groups.append(gs)
    if not unit_groups:
        raise ValueError(f"{getattr(target, 'id', target)}: no component reads an input leaf")
    if any(sg != sigs[0] for sg in sigs):
        raise ValueError(f"{getattr(target, 'id', target)}: its components are not all the same unit ({len(set(sigs))} shapes)")
    if any(ret_pos.get(g.index) for g in tail):
        raise ValueError("an output leaf is computed by the native tail")
    covered_out = sorted(p for u in outs for kind, p in u if kind == "ret")
    if covered_out != list(range(len(rets))):
        raise ValueError("the units do not cover every output leaf exactly once")
    leaves_in = [p for u in ins for kind, p in u if kind == "leaf"]
    if len(set(leaves_in)) != len(leaves_in):
        raise ValueError("two units share an input leaf")
    first = unit_groups[0]
    width = {g.index: g.prim.ret.w for g in gates}
    src_gate = lambda kind, p: args[p] if kind == "leaf" else K[p]  # noqa: E731
    arg_gates = [src_gate(k, p) for k, p in ins[0]]
    in_widths = [_leaf_width(target, p) if k == "leaf" else width[K[p]] for k, p in ins[0]]
    ret_gates = [rets[p] if k == "ret" else K[p] for k, p in outs[0]]
    return Units(target, in_widths, [width[g] for g in ret_gates], ins, outs, sigs[0], first, arg_gates, ret_gates, K, tail)


def whole(target) -> Units:
    """``target``'s one-call Program as ONE unit: its live gates, every input leaf a port and every output leaf a return, for a
    template whose gate graph is not identical components (a parse whose fields read different bytes of one body)."""
    _prog, call = standalone(target)
    args, rets = list(call.args), list(call.returns)
    gates = live_gates(call)
    local = {g.index: k for k, g in enumerate(gates)}
    port = {a: j for j, a in enumerate(args)}
    sig = tuple((g.prim.id, tuple(("g", local[o]) if o in local else ("in", port[o]) for o in g.operands)) for g in gates)
    in_widths = [_leaf_width(target, p) for p in range(len(args))]
    width = {**dict(zip(args, in_widths)), **{g.index: g.prim.ret.w for g in gates}}
    return Units(target, in_widths, [width[r] for r in rets], [[("leaf", p) for p in range(len(args))]],
                 [[("ret", p) for p in range(len(rets))]], sig, gates, args, rets)


#: tensor-core k-step primitive -> fp.tc_dot16's (groups, W, F): verity.ml.circuits.silicon.models AMPERE_BF16_M16N8K16 / HOPPER_BF16_WGMMA_K16.
#: The Ampere step has two ids, the vLLM registry's _v1 and core's _v2 (verity.ml.circuits.ops.prims): one measured total function.
TC_STEPS = {"AmpereBF16TcDot16_v1": ((8, 8), 25, -132), "AmpereBF16TcDot16_v2": ((8, 8), 25, -132), "HopperBF16WgmmaDot16_v1": ((16,), 26, -133)}
#: the FP8 steps -> fp.tc_dot_e4m3's (groups, W, F): verity.ml.circuits.silicon.total_fp8 HOPPER_E4M3_WGMMA_K32 and models.BLACKWELL_SM120_E4M3_M16N8K32
#: (a GEMM's FP8 coordinate is walked, not unit-lowered)
TC_STEPS_FP8 = {"HopperE4m3QgmmaDot32_v1": ((32,), 14, -139), "BlackwellE4m3QmmaDot32_v1": ((32,), 26, -133)}


def tc_units(target, prim: str | None = None) -> Units:
    """``target`` as tensor-core k-steps: every ``AmpereBF16TcDot16`` gate is one unit, ports in operand order: its 16 ``b``
    operands from input leaves (``("leaf", p)``), or ``("zero", 0)`` where the IR pads with a constant zero word; its 16 ``a``
    operands and its accumulator as cut words; its result a cut word. Input leaves read as ``a`` operands (the query) are public
    words (:attr:`Units.public`); every other gate is the native tail, which computes the outputs (:attr:`Units.tail_rets`)."""
    _prog, call = standalone(target)
    args, rets = list(call.args), list(call.returns)
    gates = live_gates(call)
    by = {g.index: g for g in gates}
    arg_pos = {g: i for i, g in enumerate(args)}
    tcs = [g for g in gates if (g.prim.id == prim if prim else g.prim.id in TC_STEPS)]
    if not tcs:
        raise ValueError(f"{getattr(target, 'id', target)}: no {prim or ' or '.join(TC_STEPS)} gate")
    if len({g.prim.id for g in tcs}) > 1:
        raise ValueError("tensor-core steps of two ids in one target")
    prim = tcs[0].prim.id

    def zero_word(o):
        g = by.get(o)
        return g is not None and getattr(g.prim, "family", None) == "Const" and int(g.prim.evaluate()) == 0

    public_args: set[int] = set()
    cut_set: set[int] = set()
    for g in tcs:
        acc, a, b = g.operands[0], g.operands[1:17], g.operands[17:33]
        if any(o not in arg_pos and not zero_word(o) for o in b):
            raise ValueError(f"tensor-core gate {g.index}: a b operand that is neither an input leaf nor a zero word")
        for o in (acc, *a):
            (public_args if o in arg_pos else cut_set).add(o)
        cut_set.add(g.index)
    tc_idx = {g.index for g in tcs}
    tail = [g for g in gates if g.index not in tc_idx]
    if any(o in arg_pos for g in tail for o in g.operands):
        raise ValueError("a gate outside the tensor-core steps reads an input leaf")
    if any(o in arg_pos or o in tc_idx for o in rets):
        raise ValueError("an output leaf is not computed by the tail")
    K = sorted(public_args, key=arg_pos.__getitem__) + sorted(cut_set)
    kpos = {g: k for k, g in enumerate(K)}
    ins = [[("leaf", arg_pos[o]) if o in arg_pos else ("zero", 0) for o in g.operands[17:33]] + [("cut", kpos[o]) for o in g.operands[:17][1:]]
           + [("cut", kpos[g.operands[0]])] for g in tcs]
    outs = [[("cut", kpos[g.index])] for g in tcs]

    def builder() -> Circuit:
        C = Circuit(cse=True)
        b = [C.input(f"in{j}", 16) for j in range(16)]
        a = [C.input(f"in{16 + j}", 16) for j in range(16)]
        acc = C.input("in32", 32)
        groups, W, F = TC_STEPS[prim]
        C.output("out0", fp.tc_dot16(C, acc, a, b, groups=groups, W=W, F=F))
        return C

    return Units(target, [16] * 32 + [32], [32], ins, outs, (prim,), [tcs[0]], [], [], K, tail,
                 public=[(kpos[o], arg_pos[o]) for o in sorted(public_args, key=arg_pos.__getitem__)],
                 tail_rets=list(rets), builder=builder)


def _leaf_width(target, pos: int) -> int:
    from verity.ml.circuits.evaluation.reference import signature_of
    params, _ = signature_of(target)
    widths = [w for pt in params for w in pt.leaf_widths()]
    return widths[pos]


# --- the unit netlist ------------------------------------------------------------------------------------------------------

@dataclass
class Layout:
    """A unit circuit laid out as one Flock block type (C = I; row i is ``(A_i . z)(B_i . z) = z_i``). Input and output ports
    come in groups; each group's bits are consecutive in its own aligned power-of-two range of 128-bit words."""

    rows_a: list
    rows_b: list
    const: int
    in_groups: list[tuple[int, int, list[int]]]    # (first word column, words, port widths)
    out_groups: list[tuple[int, int, list[int]]]
    counts: dict

    @property
    def useful(self) -> int:
        return len(self.rows_a)

    @property
    def in_bits(self) -> list[int]:
        return [b for _, _, bits in self.in_groups for b in bits]

    @property
    def out_bits(self) -> list[int]:
        return [b for _, _, bits in self.out_groups for b in bits]

    @property
    def in_words(self) -> int:
        col, words, _ = self.in_groups[-1]
        return col + words

    @property
    def out_col(self) -> int:
        return self.out_groups[0][0]

    @property
    def out_words(self) -> int:
        col, words, _ = self.out_groups[-1]
        return col + words - self.out_col

    @property
    def out_cols(self) -> list[int]:
        return list(range(self.out_col, self.out_col + self.out_words))

    @staticmethod
    def _cols(groups) -> list[int]:
        out = []
        for col, _, bits in groups:
            at = col * WORD
            for b in bits:
                out.append(at)
                at += b
        return out

    @property
    def in_port_cols(self) -> list[int]:
        """Per input port, the column of its bit 0."""
        return self._cols(self.in_groups)

    @property
    def out_port_cols(self) -> list[int]:
        return self._cols(self.out_groups)

    def nonzeros(self) -> tuple[np.ndarray, np.ndarray]:
        """A₀'s and B₀'s nonzeros in the unit's own coordinates: each an ``(nnz, 2)`` int64 array of ``(row, column)``, rows
        ascending and each row's columns as laid out, ``counts["nnz_a"]`` and ``counts["nnz_b"]`` long. Every row and column is
        below ``useful``, so inside the unit's slot of ``2^unit_log`` cells, the block's ``A = I ⊗ A₀ + Δ``."""
        out = []
        for rows in (self.rows_a, self.rows_b):
            cols = [np.asarray(_cols_of(r), dtype=np.int64) for r in rows]
            at = np.repeat(np.arange(len(rows), dtype=np.int64), [len(c) for c in cols])
            out.append(np.stack([at, np.concatenate(cols)], axis=1))
        return out[0], out[1]


def _groups(names: Sequence[str], split: Sequence[int] | None) -> list[list[str]]:
    cuts = [0, *(split or []), len(names)]
    return [list(names[a:b]) for a, b in zip(cuts, cuts[1:]) if b > a]


def layout(C: Circuit, in_names: Sequence[str], out_names: Sequence[str], in_split: Sequence[int] | None = None,
           out_split: Sequence[int] | None = None) -> Layout:
    """``flock-ir-unit/v2``: the input port groups first (``in_split``: the port indices where a new group starts), each
    group's bits consecutive in a power-of-two number of 128-bit words aligned to that size, unused bits forced zero; the
    live ANDs and table reads in creation order (a read ``C.lookup`` as its plain rows, :class:`_Read`); one row
    ``(L + z_j) * 1 = z_j`` per assertion; then the output port groups as copy rows, laid
    out the same way; the constant-one column last. A forced-zero row has empty A and B. So each group is one aligned
    subcube of the unit's columns (one region claim)."""
    live = C.live()
    assert all(C.kind[v] in ("in", "and", "lookup") for v in live), "hints are not laid out yet"
    col: dict[int, int] = {}
    rows_a: list = []
    rows_b: list = []

    def zero():
        rows_a.append([])
        rows_b.append([])

    def open_group(nbits: int) -> tuple[int, int]:
        words = ceil_pow2(max(1, -(-nbits // WORD)))
        while len(rows_a) % (WORD * words):
            zero()
        return len(rows_a) // WORD, words

    def close_group(c: int, words: int) -> None:
        while len(rows_a) < (c + words) * WORD:
            zero()

    in_groups = []
    for g in _groups(in_names, in_split):
        bits = [len(C.inputs[nm]) for nm in g]
        c, words = open_group(sum(bits))
        for nm in g:
            for w in C.inputs[nm]:
                (v,) = tuple(w.s)
                col[v] = len(rows_a)
                rows_a.append([len(rows_a)])
                rows_b.append([len(rows_b)])
        close_group(c, words)
        in_groups.append((c, words, bits))
    ands = []
    reads: dict[int, _Read] = {}
    const_bit: dict[int, int] = {}
    value_bit = {C.data[v]: v for v in live if C.kind[v] == "lookup"}
    for v in range(len(C.kind)):
        if v not in live or C.kind[v] == "in":
            continue
        if C.kind[v] == "and":
            ands.append(v)
            col[v] = len(rows_a)
            rows_a.append(None)
            rows_b.append(None)
            continue
        g, j = C.data[v]
        if g not in reads:
            reads[g] = r = _Read(C, g, sorted(jj for (gg, jj) in value_bit if gg == g), len(rows_a))
            rows_a.extend([None] * r.rows)
            rows_b.extend([None] * r.rows)
        if j in reads[g].const_bits:
            const_bit[v] = reads[g].const_bits[j]
        else:
            col[v] = reads[g].out_col[j]

    def form(Lf: L):
        s = [col[v] for v in Lf.s if v not in const_bit]
        if (Lf.c + sum(const_bit.get(v, 0) for v in Lf.s)) & 1:
            s.append("C")
        return s

    for r in reads.values():
        r.fill(rows_a, rows_b, form)
    for _, Lf in C.asserts:
        j = len(rows_a)
        rows_a.append(form(Lf) + [j])
        rows_b.append(["C"])
    out_groups = []
    for g in _groups(out_names, out_split):
        bits = [len(C.outputs[nm]) for nm in g]
        c, words = open_group(sum(bits))
        for nm in g:
            for w in C.outputs[nm]:
                rows_a.append(form(w))
                rows_b.append(["C"])
        close_group(c, words)
        out_groups.append((c, words, bits))
    const = len(rows_a)
    rows_a.append([const])
    rows_b.append([const])

    def fix(r):
        if r is None or isinstance(r, np.ndarray):
            return r
        par: dict[int, int] = {}
        for c in r:
            c = const if c == "C" else c
            par[c] = par.get(c, 0) ^ 1
        return sorted(c for c, p in par.items() if p)

    for rows in (rows_a, rows_b):
        for i, r in enumerate(rows):
            rows[i] = fix(r)
    at, wired = col.__getitem__, frozenset(const_bit)

    def and_row(Lf: L) -> list:
        # fix(form(Lf)): over no constant-wired value bit, the form's columns are distinct (col is one-to-one) and below const
        if wired and not wired.isdisjoint(Lf.s):
            return fix(form(Lf))
        s = sorted(map(at, Lf.s))
        if Lf.c & 1:
            s.append(const)
        return s

    for v in ands:
        a, b = C.data[v]
        rows_a[col[v]], rows_b[col[v]] = and_row(a), and_row(b)
    lay = Layout(rows_a, rows_b, const, in_groups, out_groups, {})
    read_rows = sum(r.rows for r in reads.values())
    lay.counts = {"ands": len(ands) + read_rows, "asserts": len(C.asserts), "rows": len(rows_a), "in_words": lay.in_words, "out_words": lay.out_words,
                  "nnz_a": sum(map(len, rows_a)), "nnz_b": sum(map(len, rows_b)), "reads": len(reads), "read_rows": read_rows,
                  "read_nnz_a": sum(r.nnz_a for r in reads.values())}
    return lay


#: a read's low index bits by index width, for the measured MUFU tables' widths (Project doc `compile-time-constants.md`
#: §2.4: 14 low bits for both, once a library table's constant bits are wired)
READ_LO = {24: 14, 23: 14}


def _decode_count(n: int) -> int:
    return 0 if n <= 1 else (1 << n) + _decode_count(n // 2) + _decode_count(n - n // 2)


@lru_cache(maxsize=None)
def _library_table(name: str) -> bool:
    """Whether the table ``name`` loads is on core's library list, by the SHA-512 of its raw little-endian words."""
    from verity.ml.circuits import library

    from . import gf2

    raw = np.ascontiguousarray(np.asarray(gf2.table(name), dtype="<u4")).tobytes()
    return library.table_label(hashlib.sha512(raw).hexdigest()) == library.LIBRARY


class _Read:
    """A table read ``out = table[idx]`` laid out inline as plain rows of the unit (the one-hot construction of
    :func:`fp.lookup`): the low and high index halves decode to one-hot minterms (each an AND of two half-minterms, recursively),
    row ``(j, h)`` is ``hi_h AND (XOR of the low minterms l with bit j of table[h * 2^lo + l] set)`` (omitted when that XOR is
    empty), and value bit j is the copy row ``(XOR over h of row (j, h)) AND 1``. Only the live value bits get rows, and a
    library table's bits that are constant over the whole table get none: they are wired to the constant (``const_bits``).
    A program table keeps a row for every live bit, so its layout shows nothing of its content. Every row reads earlier rows,
    the index forms or the constant, so the index fixes every row."""

    def __init__(self, C: Circuit, g: int, bits: Sequence[int], at: int):
        from . import gf2

        self.table, self.idx, _width = C.lookups[g]
        n = len(self.idx)
        words = np.asarray(gf2.table(self.table), dtype=np.uint32)
        assert len(words) == 1 << n
        self.const_bits: dict[int, int] = {}
        if _library_table(self.table):
            for j in bits:
                col = (words >> np.uint32(j)) & np.uint32(1)
                if not col.any():
                    self.const_bits[j] = 0
                elif col.all():
                    self.const_bits[j] = 1
        bits = [j for j in bits if j not in self.const_bits]
        self.lo = READ_LO.get(n) or min(range(2, n - 1), key=lambda lb: (1 << lb) + (1 << (n - lb)) * (len(bits) + 1))
        self.hi = n - self.lo
        assert self.lo >= 2 and self.hi >= 2, "a read's index halves are decoded to rows"
        self.bits = list(bits)
        self.at = at
        self.lo_first = at + _decode_count(self.lo) - (1 << self.lo)
        self.hi_first = at + _decode_count(self.lo) + _decode_count(self.hi) - (1 << self.hi)
        p = at + _decode_count(self.lo) + _decode_count(self.hi)
        seg = words.reshape(1 << self.hi, 1 << self.lo)
        self.products: list[list[tuple[int, np.ndarray]]] = []
        self.nnz_a = 0
        for j in self.bits:
            mask = ((seg >> np.uint32(j)) & np.uint32(1)).astype(bool)
            per = []
            for h in range(1 << self.hi):
                ls = np.flatnonzero(mask[h])
                if len(ls):
                    per.append((h, (self.lo_first + ls).astype(np.int32)))
                    self.nnz_a += len(ls)
            self.products.append(per)
        n_prod = sum(len(per) for per in self.products)
        self.out_col = {j: p + n_prod + t for t, j in enumerate(self.bits)}
        self.rows = _decode_count(self.lo) + _decode_count(self.hi) + n_prod + len(self.bits) if self.bits else 0

    def fill(self, rows_a: list, rows_b: list, form: Callable) -> None:
        if not self.bits:
            return
        at = [self.at]

        def emit(a, b) -> int:
            c = at[0]
            rows_a[c], rows_b[c] = a, b
            at[0] += 1
            return c

        def decode(fs: list) -> list:
            if len(fs) == 1:
                return [fs[0] + ["C"], fs[0]]
            h = len(fs) // 2
            lo, hi = decode(fs[:h]), decode(fs[h:])
            return [[emit(list(lo[i & ((1 << h) - 1)]), list(hi[i >> h]))] for i in range(1 << len(fs))]

        idx = [form(f) for f in self.idx]
        lo_m = decode(idx[:self.lo])
        hi_m = decode(idx[self.lo:])
        assert lo_m[0] == [self.lo_first] and hi_m[0] == [self.hi_first]
        prod_cols = []
        for per in self.products:
            prod_cols.append([emit(a, [self.hi_first + h]) for h, a in per])
        for j, cols in zip(self.bits, prod_cols):
            assert emit(cols, ["C"]) == self.out_col[j]
        assert at[0] == self.at + self.rows


def netlist(lay: Layout, name: str, cut: dict | None = None, leaves: dict | None = None) -> str:
    """The layout as ``flock-ir-unit/v2`` text: a header line ``FORMAT name rows const n_in_groups (col words n bits..)..
    n_out_groups (col words n bits..)..``, then one line per row ``|A| A.. |B| B..``, then ``LEAVES <canonical JSON>``
    (:attr:`Lowering.leaves`) and, for a cut lowering, ``CUT <canonical JSON>`` (:attr:`Lowering.cut`): the netlist's sha256
    pins the units' leaf maps and the cut structure."""
    head = [FORMAT, name, lay.useful, lay.const]
    for groups in (lay.in_groups, lay.out_groups):
        head.append(len(groups))
        for c, words, bits in groups:
            head += [c, words, len(bits), *bits]
    # rows go in pieces of NETLIST_ROWS lines, and the text is joined once: a GLM-5.3 norm unit's is 8.6 GB
    out = [" ".join(map(str, head)) + "\n"]
    for k in range(0, lay.useful, NETLIST_ROWS):
        out.append("\n".join([" ".join(map(str, [len(a), *_cols_of(a), len(b), *_cols_of(b)]))
                              for a, b in zip(lay.rows_a[k:k + NETLIST_ROWS], lay.rows_b[k:k + NETLIST_ROWS])]) + "\n")
    out.append(netlist_pins(cut, leaves))
    return "".join(out)


def netlist_pins(cut: dict | None, leaves: dict | None) -> str:
    """The ``LEAVES`` and ``CUT`` lines :func:`netlist` ends with: ``netlist(lay, name, cut, leaves)`` is
    ``netlist(lay, name) + netlist_pins(cut, leaves)``."""
    pins = "" if leaves is None else "LEAVES " + json.dumps(leaves, sort_keys=True, separators=(",", ":")) + "\n"
    return pins + ("" if cut is None else "CUT " + json.dumps(cut, sort_keys=True, separators=(",", ":")) + "\n")


def _cols_of(r) -> list:
    return r.tolist() if isinstance(r, np.ndarray) else r


def _sha256(text: str) -> hashlib._Hash:
    """SHA-256 of ``text``'s UTF-8, encoded :data:`HASH_CHUNK` characters at a time."""
    h = hashlib.sha256()
    for k in range(0, len(text), HASH_CHUNK):
        h.update(text[k:k + HASH_CHUNK].encode())
    return h


def digest(text: str) -> str:
    return _sha256(text).hexdigest()


def _flat_ports(t) -> list[list[int]]:
    """``[leaves, bits]`` per port of a type: a Value or (nested) Array of Values is one port; a Tuple, one per field."""
    from verity.ml.circuits.types import Array, Tuple, Value

    if isinstance(t, Tuple):
        return [x for f in t.fields for x in _flat_ports(f)]
    n = 1
    while isinstance(t, Array):
        n, t = n * t.n, t.elem
    if not isinstance(t, Value):
        raise ValueError(f"a port of type {t} is not an array of words")
    return [[n, t.w]]


def slice_words(words, nbits: int) -> list[int]:
    """Lane-major words -> bit-sliced ints: bit t of result[b] is bit b of words[t]."""
    w = np.asarray(words, dtype=np.uint64)
    out = []
    for b in range(nbits):
        bits = ((w >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
        out.append(int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little"))
    return out


def unslice_words(sliced: Sequence[int], T: int) -> np.ndarray:
    """Bit-sliced ints -> lane-major words (the inverse of :func:`slice_words`)."""
    nbytes = (T + 7) // 8
    out = np.zeros(T, dtype=np.uint64)
    for b, v in enumerate(sliced):
        bits = np.unpackbits(np.frombuffer(v.to_bytes(nbytes, "little"), dtype=np.uint8), bitorder="little")[:T]
        out |= bits.astype(np.uint64) << np.uint64(b)
    return out


def evaluate_layout(lay: Layout, ports: Sequence[Sequence[int]], T: int, flip: tuple[int, int] | None = None):
    """Run the laid-out rows bit-sliced over T lanes: ``ports[j]`` is input port j's words, one per lane. Returns
    ``(output words per port (lane-major arrays), per-lane mask of lanes whose every row is satisfied)``. ``flip = (row,
    lane)`` flips that committed bit after witness generation (a negative: a cheating prover's witness)."""
    mask = (1 << T) - 1
    z = [0] * lay.useful
    for at, width, words in zip(lay.in_port_cols, lay.in_bits, ports):
        z[at:at + width] = slice_words(words, width)
    z[lay.const] = mask

    def ev(r):
        s = 0
        for k in _cols_of(r):
            s ^= z[k]
        return s

    first_and = lay.in_words * WORD
    for i in range(first_and, lay.useful):
        if i != lay.const:
            z[i] = ev(lay.rows_a[i]) & ev(lay.rows_b[i])
    if flip is not None:
        z[flip[0]] ^= 1 << flip[1]
    ok = mask
    for i in range(lay.useful):
        ok &= ~((ev(lay.rows_a[i]) & ev(lay.rows_b[i])) ^ z[i]) & mask
    outs = [unslice_words(z[at:at + nb], T) for at, nb in zip(lay.out_port_cols, lay.out_bits)]
    return outs, ok


# --- a lowered template ------------------------------------------------------------------------------------------------------

@dataclass
class Lowering:
    """A template instance lowered to identical units: the IR Definition, its :class:`Units`, the unit circuit (None once laid
    out, as `class_statement.unit_lowering` drops it) and its ``flock-ir-unit/v2`` layout. ``evaluate`` runs the laid-out unit
    rows (what Flock proves) over every unit of every instance and assembles the instances' outputs and cut words."""

    name: str
    definition: object
    units: Units
    circuit: Circuit
    layout: Layout

    @property
    def domain(self) -> str:
        """The operand domain the statement covers (vocab ``domain``): ``total`` when the unit asserts nothing, so every input
        encoding is proven to give the IR's own output, else ``finite-only``."""
        asserts = self.layout.counts["asserts"] if self.circuit is None else len(self.circuit.asserts)
        return "finite-only" if asserts else "total"

    @property
    def cut(self) -> dict | None:
        """The cut structure, ``None`` without a cut: ``words`` (the cut word count), ``tail`` (:func:`tail_program`, its
        constants included) and, per instance-local unit, ``in`` / ``out`` its cut ports as ``[column, cut word]`` (input
        columns; output columns from the first output word); for a template whose units return no output leaf (tensor-core
        steps), ``public`` the cut words that are input leaves as ``[cut word, leaf]`` and ``outputs`` each output leaf's tail
        slot."""
        U, lay = self.units, self.layout
        if not U.cut:
            return None
        prog, slot = _tail(self)
        cut = {"words": len(U.cut), "tail": prog,
               "in": [[[lay.in_port_cols[j], p] for j, (k, p) in enumerate(src) if k == "cut"] for src in U.in_src],
               "out": [[[lay.out_port_cols[j] - lay.out_col * WORD, p] for j, (k, p) in enumerate(src) if k == "cut"]
                       for src in U.out_src]}
        if U.tail_rets:
            cut |= {"public": [list(x) for x in U.public], "outputs": [slot[g] for g in U.tail_rets]}
        return cut

    @property
    def leaves(self) -> dict:
        """The units' leaf maps: per instance-local unit, ``in`` its leaf input ports' and ``out`` its returned-output ports'
        instance leaves (flat positions in parameter / return order), and ``in_ports`` / ``out_ports`` each parameter's and
        return's ``[leaves, bits]``: what places every unit on its instance's rows and outputs."""
        params, ret = self.definition.signature()
        U = self.units
        return {"in": [[p if k == "leaf" else -1 for k, p in s if k in ("leaf", "zero")] for s in U.in_src],
                "out": [[p for k, p in s if k == "ret"] for s in U.out_src],
                "in_ports": [x for _, t in params for x in _flat_ports(t)], "out_ports": _flat_ports(ret)}

    @property
    def text(self) -> str:
        return netlist(self.layout, self.name, self.cut, self.leaves)

    @cached_property
    def sha256(self) -> str:
        """:func:`digest` of :attr:`text`, rendered on first read only: nothing changes a lowering once it is built."""
        return self.sha256_of(netlist(self.layout, self.name))

    def sha256_of(self, expanded: str) -> str:
        """:attr:`sha256` from ``expanded``, the unit's ``netlist(layout, name)`` (:attr:`text` without its LEAVES and CUT
        lines), for a caller that renders it anyway (`circuit.compose`'s unit section): the netlist is rendered once."""
        h = _sha256(expanded)
        h.update(netlist_pins(self.cut, self.leaves).encode())
        return h.hexdigest()

    def _index(self, src: list[list[tuple]], n_leaf: int) -> np.ndarray:
        """``(U, ports)`` column indices into ``[leaves | cut words | 0]`` (a zero leaf reads the last, all-zero column)."""
        zero = n_leaf + len(self.units.cut)
        return np.asarray([[p if k in ("leaf", "ret") else zero if k == "zero" else n_leaf + p for k, p in u] for u in src], dtype=np.int64)

    def unit_inputs(self, flat_inputs: np.ndarray, cuts: np.ndarray | None = None) -> list[np.ndarray]:
        """Per unit port, the lane-major words of every (instance, unit) lane, instance-major: ``flat_inputs[i]`` is instance i's
        input leaves in parameter order, ``cuts[i]`` its cut words (``cut_words``)."""
        flat_inputs = np.asarray(flat_inputs, dtype=np.uint64)
        n = flat_inputs.shape[0]
        cuts = np.zeros((n, len(self.units.cut)), dtype=np.uint64) if cuts is None else np.asarray(cuts, dtype=np.uint64)
        both = np.concatenate([flat_inputs, cuts, np.zeros((n, 1), dtype=np.uint64)], axis=1)
        idx = self._index(self.units.in_src, flat_inputs.shape[1])
        lanes = both[:, idx]                                              # (n, U, ports)
        return [lanes[:, :, j].reshape(-1) for j in range(idx.shape[1])]

    def unit_outputs(self, flat_outputs: np.ndarray, cuts: np.ndarray | None = None) -> list[np.ndarray]:
        flat_outputs = np.asarray(flat_outputs, dtype=np.uint64)
        both = flat_outputs if cuts is None else np.concatenate([flat_outputs, np.asarray(cuts, dtype=np.uint64)], axis=1)
        idx = self._index(self.units.out_src, flat_outputs.shape[1])
        lanes = both[:, idx]
        return [lanes[:, :, j].reshape(-1) for j in range(idx.shape[1])]

    def evaluate(self, flat_inputs: np.ndarray, cuts: np.ndarray | None = None, flip: tuple[int, int] | None = None):
        """``(outputs (n, output leaves), cut words the units computed (n, len(cut)), lane mask of satisfied units)`` from the
        laid-out rows; cut words no unit computes stay as given in ``cuts``."""
        flat_inputs = np.asarray(flat_inputs, dtype=np.uint64)
        n = flat_inputs.shape[0]
        U = self.units.n
        outs, ok = evaluate_layout(self.layout, self.unit_inputs(flat_inputs, cuts), n * U, flip)
        n_ret = 1 + max((p for u in self.units.out_src for k, p in u if k == "ret"), default=-1)
        res = np.zeros((n, n_ret), dtype=np.uint64)
        cut_out = np.zeros((n, len(self.units.cut)), dtype=np.uint64) if cuts is None else np.array(cuts, dtype=np.uint64)
        for j, o in enumerate(outs):
            o = o.reshape(n, U)
            for u in range(U):
                k, p = self.units.out_src[u][j]
                (res if k == "ret" else cut_out)[:, p] = o[:, u]
        return res, cut_out, ok


def lowering(definition, name: str, cut: Callable | None = None, group: int = 1, unit_fn: Callable | None = None) -> Lowering:
    """The template's units (``units``, or ``unit_fn(definition)``), their circuit and layout; cut words form their own port
    groups (after the leaf inputs and after the returned outputs), so private inputs, public outputs and cut words are separate
    aligned regions."""
    U = unit_fn(definition) if unit_fn else units(definition, cut, group)
    C = U.circuit()
    n_leaf = sum(1 for k, _ in U.in_src[0] if k in ("leaf", "zero"))
    n_ret = sum(1 for k, _ in U.out_src[0] if k == "ret")
    lay = layout(C, [f"in{j}" for j in range(len(U.in_widths))], [f"out{j}" for j in range(len(U.out_widths))],
                 [n_leaf] if n_leaf < len(U.in_widths) else None, [n_ret] if n_ret < len(U.out_widths) else None)
    return Lowering(name, definition, U, C, lay)


def cut_words(low: Lowering, flat_inputs: np.ndarray) -> np.ndarray:
    """``(n, len(cut))``: every instance's cut words by the IR reference evaluator (the verifier's own computation; the
    native tail's values included)."""
    from verity.ml.circuits.evaluate import evaluate_call

    prog, call = standalone(low.definition)
    out = np.zeros((len(flat_inputs), len(low.units.cut)), dtype=np.uint64)
    for i, row in enumerate(np.asarray(flat_inputs, dtype=np.uint64)):
        t: dict[int, int] = {}
        evaluate_call(prog.circuit, call, [int(v) for v in row], t)
        out[i] = [t[g] for g in low.units.cut]
    return out


# --- the native tail, as the verifier evaluates it --------------------------------------------------------------------------

#: The measured MUFU tables the tail primitives read, as raw u32 LE files (``write_tables``): their sha256s, pinned in the Rust
#: verifier too (``live/src/ir_tail.rs``); they are ``verity_vllm``'s ``rms_relation.tables("cuda")``, whose xz files that
#: module pins.
TABLES = {"rsq": "218fc04d271773269367dedf4cff9ff719081b1218c3d67d248d8d6126040246",
          "sqrt": "4e017678de306e0be8bed66193c2ebf09e9c60a9600cad56cd05bb7b6c3f266f",
          "rcp": "c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55",
          "ex2": "b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0"}
#: The primitives the Rust tail interpreter implements.
TAIL_PRIMS = ("F32Add_v1", "F32Mul_v1", "F32Fma_v1", "F32Div_v1", "RsqrtApprox_v1", "MufuSqrtFtz_v1", "DivFullRcp_v1", "DivFullScaleA_v1",
              "F32AddFtz_v1", "F32SubFtz_v1", "F32MulFtz_v1", "F32FmaFtz_v1", "F32FmaSubFtz_v1", "F32Max_v1", "GuardNegInfZero_v1",
              "MufuEx2Ftz_v1", "Fa2InvSum_v1", "F2fpBf16_v1")


def tail_program(low: Lowering) -> list[list]:
    """The native tail as ops ``[out slot, primitive id, operand slots...]`` in canonical order over slots ``[cut words |
    temporaries]`` (constants ``[slot, "Const", value]``): what the verifier evaluates on the units' cut words."""
    return _tail(low)[0]


def _tail(low: Lowering) -> tuple[list[list], dict[int, int]]:
    U = low.units
    slot = {g: k for k, g in enumerate(U.cut)}
    prog = []
    for g in U.tail:
        if g.index not in slot:
            slot[g.index] = len(slot)
        if getattr(g.prim, "family", None) == "Const":
            prog.append([slot[g.index], "Const", int(g.prim.evaluate())])
        elif g.prim.id in TAIL_PRIMS:
            prog.append([slot[g.index], g.prim.id, *[slot[o] for o in g.operands]])
        else:
            raise NoPiece(f"the native tail's primitive {g.prim.id} has no verifier implementation")
    return prog, slot


def write_tables(out_dir) -> dict:
    """The MUFU tables as raw u32 LE files (``rsq.u32``, ``sqrt.u32``, ``rcp.u32``, ``ex2.u32``) for the verifier; checks
    :data:`TABLES`. FA2's MUFU.RCP table (``fa2_relation.tables().rcp``) is ``rcp`` byte for byte."""
    from pathlib import Path

    from verity.ml.definitions.numerics import fa2_relation, rms_relation

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t, fa2 = rms_relation.tables("cuda"), fa2_relation.tables()
    for name, want in TABLES.items():
        b = np.ascontiguousarray(getattr(fa2 if name == "ex2" else t, name), dtype="<u4").tobytes()
        if hashlib.sha256(b).hexdigest() != want:
            raise ValueError(f"MUFU table {name}: sha256 differs from the pinned one")
        (out / f"{name}.u32").write_bytes(b)
    return dict(TABLES)


# --- instance files ----------------------------------------------------------------------------------------------------------

INSTANCES_FORMAT = "flock-ir-instances/v1"


def flat_ports(input_set, ports, lo: int, hi: int) -> np.ndarray:
    """``(hi - lo, leaves)``: each instance's words of ``ports`` (in order), concatenated."""
    return np.concatenate([np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint64).reshape(hi - lo, -1) for p in ports], axis=1)


def _pack_words(ports: Sequence[np.ndarray], widths: Sequence[int], words: int, cols: Sequence[int] | None = None) -> bytes:
    """Per lane, the ports' bits packed LSB first into ``words`` 128-bit little-endian words: port j from bit ``cols[j]``
    (default: consecutive)."""
    T = len(ports[0])
    buf = np.zeros((T, words * WORD), dtype=np.uint8)
    at = 0
    for j, (p, w) in enumerate(zip(ports, widths)):
        at = at if cols is None else cols[j]
        p = np.asarray(p, dtype=np.uint64)
        for b in range(w):
            buf[:, at + b] = ((p >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
        at += w
    return np.packbits(buf, axis=1, bitorder="little").tobytes()


def write_instances(low: Lowering, flat_in: np.ndarray, flat_out: np.ndarray, path, meta: dict, cuts: np.ndarray | None = None) -> dict:
    """``flock-ir-instances/v1``: one JSON header line, then every unit's input words (``in_words`` x 16 bytes, the netlist's
    input columns), then every unit's output words (``out_words`` x 16 bytes, the netlist's output columns); units are
    instance-major. Cut words (``cut_words``) are public words of the units that read or compute them; which ports hold them
    and the tail that relates them are the netlist's (:attr:`Lowering.cut`), never the header's."""
    n = flat_in.shape[0]
    lay, U = low.layout, low.units
    if U.cut and cuts is None:
        raise ValueError(f"{low.name}: the units read or compute cut words; pass cuts=cut_words(...)")
    head = {"format": INSTANCES_FORMAT, "unit": low.name, "unit_sha256": low.sha256, "instances": n, "units_per_instance": U.n,
            "units": n * U.n, "in_words": lay.in_words, "in_bits": lay.in_bits, "out_words": lay.out_words, "out_bits": lay.out_bits,
            "cut_words": len(U.cut), **meta}
    body = _pack_words(low.unit_inputs(flat_in, cuts), lay.in_bits, lay.in_words, lay.in_port_cols)
    body += _pack_words(low.unit_outputs(flat_out, cuts), lay.out_bits, lay.out_words, [c - lay.out_col * WORD for c in lay.out_port_cols])
    with open(path, "wb") as f:
        f.write(json.dumps(head, sort_keys=True).encode() + b"\n")
        f.write(body)
    return head


def check_staged(low: Lowering, input_set, path) -> list[str]:
    """What differs between the instance file at ``path`` and the verifier's own staging of the same range of ``input_set``
    (empty: byte-identical). A verifier proves against a file it staged itself, never a producer's."""
    import tempfile
    from pathlib import Path

    head = json.loads(Path(path).read_bytes().split(b"\n", 1)[0])
    lo, hi = head.get("range") or [0, head.get("instances", 0)]
    with tempfile.TemporaryDirectory() as d:
        mine = stage(low, input_set, lo, hi, d)
        if mine.read_bytes() == Path(path).read_bytes():
            return []
        a, b = mine.read_bytes().split(b"\n", 1), Path(path).read_bytes().split(b"\n", 1)
        return [f"{what} differs from the verifier's own staging of {input_set.name} [{lo}, {hi})"
                for what, x, y in (("header", a[0], b[0]), ("body", a[1], b[1])) if x != y]


def stage(low: Lowering, input_set, lo: int, hi: int, out_dir):
    """Write ids [lo, hi) of ``input_set`` as ``<out_dir>/inst-<n>.bin`` and the unit netlist as ``<out_dir>/net.txt``."""
    from pathlib import Path

    sub = input_set.subcircuit
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "net.txt").write_text(low.text)
    meta = {"set": input_set.name, "subcircuit": sub.id, "content_digest": input_set.content_digest, "range": [lo, hi]}
    path = out / f"inst-{hi - lo}.bin"
    fin = flat_ports(input_set, sub.inputs, lo, hi)
    cuts = cut_words(low, fin) if low.units.cut else None
    write_instances(low, fin, flat_ports(input_set, sub.outputs, lo, hi), path, meta, cuts)
    return path
