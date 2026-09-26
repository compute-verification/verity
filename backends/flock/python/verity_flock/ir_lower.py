"""C-Flock's generic IR lowering: an IR Definition becomes a Flock binary circuit by composing :mod:`fp` pieces.

:func:`lower` walks the Definition's gates in canonical order, as ``verity.ir.evaluate`` does, and maps each primitive to
its piece in :data:`PIECES` (a primitive without one is an error naming it). Constant gates, and gates whose operands are
all constant, are evaluated by the primitive's own evaluator and become constant wires.

:func:`units` splits a template instance into identical units read off its IR graph: the connected components of the gate
graph (with the input leaves) must all compute the same function of their inputs; the unit is one component, and the
arrangement lists which input and output leaves of the instance each unit takes. :func:`netlist` lays a unit circuit out as
``flock-ir-unit/v2`` (see its docstring), the format the block statement ``verity/flock-ir-block/v1`` reads.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from . import fp
from .fp import ceil_pow2
from .gf2 import Circuit, L, const_bits

FORMAT = "flock-ir-unit/v2"
WORD = 128
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
    from verity_vllm.program.registry import prims as P

    return tuple(P.f32_to_bf16_rn_bits(P._silu_f32_bits(P.bf16_to_f32_bits(g)), P.BF16_NAN_TORCH) for g in range(1 << 16))


def _silu_mul(C, g, u):
    s = fp.lookup(C, list(g), silu_table(), 16)
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(s), fp.bf16_to_f32(u)), NAN_TORCH)


def _bf16_add(nan):
    return lambda C, a, b: fp.f32_to_bf16(C, fp.f32_add(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)), nan)


def _f32_div(C, a, b):
    if not all(w.is_const() for w in b):
        raise NotImplementedError("F32Div by a non-constant divisor has no piece yet")
    return fp.f32_div_const(C, a, sum(w.c << i for i, w in enumerate(b)))


#: primitive id -> piece(C, *operand words) -> output word
PIECES: dict[str, Callable] = {
    "Bf16ToF32_v1": lambda C, a: fp.bf16_to_f32(a),
    "F2fpBf16_v1": lambda C, a: fp.f32_to_bf16(C, a, NAN_F2FP),
    "F32ToBf16Rn_v1": lambda C, a: fp.f32_to_bf16(C, a, NAN_TORCH),
    "F32Add_v1": fp.f32_add,
    "F32Mul_v1": fp.f32_mul,
    "F32Fma_v1": fp.f32_fma,
    "F32Div_v1": _f32_div,
    "Bf16Add_v1": _bf16_add(NAN_TORCH),
    "Bf16AddF2fp_v1": _bf16_add(NAN_F2FP),
    "RopeOut_v1": _rope_out,
    "RopeOutAdd_v1": _rope_out_add,
    "SiluMulBf16_v1": _silu_mul,
}


class NoPiece(NotImplementedError):
    """An IR primitive that C-Flock's lowering has no piece for."""


# --- the walk ----------------------------------------------------------------------------------------------------------

def _width(prim) -> int:
    return prim.ret.w


def _is_const(word) -> bool:
    return all(w.is_const() for w in word)


def _value(word) -> int:
    return sum(w.c << i for i, w in enumerate(word))


def lower_gates(C: Circuit, gates, wires: dict[int, list]) -> dict[int, list]:
    """Lower ``gates`` (``verity.ir.layout.Gate``s in canonical order) into C, reading operand words from ``wires`` (gate index
    -> word) and writing each gate's output word there."""
    for g in gates:
        prim = g.prim
        fam = getattr(prim, "family", None)
        if fam == "Input":
            if g.index not in wires:
                raise KeyError(f"input gate {g.index} has no wires")
            continue
        ops = [wires[o] for o in g.operands]
        if fam == "Const" or all(_is_const(o) for o in ops):
            wires[g.index] = const_bits(prim.evaluate(*[_value(o) for o in ops]), _width(prim))
            continue
        piece = PIECES.get(prim.id)
        if piece is None:
            raise NoPiece(f"C-Flock has no circuit piece for the IR primitive {prim.id}")
        out = piece(C, *ops)
        if len(out) != _width(prim):
            raise AssertionError(f"piece for {prim.id} returned {len(out)} bits, the primitive returns {_width(prim)}")
        wires[g.index] = out
    return wires


def standalone(target):
    from verity.evaluation.reference import resolve, standalone_call

    return standalone_call(resolve(target))


def lower(C: Circuit, target, inputs: Sequence[list]) -> list[list]:
    """The output words of ``target`` (a primitive or a bound composite) on the input words ``inputs`` (one per input leaf)."""
    from verity.evaluation.reference import resolve
    from verity.ir.defs import PrimitiveDefinition

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

    @property
    def n(self) -> int:
        return len(self.in_src)

    def circuit(self) -> Circuit:
        """The unit circuit: inputs ``in0 .. in{k-1}``, outputs ``out0 ..`` (one word per port)."""
        C = Circuit()
        ins = [C.input(f"in{j}", w) for j, w in enumerate(self.in_widths)]
        wires = dict(zip(self.arg_gates, ins))
        lower_gates(C, self.gates, wires)
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


def units(target, cut: Callable | None = None) -> Units:
    """Split ``target``'s one-call Program (its live gates) into the connected components of its gate graph, input leaves
    included and edges out of cut gates removed (``cut(gates, args, returns)`` returns the cut gate indices; default none).
    Every component that reads an input leaf is a unit and must have the same signature (primitive ids and operand wiring
    relative to the component, cut words as ports), else ValueError; the others form the native tail."""
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


def _leaf_width(target, pos: int) -> int:
    from verity.evaluation.reference import signature_of
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


def _groups(names: Sequence[str], split: Sequence[int] | None) -> list[list[str]]:
    cuts = [0, *(split or []), len(names)]
    return [list(names[a:b]) for a, b in zip(cuts, cuts[1:]) if b > a]


def layout(C: Circuit, in_names: Sequence[str], out_names: Sequence[str], in_split: Sequence[int] | None = None,
           out_split: Sequence[int] | None = None) -> Layout:
    """``flock-ir-unit/v2``: the input port groups first (``in_split``: the port indices where a new group starts), each
    group's bits consecutive in a power-of-two number of 128-bit words aligned to that size, unused bits forced zero; the
    live ANDs in creation order; one row ``(L + z_j) * 1 = z_j`` per assertion; then the output port groups as copy rows, laid
    out the same way; the constant-one column last. A forced-zero row has empty A and B. So each group is one aligned
    subcube of the unit's columns (one region claim)."""
    live = C.live()
    assert all(C.kind[v] in ("in", "and") for v in live), "hints are not laid out yet"
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
    ands = [v for v in range(len(C.kind)) if v in live and C.kind[v] == "and"]
    for v in ands:
        col[v] = len(rows_a)
        rows_a.append(None)
        rows_b.append(None)

    def form(Lf: L):
        s = [col[v] for v in Lf.s]
        if Lf.c:
            s.append("C")
        return s

    for v in ands:
        a, b = C.data[v]
        rows_a[col[v]], rows_b[col[v]] = form(a), form(b)
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
        par: dict[int, int] = {}
        for c in r:
            c = const if c == "C" else c
            par[c] = par.get(c, 0) ^ 1
        return sorted(c for c, p in par.items() if p)

    rows_a = [fix(r) for r in rows_a]
    rows_b = [fix(r) for r in rows_b]
    lay = Layout(rows_a, rows_b, const, in_groups, out_groups, {})
    lay.counts = {"ands": len(ands), "asserts": len(C.asserts), "rows": len(rows_a), "in_words": lay.in_words, "out_words": lay.out_words,
                  "nnz_a": sum(map(len, rows_a)), "nnz_b": sum(map(len, rows_b))}
    return lay


def netlist(lay: Layout, name: str) -> str:
    """The layout as ``flock-ir-unit/v2`` text: a header line ``FORMAT name rows const n_in_groups (col words n bits..)..
    n_out_groups (col words n bits..)..``, then one line per row ``|A| A.. |B| B..``."""
    head = [FORMAT, name, lay.useful, lay.const]
    for groups in (lay.in_groups, lay.out_groups):
        head.append(len(groups))
        for c, words, bits in groups:
            head += [c, words, len(bits), *bits]
    lines = [" ".join(map(str, head))]
    for a, b in zip(lay.rows_a, lay.rows_b):
        lines.append(" ".join(map(str, [len(a), *a, len(b), *b])))
    return "\n".join(lines) + "\n"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
        for k in r:
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
    """A template instance lowered to identical units: the IR Definition, its :class:`Units`, the unit circuit and its
    ``flock-ir-unit/v2`` layout. ``evaluate`` runs the laid-out unit rows (what Flock proves) over every unit of every
    instance and assembles the instances' outputs and cut words."""

    name: str
    definition: object
    units: Units
    circuit: Circuit
    layout: Layout

    @property
    def text(self) -> str:
        return netlist(self.layout, self.name)

    @property
    def sha256(self) -> str:
        return digest(self.text)

    def _index(self, src: list[list[tuple]], n_leaf: int) -> np.ndarray:
        """``(U, ports)`` column indices into ``[leaves | cut words]``."""
        return np.asarray([[p if k in ("leaf", "ret") else n_leaf + p for k, p in u] for u in src], dtype=np.int64)

    def unit_inputs(self, flat_inputs: np.ndarray, cuts: np.ndarray | None = None) -> list[np.ndarray]:
        """Per unit port, the lane-major words of every (instance, unit) lane, instance-major: ``flat_inputs[i]`` is instance i's
        input leaves in parameter order, ``cuts[i]`` its cut words (``cut_words``)."""
        flat_inputs = np.asarray(flat_inputs, dtype=np.uint64)
        both = flat_inputs if cuts is None else np.concatenate([flat_inputs, np.asarray(cuts, dtype=np.uint64)], axis=1)
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
        n_ret = 1 + max(p for u in self.units.out_src for k, p in u if k == "ret")
        res = np.zeros((n, n_ret), dtype=np.uint64)
        cut_out = np.zeros((n, len(self.units.cut)), dtype=np.uint64) if cuts is None else np.array(cuts, dtype=np.uint64)
        for j, o in enumerate(outs):
            o = o.reshape(n, U)
            for u in range(U):
                k, p = self.units.out_src[u][j]
                (res if k == "ret" else cut_out)[:, p] = o[:, u]
        return res, cut_out, ok


def lowering(definition, name: str, cut: Callable | None = None) -> Lowering:
    """The template's units, their circuit and layout; cut words form their own port groups (after the leaf inputs and after
    the returned outputs), so private inputs, public outputs and cut words are separate aligned regions."""
    U = units(definition, cut)
    C = U.circuit()
    n_leaf = sum(1 for k, _ in U.in_src[0] if k == "leaf")
    n_ret = sum(1 for k, _ in U.out_src[0] if k == "ret")
    lay = layout(C, [f"in{j}" for j in range(len(U.in_widths))], [f"out{j}" for j in range(len(U.out_widths))],
                 [n_leaf] if n_leaf < len(U.in_widths) else None, [n_ret] if n_ret < len(U.out_widths) else None)
    return Lowering(name, definition, U, C, lay)


def cut_words(low: Lowering, flat_inputs: np.ndarray) -> np.ndarray:
    """``(n, len(cut))``: every instance's cut words by the IR reference evaluator (the verifier's own computation; the
    native tail's values included)."""
    from verity.ir.evaluate import evaluate_call

    prog, call = standalone(low.definition)
    out = np.zeros((len(flat_inputs), len(low.units.cut)), dtype=np.uint64)
    for i, row in enumerate(np.asarray(flat_inputs, dtype=np.uint64)):
        t: dict[int, int] = {}
        evaluate_call(prog.circuit, call, [int(v) for v in row], t)
        out[i] = [t[g] for g in low.units.cut]
    return out


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
    instance-major. Cut words (``cut_words``) are public words of the units that read or compute them."""
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
