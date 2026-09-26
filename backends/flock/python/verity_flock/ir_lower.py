"""C-Flock's generic IR lowering: an IR Definition becomes a Flock binary circuit by composing :mod:`fp` pieces.

:func:`lower` walks the Definition's gates in canonical order, as ``verity.ir.evaluate`` does, and maps each primitive to
its piece in :data:`PIECES` (a primitive without one is an error naming it). Constant gates, and gates whose operands are
all constant, are evaluated by the primitive's own evaluator and become constant wires.

:func:`units` splits a template instance into identical units read off its IR graph: the connected components of the gate
graph (with the input leaves) must all compute the same function of their inputs; the unit is one component, and the
arrangement lists which input and output leaves of the instance each unit takes. :func:`netlist` lays a unit circuit out as
``flock-ir-unit/v1`` (see its docstring), the format the block statement ``verity/flock-ir-block/v1`` reads.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from . import fp
from .gf2 import Circuit, L, const_bits

FORMAT = "flock-ir-unit/v1"
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
    """A template instance as identical units: ``gates`` is one unit's gate list (IR gates of the first component) and
    ``signature`` its shape; ``inputs[u]`` / ``outputs[u]`` are the instance's input / output leaf positions unit u takes, in
    the unit's port order."""

    target: object
    in_widths: list[int]
    out_widths: list[int]
    inputs: list[list[int]]
    outputs: list[list[int]]
    signature: tuple = field(repr=False)
    gates: list = field(repr=False)
    arg_gates: list[int] = field(repr=False)

    @property
    def n(self) -> int:
        return len(self.inputs)

    def circuit(self) -> Circuit:
        """The unit circuit: inputs ``in0 .. in{k-1}``, outputs ``out0 ..`` (one word per port)."""
        C = Circuit()
        ins = [C.input(f"in{j}", w) for j, w in enumerate(self.in_widths)]
        wires = dict(zip(self.arg_gates, ins))
        lower_gates(C, self.gates, wires)
        for j, g in enumerate(self.ret_gates):
            C.output(f"out{j}", wires[g])
        return C

    ret_gates: list[int] = field(default_factory=list, repr=False)


def units(target) -> Units:
    """Split ``target``'s one-call Program into the connected components of its gate graph (input leaves included); every
    component must have the same signature (primitive ids and operand wiring relative to the component), else ValueError."""
    _prog, call = standalone(target)
    args = list(call.args)
    rets = list(call.returns)
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

    gates = list(call)
    for g in gates:
        find(g.index)
        for o in g.operands:
            parent[find(o)] = find(g.index)
    comps: dict[int, list] = {}
    for g in gates:
        comps.setdefault(find(g.index), []).append(g)
    groups = sorted(comps.values(), key=lambda gs: gs[0].index)
    sigs, ins, outs = [], [], []
    for gs in groups:
        local = {g.index: k for k, g in enumerate(gs)}
        leaves = sorted({o for g in gs for o in g.operands if o in arg_pos}, key=arg_pos.__getitem__)
        lpos = {o: j for j, o in enumerate(leaves)}
        sig = tuple((g.prim.id, tuple(("g", local[o]) if o in local else ("in", lpos[o]) for o in g.operands)) for g in gs)
        r = sorted((p, local[g.index]) for g in gs for p in ret_pos.get(g.index, []))
        sigs.append((sig, tuple(k for _, k in r)))
        ins.append([arg_pos[o] for o in leaves])
        outs.append([p for p, _ in r])
    if any(s != sigs[0] for s in sigs):
        raise ValueError(f"{getattr(target, 'id', target)}: its components are not all the same unit ({len(set(sigs))} shapes)")
    covered_in = sorted(p for u in ins for p in u)
    covered_out = sorted(p for u in outs for p in u)
    if covered_out != list(range(len(rets))):
        raise ValueError("the units do not cover every output leaf exactly once")
    if len(set(covered_in)) != len(covered_in):
        raise ValueError("two units share an input leaf")
    first = groups[0]
    leaves0 = [args[p] for p in ins[0]]
    width = {g.index: g.prim.ret.w for g in gates}
    in_widths = [_leaf_width(target, p) for p in ins[0]]
    ret_gates = [rets[p] for p in outs[0]]
    return Units(target, in_widths, [width[g] for g in ret_gates], ins, outs, sigs[0], first, leaves0, ret_gates)


def _leaf_width(target, pos: int) -> int:
    from verity.evaluation.reference import signature_of
    params, _ = signature_of(target)
    widths = [w for pt in params for w in pt.leaf_widths()]
    return widths[pos]


# --- the unit netlist ------------------------------------------------------------------------------------------------------

@dataclass
class Layout:
    """A unit circuit laid out as one Flock block type (C = I; row i is ``(A_i . z)(B_i . z) = z_i``)."""

    rows_a: list
    rows_b: list
    const: int
    in_words: int
    in_bits: list[int]      # per input port, its width; port j's bits are consecutive from column sum(in_bits[:j])
    out_cols: list[int]     # per output port, its 128-bit word column
    out_bits: list[int]
    counts: dict

    @property
    def useful(self) -> int:
        return len(self.rows_a)


def layout(C: Circuit, in_names: Sequence[str], out_names: Sequence[str]) -> Layout:
    """``flock-ir-unit/v1``: input bits first (port by port, packed into 128-bit words whose unused bits are forced zero), the
    live ANDs in creation order, one row ``(L + z_j) * 1 = z_j`` per assertion, then each output port as copy rows in a 128-bit
    word column of its own (unused rows forced zero), the constant-one column last. A forced-zero row has empty A and B."""
    live = C.live()
    assert all(C.kind[v] in ("in", "and") for v in live), "hints are not laid out yet"
    ins = [w for nm in in_names for w in C.inputs[nm]]
    col: dict[int, int] = {}
    rows_a: list = []
    rows_b: list = []
    for w in ins:
        (v,) = tuple(w.s)
        col[v] = len(rows_a)
        rows_a.append([len(rows_a)])
        rows_b.append([len(rows_b)])

    def zero():
        rows_a.append([])
        rows_b.append([])

    while len(rows_a) % WORD:
        zero()
    in_words = len(rows_a) // WORD
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
    out_cols, out_bits = [], []
    for nm in out_names:
        bits = C.outputs[nm]
        assert len(bits) <= WORD
        while len(rows_a) % WORD:
            zero()
        out_cols.append(len(rows_a) // WORD)
        out_bits.append(len(bits))
        for w in bits:
            rows_a.append(form(w))
            rows_b.append(["C"])
        for _ in range(WORD - len(bits)):
            zero()
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
    counts = {"ands": len(ands), "asserts": len(C.asserts), "rows": len(rows_a), "in_words": in_words, "out_words": len(out_cols),
              "nnz_a": sum(map(len, rows_a)), "nnz_b": sum(map(len, rows_b))}
    return Layout(rows_a, rows_b, const, in_words, [len(C.inputs[nm]) for nm in in_names], out_cols, out_bits, counts)


def netlist(lay: Layout, name: str) -> str:
    """The layout as ``flock-ir-unit/v1`` text: a header line ``FORMAT name rows const in_words n_in in_bits.. n_out
    (out_col out_bits)..``, then one line per row ``|A| A.. |B| B..``."""
    head = [FORMAT, name, lay.useful, lay.const, lay.in_words, len(lay.in_bits), *lay.in_bits, len(lay.out_cols)]
    for c, b in zip(lay.out_cols, lay.out_bits):
        head += [c, b]
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
    at = 0
    for j, width in enumerate(lay.in_bits):
        z[at:at + width] = slice_words(ports[j], width)
        at += width
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
    outs = [unslice_words(z[c * WORD:c * WORD + nb], T) for c, nb in zip(lay.out_cols, lay.out_bits)]
    return outs, ok


# --- a lowered template ------------------------------------------------------------------------------------------------------

@dataclass
class Lowering:
    """A template instance lowered to identical units: the IR Definition, its :class:`Units`, the unit circuit and its
    ``flock-ir-unit/v1`` layout. ``evaluate`` runs the laid-out unit rows (what Flock proves) over every unit of every
    instance and assembles the instances' outputs."""

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

    def unit_inputs(self, flat_inputs: np.ndarray) -> list[np.ndarray]:
        """``flat_inputs[i]`` = instance i's input leaves in parameter order -> per unit port, the lane-major words of every
        (instance, unit) lane, instance-major."""
        idx = np.asarray(self.units.inputs, dtype=np.int64)            # (U, ports)
        lanes = flat_inputs[:, idx]                                       # (n, U, ports)
        return [lanes[:, :, j].reshape(-1) for j in range(idx.shape[1])]

    def evaluate(self, flat_inputs: np.ndarray, flip: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
        """Instance outputs (``(n, output leaves)``) from the laid-out rows, and the lane mask of satisfied units."""
        flat_inputs = np.asarray(flat_inputs, dtype=np.uint64)
        n = flat_inputs.shape[0]
        U = self.units.n
        outs, ok = evaluate_layout(self.layout, self.unit_inputs(flat_inputs), n * U, flip)
        pos = np.asarray(self.units.outputs, dtype=np.int64)              # (U, out ports)
        res = np.zeros((n, int(pos.max()) + 1), dtype=np.uint64)
        for j, o in enumerate(outs):
            res[:, pos[:, j]] = o.reshape(n, U)
        return res, ok


def lowering(definition, name: str) -> Lowering:
    U = units(definition)
    C = U.circuit()
    lay = layout(C, [f"in{j}" for j in range(len(U.in_widths))], [f"out{j}" for j in range(len(U.out_widths))])
    return Lowering(name, definition, U, C, lay)


# --- instance files ----------------------------------------------------------------------------------------------------------

INSTANCES_FORMAT = "flock-ir-instances/v1"


def flat_ports(input_set, ports, lo: int, hi: int) -> np.ndarray:
    """``(hi - lo, leaves)``: each instance's words of ``ports`` (in order), concatenated."""
    return np.concatenate([np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint64).reshape(hi - lo, -1) for p in ports], axis=1)


def _pack_words(ports: Sequence[np.ndarray], widths: Sequence[int], words: int) -> bytes:
    """Per lane, the ports' bits packed LSB first (port j from bit sum(widths[:j])) into ``words`` 128-bit little-endian words."""
    T = len(ports[0])
    buf = np.zeros((T, words * WORD), dtype=np.uint8)
    at = 0
    for p, w in zip(ports, widths):
        p = np.asarray(p, dtype=np.uint64)
        for b in range(w):
            buf[:, at + b] = ((p >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
        at += w
    return np.packbits(buf, axis=1, bitorder="little").tobytes()


def write_instances(low: Lowering, flat_in: np.ndarray, flat_out: np.ndarray, path, meta: dict) -> dict:
    """``flock-ir-instances/v1``: one JSON header line, then every unit's input words (``in_words`` x 16 bytes each, the netlist's
    input columns), then every unit's output words (one 128-bit word per output port, the word in its low bits); units are
    instance-major."""
    n = flat_in.shape[0]
    lay, U = low.layout, low.units
    ins = low.unit_inputs(np.asarray(flat_in, dtype=np.uint64))
    pos = np.asarray(U.outputs, dtype=np.int64)
    outs = [np.asarray(flat_out, dtype=np.uint64)[:, pos[:, j]].reshape(-1) for j in range(pos.shape[1])]
    head = {"format": INSTANCES_FORMAT, "unit": low.name, "unit_sha256": low.sha256, "instances": n, "units_per_instance": U.n,
            "units": n * U.n, "in_words": lay.in_words, "in_bits": lay.in_bits, "out_words": len(lay.out_cols), "out_bits": lay.out_bits,
            **meta}
    body = _pack_words(ins, lay.in_bits, lay.in_words)
    for o, w in zip(outs, lay.out_bits):
        body += _pack_words([o], [w], 1)
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
    write_instances(low, flat_ports(input_set, sub.inputs, lo, hi), flat_ports(input_set, sub.outputs, lo, hi), path, meta)
    return path
