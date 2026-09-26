"""A template's tail in the circuit (lane flock-netlist, ``verity/flock-netlist/v1``): the verifier evaluates no part of the
computation, so the gates a cut lowering left to the verifier (``ir_lower.tail_program``: RMSNorm's row tail, attention's
softmax) become netlists here.

The tail program is split into **stages** at its MUFU operations. A MUFU operation is a table lookup inside exponent and
special-case logic (``live/src/ir_tail.rs``, the IR's own model): its stage computes the table index and passes the operand on
(``pre``); a **lookup slot** (a shared slot type per table, built from the pinned table by the statement, ``LOOKUP`` in the
composite netlist) maps the index to the table word; the next stage finishes it (``post``). Every value a later stage reads
is passed from stage to stage as a port. So each stage and each lookup is an ordinary slot of the block, and the wires
between them, the units and the cut words are copy constraints.

The pieces are exact on encodings: IEEE f32 ops through :mod:`fp` (whose NaN results are canonical, observable only through
the templates' final bf16 casts, as in the units), the MUFU ops bit for bit as ``ir_tail.rs`` computes them (NaN
``0x7FFFFFFF`` included).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import fp
from . import ir_lower as IL
from .gf2 import NOT, ONE, OR, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce, sub

BIG = 0x7E800000       # 2^126
TINY = 0x00800000      # 2^-126
QUARTER = 0x3E800000   # 0.25
TWO24 = 0x4B800000     # 2^24
ONE_F = 0x3F800000
QNAN_MUFU = 0x7FFFFFFF
INF = 0x7F800000

#: table name -> (index bits, the table's pinned sha256 is ir_lower.TABLES[name])
TABLE_BITS = {"rsq": 24, "sqrt": 24, "rcp": 23, "ex2": 23}
#: bits of a table word the MUFU pieces read: its exponent field and mantissa (bit 31 is zero in every measured table)
VALUE_BITS = 31


def _eq_const(C, bits, value: int):
    return and_reduce(C, [b if (value >> i) & 1 else NOT(b) for i, b in enumerate(bits)])


def _fields(C, x):
    s, e, m = x[31], list(x[23:31]), list(x[:23])
    e_max = _eq_const(C, e, 0xFF)
    e_zero = NOT(or_reduce(C, e))
    m_nz = or_reduce(C, m)
    return s, e, m, e_max, e_zero, m_nz


def _word(C, cases, default):
    """``cases`` = [(cond, word)] tried in order, else ``default`` (words as bit lists)."""
    out = list(default)
    for cond, w in reversed(cases):
        out = mux(C, cond, w, out)
    return out


def _abs_gt_big(C, x):
    """``|x| > 2^126`` as f32 compares it (false for NaN)."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    nan = C.AND(e_max, m_nz)
    ge254 = C.AND(e[7], C.AND(e[6], C.AND(e[5], C.AND(e[4], C.AND(e[3], C.AND(e[2], e[1]))))))
    is253 = _eq_const(C, e, 253)
    return C.AND(NOT(nan), OR(C, ge254, C.AND(is253, m_nz)))


def _scale(C, x, k: int):
    return fp.f32_mul(C, x, const_bits(k, 32))


# --- the MUFU pieces: pre (operand -> table index, context words), post (context, table word -> result) -------------------

def rsq_post(C, xb, v):
    s, e, m, e_max, e_zero, m_nz = _fields(C, xb)
    # re = exp(v) - (k - par) / 2, k = e - 127: (k - par) / 2 = floor(k / 2) = (e - 127) >> 1 (arithmetic), as 9-bit two's complement
    k = sub(C, e + [ZERO], const_bits(127, 9), 9)[0]
    half = k[1:] + [k[8]]
    re = sub(C, list(v[23:31]) + [ZERO], half, 9)[0]
    body = list(v[:23]) + re
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), mux(C, s, const_bits(QNAN_MUFU, 32), const_bits(0, 32)))),
                     (e_zero, const_bits(INF, 31) + [s]), (s, const_bits(QNAN_MUFU, 32))], body)


def sqrt_post(C, xb, v):
    s, e, m, e_max, e_zero, m_nz = _fields(C, xb)
    k = sub(C, e + [ZERO], const_bits(127, 9), 9)[0]
    half = k[1:] + [k[8]]
    re = add(C, list(v[23:31]) + [ZERO], half, width=9)
    body = list(v[:23]) + re
    return _word(C, [(e_max, mux(C, OR(C, m_nz, s), const_bits(QNAN_MUFU, 32), const_bits(INF, 32))),
                     (e_zero, [ZERO] * 31 + [s]), (s, const_bits(QNAN_MUFU, 32))], body)


def parity_index(C, xb):
    """``(par << 23) | mantissa``, par = (e - 127) & 1 = NOT e[0]: the rsq / sqrt tables' index."""
    return list(xb[:23]) + [NOT(xb[23])]


def rcp_post(C, xb, t):
    s, e, m, e_max, e_zero, m_nz = _fields(C, xb)
    # re = exp(t) - (e - 127) as a signed 10-bit value; re <= 0 -> signed zero, re >= 255 -> signed inf
    re, _ = sub(C, list(t[23:31]) + [ZERO, ZERO], sub(C, e + [ZERO, ZERO], const_bits(127, 10), 10)[0], 10)
    neg = re[9]
    zero = OR(C, neg, NOT(or_reduce(C, re)))
    big = C.AND(NOT(neg), OR(C, re[8], _eq_const(C, re[:8], 0xFF)))
    body = list(t[:23]) + re[:8] + [s]
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), [ZERO] * 31 + [s])), (e_zero, const_bits(INF, 31) + [s]),
                     (zero, [ZERO] * 31 + [s]), (big, const_bits(INF, 31) + [s])], body)


def _rsqrt_pre(C, args):
    x = args[0]
    p0 = NOT(_fields(C, x)[4])
    xs = mux(C, p0, x, _scale(C, x, TWO24))
    return "rsq", parity_index(C, xs), [x, xs]


def _rsqrt_post(C, ctx, v):
    x, xs = ctx
    r = rsq_post(C, xs, v)
    p0 = NOT(_fields(C, x)[4])
    rs, re_, rm, r_max, r_zero, r_mnz = _fields(C, r)
    special = OR(C, r_max, r_zero)
    scaled = list(r[:23]) + add(C, r[23:31], const_bits(12, 8), width=8) + [r[31]]
    return mux(C, OR(C, p0, special), r, scaled)


def _sqrt_pre(C, args):
    return "sqrt", parity_index(C, args[0]), [args[0]]


def _sqrt_post(C, ctx, v):
    return sqrt_post(C, ctx[0], v)


def _div_scale(C, a, b):
    """``div.full``'s range scaling of ``(a, b)``: |b| > 2^126 both * 0.25; |b| < 2^-126 both * 2^24 (after the first)."""
    big = _abs_gt_big(C, b)
    a1, b1 = mux(C, big, _scale(C, a, QUARTER), a), mux(C, big, _scale(C, b, QUARTER), b)
    tiny = _fields(C, b1)[4]
    return mux(C, tiny, _scale(C, a1, TWO24), a1), mux(C, tiny, _scale(C, b1, TWO24), b1)


def _divrcp_pre(C, args):
    b = args[0]
    big = _abs_gt_big(C, b)
    tiny = _fields(C, b)[4]
    fb = mux(C, big, _scale(C, b, QUARTER), mux(C, tiny, _scale(C, b, TWO24), b))
    a2, b2 = _div_scale(C, const_bits(ONE_F, 32), fb)
    return "rcp", list(b2[:23]), [a2, b2]


def _divrcp_post(C, ctx, t):
    a2, b2 = ctx
    return fp.f32_mul(C, a2, rcp_post(C, b2, t))


#: MUFU primitive -> (pre, post)
MUFU = {"RsqrtApprox_v1": (_rsqrt_pre, _rsqrt_post), "MufuSqrtFtz_v1": (_sqrt_pre, _sqrt_post), "DivFullRcp_v1": (_divrcp_pre, _divrcp_post)}


def _div_scale_a(C, a, b):
    big = _abs_gt_big(C, b)
    tiny = _fields(C, b)[4]
    return mux(C, big, _scale(C, a, QUARTER), mux(C, tiny, _scale(C, a, TWO24), a))


def _f32_div(C, a, b):
    if not all(w.is_const() for w in b):
        raise NotImplementedError("F32Div by a non-constant divisor has no piece yet")
    return fp.f32_div_const(C, a, sum(w.c << i for i, w in enumerate(b)))


#: every other tail primitive -> piece(C, *operand words)
OPS = {"F32Add_v1": fp.f32_add, "F32Mul_v1": fp.f32_mul, "F32Fma_v1": fp.f32_fma, "F32Div_v1": _f32_div, "DivFullScaleA_v1": _div_scale_a}


# --- stages ---------------------------------------------------------------------------------------------------------------

@dataclass
class Stage:
    """One stage netlist: its input ports (``("cut", k)`` a cut word, ``("pass", slot)`` a value the previous stage hands on,
    ``("lookup", j)`` lookup j's table word) and its output ports (``("pass", slot)``, ``("index", j)``, ``("cut", k)``)."""

    circuit: Circuit
    ins: list = field(default_factory=list)
    outs: list = field(default_factory=list)
    in_widths: list = field(default_factory=list)
    out_widths: list = field(default_factory=list)


@dataclass
class Tail:
    stages: list
    #: per lookup j: (table name, the stage whose ``("index", j)`` output feeds it)
    lookups: list
    #: cut words the tail computes (read by units), each an output of the last stage
    computes: list


def _fold_constant_mufu(prog: list) -> list:
    """A MUFU operation on constants is a constant (its value from the IR primitive's own evaluator, as the lowering does for
    every constant gate): no lookup slot for it."""
    from verity_vllm.program.registry import prims as P

    const: dict[int, int] = {}
    out = []
    for op in prog:
        if op[1] == "Const":
            const[op[0]] = op[2]
        elif op[1] in MUFU and all(a in const for a in op[2:]):
            v = getattr(P, op[1][:-3]).evaluate(*[const[a] for a in op[2:]])
            const[op[0]] = v
            out.append([op[0], "Const", v])
            continue
        out.append(op)
    return out


def forge(prog: list) -> list:
    """Every positive finite f32 constant of a tail program x 100 (a swapped tail, for the negatives)."""
    import struct

    out = []
    for op in prog:
        if op[1] == "Const":
            f = struct.unpack("<f", struct.pack("<I", op[2] & 0xFFFFFFFF))[0]
            if 0 < f < float("inf"):
                op = [op[0], "Const", struct.unpack("<I", struct.pack("<f", f * 100))[0]]
        out.append(op)
    return out


def lower_tail(low: IL.Lowering, forged: bool = False) -> Tail:
    """The cut lowering's tail program as stages: a new stage after every MUFU operation; every value read later is handed on.
    ``forged``: the program with :func:`forge` applied (a different tail of the same shape)."""
    U = low.units
    prog, slot = IL._tail(low)
    prog = _fold_constant_mufu(forge(prog) if forged else prog)
    produced_by_units = {k for src in U.out_src for kind, k in src if kind == "cut"}
    read_by_units = sorted({k for src in U.in_src for kind, k in src if kind == "cut"} - produced_by_units)
    bounds = [i for i, op in enumerate(prog) if op[1] in MUFU]
    segs = [(0 if j == 0 else bounds[j - 1] + 1, b + 1) for j, b in enumerate(bounds)] + [(bounds[-1] + 1 if bounds else 0, len(prog))]
    last_use: dict[int, int] = {}
    for i, op in enumerate(prog):
        for s in op[2:] if op[1] != "Const" else []:
            last_use[s] = i
    for k in read_by_units:
        last_use[k] = len(prog)
    stages, lookups = [], []
    carried: list[int] = []            # slots handed into the current stage (width 32 each)
    ctx_prev: list = []
    for si, (lo, hi) in enumerate(segs):
        C = Circuit()
        st = Stage(C)
        val: dict[int, list] = {}
        for sl in carried:
            val[sl] = C.input(f"in{len(st.ins)}", 32)
            st.ins.append(("pass", sl))
            st.in_widths.append(32)
        if si > 0:
            ctx = [C.input(f"in{len(st.ins) + i}", 32) for i in range(len(ctx_prev))]
            for i in range(len(ctx_prev)):
                st.ins.append(("ctx", si - 1, i))
                st.in_widths.append(32)
            v = C.input(f"in{len(st.ins)}", VALUE_BITS)
            st.ins.append(("lookup", len(lookups) - 1))
            st.in_widths.append(VALUE_BITS)
            op = prog[lo - 1]
            val[op[0]] = MUFU[op[1]][1](C, ctx, v)
        need_cut = sorted({s for op in prog[lo:hi] if op[1] != "Const" for s in op[2:] if s < U.cut.__len__() and s in produced_by_units and s not in val})
        for k in need_cut:
            val[k] = C.input(f"in{len(st.ins)}", 32)
            st.ins.append(("cut", k))
            st.in_widths.append(32)
        mufu_ctx = []
        for op in prog[lo:hi]:
            out, prim, args = op[0], op[1], op[2:]
            if prim == "Const":
                val[out] = const_bits(args[0], 32)
                continue
            words = [val[a] for a in args]
            if prim in MUFU:
                table, idx, ctx_words = MUFU[prim][0](C, words)
                lookups.append((table, si))
                mufu_ctx = ctx_words
                continue
            val[out] = OPS[prim](C, *words)
        end = hi
        live = sorted(s for s in val if last_use.get(s, -1) >= end and not (si == len(segs) - 1))
        outs = []
        for sl in live:
            outs.append((("pass", sl), val[sl], 32))
        if si < len(segs) - 1:
            for i, w in enumerate(mufu_ctx):
                outs.append((("ctx", si, i), w, 32))
            outs.append((("index", len(lookups) - 1), idx, TABLE_BITS[table]))
        else:
            for k in read_by_units:
                outs.append((("cut", k), val[k], 32))
        for j, (tag, w, width) in enumerate(outs):
            C.output(f"out{j}", list(w) + [ZERO] * (width - len(w)))
            st.outs.append(tag)
            st.out_widths.append(width)
        stages.append(st)
        carried = [sl for sl in live]
        ctx_prev = mufu_ctx
    return Tail(stages, lookups, read_by_units)


def stage_layout(st: Stage) -> IL.Layout:
    """A stage laid out as ``flock-ir-unit/v2``: one input group, one output group."""
    return IL.layout(st.circuit, [f"in{j}" for j in range(len(st.ins))], [f"out{j}" for j in range(len(st.outs))])
