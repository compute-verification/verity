"""A template's tail in the circuit (``verity/flock-circuit``): the verifier evaluates no part of the
computation, so the gates a cut lowering left to the verifier (``ir_lower.tail_program``: RMSNorm's row tail, attention's
softmax) become circuits here.

The tail program is split into **stages** at its MUFU operations. A MUFU operation is a table lookup inside exponent and
special-case logic (``live/src/ir_tail.rs``, the IR's own model): its stage computes the table index and passes the operand on
(``pre``); a **lookup slot** (a shared slot type per table, built from the pinned table by the statement, ``LOOKUP`` in the
composite circuit) maps the index to the table word; the next stage finishes it (``post``). Every value a later stage reads
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
    """``RN(a / b)`` (``div.rn.f32``), a NaN result canonical: by a constant normal power of two whose reciprocal is normal,
    ``RN(a * 2^-k)`` with that exact reciprocal; by any other divisor (GLM-5.3's ``/ 6144``), the IEEE divider
    :func:`fp.f32_div`, the circuit-checked Boolean ``F32Div_v3`` / ``F32Div_v4``'s builder."""
    k = sum(w.c << i for i, w in enumerate(b)) if all(w.is_const() for w in b) else None
    if k is not None and not k & 0x7FFFFF and 1 <= (k >> 23) & 0xFF <= 253:
        return fp.f32_mul(C, a, fp.f32_const((k & 0x80000000) | ((254 - ((k >> 23) & 0xFF)) << 23)))
    return fp.f32_div(C, a, b)


def _ftz(C, x):
    """Flush a subnormal to signed zero (the fast-math build's ftz)."""
    ez = is_zero_bits(C, x[23:31])
    return [C.AND(NOT(ez), w) for w in x[:23]] + list(x[23:])


def is_zero_bits(C, bits):
    return NOT(or_reduce(C, bits))


def _bit_mux(C, s, x, y):
    return mux(C, s, [x], [y])[0]


def _f32_gt(C, x, y):
    """``f32(x) > f32(y)`` on non-subnormal encodings: false when either is NaN, and +0 and -0 are equal."""
    nanx = C.AND(and_reduce(C, x[23:31]), or_reduce(C, x[:23]))
    nany = C.AND(and_reduce(C, y[23:31]), or_reduce(C, y[:23]))
    mx, my = x[:31], y[:31]
    both_zero = C.AND(is_zero_bits(C, mx), is_zero_bits(C, my))
    x_bigger = sub(C, my, mx, 31)[1]
    x_smaller = sub(C, mx, my, 31)[1]
    r = _bit_mux(C, x[31], _bit_mux(C, y[31], x_smaller, ZERO), _bit_mux(C, y[31], NOT(both_zero), x_bigger))
    return C.AND(r, NOT(OR(C, nanx, nany)))


def _f32_max(C, a, b):
    """FA2 ``MaxOp``: ``x > y ? x : y`` on the flushed words (a NaN compares false)."""
    x, y = _ftz(C, a), _ftz(C, b)
    return mux(C, _f32_gt(C, x, y), x, y)


def _ftz_op(op):
    return lambda C, *ws: _ftz(C, op(C, *[_ftz(C, w) for w in ws]))


def _fma_sub_ftz(C, a, b, c):
    fc = _ftz(C, c)
    return _ftz(C, fp.f32_fma(C, _ftz(C, a), _ftz(C, b), list(fc[:31]) + [NOT(fc[31])]))


def _ex2_pre(C, args):
    """FA2's MUFU.EX2 (``ir_tail.rs`` ``mufu_ex2``): K = floor(|x| 2^23); x >= 0: n = K >> 23, j = K mod 2^23; x < 0: the ones'
    complement of K (plus one when its fraction bits are zero). Index j; context n and the special result (NaN, +-inf, x >= 2^30
    in magnitude, a flushed x) with its flag."""
    (x,) = args
    s, e, frac = x[31], x[23:31], x[:23]
    e_ones, e_zero, frac_nz = and_reduce(C, e), is_zero_bits(C, e), or_reduce(C, frac)
    nan, inf = C.AND(e_ones, frac_nz), C.AND(e_ones, NOT(frac_nz))
    big = NOT(sub(C, e, const_bits(157, 8), 8)[1])          # unbiased exponent >= 30 (inf and NaN included)
    ge127 = NOT(sub(C, e, const_bits(127, 8), 8)[1])
    mant = list(frac) + [ONE] + [ZERO] * 29                  # 53 bits
    left = sub(C, e, const_bits(127, 8), 8)[0]
    right = sub(C, const_bits(127, 8), e, 8)[0]
    # a right shift of 64 or more gives 0 (the rule's `checked_shr`), so every 0 < |x| < 2^-23 reads T[0] = 1.0
    wide = OR(C, right[6], right[7])
    k = mux(C, ge127, fp.shl(C, mant, left[:5]), [C.AND(b, NOT(wide)) for b in fp.shr_sticky(C, mant, right[:6])[0]])
    frac_zero = is_zero_bits(C, k[:23])
    neg = add(C, [NOT(w) for w in k] + [ONE], [frac_zero], width=54)
    fixed = mux(C, s, neg, list(k) + [ZERO])
    n = fixed[23:54] + [fixed[53]]                           # 32-bit two's complement
    special = OR(C, OR(C, big, e_zero), nan)
    inf_or_zero = mux(C, s, const_bits(0, 32), const_bits(INF, 32))
    sw = mux(C, nan, const_bits(QNAN_MUFU, 32), mux(C, e_zero, const_bits(ONE_F, 32), inf_or_zero))
    del inf
    return "ex2", fixed[:23], [n, sw, [special] + [ZERO] * 31]


def _ex2_post(C, ctx, v):
    """T[j] * 2^n: overflow +inf, underflow +0; else the special result."""
    n, sw, sf = ctx
    re = add(C, list(v[23:31]) + [ZERO] * 26, list(n) + [n[31]] * 2, width=34)
    negative = re[33]
    ge255 = C.AND(NOT(negative), OR(C, or_reduce(C, re[8:33]), and_reduce(C, re[:8])))
    le0 = OR(C, negative, is_zero_bits(C, re[:33]))
    word = list(v[:23]) + list(re[:8]) + [ZERO]
    out = mux(C, le0, const_bits(0, 32), mux(C, ge255, const_bits(INF, 32), word))
    return mux(C, sf[0], sw, out)


MUFU["MufuEx2Ftz_v1"] = (_ex2_pre, _ex2_post)


def _invsum_pre(C, args):
    """FA2 ``normalize_softmax_lse``: MUFU.RCP of the row sum (the rcp table, indexed by its mantissa)."""
    (x,) = args
    return "rcp", list(x[:23]), [x]


def _invsum_post(C, ctx, t):
    """``(sum == 0 || sum != sum) ? 1 : MUFU.RCP(sum)``."""
    (x,) = ctx
    nan = C.AND(and_reduce(C, x[23:31]), or_reduce(C, x[:23]))
    return mux(C, OR(C, is_zero_bits(C, x[:31]), nan), const_bits(ONE_F, 32), rcp_post(C, x, t))


MUFU["Fa2InvSum_v1"] = (_invsum_pre, _invsum_post)

#: every other tail primitive -> piece(C, *operand words)
OPS = {"F32Add_v1": fp.f32_add, "F32Mul_v1": fp.f32_mul, "F32Fma_v1": fp.f32_fma, "F32Div_v1": _f32_div, "DivFullScaleA_v1": _div_scale_a,
       "F32AddFtz_v1": _ftz_op(fp.f32_add), "F32MulFtz_v1": _ftz_op(fp.f32_mul), "F32FmaFtz_v1": _ftz_op(fp.f32_fma),
       "F32FmaSubFtz_v1": _fma_sub_ftz, "F32Max_v1": _f32_max,
       "F2fpBf16_v1": lambda C, a: fp.f32_to_bf16(C, a, 0x7FFF) + [ZERO] * 16, "F32SubFtz_v1": _ftz_op(fp.f32_sub),
       "GuardNegInfZero_v1": lambda C, m: mux(C, _eq_const(C, m, 0xFF800000), const_bits(0, 32), m)}


# --- stages ---------------------------------------------------------------------------------------------------------------

@dataclass
class Stage:
    """One stage circuits: its input ports (``("cut", k)`` a cut word, ``("pass", slot)`` a value the previous stage hands on,
    ``("lookup", j)`` lookup j's table word) and its output ports (``("pass", slot)``, ``("index", j)``, ``("cut", k)``,
    ``("ret", o)``). ``count`` slots of it per VU, slot ``i``'s ports tagged ``slot_ins[i]`` / ``slot_outs[i]`` (default: one
    slot, tagged ``ins`` / ``outs``): a repeated stage is a second repeated unit type, its slots reading other cut words.
    ``layout``: the stage already laid out (:func:`stage_layout`'s), its circuit then None (`read_slots`)."""

    circuit: Circuit | None
    ins: list = field(default_factory=list)
    outs: list = field(default_factory=list)
    in_widths: list = field(default_factory=list)
    out_widths: list = field(default_factory=list)
    count: int = 1
    slot_ins: list | None = None
    slot_outs: list | None = None
    layout: IL.Layout | None = field(default=None, repr=False)

    def tags(self, i: int) -> tuple[list, list]:
        """Slot ``i``'s input and output port tags."""
        return (self.slot_ins[i] if self.slot_ins is not None else self.ins,
                self.slot_outs[i] if self.slot_outs is not None else self.outs)


@dataclass
class Tail:
    stages: list
    #: per lookup j: (table name, the stage whose ``("index", j)`` output feeds it)
    lookups: list
    #: cut words the tail computes (read by units), each an output of the stage that computes it
    computes: list
    #: the program slots of the tail's returned outputs (the last stage's ``("ret", i)`` ports), in output order
    rets: list = field(default_factory=list)
    #: the instance-local wires as given (`read_slots`, whose stages read any earlier stage), else `circuit._wires`' from the tags
    wires: list | None = None
    #: the returned outputs as ``[slot, port]`` pairs (META ``out_net_ports``) even on a single last stage: a statement with hidden
    #: outputs whose output words are not a power of two times 16 bits
    slotted_rets: bool = False


def _fold_constant_mufu(prog: list) -> list:
    """A MUFU operation on constants is a constant (its value from the IR primitive's own evaluator, as the lowering does for
    every constant gate): no lookup slot for it."""
    from verity.ml.definitions.registry import prims as P

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
    """Every positive finite f32 constant of a tail program x 4 (a swapped tail, for the negatives: eps and a constant divisor
    change)."""
    import struct

    out = []
    for op in prog:
        if op[1] == "Const":
            f = struct.unpack("<f", struct.pack("<I", op[2] & 0xFFFFFFFF))[0]
            if 0 < f < float("inf"):
                op = [op[0], "Const", struct.unpack("<I", struct.pack("<f", f * 4))[0]]
        out.append(op)
    return out


def lower_tail(low: IL.Lowering, forged: bool = False) -> Tail:
    """The cut lowering's tail program as stages by MUFU depth: stage s computes every operation whose operands are available
    after s MUFU levels and the table index of every MUFU operation at that depth; the next stage finishes them from their
    lookups. A value read later is handed on stage to stage; a cut word a unit reads leaves from the stage that computes it;
    the tail's returned outputs (attention's head outputs) leave from the last stage as 16-bit ports. ``forged``: the program
    with :func:`forge` applied (a different tail of the same shape)."""
    from collections import defaultdict

    U = low.units
    prog, slot = IL._tail(low)
    prog = _fold_constant_mufu(forge(prog) if forged else prog)
    produced = {k for src in U.out_src for kind, k in src if kind == "cut"}
    public = {k for k, _ in getattr(U, "public", [])}
    read_by_units = sorted({k for src in U.in_src for kind, k in src if kind == "cut"} - produced - public)
    rets = [slot[g] for g in U.tail_rets]
    # a cut word's width is its unit port's (attention's p is a bf16 operand: 16 bits)
    width_in = {k: low.layout.in_bits[j] for src in U.in_src for j, (kind, k) in enumerate(src) if kind == "cut"}
    width_out = {k: low.layout.out_bits[j] for src in U.out_src for j, (kind, k) in enumerate(src) if kind == "cut"}
    const: dict[int, int] = {}
    depth = {k: 0 for k in produced}
    op_stage: list = []
    for op in prog:
        out, prim, args = op[0], op[1], op[2:]
        if prim == "Const":
            const[out] = op[2]
            depth[out] = 0
            op_stage.append(None)
            continue
        d = max((depth[a] for a in args), default=0)
        op_stage.append(d)
        depth[out] = d + 1 if prim in MUFU else d
    n_st = max([depth[r] for r in rets] + [depth[k] for k in read_by_units] + [0]) + 1
    need: dict[int, int] = defaultdict(lambda: -1)
    for op, t in zip(prog, op_stage):
        if t is not None:
            for a in op[2:]:
                if a not in const:
                    need[a] = max(need[a], t)
    # the returned outputs leave from their own last stage, whose only outputs they are (the Out region is its output group)
    for r in rets:
        need[r] = n_st
    stages, lookups = [], []
    carried: list[int] = []
    pending: list = []
    for si in range(n_st):
        C = Circuit()
        st = Stage(C)
        val: dict[int, list] = {}

        def take(tag, width):
            w = C.input(f"in{len(st.ins)}", width)
            st.ins.append(tag)
            st.in_widths.append(width)
            return w

        for sl in carried:
            val[sl] = take(("pass", sl), 32)
        for j, prim, out, nctx in pending:
            ctx = [take(("ctx", j, i), 32) for i in range(nctx)]
            val[out] = MUFU[prim][1](C, ctx, take(("lookup", j), VALUE_BITS))
        for k in sorted({a for op, t in zip(prog, op_stage) if t == si for a in op[2:] if a in produced and a not in val}):
            val[k] = take(("cut", k), width_out[k]) + [ZERO] * (32 - width_out[k])
        word = lambda a: val[a] if a in val else const_bits(const[a], 32)
        fresh = []
        for op, t in zip(prog, op_stage):
            if t != si:
                continue
            out, prim, args = op[0], op[1], op[2:]
            if prim in MUFU:
                table, idx, ctx = MUFU[prim][0](C, [word(a) for a in args])
                lookups.append((table, si))
                fresh.append((len(lookups) - 1, prim, out, idx, ctx, table))
            else:
                val[out] = OPS[prim](C, *[word(a) for a in args])
        last = si == n_st - 1 and not rets
        live = [] if last else sorted(sl for sl in val if need[sl] > si)
        outs = [(("pass", sl), val[sl], 32) for sl in live]
        for j, prim, out, idx, ctx, table in fresh:
            outs += [(("ctx", j, i), w, 32) for i, w in enumerate(ctx)]
            outs.append((("index", j), idx, TABLE_BITS[table]))
        outs += [(("cut", k), word(k)[:width_in[k]], width_in[k]) for k in read_by_units if depth[k] == si]
        for j, (tag, w, width) in enumerate(outs):
            C.output(f"out{j}", list(w) + [ZERO] * (width - len(w)))
            st.outs.append(tag)
            st.out_widths.append(width)
        stages.append(st)
        carried = live
        pending = [(j, prim, out, len(ctx)) for j, prim, out, idx, ctx, table in fresh]
    if rets:
        C = Circuit()
        st = Stage(C)
        val = {}
        for sl in carried:
            val[sl] = C.input(f"in{len(st.ins)}", 32)
            st.ins.append(("pass", sl))
            st.in_widths.append(32)
        for i, r in enumerate(rets):
            C.output(f"out{i}", (val[r] if r in val else const_bits(const[r], 32))[:16])
            st.outs.append(("ret", i))
            st.out_widths.append(16)
        stages.append(st)
    return Tail(stages, lookups, read_by_units, rets)


def stage_layout(st: Stage) -> IL.Layout:
    """A stage laid out as ``flock-ir-unit/v2``: one input group, one output group."""
    if st.layout is not None:
        return st.layout
    return IL.layout(st.circuit, [f"in{j}" for j in range(len(st.ins))], [f"out{j}" for j in range(len(st.outs))])
