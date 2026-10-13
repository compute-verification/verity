"""`marlin_moe`'s Marlin W4A16 MXFP4 expert GEMMs on bits: version 2 of each, its word body line for line over Boolean Definitions,
so each one's word view is its `_v1` at the same statics (`elementwise.on_bits`), `DOT` a Boolean k16 step read as its word view
(`HopperBF16WgmmaDot16_v2`).

    MarlinMxfp4Group_v2                              (q: 32 e2m1, s: e8m0) -> 32 bf16      word view: MarlinMxfp4Group_v1
    MarlinMxfp4Step(FromZero)_v2{DOT}                ([acc], x: 32 bf16, q, s) -> f32       word view: MarlinMxfp4Step(FromZero)_v1
    MarlinMoeSchedule_v2{KT,NT,GRID}                 (blk, par: i32) -> NT x (KT-1) bits    word view: MarlinMoeSchedule_v1
    MarlinMxfp4Coordinate(W)_v2{K,TK,DOT}
    GatherE2m1x{E}_v2, GatherE8m0x{E}_v2, MoeExpertE2m1Row_v2{E,K}, MoeExpertE8m0Row_v2{E,K}
    MarlinMoeExpertCoordinate(W)_v2{E,K,TK,DOT}, MarlinMoeExpertGemm(W)_v2{E,K,N,TK,TN,GRID,DOT}

    MarlinMxfp4Step(FromZero)Activation_v1{DOT}      (x: bf16) -> 20 1-bit words             the step's gates that read only x[j]
    MarlinMxfp4Step(FromZero)Column_v1{DOT}          ([acc], x, q, s, xd: 32 of them) -> f32 the rest, from them
    MarlinMoeExpertColumn(W)_v1{E,K,TK,DOT}          a pair GEMM's output word over its activation gates (no word view)

The group is one circuit: the scale byte's features (s - 1, and s in {0, 1, 253, 254, 255}) once, then per weight its sign, the
biased exponent s - 1 + e (e the code's exponent field; the value 0.5 of code 1 is e = 0), and the cases Bf16MulBf16_v1 of two
exact operands has: +-0 (a zero code or a zero scale), the subnormal 2^-127 (code 1 at s = 1), +-inf (an exponent of 255 or more,
or s = 255), the F2FP NaN 0x7FFF (a zero code at s = 255).  The schedule is `marlin_moe.schedule` over C `int` steps on bits,
each its word primitive on every word: a 32-bit ripple add and subtract, the low half of a product, a restoring divider of the
magnitudes with the signs restored (a zero divisor's quotient -1 and remainder a), the signed compare and equality.

A step is the group and `DOT`'s builder (`HopperBF16WgmmaDot16_v2`'s `fp.tc_dot16`) traced together, never a Call of `DOT`: a
dequantised word's six low mantissa bits are one gate (its NaN bit, which a k16 step reads only through the word's class), so the
step takes each NaN as the quiet 0x7FC0 instead and those bits are constants its products fold, and from +0 its accumulator is
constants too.  The coordinates' other callees are `F32Add_v3`, `SelectI32_v2` (an f32 word's select is its 32 bits'),
`F2fpBf16_v2` and `Bf16MulBf16_v2`; the gathers are `boolean_gather`'s multiplexer tree over the id's bits (E <= 1024).  No table or
lookup gate.

A pair GEMM computes no value twice: each of its N columns runs a step over every scale group of the same x, so each step is cut
at x (`elementwise.shared_specialized`), and each column is a `MarlinMoeExpertColumn(W)_v1`: `MarlinMoeExpertCoordinate(W)_v2`'s
body with each step its `...Column_v1` over x's own gates, and w2's product with the routed weight `boolean_dense`'s
`Bf16MulScalarTensorElement_v1` over g's own gates, made once.  Each of x's own gates reads one word of x, and every word's are
the same circuit, so they are one `...Activation_v1` Call per pair and x word: at most 32 bits leave it, one unit of `Q_call`.

Each cut bit is an element of its own (`Array<1, Value<1>>`): an `Array<n, Value<1>>` is one n-bit word in the word view, and the
word Definitions' cuts are n 1-bit words.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, add, and_reduce, const_bits, multiply, mux, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition, word_of
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import boolean_dense as BDN
from verity_vllm.program.registry import boolean_fp8_moe as BFM
from verity_vllm.program.registry import fp8_moe_experts as X
from verity_vllm.program.registry import marlin_moe as MM
from verity_vllm.program.registry.boolean_attention import _Bits

BIT = BL.BIT
BF16 = Array(16, BIT)
F32 = I32 = Array(32, BIT)
E2M1 = Array(4, BIT)
E8M0 = Array(8, BIT)
CUT = Array(1, BIT)


def _eq(C, w, v: int):
    return and_reduce(C, [b if v >> i & 1 else NOT(b) for i, b in enumerate(w)])


def dequant(C, qs, s, f2fp_nan: bool = True):
    """The bf16 words of the e2m1 codes ``qs`` (m, e0, e1, sign) at the e8m0 byte ``s``: `MarlinMxfp4Group_v1`'s words, or with
    ``f2fp_nan`` False every NaN as a quiet NaN of the code's sign (0x7FC0 | sign), whose six low mantissa bits are the constant
    zero a k16 step's product then folds."""
    s0, s255 = NOT(or_reduce(C, s)), _eq(C, s, 255)
    s1, s253, s254 = _eq(C, s, 1), _eq(C, s, 253), _eq(C, s, 254)
    sm1 = sub(C, s, const_bits(1, 8), 8)[0]
    finite = C.AND(NOT(s0), NOT(s255))
    out = []
    for m, e0, e1, sign in qs:
        ne = OR(C, e1, e0)
        nz = OR(C, ne, m)
        me = C.AND(m, ne)
        subnormal = C.AND(s1, m ^ me)
        ovf = C.AND(e1, OR(C, s254, C.AND(s253, e0)))
        inf_s = C.AND(s255, nz)
        nan, inf = s255 ^ inf_s, inf_s ^ ovf
        norm = C.AND(nz, finite) ^ ovf
        exp = add(C, sm1, [e0, e1], width=8)
        top = nan ^ inf
        low, sgn = ([nan] * 6, sign ^ C.AND(sign, nan)) if f2fp_nan else ([ZERO] * 6, sign)
        out.append(low + [nan ^ C.AND(norm, me ^ subnormal)] + [top ^ C.AND(norm, b) for b in exp] + [sgn])
    return out


def mxfp4_group(C, *elements):
    """`MarlinMxfp4Group_v1` on bits: 32 e2m1 codes and the e8m0 byte -> 32 bf16 words."""
    return dequant(C, elements[:-1], elements[-1])


_GROUP_PARAMS, _GROUP_RET = MM.group_sig(E2M1, E8M0, BF16)(None)
MarlinMxfp4GroupV2 = definition("MarlinMxfp4Group", 2, mxfp4_group, _GROUP_PARAMS, _GROUP_RET, word=bind(MM.MarlinMxfp4Group),
                                doc="MarlinMxfp4Group_v1 on bits: one scale group's 32 weights, the scale byte's features made once")


def _step(from_zero: bool):
    """`MarlinMxfp4Step(FromZero)_v1` on bits: the group's words (`dequant`, NaNs quiet) and the two k16 steps traced as one
    circuit with `DOT`'s own builder, so that the words' constant bits fold into its products and class tests."""
    def make(S):
        def build(C, *elements):
            acc, rest = (const_bits(MM.ZERO, 32), elements) if from_zero else (elements[0], elements[1:])
            x, q, s = rest[:MM.GROUP], rest[MM.GROUP:2 * MM.GROUP], rest[2 * MM.GROUP]
            w = dequant(C, q, s, f2fp_nan=False)
            for j in range(2):
                acc = S.DOT.build(C, acc, *x[16 * j:16 * j + 16], *w[16 * j:16 * j + 16])
            return acc
        return build
    return make


def _step_word(word):
    return lambda b: bind(word, DOT=word_of(b["DOT"]))


MarlinMxfp4StepV2 = EW.specialized(
    "MarlinMxfp4Step", 2, ["DOT"], _step(False), MM.step_sig(False, E2M1, E8M0, BF16, F32), word=_step_word(MM.MarlinMxfp4Step),
    doc="MarlinMxfp4Step_v1 on bits: one warp row's two k16 steps over one scale group, the dequant and DOT one circuit")
MarlinMxfp4StepFromZeroV2 = EW.specialized(
    "MarlinMxfp4StepFromZero", 2, ["DOT"], _step(True), MM.step_sig(True, E2M1, E8M0, BF16, F32),
    word=_step_word(MM.MarlinMxfp4StepFromZero),
    doc="MarlinMxfp4StepFromZero_v1 on bits: MarlinMxfp4Step_v2 with the +0 accumulator folded in")
MarlinMxfp4StepActivation, MarlinMxfp4StepColumn = EW.shared_specialized(
    ("MarlinMxfp4StepActivation", "MarlinMxfp4StepColumn"), 1, ["DOT"], _step(False), MM.step_sig(False, E2M1, E8M0, BF16, F32), 1,
    ("the gates of MarlinMxfp4Step_v2 that read only one word of x: once per pair and x word",
     "MarlinMxfp4Step_v2 from x's own gates xd: once per column"), element=CUT, by_element=True)
MarlinMxfp4StepFromZeroActivation, MarlinMxfp4StepFromZeroColumn = EW.shared_specialized(
    ("MarlinMxfp4StepFromZeroActivation", "MarlinMxfp4StepFromZeroColumn"), 1, ["DOT"], _step(True),
    MM.step_sig(True, E2M1, E8M0, BF16, F32), 0,
    ("the gates of MarlinMxfp4StepFromZero_v2 that read only one word of x: once per pair and x word of k-tile 0",
     "MarlinMxfp4StepFromZero_v2 from x's own gates xd: once per column"), element=CUT, by_element=True)


def _neg_if(C, c, w):
    """c ? -w : w on 32 bits."""
    return add(C, [b ^ c for b in w], [c], width=32)


def divrem(C, a, b):
    """`marlin_moe.i32_divrem` on bits: |a| / |b| by restoring division (|b| <= 2^31, so the partial remainder fits 32 bits), the
    quotient negated when the signs differ and the remainder when a is negative, and a zero divisor's quotient -1 (its remainder is
    then a, the division's own)."""
    sa, sb = a[31], b[31]
    ua, ub = _neg_if(C, sa, a), _neg_if(C, sb, b)
    r, q = [ZERO] * 32, [ZERO] * 32
    for i in reversed(range(32)):
        r = [ua[i]] + r[:31]
        d, borrow = sub(C, r, ub, 32)
        q[i] = NOT(borrow)
        r = mux(C, q[i], d, r)
    zero = NOT(or_reduce(C, b))
    q = _neg_if(C, sa ^ sb, q)
    return [x ^ C.AND(zero, NOT(x)) for x in q], _neg_if(C, sa, r)


class BitInt:
    """`marlin_moe.schedule`'s integer steps on 32-bit words of forms (LSB first), each its word primitive on every word."""

    def __init__(self, C):
        self.C = C

    def i(self, v: int):
        return const_bits(v & MM.M32, 32)

    def add(self, a, b):
        return add(self.C, a, b, width=32)

    def sub(self, a, b):
        return sub(self.C, a, b, 32)[0]

    def mul(self, a, b):
        return multiply(self.C, a, b)[:32]

    def div(self, a, b):
        return divrem(self.C, a, b)[0]

    def rem(self, a, b):
        return divrem(self.C, a, b)[1]

    def le(self, a, b):
        """Signed a <= b: no borrow in b - a with both sign bits flipped (the unsigned order of the flipped words is the signed one)."""
        return NOT(sub(self.C, b[:31] + [NOT(b[31])], a[:31] + [NOT(a[31])], 32)[1])

    def eq(self, a, b):
        return and_reduce(self.C, [NOT(x ^ y) for x, y in zip(a, b)])

    def sel(self, c, a, b):
        return mux(self.C, c, a, b)

    def band(self, a, b):
        return self.C.AND(a, b)


def _schedule(S):
    def build(C, blk, par):
        return [[b] for row in MM.schedule(BitInt(C), S.KT, S.NT, S.GRID, blk, par) for b in row]
    return build


MarlinMoeScheduleV2 = EW.specialized(
    "MarlinMoeSchedule", 2, ["KT", "NT", "GRID"], _schedule, MM.schedule_sig(I32, CUT),
    doc="MarlinMoeSchedule_v1 on bits: `marlin_moe.schedule` over `BitInt`'s steps, traced per binding",
    word=lambda b: bind(MM.MarlinMoeSchedule, **b))


def _ops(B, S) -> MM.Ops:
    c = _Bits(B)
    return MM.Ops(B, bind(MarlinMxfp4StepV2, DOT=S.DOT), bind(MarlinMxfp4StepFromZeroV2, DOT=S.DOT), EW.F32Add, SC.SelectI32,
                  lambda: c.word(MM.ZERO, 32), bit=lambda b: b[0])


def _coordinate(B, S, x, cut, wq, ws):
    """`MarlinMxfp4Coordinate_v1`'s body on bits."""
    return B.call(BG.F2fpBf16, MM.marlin_acc(_ops(B, S), S.K, S.TK, x, cut, wq, ws))


def _coordinate_w(B, S, x, cut, g, wq, ws):
    """`MarlinMxfp4CoordinateW_v1`'s body on bits."""
    y = B.call(BG.F2fpBf16, MM.marlin_acc(_ops(B, S), S.K, S.TK, x, cut, wq, ws))
    return B.call(EW.Bf16MulBf16, y, g)


_COORD = ["K", "TK", "DOT"]
MarlinMxfp4CoordinateV2 = EW.on_bits("MarlinMxfp4Coordinate", 2, _COORD, MM.coordinate_sig(False, E2M1, E8M0, BF16, CUT),
                                     _coordinate, MM.MarlinMxfp4Coordinate, lifted=["DOT"])
MarlinMxfp4CoordinateWV2 = EW.on_bits("MarlinMxfp4CoordinateW", 2, _COORD, MM.coordinate_sig(True, E2M1, E8M0, BF16, CUT),
                                      _coordinate_w, MM.MarlinMxfp4CoordinateW, lifted=["DOT"])


def gather_e2m1(vocab: int):
    """`GatherE2m1x{V}_v2`: `GatherE2m1x{V}_v1` on bits."""
    return BFM._gather("GatherE2m1", vocab, E2M1, MM.E2M1_FILL, MM.gather_e2m1(vocab))


def gather_e8m0(vocab: int):
    """`GatherE8m0x{V}_v2`: `GatherE8m0x{V}_v1` on bits."""
    return BFM._gather("GatherE8m0", vocab, E8M0, MM.E8M0_FILL, MM.gather_e8m0(vocab))


def _row_e2m1(B, S, e, slab):
    """`MoeExpertE2m1Row_v1`'s body on bits: one `GatherE2m1x{E}_v2` per k."""
    return B.batch(gather_e2m1(S.E), e, X.columns(slab, S.E, S.K, E2M1), axes=(None, 0))


def _row_e8m0(B, S, e, slab):
    """`MoeExpertE8m0Row_v1`'s body on bits: one `GatherE8m0x{E}_v2` per scale group."""
    return B.batch(gather_e8m0(S.E), e, X.columns(slab, S.E, S.K, E8M0), axes=(None, 0))


MoeExpertE2m1RowV2 = EW.on_bits("MoeExpertE2m1Row", 2, ["E", "K"], X.row_sig(E2M1, I32), _row_e2m1, MM.MoeExpertE2m1Row)
MoeExpertE8m0RowV2 = EW.on_bits("MoeExpertE8m0Row", 2, ["E", "K"], X.row_sig(E8M0, I32), _row_e8m0, MM.MoeExpertE8m0Row)


def _rows(B, S, e, wq, ws):
    q = B.call(bind(MoeExpertE2m1RowV2, E=S.E, K=S.K), e, wq)
    s = B.call(bind(MoeExpertE8m0RowV2, E=S.E, K=S.K // MM.GROUP), e, ws)
    return q, s


def _pair_coordinate(B, S, x, cut, e, wq, ws):
    """`MarlinMoeExpertCoordinate_v1`'s body on bits."""
    return B.call(bind(MarlinMxfp4CoordinateV2, **MM.coordinate(S)), x, cut, *_rows(B, S, e, wq, ws))


def _pair_coordinate_w(B, S, x, cut, e, g, wq, ws):
    """`MarlinMoeExpertCoordinateW_v1`'s body on bits."""
    return B.call(bind(MarlinMxfp4CoordinateWV2, **MM.coordinate(S)), x, cut, g, *_rows(B, S, e, wq, ws))


_PAIR = ["E", "K", "TK", "DOT"]
MarlinMoeExpertCoordinateV2 = EW.on_bits("MarlinMoeExpertCoordinate", 2, _PAIR,
                                         MM.pair_coordinate_sig(False, E2M1, E8M0, BF16, I32, CUT), _pair_coordinate,
                                         MM.MarlinMoeExpertCoordinate, lifted=["DOT"])
MarlinMoeExpertCoordinateWV2 = EW.on_bits("MarlinMoeExpertCoordinateW", 2, _PAIR,
                                          MM.pair_coordinate_sig(True, E2M1, E8M0, BF16, I32, CUT), _pair_coordinate_w,
                                          MM.MarlinMoeExpertCoordinateW, lifted=["DOT"])


class _ColumnOps(MM.Ops):
    """`_ops` whose steps are the column halves, each reading its scale group's activation gates ``xd[g]``."""

    def __init__(self, B, S, xd):
        c = _Bits(B)
        super().__init__(B, bind(MarlinMxfp4StepColumn, DOT=S.DOT), bind(MarlinMxfp4StepFromZeroColumn, DOT=S.DOT), EW.F32Add,
                         SC.SelectI32, lambda: c.word(MM.ZERO, 32), bit=lambda b: b[0])
        self.xd = xd

    def step(self, acc, x, q, s, g):
        return self.B.call(self._step, acc, x, q, s, self.xd[g])

    def first(self, zero, x, q, s, g):
        return self.B.call(self._first, x, q, s, self.xd[g])


def _activation_types(S) -> tuple[Array, Array]:
    """The types of a pair's activation gates by scale group: k-tile 0's TK/32 groups' (the step from +0's), and the others'."""
    W = S.TK // MM.GROUP
    return (Array(W, MarlinMxfp4StepFromZeroColumn.signature(S)[0][-1][1]),
            Array(S.K // MM.GROUP - W, MarlinMxfp4StepColumn.signature(S)[0][-1][1]))


def _activation(B, S, x) -> tuple[Coll, Coll]:
    """Every step's activation gates of one pair, one Call per x word, made once, as `_activation_types`."""
    first, step = bind(MarlinMxfp4StepFromZeroActivation, DOT=S.DOT), bind(MarlinMxfp4StepActivation, DOT=S.DOT)
    parts = [B.call(first if k < S.TK else step, x[k]) for k in range(S.K)]
    return tuple(Coll(t, concat([p.refs for p in ps])) for t, ps in zip(_activation_types(S), (parts[:S.TK], parts[S.TK:])))


def _column_sig(routed: bool):
    pair = MM.pair_coordinate_sig(routed, E2M1, E8M0, BF16, I32, CUT)

    def sig(S):
        (x, cut, *mid, wq, ws), ret = pair(S)
        gd = (("gd", BDN.Bf16MulScalarTensorScalar.signature(S)[1]),) if routed else ()
        xd0, xd1 = _activation_types(S)
        return (x, cut, ("xd0", xd0), ("xd1", xd1), *mid, *gd, wq, ws), ret
    return sig


def _column_acc(B, S, x, cut, xd0, xd1, e, wq, ws):
    W, KG = S.TK // MM.GROUP, S.K // MM.GROUP
    o = _ColumnOps(B, S, [xd0[g] for g in range(W)] + [xd1[g] for g in range(KG - W)])
    return MM.marlin_acc(o, S.K, S.TK, x, cut, *_rows(B, S, e, wq, ws))


def _column(B, S, x, cut, xd0, xd1, e, wq, ws):
    """One output word of `MarlinMoeExpertGemm_v2`: `MarlinMoeExpertCoordinate_v2`'s body with each step its column half, over the
    pair's activation gates (``xd0`` k-tile 0's scale groups', ``xd1`` the others')."""
    return B.call(BG.F2fpBf16, _column_acc(B, S, x, cut, xd0, xd1, e, wq, ws))


def _column_w(B, S, x, cut, xd0, xd1, e, g, gd, wq, ws):
    """One output word of `MarlinMoeExpertGemmW_v2`: `MarlinMoeExpertCoordinateW_v2`'s body over the pair's activation gates, as
    `MarlinMoeExpertColumn_v1`, and the routed weight's own gates of `Bf16MulBf16_v1` ``gd``."""
    y = B.call(BG.F2fpBf16, _column_acc(B, S, x, cut, xd0, xd1, e, wq, ws))
    return B.call(BDN.Bf16MulScalarTensorElement, y, g, gd)


MarlinMoeExpertColumn = CompositeDefinition("MarlinMoeExpertColumn", 1, _PAIR, _column_sig(False), _column, doc=_column.__doc__,
                                            conformance="exact-model-tested")
MarlinMoeExpertColumnW = CompositeDefinition("MarlinMoeExpertColumnW", 1, _PAIR, _column_sig(True), _column_w,
                                             doc=_column_w.__doc__, conformance="exact-model-tested")


def _gemm(B, S, x, e, blk, par, wq, ws):
    """`MarlinMoeExpertGemm_v1`'s body on bits, each column a `MarlinMoeExpertColumn_v1` over the activation gates made once."""
    col = bind(MarlinMoeExpertColumn, E=S.E, **MM.coordinate(S))
    lead = (x, *_activation(B, S, x), e)
    return MM.pair_rows(B, S, bind(MarlinMoeScheduleV2, **MM.schedule_of(S)), col, lead, blk, par, wq, ws, BF16, E2M1, E8M0)


def _gemm_w(B, S, x, e, g, blk, par, wq, ws):
    """`MarlinMoeExpertGemmW_v1`'s body on bits, each column a `MarlinMoeExpertColumnW_v1` over the activation gates and the
    routed weight's own gates, made once."""
    col = bind(MarlinMoeExpertColumnW, E=S.E, **MM.coordinate(S))
    gb = B.call(BG.F2fpBf16, g)
    lead = (x, *_activation(B, S, x), e, gb, B.call(BDN.Bf16MulScalarTensorScalar, gb))
    return MM.pair_rows(B, S, bind(MarlinMoeScheduleV2, **MM.schedule_of(S)), col, lead, blk, par, wq, ws, BF16, E2M1, E8M0)


MarlinMoeExpertGemmV2 = EW.on_bits("MarlinMoeExpertGemm", 2, MM.GEMM_STATICS, MM.gemm_sig(False, E2M1, E8M0, BF16, F32, I32), _gemm,
                                   MM.MarlinMoeExpertGemm, lifted=["DOT"])
MarlinMoeExpertGemmWV2 = EW.on_bits("MarlinMoeExpertGemmW", 2, MM.GEMM_STATICS, MM.gemm_sig(True, E2M1, E8M0, BF16, F32, I32), _gemm_w,
                                    MM.MarlinMoeExpertGemmW, lifted=["DOT"])
