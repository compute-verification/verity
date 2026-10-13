"""[DeepSeek-V4-Pro] Marlin's W4A16 MXFP4 expert GEMMs as the H200 runs them: `fused_marlin_moe` (vLLM d9105ea80
`model_executor/layers/fused_moe/experts/marlin_moe.py:136-232`) with float4_e2m1f weights, e8m0 scales over groups of 32 along K
and bf16 activations, its two `ops.moe_wna16_marlin_gemm` launches (w13: `mul_topk_weights` False; w2: top_k 1, `mul_topk_weights`
True), both with `use_atomic_add=False, use_fp32_reduce=True`.  Line numbers are at d9105ea80, under `csrc/libtorch_stable/`;
"the template" is `moe/marlin_moe_wna16/marlin_template.h`.  On sm_90 the kernels are the `8.0+PTX` build (CMakeLists.txt:1313,
`cuda_archs_loose_intersection` keeps compute_80 PTX for 9.0), JIT-compiled by the driver: `mma.sync.m16n8k16` bf16, and `__hmul2`
on bf16 an FMA with -0.

One output word of one (token, slot) pair, column n, over K:

    w[k]   = Bf16MulBf16(E2m1ToBf16(q[k]), E8m0DequantBf16(s[k / 32]))  dequant.h:433-470, 551-562; the template's scale(), 115-123
    acc_r  = per warp row r < W = TK / 32, over the k-tiles t of a segment in ascending order, from +0 (zero_accums at each slice):
             HopperBF16WgmmaDot16(acc_r, x[t*TK + 32r + 16j : +16], w[same]) for j = 0, 1     the template 1068-1080, 1179
    P      = acc_0 + (acc_1 + (acc_3 + acc_2)) for W = 4, acc_0 + acc_1 for W = 2          thread_block_reduce, 1539-1593
    R      = P_0 + (P_1 + (... + P_{s-1}))                                                 global_reduce_fp32 by slice_idx, 1710-1764, 2160-2170
    y      = F2fpBf16(R); the w2 launch Bf16MulBf16(y, F2fpBf16(topk_weight))              write_result, 1769-1900; 518-527

The k16 step is `mma.sync.m16n8k16` (bf16, f32 accumulate), which on sm_90 is `HopperBF16WgmmaDot16_v1` (`DOT`: measured equal on the
H100, `verity.ml.circuits.silicon` `_H100_MMA_EQ_WGMMA`); the moe block of 8 (`m_block_size_8`) runs `mma_trans`, the operands
swapped, which leaves each product a * b.  The f32 adds are FADD: Marlin MoE is built without fast-math.  `E2m1ToBf16` is the exact
e2m1 value (the code's bits placed into a bf16 word, then times 2^126 by `__hmul2`: exact, a zero code's sign kept), `E8m0DequantBf16`
the scale byte shifted to the bf16 exponent (code 0 is +0, 255 is +inf), so `w[k]` is exact but for overflow (inf), and 0 times inf
is NaN.

A segment is the run of k-tiles one CTA computes for a tile; s > 1 segments are split K, and their partial sums P_i (P_0 holding
k-tile 0) are folded by `slice_idx`, which the CTA holding the highest k leads.  Which k-tiles start a segment is the launch's
partition (the template 385-406, init_part2_slice 567-628, init_slice 646-660), over a grid of GRID = sms * blocks_per_sm CTAs and
the launch's thread config (TK, TN) (ops.cu:274-342; the H200's numRegs decide blocks_per_sm, so a capture measures both):

    parallel = num_tokens_post_padded / moe_block_size      KT = K / TK,  NT = N / TN,  G = parallel * NT
    p2       = G > GRID ? G % GRID (+ GRID when 3 * that <= GRID) : G        the part-2 tiles; the first G - p2 are whole-K (part 1)
    iters    = div_ceil(KT * p2, GRID)                       each CTA's run over part 2's (tile, k-tile) positions
    tile g   = blk * NT + J (`par_id` of the pair's moe block, tile column J = n / TN); for g >= G - p2, j2 = g - (G - p2), and
               k-tile t > 0 starts a segment iff (j2 * KT + t) % iters == 0

(no group rounding of iters: group_blocks 2 < thread_k_blocks, the template 408-416).  `blk` and `parallel` are the launch's schedule,
`moe_align_block_size`'s sorted_token_ids (a pair's position / 8) and num_tokens_post_padded / 8: its ranks within an expert come
from atomicAdd (moe_align_sum_kernels.cu), so they are committed values of the run, not functions of the routing.

Registered functions (`_v1`):
  E2m1ToBf16, E8m0DequantBf16                 primitives: the exact e2m1 value, the e8m0 scale as bf16
  I32Mul, I32Div, I32Rem                      primitives: mul.lo.s32, div.s32, rem.s32 (C `int` * / %); a zero divisor gives
                                              quotient -1 and remainder a (a total extension: every divisor here is a positive count)
  MarlinMxfp4Group                            (q: 32 e2m1, s: e8m0) -> 32 bf16: one scale group's weights, the scale decoded once
  MarlinMxfp4Step{DOT}                        (acc: f32, x: 32 bf16, q, s) -> f32: one warp row's two k16 steps over one group
  MarlinMxfp4StepFromZero{DOT}                (x, q, s) -> f32: the same from +0 (k-tile 0)
  MarlinMoeSchedule{KT,NT,GRID}               (blk, par: i32) -> NT x (KT-1) bits: cut[J][t-1] = k-tile t of tile column J
                                              starts a segment
  MarlinMxfp4Coordinate{K,TK,DOT}             (x: K bf16, cut: KT-1 bits, wq: K e2m1, ws: K/32 e8m0) -> bf16: the w13 word
  MarlinMxfp4CoordinateW{K,TK,DOT}            (x, cut, g: bf16, wq, ws) -> bf16: the w2 word, times the routed weight
  GatherE2m1x{E}, GatherE8m0x{E}              (e: i32, column: E words) -> column[e]; e outside [0, E) -> 0 (e2m1), 0xFF (e8m0)
  MoeExpertE2m1Row{E,K}, MoeExpertE8m0Row{E,K}  (e, slab: [E][K]) -> [K] = slab[e]
  MarlinMoeExpertCoordinate(W){E,K,TK,DOT}    (x, cut, e, [g], wq: [E][K], ws: [E][K/32]) -> bf16: over the rows of expert e
  MarlinMoeExpertGemm{E,K,N,TK,TN,GRID,DOT}   (x, e, blk, par, wq: [E][N][K] e2m1, ws: [E][N][K/32] e8m0) -> [N] bf16: one pair's
                                              w13 output row
  MarlinMoeExpertGemmW{E,K,N,TK,TN,GRID,DOT}  (x, e, g, blk, par, wq, ws) -> [N] bf16: one pair's w2 output row

`wq` / `ws` are the layer's logical weights, [E][N][K] codes and [E][N][K/32] scale bytes (the checkpoint's, before Marlin's repack
and scale permutation at load: `marlin_utils_fp4.prepare_moe_mxfp4_layer_for_marlin`, `mxfp4_marlin_process_scales`, which for bf16
activations reorders scales and keeps their values); `x` the pair's bf16 activation row (w13: the token's hidden state; w2: the
pair's activation), `g` its f32 routed weight, which MarlinMoeExpertGemmW rounds to bf16 once for every coordinate.  Each
MarlinMoeExpertGemm(W) reads the schedule once and every column of a tile its cut bits.
"""
from __future__ import annotations

from verity.ml.circuits.defs import PrimitiveDefinition, bind, composite, primitive
from verity.ml.circuits.ops.prims import E2M1, F2fpBf16, HopperBF16WgmmaDot16, const
from verity.ml.circuits.ops.scalar import BIT, BitAnd, I32Add, I32Eq, I32Le, SelectF32, SelectI32
from verity.ml.circuits.refs import Coll, concat, strided_view
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.registry import fp8_moe_experts as X
from verity_vllm.program.registry.dsa import I32Sub
from verity_vllm.program.registry.prims import BF16, F32, I32, Bf16MulBf16, F32Add

M32 = 0xFFFFFFFF
E8M0 = Value(8)
#: the scale group along K (MXFP4's 32; Marlin's group_blocks 2)
GROUP = 32
#: the thread configs' TK (ops.cu:132-145): four or two warp rows
TKS = (64, 128)
#: the fills of an out-of-range expert's gathers: the zero code and e8m0's NaN code, so the dequantised word is NaN (0 * inf)
E2M1_FILL, E8M0_FILL = 0x0, 0xFF
#: the f32 bits of +0.0
ZERO = 0

_CONF = ("model-untested: read from vLLM d9105ea80's Marlin MoE (marlin_template.h, dequant.h, ops.cu) and the module docstring's "
         "lines; tests/properties/dsv4_marlin_capture_gpu.py is its hardware check")
DOT_CONF = _CONF + "; DOT = HopperBF16WgmmaDot16_v1, Hopper's mma.sync.m16n8k16 bf16"

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_to_bf16(q: int) -> int:
    """The bf16 word of e2m1 code q's value, exact (every e2m1 value is a bf16), the sign kept on zero (code 8 is -0)."""
    q &= 0xF
    sign, em = q >> 3, q & 7
    if em == 0:
        return sign << 15
    e, m = em >> 1, em & 1
    exp = 126 + e if e else 126
    mant = (m << 6) if e else 0
    return (sign << 15) | (exp << 7) | mant


def e8m0_to_bf16(s: int) -> int:
    """`dequant_fp8_scales<nv_bfloat162, kFE8M0fnu>` (dequant.h:551-562): the byte as the bf16 exponent field, sign and mantissa
    zero: code 0 is +0, 255 is +inf, else 2^(code - 127)."""
    return (s & 0xFF) << 7


@primitive("E2m1ToBf16", 1, [("q", E2M1)], BF16, conformance=_CONF)
def E2m1ToBf16(q: int) -> int:
    """`dequant<nv_bfloat162, kFE2M1f, false>` (dequant.h:433-470) on one code: its sign, exponent and mantissa bits placed into a
    bf16 word (exponent field e, mantissa m << 6), then `__hmul2` by 2^126, which is exact: the e2m1 value, -0 for code 8."""
    return e2m1_to_bf16(q)


@primitive("E8m0DequantBf16", 1, [("s", E8M0)], BF16, conformance=_CONF)
def E8m0DequantBf16(s: int) -> int:
    """`dequant_fp8_scales<nv_bfloat162, kFE8M0fnu>` on one byte: `e8m0_to_bf16`."""
    return e8m0_to_bf16(s)


def _signed(a: int) -> int:
    a &= M32
    return a - (1 << 32) if a & 0x80000000 else a


def i32_divrem(a: int, b: int) -> tuple[int, int]:
    """C `int` a / b and a % b (truncating toward zero, the remainder a's sign), modulo 2^32; INT_MIN / -1 wraps to INT_MIN with
    remainder 0, and b = 0 gives (-1, a)."""
    sa, sb = _signed(a), _signed(b)
    if sb == 0:
        return M32, a & M32
    q = abs(sa) // abs(sb)
    if (sa < 0) != (sb < 0):
        q = -q
    return q & M32, (sa - q * sb) & M32


@primitive("I32Mul", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Mul(a: int, b: int) -> int:
    """`mul.lo.s32`: a * b modulo 2^32."""
    return (a * b) & M32


@primitive("I32Div", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Div(a: int, b: int) -> int:
    """`div.s32` (C `int` /): `i32_divrem`'s quotient."""
    return i32_divrem(a, b)[0]


@primitive("I32Rem", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Rem(a: int, b: int) -> int:
    """`rem.s32` (C `int` %): `i32_divrem`'s remainder."""
    return i32_divrem(a, b)[1]


MARLIN_PRIMITIVES = [E2m1ToBf16, E8m0DequantBf16, I32Mul, I32Div, I32Rem]


def group_sig(e2m1=E2M1, e8m0=E8M0, bf16=BF16):
    return lambda S: ((("q", Array(GROUP, e2m1)), ("s", e8m0)), Array(GROUP, bf16))


@composite("MarlinMxfp4Group", 1, [], group_sig(), conformance=_CONF)
def MarlinMxfp4Group(B, S, q, s):
    """One scale group's 32 dequantised weights: `scale()`'s `__hmul2` (the template 115-123) of each e2m1 value and the group's
    e8m0 scale, a single bf16 rounding, which Bf16MulBf16_v1's f32 product then RNE equals (the product of two bf16 words is exact
    in f32)."""
    sb = B.call(E8m0DequantBf16, s)
    return B.batch(Bf16MulBf16, B.batch(E2m1ToBf16, q, axes=(0,)), sb, axes=(0, None))


def step_sig(from_zero: bool, e2m1=E2M1, e8m0=E8M0, bf16=BF16, f32=F32):
    acc = () if from_zero else (("acc", f32),)
    return lambda S: (acc + (("x", Array(GROUP, bf16)), ("q", Array(GROUP, e2m1)), ("s", e8m0)), f32)


def _two_steps(B, dot, acc, x, w):
    for j in range(2):
        acc = B.call(dot, acc, x[16 * j:16 * j + 16], w[16 * j:16 * j + 16])
    return acc


@composite("MarlinMxfp4Step", 1, ["DOT"], step_sig(False), conformance=DOT_CONF)
def MarlinMxfp4Step(B, S, acc, x, q, s):
    """One warp row's run over one k-tile: its two k16 `DOT` steps (j = 0, 1) over the 32 k of one scale group, the weights
    `MarlinMxfp4Group`'s."""
    return _two_steps(B, S.DOT, acc, x, B.call(MarlinMxfp4Group, q, s))


@composite("MarlinMxfp4StepFromZero", 1, ["DOT"], step_sig(True), conformance=DOT_CONF)
def MarlinMxfp4StepFromZero(B, S, x, q, s):
    """`MarlinMxfp4Step_v1` from the +0 accumulator (a warp row's run over k-tile 0): the Boolean version's own constant, so that
    its circuit folds the accumulator rather than reading 32 copies of one zero bit."""
    return _two_steps(B, S.DOT, B.call(const(32, ZERO)), x, B.call(MarlinMxfp4Group, q, s))


def check_statics(K: int, TK: int) -> None:
    if TK not in TKS or K % TK or K < 2 * TK:
        raise ValueError(f"a Marlin coordinate reads K = TK * KT with TK in {TKS} and KT >= 2, not K = {K}, TK = {TK}")


def check_schedule(KT: int, NT: int, GRID: int) -> None:
    if KT < 2 or NT < 1 or GRID < 1:
        raise ValueError(f"a Marlin schedule has KT >= 2 k-tiles, NT >= 1 tile columns and GRID >= 1 CTAs, not {KT}, {NT}, {GRID}")


def warp_tree(add, acc: list):
    """`thread_block_reduce` (the template 1539-1593) over the W warp rows' accumulators: for i = W/2, W/4, ..., 1 the rows
    [i, 2i) store, those below W/2 having first added (stored[r] + stored[r - i]); row 0 then adds the last stored value."""
    W = len(acc)
    vals, slots = list(acc), [None] * W
    i = W // 2
    while i > 0:
        for r in range(i, 2 * i):
            if i < W // 2:
                vals[r] = add(vals[r], add(slots[r], slots[r - i]))
            slots[r - i] = vals[r]
        i //= 2
    return add(vals[0], slots[0])


class Ops:
    """The steps a coordinate takes: ``step(acc, x32, q32, s, g)`` (`MarlinMxfp4Step` over scale group g),
    ``first(zero, x32, q32, s, g)`` (the step from +0: ``first`` the Definition that starts there, else ``step`` on ``zero``),
    ``add``, ``select(c, a, b)`` (32-bit words, c a cut bit) and ``zero()`` (+0.0, made once per coordinate); `word_ops` binds the
    word Definitions, `boolean_marlin_moe` their Boolean ones, so the body is written once."""

    def __init__(self, B, step, first, add, select, zero, bit=lambda c: c):
        self.B, self._step, self._first, self._add, self._select, self._zero = B, step, first, add, select, zero
        self._bit = bit

    def step(self, acc, x, q, s, g):
        return self.B.call(self._step, acc, x, q, s)

    def first(self, zero, x, q, s, g):
        return self.step(zero, x, q, s, g) if self._first is None else self.B.call(self._first, x, q, s)

    def add(self, a, b):
        return self.B.call(self._add, a, b)

    def select(self, c, a, b):
        return self.B.call(self._select, self._bit(c), a, b)

    def zero(self):
        return self._zero()


def word_ops(B, S) -> Ops:
    return Ops(B, bind(MarlinMxfp4Step, DOT=S.DOT), None, F32Add, SelectF32, lambda: B.call(const(32, ZERO)))


def marlin_acc(o: Ops, K: int, TK: int, x, cut, wq, ws):
    """The f32 R of one coordinate: each warp row's k16 chain, one `MarlinMxfp4Step` per k-tile (its 32 k are one scale group),
    zeroed where a segment starts (`cut[t-1]` for k-tile t), the warp tree at every k-tile that ends a segment, and those trees
    folded from the highest k down."""
    check_statics(K, TK)
    W, KT = TK // 32, K // TK
    zero = o.zero()
    acc: list = [None] * W
    trees = []
    for t in range(KT):
        for r in range(W):
            k = t * TK + GROUP * r
            g = k // GROUP
            xs, q, s = x[k:k + GROUP], wq[k:k + GROUP], ws[g]
            acc[r] = o.first(zero, xs, q, s, g) if t == 0 else o.step(o.select(cut[t - 1], zero, acc[r]), xs, q, s, g)
        trees.append(warp_tree(o.add, acc))
    R = trees[-1]
    for t in reversed(range(KT - 1)):
        R = o.select(cut[t], o.add(trees[t], R), R)
    return R


def coordinate_sig(routed: bool, e2m1=E2M1, e8m0=E8M0, bf16=BF16, bit=BIT):
    g = (("g", bf16),) if routed else ()
    return lambda S: ((("x", Array(S.K, bf16)), ("cut", Array(S.K // S.TK - 1, bit))) + g
                      + (("wq", Array(S.K, e2m1)), ("ws", Array(S.K // GROUP, e8m0))), bf16)


@composite("MarlinMxfp4Coordinate", 1, ["K", "TK", "DOT"], coordinate_sig(False), conformance=DOT_CONF)
def MarlinMxfp4Coordinate(B, S, x, cut, wq, ws):
    """One output word of a w13 launch: F2fpBf16 of `marlin_acc` (the module docstring)."""
    return B.call(F2fpBf16, marlin_acc(word_ops(B, S), S.K, S.TK, x, cut, wq, ws))


@composite("MarlinMxfp4CoordinateW", 1, ["K", "TK", "DOT"], coordinate_sig(True), conformance=DOT_CONF)
def MarlinMxfp4CoordinateW(B, S, x, cut, g, wq, ws):
    """One output word of a w2 launch: the bf16 word times the routed weight g, already bf16 (`__hmul2`, write_result 1880-1887;
    `MarlinMoeExpertGemmW_v1` rounds the router's f32 weight once per pair)."""
    y = B.call(F2fpBf16, marlin_acc(word_ops(B, S), S.K, S.TK, x, cut, wq, ws))
    return B.call(Bf16MulBf16, y, g)


class IntOps:
    """The schedule's integer steps, ``i(v)`` a constant word; `word_int_ops` binds the word primitives."""

    def __init__(self, B, word, defs: dict):
        self.B, self._word, self._defs, self._words = B, word, defs, {}

    def i(self, v: int):
        v &= M32
        if v not in self._words:
            self._words[v] = self._word(v)
        return self._words[v]

    def __getattr__(self, name: str):
        d = self._defs[name]
        return lambda *a: self.B.call(d, *a)


WORD_INT = dict(add=I32Add, sub=I32Sub, mul=I32Mul, div=I32Div, rem=I32Rem, le=I32Le, eq=I32Eq, sel=SelectI32, band=BitAnd)


def word_int_ops(B) -> IntOps:
    return IntOps(B, lambda v: B.call(const(32, v)), WORD_INT)


def schedule(o: IntOps, KT: int, NT: int, GRID: int, blk, par) -> list[list]:
    """The cut bits of every tile column J of one pair's launch (the module docstring's partition): G, p2, iters and the first
    part-2 tile once, then per J the segment phase r = (j2 * KT) % iters and, per k-tile t > 0, r = r + 1 wrapping at iters, the cut
    where it wraps, in part 2 only."""
    check_schedule(KT, NT, GRID)
    G = o.mul(par, o.i(NT))
    r = o.rem(G, o.i(GRID))
    r = o.sel(o.le(o.mul(r, o.i(3)), o.i(GRID)), o.add(r, o.i(GRID)), r)
    p2 = o.sel(o.le(G, o.i(GRID)), G, r)
    iters = o.div(o.add(o.mul(p2, o.i(KT)), o.i(GRID - 1)), o.i(GRID))
    first = o.sub(G, p2)
    b0 = o.mul(blk, o.i(NT))
    out = []
    for J in range(NT):
        g = o.add(b0, o.i(J)) if J else b0
        part2 = o.le(first, g)
        r = o.rem(o.mul(o.sub(g, first), o.i(KT)), iters)
        row = []
        for t in range(1, KT):
            r = o.add(r, o.i(1))
            wrap = o.eq(r, iters)
            row.append(o.band(part2, wrap))
            if t < KT - 1:
                r = o.sel(wrap, o.i(0), r)
        out.append(row)
    return out


def schedule_sig(i32=I32, bit=BIT):
    return lambda S: ((("blk", i32), ("par", i32)), Array(S.NT, Array(S.KT - 1, bit)))


def schedule_of(S) -> dict:
    return {"KT": S.K // S.TK, "NT": S.N // S.TN, "GRID": S.GRID}


@composite("MarlinMoeSchedule", 1, ["KT", "NT", "GRID"], schedule_sig(), conformance=_CONF)
def MarlinMoeSchedule(B, S, blk, par):
    """The segment starts of one pair's launch: `schedule` (the module docstring's partition) over its moe block `blk` and the
    launch's `par` = num_tokens_post_padded / moe_block_size."""
    rows = schedule(word_int_ops(B), S.KT, S.NT, S.GRID, blk, par)
    return Coll(Array(S.NT, Array(S.KT - 1, BIT)), concat([b.refs for row in rows for b in row]))


def gather_e2m1(vocab: int) -> PrimitiveDefinition:
    """`GatherE2m1x{V}_v1`: column[token] of V e2m1 codes, else 0."""
    return X._gather("GatherE2m1", vocab, E2M1, E2M1_FILL)


def gather_e8m0(vocab: int) -> PrimitiveDefinition:
    """`GatherE8m0x{V}_v1`: column[token] of V e8m0 bytes, else 0xFF."""
    return X._gather("GatherE8m0", vocab, E8M0, E8M0_FILL)


@composite("MoeExpertE2m1Row", 1, ["E", "K"], X.row_sig(E2M1), conformance="exact-model-tested")
def MoeExpertE2m1Row(B, S, e, slab):
    """`wrow[k] = slab[e][k]`: one `GatherE2m1x{E}_v1` per k over the E experts' codes."""
    return B.batch(gather_e2m1(S.E), e, X.columns(slab, S.E, S.K, E2M1), axes=(None, 0))


@composite("MoeExpertE8m0Row", 1, ["E", "K"], X.row_sig(E8M0), conformance="exact-model-tested")
def MoeExpertE8m0Row(B, S, e, slab):
    """`srow[k] = slab[e][k]`: one `GatherE8m0x{E}_v1` per scale group over the E experts' scale bytes."""
    return B.batch(gather_e8m0(S.E), e, X.columns(slab, S.E, S.K, E8M0), axes=(None, 0))


def pair_coordinate_sig(routed: bool, e2m1=E2M1, e8m0=E8M0, bf16=BF16, i32=I32, bit=BIT):
    g = (("g", bf16),) if routed else ()
    return lambda S: ((("x", Array(S.K, bf16)), ("cut", Array(S.K // S.TK - 1, bit)), ("e", i32)) + g
                      + (("wq", Array(S.E, Array(S.K, e2m1))), ("ws", Array(S.E, Array(S.K // GROUP, e8m0)))), bf16)


def coordinate(S) -> dict:
    return {"K": S.K, "TK": S.TK, "DOT": S.DOT}


@composite("MarlinMoeExpertCoordinate", 1, ["E", "K", "TK", "DOT"], pair_coordinate_sig(False), conformance=DOT_CONF)
def MarlinMoeExpertCoordinate(B, S, x, cut, e, wq, ws):
    """One output word of the w13 launch of a pair routed to expert e: `MarlinMxfp4Coordinate_v1` over expert e's rows (`wq`, `ws`
    the column's codes and scale bytes of every expert)."""
    q = B.call(bind(MoeExpertE2m1Row, E=S.E, K=S.K), e, wq)
    s = B.call(bind(MoeExpertE8m0Row, E=S.E, K=S.K // GROUP), e, ws)
    return B.call(bind(MarlinMxfp4Coordinate, **coordinate(S)), x, cut, q, s)


@composite("MarlinMoeExpertCoordinateW", 1, ["E", "K", "TK", "DOT"], pair_coordinate_sig(True), conformance=DOT_CONF)
def MarlinMoeExpertCoordinateW(B, S, x, cut, e, g, wq, ws):
    """One output word of the w2 launch of a pair: `MarlinMxfp4CoordinateW_v1` over expert e's rows."""
    q = B.call(bind(MoeExpertE2m1Row, E=S.E, K=S.K), e, wq)
    s = B.call(bind(MoeExpertE8m0Row, E=S.E, K=S.K // GROUP), e, ws)
    return B.call(bind(MarlinMxfp4CoordinateW, **coordinate(S)), x, cut, g, q, s)


GEMM_STATICS = ["E", "K", "N", "TK", "TN", "GRID", "DOT"]


def check_gemm(S) -> None:
    check_statics(S.K, S.TK)
    if S.TN < 1 or S.N % S.TN:
        raise ValueError(f"a Marlin launch's N = {S.N} is a whole number of TN = {S.TN} tile columns")


def gemm_sig(routed: bool, e2m1=E2M1, e8m0=E8M0, bf16=BF16, f32=F32, i32=I32):
    g = (("g", f32),) if routed else ()
    return lambda S: ((("x", Array(S.K, bf16)), ("e", i32)) + g + (("blk", i32), ("par", i32))
                      + (("wq", Array(S.E, Array(S.N, Array(S.K, e2m1)))), ("ws", Array(S.E, Array(S.N, Array(S.K // GROUP, e8m0))))),
                      Array(S.N, bf16))


def pair_rows(B, S, sched, coord, lead: tuple, blk, par, wq, ws, word=BF16, e2m1=E2M1, e8m0=E8M0) -> Coll:
    """The N words of one pair: the schedule once, then per tile column J `coord` over its TN columns, each reading `lead` and J's
    cut bits whole and its [E][K] code and [E][K/32] scale slabs."""
    check_gemm(S)
    E, K, N, TN = S.E, S.K, S.N, S.TN
    KG, NT = K // GROUP, N // TN
    cuts = B.call(sched, blk, par)
    Lq, Ls = e2m1.leaves, e8m0.leaves
    parts = []
    for J in range(NT):
        n0, n1 = J * TN, (J + 1) * TN
        q = Coll(Array(TN, Array(E, Array(K, e2m1))),
                 concat([strided_view(wq, Lq * n * K, Lq * K, Lq * N * K, 1, Lq * E * K) for n in range(n0, n1)]))
        s = Coll(Array(TN, Array(E, Array(KG, e8m0))),
                 concat([strided_view(ws, Ls * n * KG, Ls * KG, Ls * N * KG, 1, Ls * E * KG) for n in range(n0, n1)]))
        parts.append(B.batch(coord, *lead[:1], cuts[J], *lead[1:], q, s, axes=(None,) * (len(lead) + 1) + (0, 0)))
    return Coll(Array(N, word), concat([p.refs for p in parts]))


@composite("MarlinMoeExpertGemm", 1, GEMM_STATICS, gemm_sig(False), conformance=DOT_CONF)
def MarlinMoeExpertGemm(B, S, x, e, blk, par, wq, ws):
    """The w13 launch (`mul_topk_weights` False) on one (token, slot) pair routed to expert e, its moe block `blk` of `par`:
    y[n] = `MarlinMoeExpertCoordinate_v1`(x, cut[n / TN], e, wq[:, n], ws[:, n]), the cuts `MarlinMoeSchedule_v1`'s."""
    coord = bind(MarlinMoeExpertCoordinate, E=S.E, **coordinate(S))
    return pair_rows(B, S, bind(MarlinMoeSchedule, **schedule_of(S)), coord, (x, e), blk, par, wq, ws)


@composite("MarlinMoeExpertGemmW", 1, GEMM_STATICS, gemm_sig(True), conformance=DOT_CONF)
def MarlinMoeExpertGemmW(B, S, x, e, g, blk, par, wq, ws):
    """The w2 launch (top_k 1, `mul_topk_weights` True) on one pair: y[n] = `MarlinMoeExpertCoordinateW_v1`(x, cut[n / TN], e,
    F2fpBf16(g), wq[:, n], ws[:, n]), the router's f32 weight g rounded once."""
    coord = bind(MarlinMoeExpertCoordinateW, E=S.E, **coordinate(S))
    gb = B.call(F2fpBf16, g)
    return pair_rows(B, S, bind(MarlinMoeSchedule, **schedule_of(S)), coord, (x, e, gb), blk, par, wq, ws)


#: the step DSV4's Programs bind (`DOT`)
DOT = HopperBF16WgmmaDot16
