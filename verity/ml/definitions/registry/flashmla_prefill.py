"""[DeepSeek-V4-Pro] The sparse MLA prefill attention as the H200 runs it: DeepSeek-V4-Pro at TP8 on sm_90, vLLM d9105ea80's
`DeepseekV4FlashMLA._forward_prefill` (vllm/models/deepseek_v4/nvidia/flashmla.py:266-398) over its K-cache gather
`DequantGatherKCacheKernel` (vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py, the path `has_cutedsl()` takes,
common/ops/cache_utils.py:390-408) and the FlashMLA it vendors (6bc49418): `flash_mla_sparse_fwd`'s sm_90 kernel
`sm90::fwd::KernelTemplate<512, true>` (csrc/sm90/prefill/sparse/phase1.cuh, config.h; `phase1_k512_topklen`).  Line numbers are
at those commits; "the SASS" is that instance's sm_90a SASS (nvcc without fast-math: FADD / FMUL / FFMA round to nearest with
subnormals, every NaN result 0x7FFFFFFF).  `registry.flashmla` is the decode; this module reuses its libm pieces and steps.

The call (flashmla.py:316-398): per chunk of prefill requests the gather writes each request's compressed rows and window rows
into a bf16 workspace `kv` of s_kv = chunk_size * chunk_M rows of 512; `combine_topk_swa_indices` gives each query token its
`combined_topk` = round_up(top_k + 128 + max_image_tokens, 128) workspace indices and its length `topk_length`;
`flash_mla_sparse_fwd(q, kv, indices, softmax_scale, attn_sink, topk_length, out)` then runs one CTA per token and 64 heads (the
rank's 16 real heads padded to 64, as in the decode).  K = V = a workspace row of D = 512 bf16; the output is 512 bf16 per head
(the kernel's lse and max_logits are not read).

1. The gather (dequant_gather_k_cutedsl.py:243-298), per cache row: NoPE element d is the bf16 multiply (`mul.rn.bf16x2`, :277-280)
   of the E4M3 byte widened to bf16 (`cvt.rn.f16x2.e4m3x2` then `cvt.rn.bf16.f16`, vllm/cute_utils/cvt.py:58-90: exact on every
   E4M3 word but a NaN, whose product is the multiply's NaN either way, so `F2fpBf16(E4m3ToF32(x))`) by the scale of its 64
   elements, the UE8M0 byte as the bf16 word `s << 7` (:261-266: 0 is +0 and 0xFF +inf, unlike the decode's
   `__nv_cvt_e8m0_to_bf16raw`); the RoPE elements are copied (:285-291): `FlashMlaGatherRow_v1`.
2. Rows (the producer warpgroup, phase1.cuh:477-513).  Row r < N = 2 * NP * 64 of a token in kernel order is valid iff its index
   t satisfies t >= 0 and t < s_kv and r < topk_length, signed (:485-488): `FlashMlaPrefillValid_v1`; an invalid row's cp.async
   reads nothing and zero-fills (:505-509), so its K = V row is +0 and its score is masked to -inf (:119-131).  The token's
   NB = ceil(topk_length / 64) blocks (:80-81) are taken in pairs, the pair's second block past topk_length when NB is odd: NP =
   ceil(NB / 2) pairs, every row of such a block invalid.
3. A pair (:306-438), per head, from the state m = -1e30, l = 0, o = 0 (:96-101).  Warpgroup 0 takes the pair's first block and
   output columns d < 256, warpgroup 1 the second block and d >= 256 (the kernel's D_V / 2; D / 2 here):
       s0[c] = DotBf16{K = 512, DOT}(q, k0[c]) over d ascending, s1[c] the same chain over d = 256..511 then 0..255 (:262-281,
               :380-395; each a zero-initialised wgmma chain), invalid rows -inf
       c_i   = FMUL(fmaxf over {-inf} and s_i, scale_log2) (:144-152; the fold is order-free)
       m0 = fmaxf(m, c0);  m1 = fmaxf(m0, c1)  (warpgroup 1 reads m0 from sM, :137-157)
       so_i  = exp2f(FADD(m, -m_i));  P_i[c] = exp2f(FFMA(s_i[c], scale_log2, -m_i));  sf = exp2f(FADD(m0, -m1)) (:339)
       lane t of warpgroup i: u_i[t] = the FADD chain over j ascending of FADD(P_i[8j + 2t], P_i[8j + 2t + 1]) (:166-175; its
               leading FADD(+0, x) is x, x never -0, so it is not written)
       l0[t] = FMUL(FFMA(l0[t], so0, u0[t]), sf) (:175, :302);  l1[t] = FFMA(l1[t], so1, u1[t])
       o[d < 256]  = FMUL(o, so0), the 4 k16 wgmma steps of bf16(P0) . V0, FMUL(., sf), then bf16(P1) . V1 (:160-163, :326, :351-352)
       o[d >= 256] = FMUL(o, so1), bf16(P1) . V1, then bf16(FMUL(P0, sf)) . V0 (:342, :416, :422)
       m = m1: `FlashMlaPrefillPair_v1`.  On the first pair m = -1e30 makes the rescale an identity (FMUL(0, so) = +0 and
       FFMA(0, so, u) = u: so is finite there), so it is not written.
4. The end (reduce_L :186-199, store_O :201-229): L = FADD(FADD(FADD(l0[0], l0[1]), FADD(l0[2], l0[3])), the same over l1)
   (the xor butterfly, then the peer warpgroup's sum: FADD is commutative, so both warpgroups hold one word); out =
   bf16(FMUL(o, L == 0 ? 0 : 1.0f / FADD(L, exp2f(FADD(FMUL(sink, L2E), -m))))), the divide IEEE `div.rn.f32` (no fast-math):
   `FlashMlaPrefillHead_v1`, and over a token's heads `FlashMlaPrefill_v1`.

Registered functions (all `Name_v1`):
  E8m0ShlBf16                                  primitive: the gather's scale word, s << 7
  FlashMlaGatherRow{DN,DR,QT}                  (nope: DN e4m3, scale: DN/QT e8m0, rope: DR bf16) -> DN+DR bf16
  FlashMlaPrefillValid                         (idx: i32, len: i32, pos: i32, skv: i32) -> bit
  FlashMlaPrefillPair{D,BN,FIRST,DOT,SCALE}    (q: D bf16, k: 2*BN rows[, -m, l[8], o[D]]) -> (-m, l[8], o[D])
  FlashMlaPrefillHead{NP,D,BN,DOT,SCALE}       (q: D bf16, k: N rows, sink: f32) -> D bf16
  FlashMlaPrefill{NH,NP,D,BN,DOT,SCALE}        (q: NH x D bf16, kv: N x D bf16, idx: N i32, len, skv: i32, sink: NH f32)
                                               -> NH x D bf16

A row is a staged row (`flashmla`: D bf16, valid bit); N = 2 * NP * BN, NP the token's pairs (ceil(ceil(topk_length / 64) / 2):
the statics of a token's launch).  l[8] is warpgroup 0's four lanes, then warpgroup 1's; the state carries the running max
negated, as the decode's block does.  `FLASHMLA_PREFILL_DSV4` holds DeepSeek-V4-Pro's statics.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.fp32 import F32DivV2
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity.ml.definitions.numerics import flashmla_prefill_twin
from verity.ml.definitions.registry import flashmla as F
from verity.ml.definitions.registry.prims import BF16, BIT, I32

_CONF = ("model-untested: read from vLLM d9105ea80's sparse MLA prefill and its K-cache gather and FlashMLA 6bc49418's Hopper sparse "
         "prefill (the module docstring's files) and the prefill kernel's SASS; tests/properties/dsv4_flashmla_capture_gpu.py is its "
         "hardware check")

#: DeepSeek-V4-Pro at TP8 on the H200: 16 of the 64 padded heads per rank, the 512-element workspace row (NoPE 448 + RoPE 64, one
#: scale per 64 NoPE elements), 64-row blocks, softmax_scale 512^-0.5
FLASHMLA_PREFILL_DSV4 = {"NH": 16, "D": 512, "BN": 64, "DOT": HopperBF16WgmmaDot16, "SCALE": 512 ** -0.5}
FLASHMLA_GATHER_DSV4 = {"DN": 448, "DR": 64, "QT": 64}


@primitive("E8m0ShlBf16", 1, [("x", F.E8M0)], BF16, conformance=_CONF)
def E8m0ShlBf16(x: int) -> int:
    """A UE8M0 scale byte as the bf16 word the prefill gather multiplies by: x << 7, the bf16 2^(x - 127) for x in 1..254, +0 for 0
    and +inf for 0xFF (dequant_gather_k_cutedsl.py:261-266)."""
    return (x & 0xFF) << 7


FLASHMLA_PREFILL_PRIMITIVES = [E8m0ShlBf16]


# ---- the gather -------------------------------------------------------------------------------------------------------------

def gather_row(o: F.Ops, S, nope, scale, rope):
    """`FlashMlaGatherRow_v1`'s body."""
    DN, DR, QT = S.DN, S.DR, S.QT
    F.check_row_statics(DN, DR, QT, "FlashMlaGatherRow")
    sc = [o.shl(scale[j]) for j in range(DN // QT)]
    return array_of([o.bmul(o.cvt(o.e4m3(nope[d])), sc[d // QT]) for d in range(DN)] + [rope[d] for d in range(DR)])


def _gather_type(S, T: F.Types = F.WORD_T):
    return ((("nope", Array(S.DN, T.e4m3)), ("scale", Array(S.DN // S.QT, T.e8m0)), ("rope", Array(S.DR, T.bf16))),
            Array(S.DN + S.DR, T.bf16))


@composite("FlashMlaGatherRow", 1, ["DN", "DR", "QT"], _gather_type, conformance=_CONF)
def FlashMlaGatherRow(B, S, nope, scale, rope):
    """One cache row as the prefill gather writes it to the workspace: NoPE element d `Bf16MulBf16(F2fpBf16(E4m3ToF32(nope[d])),
    E8m0ShlBf16(scale[d / QT]))`, then the RoPE elements as stored (the module docstring's step 1)."""
    return gather_row(word_ops(B), S, nope, scale, rope)


# ---- the rows ---------------------------------------------------------------------------------------------------------------

def prefill_valid(o: F.Ops, idx, n, pos, skv):
    """`FlashMlaPrefillValid_v1`'s body."""
    return o.band(o.band(o.ile(o.word(0), idx), o.bnot(o.ile(skv, idx))), o.bnot(o.ile(n, pos)))


@composite("FlashMlaPrefillValid", 1, [], lambda S: ((("idx", I32), ("len", I32), ("pos", I32), ("skv", I32)), BIT), conformance=_CONF)
def FlashMlaPrefillValid(B, S, idx, n, pos, skv):
    """The row at position pos of a token whose length is len is valid iff its index lies in [0, skv) and pos < len, signed
    (phase1.cuh:485-488)."""
    return prefill_valid(word_ops(B), idx, n, pos, skv)


# ---- one pair, one head -------------------------------------------------------------------------------------------------------

def check_pair_statics(D_: int, BN: int) -> None:
    if D_ % 32 or D_ < 32 or BN % 16 or BN < 16:
        raise ValueError(f"FlashMlaPrefillPair: D={D_} must be a multiple of 32 (two halves of k16 steps) and BN={BN} of 16")


def lane_sums(o: F.Ops, p, BN: int) -> list:
    """The four lanes' sums of one warpgroup's P (lane t: columns 8j + 2t, 8j + 2t + 1), pairs first."""
    lanes = []
    for t in range(4):
        pairs = [o.add(p[8 * j + 2 * t], p[8 * j + 2 * t + 1]) for j in range(BN // 8)]
        acc = pairs[0]
        for x in pairs[1:]:
            acc = o.add(acc, x)
        lanes.append(acc)
    return lanes


def pv(o: F.Ops, S, acc, pb, k, b: int, cols: range) -> list:
    """``acc`` (the f32 columns ``cols``) after the k16 wgmma steps, rows ascending, of the bf16 P ``pb`` against the V columns
    ``cols`` of block ``b`` of the staged rows ``k``."""
    B, BN, T, L = o.B, S.BN, o.T, o.leaf
    RS = S.D * L + 1
    for ks in range(BN // 16):
        r0 = b * BN + 16 * ks
        prow = Coll(Array(16, T.bf16), concat([pb[16 * ks + j].refs for j in range(16)]))
        vt = Coll(Array(len(cols), Array(16, T.bf16)), concat([strided_view(k, r0 * RS + d * L, L, RS, 1, 16 * L) for d in cols]))
        acc = list(B.batch(S.DOT, array_of(acc), prow, vt, axes=(0, None, 0)))
    return acc


def pair(o: F.Ops, S, dot, q, k, nm_prev=None, l_prev=None, o_prev=None):
    """`FlashMlaPrefillPair_v1`'s body over ``o``'s steps, ``dot`` the score chain (`DotBf16{K = D, DOT}`)."""
    B, D_, BN, FIRST = o.B, S.D, S.BN, S.FIRST
    check_pair_statics(D_, BN)
    H = D_ // 2
    scale, ninf, mul = o.word(F.scale_log2_bits(S.SCALE)), o.word(F.F_NINF), o._defs["mul"]
    rot = lambda x: Coll(x.type, concat([x[H:].refs, x[:H].refs]))  # noqa: E731
    qr = rot(q)
    s0 = [o.sel(k[c][1], B.call(dot, q, k[c][0]), ninf) for c in range(BN)]
    s1 = [o.sel(k[BN + c][1], B.call(dot, qr, rot(k[BN + c][0])), ninf) for c in range(BN)]
    cur = []
    for s in (s0, s1):
        mx = ninf
        for x in s:
            mx = o.fmax(mx, x)
        cur.append(o.mul(mx, scale))
    m = o.word(F.F_MAX_INIT) if FIRST else o.neg(nm_prev)
    m0 = o.fmax(m, cur[0])
    nm0 = o.neg(m0)
    m1 = o.fmax(m0, cur[1])
    nm1 = o.neg(m1)
    sf = o.exp2(o.add(m0, nm1))
    p0 = [o.exp2(o.fma(x, scale, nm0)) for x in s0]
    p1 = [o.exp2(o.fma(x, scale, nm1)) for x in s1]
    u0, u1 = lane_sums(o, p0, BN), lane_sums(o, p1, BN)
    if FIRST:
        l0, l1 = [o.mul(u, sf) for u in u0], u1
        lo, hi = [o.word(0)] * H, [o.word(0)] * (D_ - H)
    else:
        so0, so1 = o.exp2(o.add(m, nm0)), o.exp2(o.add(m, nm1))
        l0 = [o.mul(o.fma(l_prev[t], so0, u0[t]), sf) for t in range(4)]
        l1 = [o.fma(l_prev[4 + t], so1, u1[t]) for t in range(4)]
        lo = list(B.batch(mul, o_prev[:H], so0, axes=(0, None)))
        hi = list(B.batch(mul, o_prev[H:], so1, axes=(0, None)))
    pb1 = [o.cvt(x) for x in p1]
    lo = pv(o, S, lo, [o.cvt(x) for x in p0], k, 0, range(H))
    lo = pv(o, S, list(B.batch(mul, array_of(lo), sf, axes=(0, None))), pb1, k, 1, range(H))
    hi = pv(o, S, hi, pb1, k, 1, range(H, D_))
    hi = pv(o, S, hi, [o.cvt(o.mul(x, sf)) for x in p0], k, 0, range(H, D_))
    return tuple_of(nm1, array_of(l0 + l1), array_of(lo + hi))


def _pair_type(S, T: F.Types = F.WORD_T):
    state = () if S.FIRST else (("nm_prev", T.f32), ("l_prev", Array(8, T.f32)), ("o_prev", Array(S.D, T.f32)))
    return ((("q", Array(S.D, T.bf16)), ("k", Array(2 * S.BN, F._staged(S.D, T)))) + state,
            Tuple(T.f32, Array(8, T.f32), Array(S.D, T.f32)))


@composite("FlashMlaPrefillPair", 1, ["D", "BN", "FIRST", "DOT", "SCALE"], _pair_type, conformance=_CONF)
def FlashMlaPrefillPair(B, S, q, k, nm_prev=None, l_prev=None, o_prev=None):
    """One pair of BN-row blocks of the prefill's online softmax for one head, the two warpgroups' halves of the output (the
    module docstring's step 3).  The first pair (FIRST) starts from the constant state m = -1e30, l = 0, o = 0 and takes no state."""
    return pair(word_ops(B), S, bind(DotBf16V2, K=S.D, DOT=S.DOT), q, k, nm_prev, l_prev, o_prev)


def check_head_statics(NP: int) -> None:
    if NP < 1:
        raise ValueError(f"FlashMlaPrefillHead: NP={NP} pairs (a token's topk_length is at least 1)")


def head(o: F.Ops, S, pair_of, q, k, sink):
    """`FlashMlaPrefillHead_v1`'s body, ``pair_of(first)`` the bound pair."""
    B, NP, BN = o.B, S.NP, S.BN
    check_head_statics(NP)
    nm = l = acc = None
    for j in range(NP):
        kj = k[2 * j * BN:2 * (j + 1) * BN]
        res = B.call(pair_of(True), q, kj) if j == 0 else B.call(pair_of(False), q, kj, nm, l, acc)
        nm, l, acc = res[0], res[1], res[2]
    total = o.add(o.add(o.add(l[0], l[1]), o.add(l[2], l[3])), o.add(o.add(l[4], l[5]), o.add(l[6], l[7])))
    den = o.add(total, o.exp2(o.add(o.mul(sink, o.word(F.L2E)), nm)))
    scale = o.sel(o.eq(total, o.word(0)), o.word(0), o.div(o.word(F.F_ONE), den))
    return B.batch(o._defs["cvt"], B.batch(o._defs["mul"], acc, scale, axes=(0, None)), axes=(0,))


def _head_type(S, T: F.Types = F.WORD_T):
    return ((("q", Array(S.D, T.bf16)), ("k", Array(2 * S.NP * S.BN, F._staged(S.D, T))), ("sink", T.f32)), Array(S.D, T.bf16))


@composite("FlashMlaPrefillHead", 1, ["NP", "D", "BN", "DOT", "SCALE"], _head_type, conformance=_CONF)
def FlashMlaPrefillHead(B, S, q, k, sink):
    """One head of one prefill token over its N staged rows: NP `FlashMlaPrefillPair_v1`s in ascending order from the constant
    state, then the module docstring's step 4, the bf16 head output with the sink folded in."""
    def pair_of(first):
        return bind(FlashMlaPrefillPair, D=S.D, BN=S.BN, FIRST=first, DOT=S.DOT, SCALE=S.SCALE)
    return head(word_ops(B), S, pair_of, q, k, sink)


# ---- the token ----------------------------------------------------------------------------------------------------------------

def prefill(o: F.Ops, S, valid_of, head_of, q, kv, idx, n, skv, sink):
    """`FlashMlaPrefill_v1`'s body: each row's validity and its masking once for every head."""
    B, T, D_ = o.B, o.T, S.D
    N = 2 * S.NP * S.BN
    z = o.half(0)
    valid = [B.call(valid_of, idx[r], n, o.word(r), skv) for r in range(N)]
    rows = [B.batch(o._defs["bsel"], valid[r], kv[r], z, axes=(None, 0, None)) for r in range(N)]
    staged = Coll(Array(N, F._staged(D_, T)), concat([x for r in range(N) for x in (rows[r].refs, valid[r].refs)]))
    return B.batch(head_of, q, staged, sink, axes=(0, None, 0))


def _prefill_type(S, T: F.Types = F.WORD_T):
    N = 2 * S.NP * S.BN
    return ((("q", Array(S.NH, Array(S.D, T.bf16))), ("kv", Array(N, Array(S.D, T.bf16))), ("idx", Array(N, T.i32)), ("len", T.i32),
             ("skv", T.i32), ("sink", Array(S.NH, T.f32))), Array(S.NH, Array(S.D, T.bf16)))


@composite("FlashMlaPrefill", 1, ["NH", "NP", "D", "BN", "DOT", "SCALE"], _prefill_type, conformance=_CONF)
def FlashMlaPrefill(B, S, q, kv, idx, n, skv, sink):
    """One prefill token's NH heads over its N workspace rows in kernel order, given each row's index: each row
    `FlashMlaPrefillValid_v1` and its masking (an invalid row is +0) once, then head h `FlashMlaPrefillHead_v1` over q[h] and
    sink[h]."""
    head_of = bind(FlashMlaPrefillHead, NP=S.NP, D=S.D, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return prefill(word_ops(B), S, FlashMlaPrefillValid, head_of, q, kv, idx, n, skv, sink)


WORD = {**F.WORD, "div": F32DivV2, "shl": E8m0ShlBf16}


def word_ops(B) -> F.Ops:
    return F.word_ops(B, WORD)


FLASHMLA_PREFILL_COMPOSITES = [FlashMlaGatherRow, FlashMlaPrefillValid, FlashMlaPrefillPair, FlashMlaPrefillHead, FlashMlaPrefill]

flashmla_prefill_twin.register_twin({d.id: d for d in FLASHMLA_PREFILL_COMPOSITES})
