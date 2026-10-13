"""[DSV4] DeepSeek-V4-Pro's manifold-constrained hyper-connections (mHC) as the H200 runs them: 8xH200 TP8 on sm_90, vLLM
d9105ea80, TileLang 0.1.12 (nvcc without fast-math), DeepGEMM 8b1392b9.  Line numbers are at those commits; "the CUDA" is
TileLang's generated source for each kernel at DeepSeek-V4-Pro's statics (hc_mult 4, hidden 7168, Sinkhorn 20, the eps 1e-6).

DeepSeek-V4-Pro keeps HC = 4 residual streams [HC, H] bf16 per token.  Each layer runs mHC twice, before attention (fused with
`attn_norm`) and before the MLP (with `ffn_norm`), each time the previous sublayer's post and the next one's pre
(`deepseek_v4/nvidia/model.py:1186-1267`); the model's last streams go through hc_head before the final norm (model.py:1471):

1. post + pre, at most 16 tokens (`mhc_fused_post_pre_tilelang`, mhc/tilelang.py:433, dispatch 518-600: `use_small_fma`, n_splits
   4 at hidden 7168): `mhc_fused_tilelang` (tilelang_kernels.py:494) per (token, split) computes the new residual and its share of
   the 24 prenorm products and the squared sum, then `mhc_pre_big_fuse_with_norm_tilelang` (:172) the mixes, the Sinkhorn
   normalisation and the fused RMSNorm: `MhcFusedPostPre_v1`.
2. post + pre above 16 tokens: `mhc_post_tilelang` (:615), DeepGEMM `tf32_hc_prenorm_gemm` (TF32 wgmma, split-K over
   `compute_num_split(64, 28672, cdiv(T, 64))` splits, :19), then the same big fuse: `MhcPostPreTf32_v1`.
3. layer 0's attention pre on the embedding broadcast to the streams (`mhc_pre_broadcast_tilelang`, tilelang.py:303): DeepGEMM
   `tf32_hc_prenorm_gemm` over `fn_broadcast` [24, H] at K = H, at every token count (decode included), then
   `mhc_pre_big_fuse_broadcast_with_norm_tilelang` (:332): `MhcPreBroadcast_v1`.  Its `residual_out` (the four copies of the
   embedding) is a copy no Definition states.
4. `hc_head_fuse_tilelang` (:847) on the last streams: `HcHead_v1`.

What nvcc makes of TileLang's expressions, in the CUDA and its PTX: `a + b * c` is `fma.rn.f32`; `a / b` is `div.rn.f32` and
`1 / x` `rcp.rn.f32` (the same word as `div.rn.f32(1, x)`); `T.rsqrt` is `rsqrtf`, `rsqrt.approx.f32` (`RsqrtApprox_v1`);
`T.exp` is `expf` (`NvExpf_v1`); `T.sigmoid(z)` is `1 / (1 + expf(0 - z))` (`MhcSigmoid_v1`); `T.max` is `max.f32`
(`F32Fmaxf_v1`); a bf16 store is `cvt.rn.bf16.f32` (`F2fpBf16_v1`).  `a - b` is `sub.f32`, written `add(a, neg(b))`: the same
word but for a NaN's payload.  Each `T.reduce_*` / reducer finalisation and each `warp_reduce` is TileLang's xor butterfly (lane
t takes `op(v[t], v[t ^ h])` for h = n/2 .. 1: `dsa.halving_tree`), each accumulator and each cross-warp sum a chain from +0
in index order (`chain`).  A NaN word is IR's (0x7FC00000) where a leaf's is, and the GPU's 0x7FFFFFFF elsewhere; no output reads
a NaN's payload on finite operands.

The Sinkhorn normalisation (tilelang_kernels.py:247-266, and :410-429 in the broadcast), on the 4 x 4 logits x[r][c] = `fma(mixes[8 + 4r + c], scale[2],
base[8 + 4r + c])`: a row softmax (`max` tree from -inf, `expf(x - max)`, the row sum tree from +0, `e / sum + eps`), a column
normalisation `cm / (colsum + eps)`, then REPEAT - 1 times a row and a column normalisation `cm / (sum + eps)`.

TF32 (`tf32_hc_prenorm_gemm`, DeepGEMM `sm90_tf32_hc_prenorm_gemm.cuh:150-240`, `mma/sm90.cuh:150-195`): BLOCK_M 64, BLOCK_N 32,
BLOCK_K 64; each split s takes `per + (s < rem)` K-blocks of 64 from `(s * per + min(s, rem)) * 64` (per, rem = divmod(K/64,
NS)); `wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32` (A from registers, the bf16 streams widened to f32; B the f32 `fn`
from shared memory), both operands rounded RNE to TF32, from a zero accumulator, eight k8 steps a block, `scale-d` 1:
`HopperTf32WgmmaDot8_v1`.  A's conversion is unobservable here because widened bf16 has thirteen zero low bits.  The squared sum
of each block: thread lane e < 4 of a quad adds `fma(x[k], x[k], x[k+4] * x[k+4])`, k = 64 st + 8 i + e, into its own chain
from +0, and the quad's shuffles add `(v0 + v2) + (v1 + v3)`; these are GPU FMUL/FFMA/FADD operations, whose NaN result is
0x7FFFFFFF.  The k8 step itself is unprobed on Hopper (core's
`instructions.py` lists `sm90.wgmma.m64n8k8.tf32` so): `HopperTf32WgmmaDot8_v1` is a candidate, the Hopper BF16 step's adder
on TF32 products; tests/properties/dsv4_mhc_capture_gpu.py pins it or names the mismatch.

Registered functions (all `Name_v1`; OUT = HC * (2 + HC), the 24 mixes; `yp`, `rp` the split partials):
  MhcSigmoid                                  (z: f32) -> f32: `1 / (1 + expf(0 - z))`
  MhcSinkhorn{HC,REPEAT,EPS}                  (x: HC x HC f32) -> HC x HC f32
  MhcMixes{HC,NS,K,REPEAT,EPS,HC_EPS}         (yp: NS x OUT, rp: NS, scale: 3, base: OUT f32) -> (pre: HC, post: HC, comb: HC x HC)
  MhcPreNorm{HC,H,TN,EPS}                     (residual: HC x H bf16, pre: HC f32, w: H bf16) -> H bf16
  MhcPreNormBroadcast{HC,H,TN,EPS}            (x: H bf16, pre: HC f32, w: H bf16) -> H bf16
  MhcPreBigFuse{HC,H,NS,TN,REPEAT,EPS,HC_EPS} (yp, rp, scale, base, residual: HC x H bf16, w) -> (out: H bf16, post, comb)
  MhcPreBigFuseBroadcast{...}                 (yp, rp, scale, base, x: H bf16, w) -> (out: H bf16, post, comb)
  MhcPost{HC,N}                               (x: N bf16, residual: HC x N bf16, post: HC, comb: HC x HC) -> HC x N bf16
  MhcFusedSplit{HC,N,OUT,TH}                  (x: N, residual: HC x N bf16, post, comb, fn: OUT x HC x N f32)
                                              -> (residual': HC x N bf16, yp: OUT, rp)
  MhcFusedPostPre{HC,H,NS,TH,TN,REPEAT,EPS,HC_EPS}
                                              (x: H bf16, residual: HC x H bf16, post: HC, comb: HC x HC, fn: OUT x HC*H,
                                               scale: 3, base: OUT f32, w: H bf16) -> (residual': HC x H bf16, out: H bf16,
                                               post', comb')
  HopperTf32WgmmaDot8                         primitive: (acc: f32, a, b: 8 f32) -> f32, the candidate k8 step
  MhcPrenormTf32Split{K,OUT}                  (a: K bf16, b: OUT x K f32) -> (yp: OUT, rp)
  MhcPrenormTf32{K,NS,OUT}                    (a: K bf16, b: OUT x K f32) -> (yp: NS x OUT, rp: NS)
  MhcPostPreTf32{HC,H,NS,TN,REPEAT,EPS,HC_EPS}  as MhcFusedPostPre, above 16 tokens
  MhcPreBroadcast{HC,H,NS,TN,REPEAT,EPS,HC_EPS} (x: H bf16, fn: OUT x H, scale: 3, base: OUT f32, w: H bf16) -> (out, post, comb)
  HcHead{HC,H,TH,EPS,HC_EPS}                  (residual: HC x H bf16, fn: HC x HC*H, scale: 1, base: HC f32) -> H bf16

DeepSeek-V4-Pro's statics: HC 4, H 7168, REPEAT 20, EPS (`rms_eps`, and `norm_eps`, the same `rms_norm_eps`) and HC_EPS
(`hc_pre_eps` and `hc_sinkhorn_eps`, both `hc_eps`) 1e-6, TH 256 (`mhc_fused`) and 128 (hc_head), TN 64 (the fused norm), NS 4
at most 16 tokens, 112 (T <= 64), 66 (T <= 128), ... above, 28 for the broadcast.  `hc_post_mult_value` 2.0 is `s + s`.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import F32Neg
from verity.ml.circuits.silicon.models import GroupSum
from verity.ml.circuits.silicon.term import TF32, tf32_product
from verity.ml.circuits.silicon.total import NAN, Inf, fp32_value, group_step_total, pack_value
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import moe as M
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16, F32

M32 = 0xFFFFFFFF
F_ONE, F_NEG_INF = 0x3F800000, 0xFF800000

_CONF = ("model-untested: read from the TileLang kernels' generated CUDA and PTX at the commits the module docstring pins; "
         "tests/properties/dsv4_mhc_capture_gpu.py is its hardware check")
_TF32_CONF = ("model-untested: the k8 TF32 step is unprobed on sm_90 (core's instructions.py); the Hopper BF16 step's adder (one "
              "group, 26 bits, floor -133) on exact products after each f32 operand is rounded RNE to TF32; "
              "tests/properties/dsv4_mhc_capture_gpu.py probes it")
_DG_CONF = ("model-untested: read from DeepGEMM 8b1392b9 sm90_tf32_hc_prenorm_gemm.cuh:150-240 on the candidate "
            "HopperTf32WgmmaDot8_v1; tests/properties/dsv4_mhc_capture_gpu.py is its hardware check")

#: The candidate k8 step's grouped sum: the sm_90 BF16 step's adder (`HOPPER_BF16_WGMMA_K16`) over the eight products
TF32_GROUPS, TF32_WIDTH, TF32_FLOOR = (8,), 26, -133
HOPPER_TF32_WGMMA_K8_CANDIDATE = GroupSum(
    name="hopper_tf32_wgmma_k8_candidate", arch="sm_90", dtype="tf32", operand_bits=19, groups=TF32_GROUPS, width=TF32_WIDTH,
    zero_exponent=TF32_FLOOR, validated="unprobed: tests/properties/dsv4_mhc_capture_gpu.py")


class Ops(D.Ops):
    """`dsa.Ops` with ``sub(name, **statics)``: the composite ``defs[name]`` bound, the word or the Boolean one by the defs."""

    def sub(self, name: str, **statics):
        return bind(self._defs[name], **statics)


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x))


def f32_bits(v: float) -> int:
    return D.f32_bits(v)


def chain(o: D.Ops, vals: list):
    """``((0 + v0) + v1) + ...``: an accumulator from +0, or a cross-warp sum, in index order."""
    acc = o.word(0)
    for v in vals:
        acc = o.add(acc, v)
    return acc


def stack(colls: list) -> Coll:
    """Colls of one type as an Array of them (no gates): words as a row, rows as a grid, on words or on bits."""
    return Coll(Array(len(colls), colls[0].type), concat([c.refs for c in colls]))


def _grid(rows: list) -> Coll:
    return stack([stack(r) for r in rows])


def _check_hc(HC: int) -> None:
    if not (HC > 0 and HC & (HC - 1) == 0):
        raise ValueError(f"mHC: the Sinkhorn trees take a power-of-two hc_mult (HC={HC})")


# --- the sigmoid and the Sinkhorn normalisation ------------------------------------------------------------------------------

def sigmoid(o: D.Ops, z):
    """`MhcSigmoid_v1`'s body: `div.rn(1, add(1, expf(add(0, -z))))`."""
    one = o.word(F_ONE)
    return o.div(one, o.add(one, o.expf(o.add(o.word(0), o.neg(z)))))


@composite("MhcSigmoid", 1, [], lambda S: ((("z", F32),), F32), conformance=_CONF)
def MhcSigmoid(B, S, z):
    """`T.sigmoid(z)` as TileLang 0.1.12 writes it, `1.f / (1.f + expf(0.f - z))`: `sub.f32` from +0, `NvExpf_v1`, `add.f32`,
    `rcp.rn.f32` (`F32Div_v1` from 1)."""
    return sigmoid(word_ops(B), z)


def _normalize(o: D.Ops, cm: list, eps, rows: bool) -> list:
    """Each row (``rows``) or column of ``cm`` divided by its sum plus eps: the sum a halving tree over `add(0, v)`."""
    HC = len(cm)
    zero = o.word(0)
    if rows:
        sums = [o.add(D.halving_tree([o.add(zero, cm[r][c]) for c in range(HC)], o.add), eps) for r in range(HC)]
        return [[o.div(cm[r][c], sums[r]) for c in range(HC)] for r in range(HC)]
    sums = [o.add(D.halving_tree([o.add(zero, cm[r][c]) for r in range(HC)], o.add), eps) for c in range(HC)]
    return [[o.div(cm[r][c], sums[c]) for c in range(HC)] for r in range(HC)]


def sinkhorn(o: D.Ops, HC: int, REPEAT: int, eps, x: list) -> list:
    """`MhcSinkhorn_v1`'s body on the logits ``x`` [r][c]."""
    _check_hc(HC)
    if REPEAT < 1:
        raise ValueError(f"MhcSinkhorn: at least one iteration (REPEAT={REPEAT})")
    ninf, zero = o.word(F_NEG_INF), o.word(0)
    cm = []
    for r in range(HC):
        nmax = o.neg(D.halving_tree([o.fmaxf(ninf, x[r][c]) for c in range(HC)], o.fmaxf))
        e = [o.expf(o.add(x[r][c], nmax)) for c in range(HC)]
        s = D.halving_tree([o.add(zero, v) for v in e], o.add)
        cm.append([o.add(o.div(v, s), eps) for v in e])
    cm = _normalize(o, cm, eps, rows=False)
    for _ in range(REPEAT - 1):
        cm = _normalize(o, _normalize(o, cm, eps, rows=True), eps, rows=False)
    return cm


def _sig_sinkhorn(S, f32=F32):
    return (("x", Array(S.HC, Array(S.HC, f32))),), Array(S.HC, Array(S.HC, f32))


def _sinkhorn(o: D.Ops, S, x):
    return _grid(sinkhorn(o, S.HC, S.REPEAT, o.word(f32_bits(S.EPS)), [[x[r][c] for c in range(S.HC)] for r in range(S.HC)]))


@composite("MhcSinkhorn", 1, ["HC", "REPEAT", "EPS"], _sig_sinkhorn, conformance=_CONF)
def MhcSinkhorn(B, S, x):
    """The Sinkhorn normalisation inside the big fuse (tilelang_kernels.py:247-266, 410-429): per row the max tree from -inf
    (`F32Fmaxf_v1`), `e = expf(x - max)`, the sum tree from +0, `cm = e / sum + EPS`; then the column normalisation `cm / (colsum
    + EPS)`; then REPEAT - 1 times the row and the column normalisation, each `cm / (sum + EPS)`.  Each sum over HC is a halving
    tree over `add(0, v)` (the `T.reduce_sum` of a fragment whose four lanes hold one element each)."""
    return _sinkhorn(word_ops(B), S, x)


# --- the mixes and the fused RMSNorm -------------------------------------------------------------------------------------------

def _out(HC: int) -> int:
    return HC * (2 + HC)


def _sig_mixes(S, f32=F32):
    OUT = _out(S.HC)
    return ((("yp", Array(S.NS, Array(OUT, f32))), ("rp", Array(S.NS, f32)), ("scale", Array(3, f32)), ("base", Array(OUT, f32))),
            Tuple(Array(S.HC, f32), Array(S.HC, f32), Array(S.HC, Array(S.HC, f32))))


def mixes(o: Ops, S, yp, rp, scale, base):
    """`MhcMixes_v1`'s body: (pre, post, comb) as lists."""
    HC, OUT = S.HC, _out(S.HC)
    _check_hc(HC)
    hc_eps = o.word(f32_bits(S.HC_EPS))
    rms = o.rsqrt(o.add(o.div(chain(o, [rp[s] for s in range(S.NS)]), o.word(f32_bits(S.K))), o.word(f32_bits(S.EPS))))
    m = [o.mul(chain(o, [yp[s][j] for s in range(S.NS)]), rms) for j in range(OUT)]
    pre = [o.add(o.sigmoid(o.fma(m[j], scale[0], base[j])), hc_eps) for j in range(HC)]
    post = []
    for j in range(HC):
        s = o.sigmoid(o.fma(m[HC + j], scale[1], base[HC + j]))
        post.append(o.add(s, s))
    x = _grid([[o.fma(m[2 * HC + HC * r + c], scale[2], base[2 * HC + HC * r + c]) for c in range(HC)] for r in range(HC)])
    comb = o.B.call(o.sub("MhcSinkhorn", HC=HC, REPEAT=S.REPEAT, EPS=S.HC_EPS), x)
    return pre, post, comb


def _mixes(o: Ops, S, yp, rp, scale, base):
    pre, post, comb = mixes(o, S, yp, rp, scale, base)
    return tuple_of(stack(pre), stack(post), comb)


@composite("MhcMixes", 1, ["HC", "NS", "K", "REPEAT", "EPS", "HC_EPS"], _sig_mixes, conformance=_CONF)
def MhcMixes(B, S, yp, rp, scale, base):
    """The head of both big fuses (tilelang_kernels.py:218-279, 380-441) on the NS split partials of one token: the squared sum
    and each of the OUT = HC (2 + HC) products a chain from +0 over the splits; `rms = rsqrt(sqrsum / K + EPS)`; `mixes[j] =
    sum_j * rms`; pre[j] = `MhcSigmoid(fma(mixes[j], scale[0], base[j])) + HC_EPS`; post[j] = `s + s` of `MhcSigmoid(fma(mixes[HC
    + j], scale[1], base[HC + j]))` (`hc_post_mult_value` 2.0); comb = `MhcSinkhorn{HC, REPEAT, HC_EPS}` of `fma(mixes[2HC + HC r
    + c], scale[2], base[2HC + HC r + c])`.  K is the products' length: HC * H, or H for the broadcast."""
    return _mixes(word_ops(B), S, yp, rp, scale, base)


def _check_norm(H: int, TN: int) -> None:
    if not (TN > 0 and TN & (TN - 1) == 0 and H % (16 * TN) == 0):
        raise ValueError(f"the fused RMSNorm takes 16 elements per thread a block over a power-of-two thread count (H={H}, TN={TN})")


def prenorm(o: D.Ops, HC: int, H: int, TN: int, eps, rows: list, pre: list, w) -> list:
    """The fused RMSNorm's bf16 words: ``rows`` the HC streams' f32 words (one list repeated for the broadcast)."""
    _check_norm(H, TN)
    zero = o.word(0)
    ol = []
    for p in range(H):
        acc = zero
        for j in range(HC):
            acc = o.fma(pre[j], rows[j][p], acc)
        ol.append(acc)
    ss = [[zero] * 16 for _ in range(TN)]
    for b in range(H // (16 * TN)):
        for q in range(TN):
            for half in (0, 1):
                for e in range(8):
                    p = 16 * TN * b + 8 * TN * half + 8 * q + e
                    ss[q][8 * half + e] = o.fma(ol[p], ol[p], ss[q][8 * half + e])
    total = D.halving_tree([chain(o, [ss[q][8 * (rv & 1) + (rv >> 1)] for rv in range(16)]) for q in range(TN)], o.add)
    rn = o.rsqrt(o.add(o.div(total, o.word(f32_bits(H))), eps))
    return [o.bf16(o.mul(o.mul(o.cvt(o.bf16(ol[p])), rn), o.cvt(w[p]))) for p in range(H)]


def _sig_norm(S, bf16=BF16, f32=F32):
    return (("residual", Array(S.HC, Array(S.H, bf16))), ("pre", Array(S.HC, f32)), ("w", Array(S.H, bf16))), Array(S.H, bf16)


def _sig_norm_b(S, bf16=BF16, f32=F32):
    return (("x", Array(S.H, bf16)), ("pre", Array(S.HC, f32)), ("w", Array(S.H, bf16))), Array(S.H, bf16)


def _norm(o: D.Ops, S, residual, pre, w):
    rows = [[o.cvt(residual[j][p]) for p in range(S.H)] for j in range(S.HC)]
    return stack(prenorm(o, S.HC, S.H, S.TN, o.word(f32_bits(S.EPS)), rows, [pre[j] for j in range(S.HC)], w))


def _norm_b(o: D.Ops, S, x, pre, w):
    xs = [o.cvt(x[p]) for p in range(S.H)]
    return stack(prenorm(o, S.HC, S.H, S.TN, o.word(f32_bits(S.EPS)), [xs] * S.HC, [pre[j] for j in range(S.HC)], w))


@composite("MhcPreNorm", 1, ["HC", "H", "TN", "EPS"], _sig_norm, conformance=_CONF)
def MhcPreNorm(B, S, residual, pre, w):
    """The tail of `mhc_pre_big_fuse_with_norm` (tilelang_kernels.py:281-327, threads 32..95; TN threads, thread q holding elements p = 16 TN b
    + 8 TN half + 8 q + e): `ol[p]` = the chain from +0 of `fma(pre[j], f32(residual[j][p]), ol)` over j; each thread's sixteen
    squared sums `ss[8 half + e]` a chain from +0 of `fma(ol, ol, ss)` over the blocks b; the thread's total the chain from +0
    over `ss[8 (r & 1) + (r >> 1)]`, r = 0..15 (the reducer's layout); the threads' halving tree; `rn = rsqrt(total / H + EPS)`;
    out[p] = `bf16(f32(bf16(ol[p])) * rn * f32(w[p]))` (the layer input is stored as bf16 and read back before the scale)."""
    return _norm(word_ops(B), S, residual, pre, w)


@composite("MhcPreNormBroadcast", 1, ["HC", "H", "TN", "EPS"], _sig_norm_b, conformance=_CONF)
def MhcPreNormBroadcast(B, S, x, pre, w):
    """`MhcPreNorm_v1` with every stream x (tilelang_kernels.py:443-490): `f32(x[p])` once, `ol[p]` its chain over the HC pre."""
    return _norm_b(word_ops(B), S, x, pre, w)


def _sig_fuse(S, bf16=BF16, f32=F32):
    (yp, rp, scale, base), ret = _sig_mixes(S, f32)
    return ((yp, rp, scale, base, ("residual", Array(S.HC, Array(S.H, bf16))), ("w", Array(S.H, bf16))),
            Tuple(Array(S.H, bf16), Array(S.HC, f32), Array(S.HC, Array(S.HC, f32))))


def _sig_fuse_b(S, bf16=BF16, f32=F32):
    (yp, rp, scale, base), ret = _sig_mixes(S, f32)
    return ((yp, rp, scale, base, ("x", Array(S.H, bf16)), ("w", Array(S.H, bf16))),
            Tuple(Array(S.H, bf16), Array(S.HC, f32), Array(S.HC, Array(S.HC, f32))))


def big_fuse(o: Ops, S, yp, rp, scale, base, rows, w, broadcast: bool):
    """Both big fuses' bodies: `MhcMixes`, then `MhcPreNorm` (or `MhcPreNormBroadcast`) on its pre."""
    K = S.H if broadcast else S.HC * S.H
    mx = o.B.call(o.sub("MhcMixes", HC=S.HC, NS=S.NS, K=K, REPEAT=S.REPEAT, EPS=S.EPS, HC_EPS=S.HC_EPS), yp, rp, scale, base)
    norm = o.sub("MhcPreNormBroadcast" if broadcast else "MhcPreNorm", HC=S.HC, H=S.H, TN=S.TN, EPS=S.EPS)
    return tuple_of(o.B.call(norm, rows, mx[0], w), mx[1], mx[2])


_FUSE = ["HC", "H", "NS", "TN", "REPEAT", "EPS", "HC_EPS"]


@composite("MhcPreBigFuse", 1, _FUSE, _sig_fuse, conformance=_CONF)
def MhcPreBigFuse(B, S, yp, rp, scale, base, residual, w):
    """`mhc_pre_big_fuse_with_norm_tilelang` (tilelang_kernels.py:172) for one token: `MhcMixes_v1{HC, NS, K = HC * H, REPEAT,
    EPS, HC_EPS}` on the split partials, then `MhcPreNorm_v1{HC, H, TN, EPS}` of the residual (the bf16 streams the post wrote) at
    its pre: (layer input, post, comb)."""
    return big_fuse(word_ops(B), S, yp, rp, scale, base, residual, w, False)


@composite("MhcPreBigFuseBroadcast", 1, _FUSE, _sig_fuse_b, conformance=_CONF)
def MhcPreBigFuseBroadcast(B, S, yp, rp, scale, base, x, w):
    """`mhc_pre_big_fuse_broadcast_with_norm_tilelang` (tilelang_kernels.py:332) for one token: `MhcMixes_v1` at K = H (the
    squared sum of the broadcast streams over H, the GEMM's over `fn_broadcast`), then `MhcPreNormBroadcast_v1` of x."""
    return big_fuse(word_ops(B), S, yp, rp, scale, base, x, w, True)


# --- the post and the fused split (<= 16 tokens) -------------------------------------------------------------------------------

def new_residual(o: D.Ops, HC: int, xf, rf: list, post: list, comb) -> list:
    """The post's f32 streams at one element: `nr[j] = fma(comb[HC-1][j], r[HC-1], ... fma(comb[1][j], r[1], fma(post[j], x,
    comb[0][j] * r[0])))` (tilelang_kernels.py:567-570, 658-660 and their PTX: nvcc contracts `pm * x + cm[0] * r[0]` into the post's FMA)."""
    out = []
    for j in range(HC):
        acc = o.fma(post[j], xf, o.mul(comb[0][j], rf[0]))
        for k in range(1, HC):
            acc = o.fma(comb[k][j], rf[k], acc)
        out.append(acc)
    return out


def _sig_post(S, bf16=BF16, f32=F32):
    return ((("x", Array(S.N, bf16)), ("residual", Array(S.HC, Array(S.N, bf16))), ("post", Array(S.HC, f32)),
             ("comb", Array(S.HC, Array(S.HC, f32)))), Array(S.HC, Array(S.N, bf16)))


def _post(o: D.Ops, S, x, residual, post, comb):
    cm = [[comb[k][j] for j in range(S.HC)] for k in range(S.HC)]
    pj = [post[j] for j in range(S.HC)]
    cols = [new_residual(o, S.HC, o.cvt(x[n]), [o.cvt(residual[k][n]) for k in range(S.HC)], pj, cm) for n in range(S.N)]
    return _grid([[o.bf16(cols[n][j]) for n in range(S.N)] for j in range(S.HC)])


@composite("MhcPost", 1, ["HC", "N"], _sig_post, conformance=_CONF)
def MhcPost(B, S, x, residual, post, comb):
    """`mhc_post_tilelang` (tilelang_kernels.py:615) on N elements of one token: residual'[j][n] = `bf16(new_residual)`, the
    f32 sum of `post[j] * x[n]` and `comb[k][j] * residual[k][n]` in `new_residual`'s order (`comb` row k is stream k's)."""
    return _post(word_ops(B), S, x, residual, post, comb)


def _check_split(N: int, TH: int) -> None:
    if not (TH > 0 and (TH % 32 == 0 or 32 % TH == 0) and N % TH == 0):
        raise ValueError(f"mhc_fused: each thread takes N / TH elements, TH whole warps or a power of two below 32 (N={N}, TH={TH})")


def _sig_split(S, bf16=BF16, f32=F32):
    return ((("x", Array(S.N, bf16)), ("residual", Array(S.HC, Array(S.N, bf16))), ("post", Array(S.HC, f32)),
             ("comb", Array(S.HC, Array(S.HC, f32))), ("fn", Array(S.OUT, Array(S.HC, Array(S.N, f32))))),
            Tuple(Array(S.HC, Array(S.N, bf16)), Array(S.OUT, f32), f32))


def fused_split(o: D.Ops, S, x, residual, post, comb, fn):
    """`MhcFusedSplit_v1`'s body."""
    HC, N, OUT, TH = S.HC, S.N, S.OUT, S.TH
    _check_split(N, TH)
    zero = o.word(0)
    cm = [[comb[k][j] for j in range(HC)] for k in range(HC)]
    pj = [post[j] for j in range(HC)]
    nr = [new_residual(o, HC, o.cvt(x[h]), [o.cvt(residual[k][h]) for k in range(HC)], pj, cm) for h in range(N)]
    sqr = [zero] * TH
    acc = [[zero] * TH for _ in range(OUT)]
    for it in range(N // TH):
        for t in range(TH):
            h = TH * it + t
            for j in range(HC):
                sqr[t] = o.fma(nr[h][j], nr[h][j], sqr[t])
            for oi in range(OUT):
                for j in range(HC):
                    acc[oi][t] = o.fma(fn[oi][j][h], nr[h][j], acc[oi][t])

    lanes = min(32, TH)

    def reduce(v):
        return chain(o, [D.halving_tree(v[lanes * w:lanes * w + lanes], o.add) for w in range(TH // lanes)])
    res = _grid([[o.bf16(nr[h][j]) for h in range(N)] for j in range(HC)])
    return tuple_of(res, stack([reduce(acc[oi]) for oi in range(OUT)]), reduce(sqr))


@composite("MhcFusedSplit", 1, ["HC", "N", "OUT", "TH"], _sig_split, conformance=_CONF)
def MhcFusedSplit(B, S, x, residual, post, comb, fn):
    """One (token, split) of `mhc_fused_tilelang` (tilelang_kernels.py:531-612; TH threads, thread t elements h = TH it + t):
    the new residual `new_residual` at each element, stored as bf16; the f32 values themselves, unrounded, feed the thread's
    chains from +0, per element `sqr = fma(nr[j], nr[j], sqr)` over j, then `acc[o] = fma(fn[o][j][h], nr[j], acc[o])` over j for
    each o; each sum a warp's halving tree, then the chain from +0 over the TH / 32 warps: (residual', yp, rp).  A TH below 32 (a
    power of two) is one partial warp, its halving tree over TH lanes."""
    return fused_split(word_ops(B), S, x, residual, post, comb, fn)


def _sig_layer(S, bf16=BF16, f32=F32):
    OUT = _out(S.HC)
    return ((("x", Array(S.H, bf16)), ("residual", Array(S.HC, Array(S.H, bf16))), ("post", Array(S.HC, f32)),
             ("comb", Array(S.HC, Array(S.HC, f32))), ("fn", Array(OUT, Array(S.HC * S.H, f32))), ("scale", Array(3, f32)),
             ("base", Array(OUT, f32)), ("w", Array(S.H, bf16))),
            Tuple(Array(S.HC, Array(S.H, bf16)), Array(S.H, bf16), Array(S.HC, f32), Array(S.HC, Array(S.HC, f32))))


def fused_post_pre(o: Ops, S, x, residual, post, comb, fn, scale, base, w):
    """`MhcFusedPostPre_v1`'s body."""
    HC, H, NS, OUT = S.HC, S.H, S.NS, _out(S.HC)
    if H % NS:
        raise ValueError(f"mhc_fused splits H evenly (H={H}, NS={NS})")
    N = H // NS
    split = o.sub("MhcFusedSplit", HC=HC, N=N, OUT=OUT, TH=S.TH)
    parts = []
    for s in range(NS):
        lo = N * s
        rs = stack([residual[j][lo:lo + N] for j in range(HC)])
        fs = stack([stack([fn[oi][H * j + lo:H * j + lo + N] for j in range(HC)]) for oi in range(OUT)])
        parts.append(o.B.call(split, x[lo:lo + N], rs, post, comb, fs))
    res = stack([Coll(Array(H, x.type.elem), concat([p[0][j].refs for p in parts])) for j in range(HC)])
    yp = stack([p[1] for p in parts])
    rp = stack([p[2] for p in parts])
    fuse = o.sub("MhcPreBigFuse", HC=HC, H=H, NS=NS, TN=S.TN, REPEAT=S.REPEAT, EPS=S.EPS, HC_EPS=S.HC_EPS)
    out = o.B.call(fuse, yp, rp, scale, base, res, w)
    return tuple_of(res, out[0], out[1], out[2])


@composite("MhcFusedPostPre", 1, ["HC", "H", "NS", "TH", "TN", "REPEAT", "EPS", "HC_EPS"], _sig_layer, conformance=_CONF)
def MhcFusedPostPre(B, S, x, residual, post, comb, fn, scale, base, w):
    """`mhc_fused_post_pre_tilelang` at most 16 tokens (mhc/tilelang.py:518-575) for one token: the previous sublayer's output x
    and its (post, comb) into the streams, and the next sublayer's pre.  Split s of NS takes elements [s H/NS, (s+1) H/NS) of
    every stream (`fn` [OUT, HC*H] read as [OUT, HC, H]): `MhcFusedSplit_v1{HC, H/NS, OUT, TH}`; then `MhcPreBigFuse_v1` on the
    partials and the bf16 streams: (residual', layer input, post', comb')."""
    return fused_post_pre(word_ops(B), S, x, residual, post, comb, fn, scale, base, w)


# --- the TF32 path (above 16 tokens, and layer 0's broadcast) -------------------------------------------------------------------

def tf32_operand(word: int) -> int:
    """The 19-bit TF32 word an HGMMA TF32 operand reads: f32 rounded to nearest, ties to even.

    A NaN remains a NaN even when its payload lives entirely in the thirteen discarded bits.  The tensor-core step only
    observes its class, so one low TF32 payload bit is sufficient.
    """
    word &= M32
    high = word >> 13
    if word & 0x7F800000 == 0x7F800000:
        return high | int(bool(word & 0x7FFFFF))
    return ((word + 0xFFF + (high & 1)) & M32) >> 13


def tf32_product_total(a: int, b: int):
    """The product of two TF32 words as a `silicon.total` Value: NaN for a NaN operand or inf times zero, else a signed
    infinity, else exact (`term.tf32_product`)."""
    da, db = TF32.decode(a), TF32.decode(b)
    if da == "nan" or db == "nan":
        return NAN
    if isinstance(da, str) or isinstance(db, str):
        if (not isinstance(da, str) and da[2] == 0) or (not isinstance(db, str) and db[2] == 0):
            return NAN
        return Inf(bool(((a ^ b) >> 18) & 1))
    return tf32_product(a, b)


def tf32_dot_total(acc: int, a, b, groups=TF32_GROUPS, width: int = TF32_WIDTH, floor: int = TF32_FLOOR) -> int:
    """``acc + sum a[i] b[i]`` on f32 words at the given grouped-sum parameters (the candidate's by default)."""
    v, start = fp32_value(acc & M32), 0
    for size in groups:
        prods = [tf32_product_total(tf32_operand(a[i]), tf32_operand(b[i])) for i in range(start, start + size)]
        v = group_step_total(v, prods, width=width, zero_exponent=floor)
        start += size
    return pack_value(v)


@primitive("HopperTf32WgmmaDot8", 1, [("acc", F32), ("a", Array(8, F32)), ("b", Array(8, F32))], F32, conformance=_TF32_CONF)
def HopperTf32WgmmaDot8(acc: int, *ab: int) -> int:
    """Candidate: one accumulator coordinate of `wgmma.mma_async.sync.aligned.m64n{N}k8.f32.tf32.tf32` on sm_90 with f32 words
    as operands: each operand rounded RNE to a 19-bit TF32 word, the eight exact products and `acc` in one group through the Hopper BF16
    step's 26-bit truncating adder with exponent floor -133 (`HOPPER_TF32_WGMMA_K8_CANDIDATE`); total on every encoding as that
    step (NaN 0x7FFFFFFF, sticky infinities, a zero +0)."""
    return tf32_dot_total(acc, ab[:8], ab[8:16])


def _check_tf32(K: int) -> None:
    if not (K > 0 and K % 64 == 0):
        raise ValueError(f"tf32_hc_prenorm_gemm: K is a whole number of 64-wide blocks (K={K})")


def _sig_tf32_split(S, bf16=BF16, f32=F32):
    return (("a", Array(S.K, bf16)), ("b", Array(S.OUT, Array(S.K, f32)))), Tuple(Array(S.OUT, f32), f32)


def tf32_split(o: D.Ops, S, a, b):
    """`MhcPrenormTf32Split_v1`'s body."""
    _check_tf32(S.K)
    zero = o.word(0)
    x = [o.cvt(a[k]) for k in range(S.K)]
    yp = []
    for n in range(S.OUT):
        acc = zero
        for s in range(S.K // 8):
            acc = o.tf32(acc, stack(x[8 * s:8 * s + 8]), b[n][8 * s:8 * s + 8])
        yp.append(acc)
    v = [zero] * 4
    for st in range(S.K // 64):
        for i in range(8):
            for e in range(4):
                k = 64 * st + 8 * i + e
                v[e] = o.gpu_add(v[e], o.gpu_fma(x[k], x[k], o.gpu_mul(x[k + 4], x[k + 4])))
    return tuple_of(stack(yp), D.halving_tree(v, o.gpu_add))


@composite("MhcPrenormTf32Split", 1, ["K", "OUT"], _sig_tf32_split, conformance=_DG_CONF)
def MhcPrenormTf32Split(B, S, a, b):
    """One split of `tf32_hc_prenorm_gemm` for one token over its K = 64 * stages elements: yp[n] = the K/8 k8 steps
    `HopperTf32WgmmaDot8_v1` in ascending k from +0 (`scale-d` 1 on a zeroed accumulator), a the bf16 elements widened to f32;
    rp the quad's four squared-sum chains (GPU `FFMA(x[k], x[k], FMUL(x[k+4], x[k+4]))` then GPU `FADD` from +0,
    k = 64 st + 8 i + e) and GPU FADDs `(v0 + v2) + (v1 + v3)`.  These operations canonicalise every NaN to 0x7FFFFFFF."""
    return tf32_split(word_ops(B), S, a, b)


def tf32_ranges(K: int, NS: int) -> list[tuple[int, int]]:
    """Each split's (offset, length) in elements: `per + (s < rem)` blocks of 64 from block `s * per + min(s, rem)`."""
    _check_tf32(K)
    blocks = K // 64
    if not 1 <= NS <= blocks:
        raise ValueError(f"tf32_hc_prenorm_gemm: 1 <= NS <= K / 64 (K={K}, NS={NS})")
    per, rem = divmod(blocks, NS)
    return [(64 * (s * per + min(s, rem)), 64 * (per + (s < rem))) for s in range(NS)]


def _sig_tf32(S, bf16=BF16, f32=F32):
    return ((("a", Array(S.K, bf16)), ("b", Array(S.OUT, Array(S.K, f32)))),
            Tuple(Array(S.NS, Array(S.OUT, f32)), Array(S.NS, f32)))


def prenorm_tf32(o: Ops, S, a, b):
    parts = []
    for off, n in tf32_ranges(S.K, S.NS):
        bs = stack([b[i][off:off + n] for i in range(S.OUT)])
        parts.append(o.B.call(o.sub("MhcPrenormTf32Split", K=n, OUT=S.OUT), a[off:off + n], bs))
    return tuple_of(stack([p[0] for p in parts]), stack([p[1] for p in parts]))


@composite("MhcPrenormTf32", 1, ["K", "NS", "OUT"], _sig_tf32, conformance=_DG_CONF)
def MhcPrenormTf32(B, S, a, b):
    """`tf32_hc_prenorm_gemm` for one token: split s `MhcPrenormTf32Split_v1` over `tf32_ranges(K, NS)[s]`."""
    return prenorm_tf32(word_ops(B), S, a, b)


def post_pre_tf32(o: Ops, S, x, residual, post, comb, fn, scale, base, w):
    HC, H = S.HC, S.H
    res = o.B.call(o.sub("MhcPost", HC=HC, N=H), x, residual, post, comb)
    flat = Coll(Array(HC * H, x.type.elem), res.refs)
    g = o.B.call(o.sub("MhcPrenormTf32", K=HC * H, NS=S.NS, OUT=_out(HC)), flat, fn)
    fuse = o.sub("MhcPreBigFuse", HC=HC, H=H, NS=S.NS, TN=S.TN, REPEAT=S.REPEAT, EPS=S.EPS, HC_EPS=S.HC_EPS)
    out = o.B.call(fuse, g[0], g[1], scale, base, res, w)
    return tuple_of(res, out[0], out[1], out[2])


@composite("MhcPostPreTf32", 1, _FUSE, _sig_layer, conformance=_DG_CONF)
def MhcPostPreTf32(B, S, x, residual, post, comb, fn, scale, base, w):
    """`mhc_fused_post_pre_tilelang` above 16 tokens (mhc/tilelang.py:577-600) for one token: `MhcPost_v1{HC, H}`, then
    `MhcPrenormTf32_v1{HC * H, NS, OUT}` on the bf16 streams it wrote, then `MhcPreBigFuse_v1`; NS is `compute_num_split(64,
    HC * H, cdiv(T, 64))`, the batch's."""
    return post_pre_tf32(word_ops(B), S, x, residual, post, comb, fn, scale, base, w)


def _sig_broadcast(S, bf16=BF16, f32=F32):
    OUT = _out(S.HC)
    return ((("x", Array(S.H, bf16)), ("fn", Array(OUT, Array(S.H, f32))), ("scale", Array(3, f32)), ("base", Array(OUT, f32)),
             ("w", Array(S.H, bf16))), Tuple(Array(S.H, bf16), Array(S.HC, f32), Array(S.HC, Array(S.HC, f32))))


def pre_broadcast(o: Ops, S, x, fn, scale, base, w):
    g = o.B.call(o.sub("MhcPrenormTf32", K=S.H, NS=S.NS, OUT=_out(S.HC)), x, fn)
    fuse = o.sub("MhcPreBigFuseBroadcast", HC=S.HC, H=S.H, NS=S.NS, TN=S.TN, REPEAT=S.REPEAT, EPS=S.EPS, HC_EPS=S.HC_EPS)
    return o.B.call(fuse, g[0], g[1], scale, base, x, w)


@composite("MhcPreBroadcast", 1, _FUSE, _sig_broadcast, conformance=_DG_CONF)
def MhcPreBroadcast(B, S, x, fn, scale, base, w):
    """`mhc_pre_broadcast_tilelang` (mhc/tilelang.py:303) for one token at layer 0: `MhcPrenormTf32_v1{H, NS, OUT}` of the
    embedding x over `fn_broadcast` [OUT, H], at every token count, then `MhcPreBigFuseBroadcast_v1`: (layer input, post,
    comb).  NS is `compute_num_split(64, H, cdiv(T, 64))`."""
    return pre_broadcast(word_ops(B), S, x, fn, scale, base, w)


# --- hc_head ---------------------------------------------------------------------------------------------------------------------

def _check_head(H: int, TH: int) -> None:
    if not (TH > 0 and TH & (TH - 1) == 0 and H % (8 * TH) == 0):
        raise ValueError(f"hc_head: 8 elements per thread a block over a power-of-two thread count (H={H}, TH={TH})")


def _sig_head(S, bf16=BF16, f32=F32):
    return ((("residual", Array(S.HC, Array(S.H, bf16))), ("fn", Array(S.HC, Array(S.HC * S.H, f32))), ("scale", Array(1, f32)),
             ("base", Array(S.HC, f32))), Array(S.H, bf16))


def hc_head(o: Ops, S, residual, fn, scale, base):
    """`HcHead_v1`'s body."""
    HC, H, TH = S.HC, S.H, S.TH
    _check_head(H, TH)
    zero = o.word(0)
    xs = [[o.cvt(residual[m][p]) for p in range(H)] for m in range(HC)]
    sqr = [zero] * TH
    mix = [[zero] * TH for _ in range(HC)]
    for mc in range(HC):
        for ih in range(H // (8 * TH)):
            for t in range(TH):
                for e in range(8):
                    p = 8 * TH * ih + 8 * t + e
                    sqr[t] = o.fma(xs[mc][p], xs[mc][p], sqr[t])
            for m in range(HC):
                for t in range(TH):
                    for e in range(8):
                        p = 8 * TH * ih + 8 * t + e
                        mix[m][t] = o.fma(xs[mc][p], fn[m][H * mc + p], mix[m][t])
    rs = o.rsqrt(o.add(o.div(D.halving_tree(sqr, o.add), o.word(f32_bits(HC * H))), o.word(f32_bits(S.EPS))))
    hc_eps = o.word(f32_bits(S.HC_EPS))
    pre = [o.add(o.sigmoid(o.fma(o.mul(D.halving_tree(mix[m], o.add), rs), scale[0], base[m])), hc_eps) for m in range(HC)]
    out = []
    for p in range(H):
        acc = zero
        for k in range(HC):
            acc = o.fma(pre[k], xs[k][p], acc)
        out.append(o.bf16(acc))
    return stack(out)


@composite("HcHead", 1, ["HC", "H", "TH", "EPS", "HC_EPS"], _sig_head, conformance=_CONF)
def HcHead(B, S, residual, fn, scale, base):
    """`hc_head_fuse_tilelang` (tilelang_kernels.py:847) for one token (TH threads, h_block 8 TH, thread t elements p = 8 TH ih
    + 8 t + e): over (stream mc, block ih, e) the chains from +0 `sqr = fma(x, x, sqr)` and `mix[m] = fma(x, fn[m][H mc + p],
    mix[m])`, x = f32(residual[mc][p]); each the threads' halving tree; `rs = rsqrt(sqr / (HC H) + EPS)`; `pre[m] =
    MhcSigmoid(fma(mix[m] * rs, scale, base[m])) + HC_EPS`; out[p] = `bf16` of the chain from +0 of `fma(pre[k], x_k, .)`."""
    return hc_head(word_ops(B), S, residual, fn, scale, base)


WORD = dict(D.WORD, expf=M.NvExpf, bf16=P.F2fpBf16, gpu_add=P.F32AddV2, gpu_mul=P.F32MulV2, gpu_fma=P.F32FmaV2,
            sigmoid=MhcSigmoid, tf32=HopperTf32WgmmaDot8, MhcSinkhorn=MhcSinkhorn,
            MhcMixes=MhcMixes, MhcPreNorm=MhcPreNorm, MhcPreNormBroadcast=MhcPreNormBroadcast, MhcPreBigFuse=MhcPreBigFuse,
            MhcPreBigFuseBroadcast=MhcPreBigFuseBroadcast, MhcPost=MhcPost, MhcFusedSplit=MhcFusedSplit,
            MhcPrenormTf32Split=MhcPrenormTf32Split, MhcPrenormTf32=MhcPrenormTf32)

MHC_PRIMITIVES = [HopperTf32WgmmaDot8]
MHC_COMPOSITES = [MhcSigmoid, MhcSinkhorn, MhcMixes, MhcPreNorm, MhcPreNormBroadcast, MhcPreBigFuse, MhcPreBigFuseBroadcast,
                  MhcPost, MhcFusedSplit, MhcFusedPostPre, MhcPrenormTf32Split, MhcPrenormTf32, MhcPostPreTf32, MhcPreBroadcast,
                  HcHead]
