"""`dsa_norm`'s RMSNorms on bits: `DsaRMSNorm_v2{N, EPS}` and `TritonBlockRMSNorm_v2{N, BLOCK, WARPS, EPS}`, `dsa_norm.rms_norm` over
`boolean_dsa.bit_ops` (core's Boolean FP32 pieces: `Bf16ToF32`, `F32Fma`, `F32Mul`, `F32Add`, `DivFullScaleA`, `DivFullRcp`,
`RsqrtApprox`) and `F2fpBf16_v2`, and `DsaRMSNormSumSq_v2{N, EPS}` and `DsaRMSNormApply_v2{G}`, that row cut at its scale
(`dsa_norm.sum_sq`, `dsa_norm.apply`), each traced as one circuit (`boolean_dsa.flat`), so its word view is the `_v1` at the same
statics.  `DsaLatentRow_v2{KVL, R, EPS}` is `DsaLatentRow_v1` on bits, a composite of `DsaRMSNorm_v2` and
`boolean_rope.RoPEHeadGptJ_v3`.

    DsaRMSNormSumSq_v3{N, EPS}, DsaRMSNormApply_v3{G}, DsaRMSNorm_v3{N, EPS}   the `_v2`s' functions written as Calls; ``opt_in``
    DsaRMSNormSquares_v1            (x0, x1: bf16) -> f32     a thread's first two registers: fma(f32(x0), f32(x0), f32(x1)^2)
    DsaRMSNormSquareFma_v1          (x: bf16, s: f32) -> f32  each next register: fma(f32(x), f32(x), s)
    DsaRMSNormMeanRsqrt_v1{N, EPS}  (sum: f32) -> f32         rsqrt.approx.ftz(eps + div.full(sum, N)), N and eps folded
    DsaRMSNormScale_v1              (r: f32) -> rd            the gates of an output element that read only the scale r
    DsaRMSNormOut_v1                (x, w: bf16, r: f32, rd) -> bf16   the rest of bf16(r * f32(x) * f32(w)), from r's own gates

A traced unit's description is every gate of it, so it grows with the row.  The `_v3`s are the units whose description does not:
the scale is a batch of `DsaRMSNormSquares` over the 128 threads, then one of `DsaRMSNormSquareFma` for each further register
(N / 128 - 2 of them), the 127 `F32Add_v3` of the warps' butterflies and their halving tree, and `DsaRMSNormMeanRsqrt`; the
apply unit is `DsaRMSNormScale` once and a batch of `DsaRMSNormOut` over its G elements, an output element's circuit cut at the
scale every element reads (`EW.shared_operand`), so r is decoded once, as one trace of the unit decodes it.  Each piece is
traced from the word body's steps over `bit_ops`, as the `_v2`s are, so a bf16 operand's 16 zero bits, a square's equal
operands and the constants N and eps are folded into its gates, and the pieces of a scale or apply unit together hold the
`_v2`'s gates, count for count.  `DsaRMSNorm_v3` is the scale and then the apply unit over the row; an element's own gates,
which one trace of the row shares between its square and its output, are computed by each (circuit-check counts them as
redundant gates, a warning).  They are ``opt_in`` (`verity.definitions.boolean`): `boolean_version` keeps the `_v2`s, and a
`_v3` serves only a caller that binds it."""
from __future__ import annotations

from types import SimpleNamespace

from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation import bits as BS
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.ops.boolean import gemm as BG
from verity.circuits.refs import Coll, concat, strided_view
from verity.circuits.types import Array
from verity.definitions.registry import boolean_rope as BR
from verity.definitions.registry import dsa as D
from verity.definitions.registry import dsa_norm as DN
from verity.definitions.registry.boolean_dsa import F32, bit_ops, flat, replay, replayed

BF16 = Array(16, BL.BIT)


def _rms(B, S, x, w):
    """`DsaRMSNorm_v1`'s body on bits."""
    return DN.rms_norm(bit_ops(B), S, x, w, BG.F2fpBf16, BF16)


DsaRMSNormV2 = flat("DsaRMSNorm", 2, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
                    _rms, DN.DsaRMSNorm)


def _sum_sq(B, S, x):
    """`DsaRMSNormSumSq_v1`'s body on bits."""
    return DN.sum_sq(bit_ops(B), S, x)


DsaRMSNormSumSqV2 = flat("DsaRMSNormSumSq", 2, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)),), F32), _sum_sq,
                         DN.DsaRMSNormSumSq)


def _apply(B, S, x, w, r):
    """`DsaRMSNormApply_v1`'s body on bits."""
    return DN.apply(bit_ops(B), S.G, x, w, r, BG.F2fpBf16, BF16)


DsaRMSNormApplyV2 = flat("DsaRMSNormApply", 2, ["G"],
                         lambda S: ((("x", Array(S.G, BF16)), ("w", Array(S.G, BF16)), ("r", F32)), Array(S.G, BF16)), _apply,
                         DN.DsaRMSNormApply)


def _block_rms(B, S, x, w):
    """`TritonBlockRMSNorm_v1`'s body on bits."""
    return DN.block_rms_norm(bit_ops(B), S, x, w, BG.F2fpBf16, BF16)


TritonBlockRMSNormV2 = flat("TritonBlockRMSNorm", 2, ["N", "BLOCK", "WARPS", "EPS"],
                            lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)), _block_rms,
                            DN.TritonBlockRMSNorm)

def _latent(B, S, kv, w, cs):
    """`DsaLatentRow_v1` on bits: `DsaRMSNorm_v2{KVL, EPS}` of kv's first KVL words, then `boolean_rope.RoPEHeadGptJ_v3{R}` of its
    last R, as one row."""
    kvc = B.call(bind(DsaRMSNormV2, N=S.KVL, EPS=S.EPS), kv[0:S.KVL], w)
    kpe = B.call(bind(BR.RoPEHeadGptJ, D=S.R), kv[S.KVL:S.KVL + S.R], cs)
    return Coll(Array(S.KVL + S.R, BF16), concat([kvc.refs, kpe.refs]))


DsaLatentRowV2 = EW.on_bits(
    "DsaLatentRow", 2, ["KVL", "R", "EPS"],
    lambda S: ((("kv", Array(S.KVL + S.R, BF16)), ("w", Array(S.KVL, BF16)), ("cs", Array(S.R, BF16))), Array(S.KVL + S.R, BF16)),
    _latent, DN.DsaLatentRow)


def _own_gates(part: CompositeDefinition):
    """A builder for the traced ``part``, its own gates on a circuit: with one, Flock's flat lowering runs each activation of
    the part (`ir_lower.lower_gates`) and keeps a word for what it returns, not for each of its gates. The circuit it lowers
    to is the same. A part with statics takes them first."""
    def gates(C, S, *elements):
        return replay(BS.compile(bind(part, **{k: getattr(S, k) for k in part.statics})), C, [f for e in elements for f in e])
    return gates if part.statics else lambda C, *elements: gates(C, None, *elements)


def _piece(name: str, statics: tuple[str, ...], signature, body) -> CompositeDefinition:
    """``name_v1{statics}``, a piece of a norm written as Calls: ``body`` (over `bit_ops`) compiled and traced as one circuit per
    binding, as `flat` traces a whole unit, with no word view; Flock lowers an activation by its own gates (`_own_gates`)."""
    inner = CompositeDefinition(name, 1, statics, signature, body, doc=body.__doc__ or "", conformance="exact-model-tested",
                                register=False)
    d = EW.specialized(name, 1, statics, replayed(inner, statics, signature), signature, doc=body.__doc__ or "")
    d.build = _own_gates(d)  # type: ignore[attr-defined]
    return d


def _squares(B, S, x0, x1):
    """A thread's sum of squares over its first two registers, `dsa_norm.rms_scale`'s first step:
    ``fma.rn(f32(x0), f32(x0), mul.f32(f32(x1), f32(x1)))``."""
    o = bit_ops(B)
    v0, v1 = o.cvt(x0), o.cvt(x1)
    return o.fma(v0, v0, o.mul(v1, v1))


def _square_fma(B, S, x, s):
    """A thread's sum of squares taking one more register, each later step of `dsa_norm.rms_scale`'s chain:
    ``fma.rn(f32(x), f32(x), s)``."""
    o = bit_ops(B)
    v = o.cvt(x)
    return o.fma(v, v, s)


def _mean_rsqrt(B, S, total):
    """`dsa_norm.rms_scale`'s epilogue: ``rsqrt.approx.ftz(add.f32(eps, div.full.f32(sum, N)))``, `div.full` its two MUFU pieces
    and their product, the words N and eps constants of the binding."""
    o = bit_ops(B)
    n = o.word(D.f32_bits(S.N))
    return o.rsqrt(o.add(o.word(D.f32_bits(S.EPS)), o.mul(o.scale_a(total, n), o.rcp(n))))


DsaRMSNormSquares = _piece("DsaRMSNormSquares", (), lambda S: ((("x0", BF16), ("x1", BF16)), F32), _squares)
DsaRMSNormSquareFma = _piece("DsaRMSNormSquareFma", (), lambda S: ((("x", BF16), ("s", F32)), F32), _square_fma)
DsaRMSNormMeanRsqrt = _piece("DsaRMSNormMeanRsqrt", ("N", "EPS"), lambda S: ((("sum", F32),), F32), _mean_rsqrt)


def _sum_sq_calls(B, S, x):
    """`DsaRMSNormSumSq_v1`'s body on bits as Calls: register k of the 128 threads (`dsa_norm.layout`) is one row of the batch
    that takes it, `DsaRMSNormSquares` for the first two and `DsaRMSNormSquareFma` for each next one; the warps' butterflies
    and their halving tree are `F32Add_v3`s, in `dsa_norm.rms_scale`'s order; then `DsaRMSNormMeanRsqrt{N, EPS}`."""
    DN.check_statics(S.N, S.EPS)
    regs = DN.layout(S.N)
    step = regs[1][0] - regs[0][0]

    def register(k: int) -> Coll:
        """Register k of each thread: element ``regs[t][k]``, a thread's elements ``step`` apart from the next thread's."""
        return Coll(Array(DN.THREADS, BF16), strided_view(x, 16 * regs[0][k], 16, 16 * step, 1, 16 * DN.THREADS))
    s = B.batch(DsaRMSNormSquares, register(0), register(1), axes=(0, 0))
    for k in range(2, len(regs[0])):
        s = B.batch(DsaRMSNormSquareFma, register(k), s, axes=(0, 0))
    add = bit_ops(B).add
    warps = [D.halving_tree([s[t] for t in range(DN.WARP * g, DN.WARP * (g + 1))], add) for g in range(DN.THREADS // DN.WARP)]
    return B.call(bind(DsaRMSNormMeanRsqrt, N=S.N, EPS=S.EPS), D.halving_tree(warps, add))


DsaRMSNormSumSqV3 = EW.on_bits("DsaRMSNormSumSq", 3, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)),), F32), _sum_sq_calls,
                               DN.DsaRMSNormSumSq)
DsaRMSNormSumSqV3.opt_in = True  # type: ignore[attr-defined]

_ELEMENT = (("x", BF16), ("w", BF16), ("r", F32))


def _element(B, S, x, w, r):
    """One output element of `DsaRMSNormApply_v1`'s body on bits: ``bf16(r * f32(x) * f32(w))``, the products in that order."""
    y = DN.apply(bit_ops(B), 1, Coll(Array(1, BF16), x.refs), Coll(Array(1, BF16), w.refs), r, BG.F2fpBf16, BF16)
    return Coll(BF16, y.refs)


def _element_gates(C, *elements):
    """`_element`'s circuit over its operands' bits, the scale's own gates among them."""
    inner = CompositeDefinition("DsaRMSNormOut", 1, (), lambda S: (_ELEMENT, BF16), _element, doc=_element.__doc__,
                                conformance="exact-model-tested", register=False)
    return replayed(inner, (), lambda S: (_ELEMENT, BF16))(None)(C, *elements)


DsaRMSNormScale, DsaRMSNormOut = EW.shared_operand(
    ("DsaRMSNormScale", "DsaRMSNormOut"), 1, _element_gates, _ELEMENT, BF16, 2,
    ("the gates of bf16(r * f32(x) * f32(w)) that read only the scale r, once for the elements that share it",
     "bf16(r * f32(x) * f32(w)) on bits, one output element of DsaRMSNormApply, from r's own gates rd (DsaRMSNormScale_v1)"))
DsaRMSNormScale.build = _own_gates(DsaRMSNormScale)  # type: ignore[attr-defined]
DsaRMSNormOut.build = _own_gates(DsaRMSNormOut)  # type: ignore[attr-defined]


def _apply_calls(B, S, x, w, r):
    """`DsaRMSNormApply_v1`'s body on bits as Calls: the scale's own gates once (`DsaRMSNormScale`), then `DsaRMSNormOut` of each
    element, its weight, the scale and those gates, a batch over the G elements."""
    return B.batch(DsaRMSNormOut, x, w, r, B.call(DsaRMSNormScale, r), axes=(0, 0, None, None))


DsaRMSNormApplyV3 = EW.on_bits("DsaRMSNormApply", 3, ["G"],
                               lambda S: ((("x", Array(S.G, BF16)), ("w", Array(S.G, BF16)), ("r", F32)), Array(S.G, BF16)),
                               _apply_calls, DN.DsaRMSNormApply)
DsaRMSNormApplyV3.opt_in = True  # type: ignore[attr-defined]


def _rms_calls(B, S, x, w):
    """`DsaRMSNorm_v1`'s body on bits as Calls: `DsaRMSNormSumSq_v3{N, EPS}`'s body for the scale, then `DsaRMSNormApply_v3`'s
    over the row's N elements."""
    return _apply_calls(B, SimpleNamespace(G=S.N), x, w, _sum_sq_calls(B, S, x))


DsaRMSNormV3 = EW.on_bits("DsaRMSNorm", 3, ["N", "EPS"],
                          lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)), _rms_calls, DN.DsaRMSNorm)
DsaRMSNormV3.opt_in = True  # type: ignore[attr-defined]

BOOLEAN_DSA_NORM = [DsaRMSNormV2, DsaRMSNormSumSqV2, DsaRMSNormApplyV2, TritonBlockRMSNormV2, DsaLatentRowV2, DsaRMSNormSquares,
                    DsaRMSNormSquareFma, DsaRMSNormMeanRsqrt, DsaRMSNormSumSqV3, DsaRMSNormScale, DsaRMSNormOut,
                    DsaRMSNormApplyV3, DsaRMSNormV3]
