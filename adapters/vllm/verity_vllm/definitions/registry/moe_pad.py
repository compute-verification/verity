"""The fixed-slot MoE block as Definitions: `MoeFfnFixed_v1` (the router GEMM, then the shared core) and `MoeFfnFixedL_v1` (the
same core over given router logits, the unit the frontend binds), both over `MoeExpertBody_v1{H,I}`, the fused_moe_kernel pair body
without the expert-axis gather: `Gemm{K=H,N=2I}` (the w13 launch) -> `SiluMul{I}` -> `MoeSlotDown_v1{K=I,N=H}` (the w2 launch with
MUL_ROUTED_WEIGHT=True: `bf16_rne(f32mul(DotBf16(a, W2[e][n,:]), g))`, the routed weight multiplied into the f32 accumulator before
the one bf16 conversion).  Each slot owns its weights: E expert bodies run per token, `g_e` the routed weight of the slot the router
picked (`0.0f` otherwise), the TOPK slot outputs are selected by the router ids and summed by `MoeSum_v1`.  The indexed-read
construction (`registry/moe.py`) stays registered as the labelled alternative the OLMoE rows of record were built on.
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.circuits.refs import array_of, coll_concat
from verity.circuits.types import Array

from verity_vllm.definitions.registry import b1, moe
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.prims import BF16, F32


# ---------------------------------------------------------------------------------------------------------
# the unlifted per-expert body with FIXED weights (new ids; the frozen indexed-read pair bodies in `moe.py` are untouched)
# ---------------------------------------------------------------------------------------------------------


@composite("MoeSlotDownCoordinate", 1, ["K"], lambda S: ((("a", Array(S.K, BF16)), ("g", F32), ("wrow", Array(S.K, BF16))), BF16),
           conformance="not-yet-established")
def MoeSlotDownCoordinate(B, S, a, g, wrow):
    """One output word of the down projection of a FIXED expert: `bf16_rne(f32mul(DotBf16(a, W2[e][n, :]), g))` -- the routed weight
    `g` multiplied into the f32 accumulator before the single bf16 conversion (fused_moe_kernel MUL_ROUTED_WEIGHT=True:
    `accumulator = accumulator * moe_weight[:, None]; accumulator.to(compute_type)`) = `MoeExpertCoordinateW_v1` with the row given."""
    acc = B.call(bind(b1.DotBf16, K=S.K), a, wrow)
    return B.call(P.F2fpBf16, B.call(P.F32Mul, acc, g))


@composite("MoeSlotDown", 1, ["K", "N"], lambda S: ((("a", Array(S.K, BF16)), ("g", F32), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="not-yet-established")
def MoeSlotDown(B, S, a, g, w):
    """`y = g * (a @ W2[e]^T)` for one fixed expert slot: the w2 launch over the slot's own [N, K] weight."""
    return B.batch(bind(MoeSlotDownCoordinate, K=S.K), a, g, w, axes=(None, None, 0))


@composite("MoeExpertBody", 1, ["H", "I"],
           lambda S: ((("x", Array(S.H, BF16)), ("g", F32), ("w13", Array(2 * S.I, Array(S.H, BF16))), ("w2", Array(S.H, Array(S.I, BF16)))),
                      Array(S.H, BF16)),
           conformance="not-yet-established")
def MoeExpertBody(B, S, x, g, w13, w2):
    """`E_e(x, W_e)` (s3): the expert's full body with ITS weights -- up/gate `Gemm{K=H,N=2I}` (the w13 launch; `MoeExpertCoordinate_v1`
    is `GemmCoordinate_v1` once the row is fixed) -> `SiluMul{I}` (`_C.silu_and_mul`, gate = [:I], up = [I:]) -> `MoeSlotDown{K=I,N=H}`
    (the w2 launch with the routed weight).  No gather over the expert axis anywhere: the slot's weights are structural inputs."""
    gu = B.call(bind(b1.Gemm, K=S.H, N=2 * S.I), x, w13, name="w13")
    a = B.call(bind(b1.SiluMul, I=S.I), gu, name="silu_mul")
    return B.call(bind(MoeSlotDown, K=S.I, N=S.H), a, g, w2, name="w2")


@composite("MoeFfnFixed", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: ((("x", Array(S.H, BF16)), ("wg", Array(S.E, Array(S.H, BF16))), ("w13", Array(S.E, Array(2 * S.I, Array(S.H, BF16)))),
                       ("w2", Array(S.E, Array(S.H, Array(S.I, BF16))))), Array(S.H, BF16)),
           conformance="protocol-defined")
def MoeFfnFixed(B, S, x, wg, w13, w2):
    """The fixed-slot twin (Stage 3 builder, route A (1)): router -> ALL `E` expert bodies on their own fixed
    weights (`g_e` = the routed weight of the slot the router picked, `0.0f` otherwise) -> the `TOPK` slot outputs selected by the router
    ids -> `MoeSum_v1`.  Value-equal to the indexed-read construction on every input (the same pair body, `MoeExpertBody_v1`; the selection is by the same ids) and
    route-independent in structure; the frontend binds one call per token to it,."""
    logits = B.call(bind(b1.Gemm, K=S.H, N=S.E), x, wg, name="router/logits")
    return _fixed_core(B, S, x, logits, w13, w2)


def _fixed_core(B, S, x, logits, w13, w2):
    """The unlifted twin's body after the router GEMM (shared by `MoeFfnFixed_v1` and `MoeFfnFixedL_v1`; same nodes, same order)."""
    E, TOPK, H, I = S.E, S.TOPK, S.H, S.I
    r = B.call(bind(moe.MoeRouterTopK, E=E, TOPK=TOPK, VPT=S.VPT), logits, name="router/topk")
    alpha, ids = r[0], r[1]
    zero = B.call(b1.ZERO32)
    consts = [B.call(b1.const(32, e)) for e in range(E)]
    body = bind(MoeExpertBody, H=H, I=I)
    zs = []
    for e in range(E):
        g = zero
        for j in reversed(range(TOPK)):                                              # the ids are distinct: at most one j hits slot e
            g = B.call(P.SelectF32, B.call(P.I32Eq, ids[j], consts[e], name=f"slot{e}/eq{j}"), alpha[j], g, name=f"slot{e}/g{j}")
        zs.append(B.call(body, x, g, w13[e], w2[e], name=f"slot{e}/expert"))
    cols = coll_concat(Array(H, Array(E, BF16)), [zs[e][n] for n in range(H) for e in range(E)])
    us = [B.batch(P.gather_bf16(E), ids[j], cols, axes=(None, 0), name=f"out{j}/select") for j in range(TOPK)]
    return B.call(bind(moe.MoeSum, TOPK=TOPK, H=H), array_of(us), name="moe_sum")


@composite("MoeFfnFixedL", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: ((("x", Array(S.H, BF16)), ("logits", Array(S.E, BF16)), ("w13", Array(S.E, Array(2 * S.I, Array(S.H, BF16)))),
                       ("w2", Array(S.E, Array(S.H, Array(S.I, BF16))))), Array(S.H, BF16)),
           conformance="protocol-defined")
def MoeFfnFixedL(B, S, x, logits, w13, w2):
    """The frontend's fixed-slot MoE Definition (route A (3), `moe_construction = padded`): ONE call per token binds vLLM's fused block
    `vllm.moe_forward(hidden_states, router_logits)` -- `_moe_C.topk_softmax` -> `moe_align_block_size` (schedule marker) ->
    `fused_moe_kernel` x2 -> `_C.silu_and_mul` -> `_moe_C.moe_sum` -- as `MoeFfnFixed_v1` minus the router GEMM: the router top-k on the
    given logits, ALL E expert bodies on their own weights, the TOPK slot outputs selected by the ids, `MoeSum_v1`.  Value-equal to
    the indexed-read construction's six-Compute reading of the same launches on every input (the same pair body, the same ids); route-independent in structure
    (N-pad-09: the schedule's shapes -- blocks per expert, num_tokens_post_padded -- never enter the statics).."""
    return _fixed_core(B, S, x, logits, w13, w2)


ALL = [MoeSlotDownCoordinate, MoeSlotDown, MoeExpertBody, MoeFfnFixed, MoeFfnFixedL]
