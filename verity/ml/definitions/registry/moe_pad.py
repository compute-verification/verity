"""The PADDED fixed-slot MoE block, padding first, under EXPLICIT ENABLEMENT, as Definitions on the lifted registry
(`verity/ml/definitions/registry/lifted.py`: the `D + bot` encoding, `Lifted[<id>]_v2` = every lifted computed gate takes its
owner's ENABLE as its LAST operand, the CLOSED specified class `LSelect / LIsActive / LBot / LGatherRow / LStepLive /
LFixedSelect / LEnable`).  Nothing here edits a frozen Definition and nothing new is asked of the specified class.

The construction, per token and MoE layer, with E expert slots FIXED before any route is known, `a` the block's own enable
(the enable of the call that owns the block: the row's continuation bit `run_t` in a decode Program, the root's constant 1 at a root):

    logits = Lifted[Gemm{K=H,N=E}]_v2(x, W_g, a)                     the router logits (x @ W_g^T), a lifted GEMM owned by the block
    (alpha, r) = Lifted[MoeRouterTopK_v1{E,TOPK,VPT}]_v2(logits, a)    the router's ids r_j and weights alpha_j: COMPUTED lifted values
    eq_je  = Lifted[I32Eq_v1]_v2(r_j, Lifted[Const32[e]]_v2(a), a)     one lifted bit per (slot j, expert e), owned by the block
    s_e    = OR_j eq_je                                                the selection condition (lifted BitNot/BitAnd, owned by the block)
    a_e    = LEnable_v1(a, s_e)                                        THE SLOT'S ENABLE: a_parent AND [e selected]; total, never bot (A9.2)
    g_e    = Lifted[LSelect32]_v2(eq_1e, alpha_1, ... Lifted[LBot32]_v2(a_e) ..., a_e)   the routed weight, produced INSIDE the slot
    z_e    = Lifted[MoeExpertBody_v1{H,I}]_v2(x, g_e, W13[e], W2[e], a_e)  the expert body with the slot's FIXED weights; x wired in UNCHANGED
    u_j    = Lifted[LFixedSelect<E>x16]_v2(r_j, [z_0[n], ..., z_{E-1}[n]], a)  per word n  over the E computed slot OUTPUTS (pad's G2)
    y      = Lifted[MoeSum_v1{TOPK,H}]_v2(u, a)                        the REGISTERED combine: ascending j from a zero f32 acc, one RNE

A12 as written: the slot input `x` is NOT `Select(s_e, x, bot)` any more -- it is wired in unchanged, and every gate of the slot's body is
owned by the slot call: `a_e = 1` -> the registered relation on its operands (bit for bit the indexed-read construction's pair on the
selected slot, (b));
`a_e = 0` -> `enc(bot)` whatever the operands (A9.3; the slot's internally produced values -- g_e, the w13 rows, the activation, z_e -- are
the bot words the integration materializes and commits, A14).  The E - TOPK disabled slots per token are therefore DISABLED
computations, not strict propagation of a bot input (the v1 reading, superseded).  Under `a = 0` (a disabled row: the continuation's
post-prefix rows) every value the block produces is bot -- ids, weights, conditions, every slot, u, y -- and every `a_e` is Active(0)
(LEnable(0, bot) = 0: an enable is never bot).

`MoeExpertBody_v1{H,I}` (new, unlifted, then lifted by node replay with the slot's enable threaded to every node) is the fused_moe_kernel
pair body WITHOUT the expert-axis gather: `Gemm{K=H,N=2I}` (the w13 launch, `MoeExpertCoordinate_v1` = `GemmCoordinate_v1` once the row is
fixed) -> `SiluMul{I}` (`_C.silu_and_mul`) -> `MoeSlotDown_v1{K=I,N=H}` (the w2 launch with MUL_ROUTED_WEIGHT=True:
`bf16_rne(f32mul(DotBf16(a, W2[e][n,:]), g))`, the routed weight multiplied into the f32 accumulator before the ONE bf16 conversion --
`MoeExpertCoordinateW_v1` minus `MoeExpertRow_v1`).  The multiplexing of the whole expert weight bank per pair (`GatherBf16x{E}` per
weight element, the indexed-read construction) is gone: each slot owns its weights.  The indexed-read construction (`registry/moe.py`)
stays registered as the labelled ALTERNATIVE the OLMoE rows of record were built on.

Weights enter as lifted `Input<17>` words checked Active at the boundary (pad's `LServe` convention, `check_boundary(prescribed=True)`);
an `Input` owns no enable (it is the Program's boundary, produced by no body).  The block returns the whole small assignment (y, ids,
weights, s_e, g_e, every slot's z_e, the selected u_j, and the E slot enables a_e) so the reference evaluator exposes it, as Stage 1 asks;
a Program that composes the block uses `LMoeFfn_v1` (y only).  Both take the enable as their LAST parameter (`lifted.ENABLE_PARAM`), so
`lifted.has_enable` holds and `replay_lifted` / `emit_lifted_node` thread a caller's enable to them like any lifted callee.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, array_of, coll_concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import lifted as L
from verity.ml.definitions.registry import moe
from verity.ml.definitions.registry import pad_prims as PP
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, F32

SEMANTIC_ID = "moe/PaddedFixedSlots@v1"
LABEL = "padded-fixed-slots"          # the baseline representation; the indexed-read rows keep their "alternative" label
STAGE = 2                             # Stage 2: the complete small padded MoE (E=4, top-1 and top-2)


def V(w: int) -> Value:
    """The lifted word of payload width `w` (`w+1` bits, tag bit `w`)."""
    return Value(w + 1)


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


# ---------------------------------------------------------------------------------------------------------
# the padded block: E fixed slots, computed guards, DISABLED unselected slots (s3 / A12), authored on Padded_v1 under B7
# ---------------------------------------------------------------------------------------------------------


def fixed_select(E: int, w: int = 16):
    """The `FixedSelect` RELATION over the E computed slot outputs: the lifted registry's `LFixedSelect<E>x<w>_v1` (G2 -- an Active
    out-of-range route id is `INV_w` NAMED, never a default slot; `r = bot -> bot`; in range -> `z_r` bit for bit including bot).  This is
    the v1 member = the relation at enable 1, what a verifier evaluates a committed `(r, z_0..z_{E-1}, u)` against; the GATE the block
    emits is its enabled form `fixed_select_enabled` (B7: the block's enable is the last operand)."""
    return L.l_fixed_select(E, w)


def fixed_select_enabled(E: int, w: int = 16):
    """`Lifted[LFixedSelect<E>x<w>_v1]_v2(r, z, enable)`: the combine gate as emitted (A12 under B7) -- bot under a disabled parent whatever `r`."""
    return L.lifted_prim(L.l_fixed_select(E, w))


FIXED_SELECT_MEMBER = "LFixedSelect (Padded_v1 G2) under B7 enablement"


def _lor(B, a: Coll, b: Coll, enable: Coll) -> Coll:
    """Lifted OR on `Value<2>` from pad_prims' `BitNot` / `BitAnd` under the owner's enable: `NOT(AND(NOT a, NOT b))`.  bot under a
    disabled owner; INV propagates; on Active bits the Boolean OR."""
    ln, la = L.lifted_prim(PP.BitNot), L.lifted_prim(PP.BitAnd)
    return B.call(ln, B.call(la, B.call(ln, a, enable), B.call(ln, b, enable), enable), enable)


def block_signature(S):
    return ((("x", Array(S.H, V(16))),
             ("wg", Array(S.E, Array(S.H, V(16)))),
             ("w13", Array(S.E, Array(2 * S.I, Array(S.H, V(16))))),
             ("w2", Array(S.E, Array(S.H, Array(S.I, V(16))))),
             (L.ENABLE_PARAM, L.ENABLE_T)),                # a: the block's enable, LAST (B7; `lifted.has_enable` holds for the block)
            Tuple(Array(S.H, V(16)),                      # y: the block output
                  Array(S.TOPK, V(32)),                   # ids r_j (lifted I32)
                  Array(S.TOPK, V(32)),                   # weights alpha_j (lifted F32)
                  Array(S.E, V(1)),                       # s_e: the selection condition per slot (lifted bit; bot under a disabled block)
                  Array(S.E, V(32)),                      # g_e: the slot's routed weight (bot when the slot is disabled)
                  Array(S.E, Array(S.H, V(16))),          # z_e: every slot's output (the disabled slots' bot words are HERE, exposed)
                  Array(S.TOPK, Array(S.H, V(16))),       # u_j: the selected outputs in slot order
                  Array(S.E, L.ENABLE_T)))                # a_e: the slot enables (A12: a AND s_e; Active(0)/Active(1), never bot)


@composite("LMoeBlock", 1, ["E", "TOPK", "VPT", "H", "I"], block_signature, conformance="protocol-defined")
def LMoeBlock(B, S, x, wg, w13, w2, a):
    """The fixed-slot padded MoE block of one token under its enable `a` (module docstring; A12): the Program is a function of
    (E, TOPK, VPT, H, I) alone; the route (ids, weights) is a value the block computes, the slot enables `a_e = LEnable(a, s_e)` are
    values the block computes, and every admitted route is an ASSIGNMENT to the same circuit."""
    logits = B.call(L.lift_def(bind(b1.Gemm, K=S.H, N=S.E)), x, wg, a, name="router/logits")
    return _lifted_core(B, S, x, logits, w13, w2, a)


def _lifted_core(B, S, x, logits, w13, w2, a):
    """Everything of the padded block AFTER the router GEMM, from the lifted `logits` (module docstring lines 2-9): the router top-k, the
    E slots with their enables `a_e`, the routed weights produced inside the slots, the bodies, the combine and the sum.  ONE emission
    shared by `LMoeBlock_v1` (which owns the router GEMM, `logits = Gemm(x, W_g)`) and `LMoeBlockL_v1` (which takes the logits as an
    input: the frontend's construction, route A (3), where the router GEMM is the layer's own `Gemm_v1{K=H,N=E}` call over `mlp.gate`
    and the fused block starts at `_moe_C.topk_softmax`) -- the same nodes in the same order, so `LMoeBlock_v1`'s digest stays
    1dd831feee7ac882 at E=4, K=2."""
    E, TOPK, H, I = S.E, S.TOPK, S.H, S.I
    r = B.call(L.lift_def(bind(moe.MoeRouterTopK, E=E, TOPK=TOPK, VPT=S.VPT)), logits, a, name="router/topk")
    alpha, ids = r[0], r[1]
    leq, len_ = L.lifted_prim(P.I32Eq), L.l_enable()
    lsel32, lbot32 = L.lifted_prim(L.l_select(32)), L.lifted_prim(L.l_bot(32))
    consts = [B.call(L.lifted_prim(b1.const(32, e)), a, name=f"const/{e}") for e in range(E)]   # Active(e) under the block's enable
    body = L.lift_def(bind(MoeExpertBody, H=H, I=I))                                           # ..., enable) -- the slot's enable threaded
    ss, aes, gs, zs = [], [], [], []
    for e in range(E):
        eqs = [B.call(leq, ids[j], consts[e], a, name=f"slot{e}/eq{j}") for j in range(TOPK)]
        s = eqs[0]
        for j in range(1, TOPK):
            s = _lor(B, s, eqs[j], a)
        ss.append(s)
        ae = B.call(len_, a, s, name=f"slot{e}/enable")                                        # a_e = a AND s_e (A9.2): total, never bot
        aes.append(ae)
        g = B.call(lbot32, ae, name=f"slot{e}/g_none")
        for j in reversed(range(TOPK)):                                                        # the ids are distinct: at most one j hits
            g = B.call(lsel32, eqs[j], alpha[j], g, ae, name=f"slot{e}/g{j}")                 # produced inside the slot: bot when disabled
        gs.append(g)
        zs.append(B.call(body, x, g, w13[e], w2[e], ae, name=f"slot{e}/expert"))              # x wired in UNCHANGED; the body owned by a_e
    zstack = array_of(zs)                                                                      # Array<E, Array<H>>
    cols = coll_concat(Array(H, Array(E, V(16))), [zs[e][n] for n in range(H) for e in range(E)])   # per word n: the E candidates
    lg = fixed_select_enabled(E, 16)                                                           # the combine, owned by the block
    us = [B.batch(lg, ids[j], cols, a, axes=(None, 0, None), name=f"out{j}/select") for j in range(TOPK)]
    y = B.call(L.lift_def(bind(moe.MoeSum, TOPK=TOPK, H=H)), array_of(us), a, name="moe_sum")
    return tuple_of(y, ids, alpha, array_of(ss), array_of(gs), zstack, array_of(us), array_of(aes))


@composite("LMoeFfn", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: (block_signature(S)[0], Array(S.H, V(16))), conformance="protocol-defined")
def LMoeFfn(B, S, x, wg, w13, w2, a):
    """The block as a layer would call it: `y` only (the assignment record stays inside the callee's nodes, addressable as proof blocks); the
    caller's enable is its last operand, as for every lifted callee."""
    return B.call(bind(LMoeBlock, E=S.E, TOPK=S.TOPK, VPT=S.VPT, H=S.H, I=S.I), x, wg, w13, w2, a, name="moe")[0]


@composite("MoeFfnFixed", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: ((("x", Array(S.H, BF16)), ("wg", Array(S.E, Array(S.H, BF16))), ("w13", Array(S.E, Array(2 * S.I, Array(S.H, BF16)))),
                       ("w2", Array(S.E, Array(S.H, Array(S.I, BF16))))), Array(S.H, BF16)),
           conformance="protocol-defined")
def MoeFfnFixed(B, S, x, wg, w13, w2):
    """The UNLIFTED fixed-slot twin of `LMoeFfn_v1` (Stage 3 builder, route A (1)): router -> ALL `E` expert bodies on their own fixed
    weights (`g_e` = the routed weight of the slot the router picked, `0.0f` otherwise) -> the `TOPK` slot outputs selected by the router
    ids -> `MoeSum_v1`.  Value-equal to the indexed-read construction on every input (the same pair body, `MoeExpertBody_v1`; the selection is by the same ids) and
    route-independent in structure; the frontend binds one call per token to it, and its lift is the HAND-AUTHORED `LMoeFfn_v1` (the
    unselected slots DISABLED by `a_e = LEnable(a, s_e)`, their values bot) -- the strict replay of THIS body would demand Active values nobody computed for the
    unselected bodies, which is why the lift is registered by hand and never replayed."""
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


# Route A (2) (`lifted.register_hand_lift`): `LMoeFfn_v1` IS the padded-v1 lift of the twin --
# `lift_def(bind(MoeFfnFixed, ...))` on the registry path AND on the descriptor-decoded path (the workload's request_component /
# codec.decode_program:
# the Build path) yields `bind(LMoeFfn, **same bindings)` in place of the node replay `Lifted[MoeFfnFixed_v1]_v2` (which would be an all-Active
# Program).  Checked by name at registration (same statics, composites, one lift per id, before any replay lift of the id) and per binding
# (the lift's signature == the twin's lifted params + ('enable', Value<2>) last).  The body is this module's.
HAND_LIFT = L.register_hand_lift(MoeFfnFixed, LMoeFfn)


# ---------------------------------------------------------------------------------------------------------
# Route A (3): the FRONTEND's unit -- the block from the router LOGITS.  In vLLM's OLMoE the router
# GEMM is the layer's own `mlp.gate` linear (`Gemm_v1{K=H,N=E}`, bound by the base GEMM rule; its output `router_logits` is a boundary value
# of the record) and the fused block `vllm.moe_forward(hidden_states, router_logits)` starts at `_moe_C.topk_softmax`.  One resolution
# per dispatcher root (resolver / frontend alike) means the block's Definition takes the logits as an INPUT rather than recomputing them:
# `MoeFfnFixedL_v1(x, logits, w13, w2)` is `MoeFfnFixed_v1` minus its router GEMM (the same `_fixed_core`, node for node), and its lift
# of record is the hand-authored `LMoeFfnL_v1` = `LMoeBlockL_v1[0]` (the same `_lifted_core` as `LMoeBlock_v1`: the E slots with their
# enables, the routed weights produced inside the slots, the fixed-select combine, the registered sum).  `LMoeBlock_v1` / `LMoeFfn_v1` /
# `MoeFfnFixed_v1` stay registered unchanged (Stage 2, route A (1)(2)) as the router-owning form of the same block.
# ---------------------------------------------------------------------------------------------------------


def block_l_signature(S):
    sig, out = block_signature(S)
    return ((sig[0], ("logits", Array(S.E, V(16)))) + sig[2:], out)          # x, logits (in place of wg), w13, w2, enable LAST


@composite("LMoeBlockL", 1, ["E", "TOPK", "VPT", "H", "I"], block_l_signature, conformance="protocol-defined")
def LMoeBlockL(B, S, x, logits, w13, w2, a):
    """`LMoeBlock_v1` from the router LOGITS (the layer's `Gemm_v1{K=H,N=E}` over `mlp.gate`, a lifted input here): the same assignment
    record (y, ids, alpha, s_e, g_e, z_e, u_j, a_e), the same nodes after the GEMM (`_lifted_core`), under the block's enable `a`."""
    return _lifted_core(B, S, x, logits, w13, w2, a)


@composite("LMoeFfnL", 1, ["E", "TOPK", "VPT", "H", "I"],
           lambda S: (block_l_signature(S)[0], Array(S.H, V(16))), conformance="protocol-defined")
def LMoeFfnL(B, S, x, logits, w13, w2, a):
    """`y` only, from the logits: the lift of record of the frontend's `MoeFfnFixedL_v1` call (route A (3)); the caller's enable LAST."""
    return B.call(bind(LMoeBlockL, E=S.E, TOPK=S.TOPK, VPT=S.VPT, H=S.H, I=S.I), x, logits, w13, w2, a, name="moe")[0]


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
    (N-pad-09: the schedule's shapes -- blocks per expert, num_tokens_post_padded -- never enter the statics).  Its lift of record is the
    HAND-AUTHORED `LMoeFfnL_v1` (registered below), never the node replay."""
    return _fixed_core(B, S, x, logits, w13, w2)


HAND_LIFT_L = L.register_hand_lift(MoeFfnFixedL, LMoeFfnL)


def block(E: int, TOPK: int, VPT: int, H: int, I: int):
    """The fully bound fixed-slot block: the ONE Program of the (E, TOPK, VPT, H, I) envelope (for fixed construction inputs the
    builder produces one Program; routes are assignments).  `admitted_routes` enumerates its value space."""
    if E % VPT:
        raise ValueError("E must be a multiple of VPT (the router's reduction geometry)")
    if not (1 <= TOPK <= E):
        raise ValueError("TOPK must be in [1, E]")
    return bind(LMoeBlock, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)


def admitted_routes(E: int, TOPK: int) -> list[tuple[int, ...]]:
    """Every ordered route the router can emit: TOPK DISTINCT expert ids (the winner of each argmax round is masked to -10000.f),
    in selection order.  E=4: 4 routes at top-1, 12 at top-2 -- the Stage 2 population every test walks in full."""
    from itertools import permutations
    return [tuple(p) for p in permutations(range(E), TOPK)]


ALL = [MoeSlotDownCoordinate, MoeSlotDown, MoeExpertBody, LMoeBlock, LMoeFfn, MoeFfnFixed, LMoeBlockL, LMoeFfnL, MoeFfnFixedL]
