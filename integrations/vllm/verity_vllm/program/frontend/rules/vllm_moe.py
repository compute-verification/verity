"""Bet A rules for vLLM's fused-MoE block under the batch-invariant profile — the leaf kernels `export_compat.moe_ops_traced`
makes visible in place of the opaque `vllm.moe_forward` dispatcher: generic in the block's shape parameters
(E experts, TOPK, hidden H, intermediate I, tokens M), nothing about a model.

The block, in launch order, and what each rule binds (numerics: `verity.ml.definitions.registry.moe`, difftested bit-exact in
`verity_vllm/check/moe_kernel_xcheck.py` against the same kernels on cp-moe):

    _moe_C.topk_softmax(w!, ids!, tei!, logits, renormalize, bias=None, is_padding=None)
        -> MoeRouterTopK{E,TOPK,VPT} per token row: f32 softmax in the CUDA kernel's reduction geometry, TOPK rounds of lowest-index argmax.
           `w` [M,TOPK] f32 and `ids` [M,TOPK] i32 are rebound; `tei` (token_expert_indices) is dead (its readers refuse).
    _moe_C.moe_align_block_size(ids, E, BLOCK_M, sorted_ids!, expert_ids!, num_post_padded!)
        -> a schedule MARKER, not values: the sorted order is a permutation the two GEMM launches only use for addressing (every token/slot pair
           is computed exactly once, the same way, wherever it lands in the sorted stream) -- the Program is the unpermuted pair set.
    verity_cba::triton_launch(fused_moe_kernel; a, b, c, [bias], w, sorted_ids, expert_ids, num_post_padded, ...)  [x2]
        -> MoeExpertGemm{E,K,N} (MUL_ROUTED_WEIGHT=False; top_k=TOPK: pair p reads a[p // TOPK]) resp. MoeExpertGemmW{E,K,N}
           (MUL_ROUTED_WEIGHT=True; top_k=1: pair p reads a[p]) once per (token, slot) PAIR p, with the pair's expert id `ids[p]` and (W) its
           routed weight `w[p]` as VALUE operands; `c` is rebound.  `b` is the [E, N, K] expert weight stack (committed weight input).
    _C.silu_and_mul(out!, x)                     -> the profile's SiluMul{I} rule (profile.SiluAndMulRule), per pair row.
    _moe_C.moe_sum(y [M,TOPK,H], out! [M,H])     -> MoeSum{TOPK,H} per token.

Both fused_moe_kernel schedules are accepted: the sorted one (`sorted_token_ids_ptr` = the align marker's output) and vLLM's
`naive_block_assignment` (`expert_ids_ptr` = the router ids themselves, one block per pair); the emitted Program is the same.
Everything else the launch could be (SPLIT_K > 1, fused bias, fp8/int8 paths, SWAP_AB, group quant, another compute_type, another source
body) refuses with the pinned field named, as `TritonGemm[vllm-bi]` does.

[route A (3): `TargetProfile.moe_construction`] TWO rule families own the same six leaf kernels, exclusively, by the
DECLARED construction (`vllm_bindings.active_target_profile().effective_moe_construction()`; a Program static, never attested):

  * `indexed-read` (the default; MOE-01, the four rules above): the block as SIX kinds -- router, the pair-major expert GEMMs with the
    expert id as a VALUE operand, the activation, the routed sum; every intermediate (`topk_weights`, `topk_ids`, the pair caches) is a
    value of the Program.
  * `padded` (the `Padded*` rules below): the block as ONE kind per token, `MoeFfnFixedL_v1{E,TOPK,VPT,H,I}(x_t, logits_t, W13, W2)`
    (`registry.moe_pad`: ALL E expert bodies on their own fixed weights, the TOPK slot outputs selected by the router ids, `MoeSum_v1`;
    its lift of record the hand-authored `LMoeFfnL_v1`, the unselected slots DISABLED).  The rules bind the same six launches in the same
    order with the SAME pins (`_launch_pins`), but the intermediates are not values: `topk_softmax` rebinds `w` / `ids` to a `PaddedRoute`
    MARKER (the logits rows it was computed from), the align kernel to a `PaddedSchedule` marker over it, the up GEMM's `c` to `PaddedUp`
    (the hidden rows + W13 it read), `silu_and_mul`'s `out` to `PaddedAct`, the down GEMM's `c` to `PaddedDown` (+ W2), and `moe_sum` --
    the block's last kernel, where every operand of the token's Definition is known -- EMITS the M calls and rebinds `out` to their rows.
    Any OTHER reader of an intermediate meets a marker and refuses (`Ctx.coll`: "is not a tensor value (Padded...)") -- the block's
    interior is the Definition's, not the Program's (fail closed, never a silent mix).  Under `padded` the MOE-01 rules decline by name
    (and vice versa), so a Program is of ONE construction; two Programs of the same trace differ in digest by construction (G1 by name).
    The schedule's shapes (blocks per expert, `num_tokens_post_padded`) are read as the marker and never enter the statics.
"""

from __future__ import annotations

import math

import torch

from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity_vllm.program.frontend.derive import Alloc, StaticIndex, refuse
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import InPlaceRule, RuleInfo, ViewRule
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.triton_iface import launch_meta, launch_tensor, triton_launch
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_bindings.pins import moe_expert_refusal, moe_expert_role

aten = torch.ops.aten

#: the fused_moe_kernel launch fields pinned to the bf16 unquantized path the Definitions model.  `src_sha256` is the kernel body of the
#: pinned vLLM (recorded from the cp-moe export; another body with the same name refuses).  Tile constexprs are geometry the f32 accumulator
#: is invariant to along K only in k16 chunks (BLOCK_SIZE_K % 16 == 0, checked), so they are recorded, not pinned.
MOE_GEMM_PINS = {
    "kernel": "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel",
    "src_sha256": "90674f19ec8363c7321df3e29e29ad1e2137720c2005b561e087ddb5e823e48b",  # vLLM d9105ea80 fused_moe_kernel.src (cp-moe)
    "constexprs": {"SPLIT_K": 1, "HAS_BIAS": False, "SWAP_AB": False, "USE_TD": False, "use_fp8_w8a8": False, "use_int8_w8a8": False,
                   "use_int8_w8a16": False, "per_channel_quant": False, "group_n": 0, "group_k": 0, "compute_type": "<dtype:bf16>"},
    "none_ptrs": ("b_bias_ptr", "a_scale_ptr", "b_scale_ptr"),
    # the launch geometry vLLM picks under VLLM_BATCH_INVARIANT=1 on the L40S for the OLMoE stacks (`try_get_optimal_moe_config`: no tuned table
    # -> the BI default, the SAME at every M from 1 to 8192, checked on cp-moe) and the only one the xcheck has evidence for: any other tiling
    # refuses (numerics_unregistered) until a difftest row exists -- the k-chain claim is tile-independent by the kernel body, but it is
    # CHECKED, not assumed (registry.moe header).
    "tiles": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
    "num_warps": 4, "num_stages": 3,
}

#: `topk_softmax` fused kernel geometry for bf16 gating logits: 16-byte loads -> 8 elements per thread-vector; E/VPT threads per row.
#: Difftested for E in {64, 128} (`moe_kernel_xcheck.py`); other E refuse (`numerics_unregistered`) until an xcheck row exists.
TOPK_SOFTMAX_VPT = {torch.bfloat16: 8, torch.float16: 8, torch.float32: 4}
TOPK_SOFTMAX_E_TESTED = frozenset({64, 128})


class MoeSchedule:
    """The result of `moe_align_block_size`: the router ids node it sorted plus the launch geometry it padded to.  Not a value."""

    def __init__(self, ids_node, E: int, block_m: int, num_valid: int) -> None:
        self.ids_node, self.E, self.block_m, self.num_valid = ids_node, E, block_m, num_valid

    def __repr__(self) -> str:
        return f"MoeSchedule(E={self.E}, BLOCK_M={self.block_m}, pairs={self.num_valid})"


class DeadOutput:
    """An output buffer the block writes and nobody reads (`token_expert_indices`, the align kernel's counters): reading it refuses."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __repr__(self) -> str:
        return f"DeadOutput({self.what})"


def _moe_C_op(name: str) -> set:
    try:
        import vllm  # noqa: F401
        return {getattr(torch.ops._moe_C, name).default}
    except Exception:  # noqa: BLE001 — vLLM not importable: no targets
        return set()


def _val_matches(v, node) -> bool:
    return isinstance(v, torch.fx.Node) and v is node


# -- the declared construction: which family owns the block ------------------------------------------------------------------

INDEXED_READ, INDEXED_READ_ORDERED = "indexed-read", "indexed-read-ordered"   # MOE-01 with the router in selection order
PADDED = "padded"


def active_moe_construction() -> str:
    """The construction the ACTIVE TargetProfile declares (`TargetProfile.DEFAULT_MOE_CONSTRUCTION` when undeclared); read at `applies` time, never cached."""
    return active_target_profile().effective_moe_construction()


def _not_owned(family: str) -> str | None:
    """A refusal string when the active construction is not `family`'s, else None: the two families are exclusive BY DECLARATION."""
    c = active_moe_construction()
    if c != family and (c, family) != (INDEXED_READ_ORDERED, INDEXED_READ):
        return f"moe_construction={c!r} (declared TargetProfile): the {c} rules own the fused-MoE block; this rule is the {family} family"
    return None


def _pinned_tile_refusal(sc, geom) -> str | None:
    for k, v in MOE_GEMM_PINS["tiles"].items():
        if sc.get(k) != v:
            return f"numerics_unregistered: tiling {k}={sc.get(k)!r} is not the difftested launch geometry ({MOE_GEMM_PINS['tiles']})"
    if geom.get("num_warps", sc.get("num_warps")) != MOE_GEMM_PINS["num_warps"] or geom.get("num_stages", sc.get("num_stages")) != MOE_GEMM_PINS["num_stages"]:
        return f"numerics_unregistered: num_warps/num_stages {geom.get('num_warps')}/{geom.get('num_stages')} are not the difftested {MOE_GEMM_PINS['num_warps']}/{MOE_GEMM_PINS['num_stages']}"
    return None


def _launch_pins(node, meta, sc) -> str | None:
    """`fused_moe_kernel` launch pins shared by both families (`MOE_GEMM_PINS`: kernel, source body, numeric constexprs, no bias / scale
    pointers, the difftested tiling and warps / stages, or a tuned tile `frontend.targets.moe_tuned_tile_refusal` accepts): a refusal
    string or None.  Operand shapes / schedule are the family's own."""
    if meta.get("kernel") != MOE_GEMM_PINS["kernel"]:
        return f"kernel {meta.get('kernel')} is not {MOE_GEMM_PINS['kernel']}"
    pin_sha = MOE_GEMM_PINS["src_sha256"]
    if meta.get("src_sha256") != pin_sha:
        return f"kernel source sha256 {meta.get('src_sha256')} differs from the pin {pin_sha} (same name, another kernel body)"
    for k, v in MOE_GEMM_PINS["constexprs"].items():
        if sc.get(k) != v:
            return f"numerics_unregistered: constexpr {k}={sc.get(k)!r} differs from the modelled path ({v!r})"
    for nm in MOE_GEMM_PINS["none_ptrs"]:
        if sc.get(nm) is not None or launch_tensor(node, meta, nm) is not None:
            return f"numerics_unregistered: {nm} is set (bias / scale epilogue is a different numeric)"
    if (r := _pinned_tile_refusal(sc, meta.get("geometry") or {})) is not None:
        from verity_vllm.program.frontend import targets as T
        if T.moe_tuned_tile_refusal(active_target_profile(), sc.get("BLOCK_SIZE_K")) is not None:
            return r
    bk = sc.get("BLOCK_SIZE_K")
    if not isinstance(bk, int) or bk % 16:
        return f"BLOCK_SIZE_K={bk!r} (need a multiple of 16)"
    return None


def _topk_softmax_checks(node, ctx) -> str | None:
    """The `_moe_C.topk_softmax` preconditions both families share (argument shapes / dtypes, the difftested E, no bias, no padding): a
    refusal string or None."""
    if len(node.args) < 5:
        return f"{len(node.args)} args"
    w, ids, tei, logits, renorm = node.args[:5]
    bias = node.args[5] if len(node.args) > 5 else None
    pad = node.args[6] if len(node.args) > 6 else None
    if renorm not in (False, True):
        return f"renormalize={renorm!r} is not a static bool"
    if bias is not None:
        return "numerics_unregistered: e_score_correction_bias is set (a different routing numeric)"
    if pad is not None:
        return "is_padding is set (padded rows route to -1: not modelled)"
    if not isinstance(logits, torch.fx.Node) or not isinstance(ctx.val(logits), Coll):
        return "logits is not a value collection"
    dt = ctx.dtype_of(logits)
    if dt not in TOPK_SOFTMAX_VPT:
        return f"numerics_unregistered: logits dtype {dt}"
    ls = shape(ctx, logits)
    if len(ls) != 2:
        return f"logits rank {len(ls)}"
    M, E = ls
    if E not in TOPK_SOFTMAX_E_TESTED:
        return f"numerics_unregistered: E={E} is not in the difftested set {sorted(TOPK_SOFTMAX_E_TESTED)} (MoeRouterTopK_v1 reduction geometry)"
    if E % TOPK_SOFTMAX_VPT[dt]:
        return f"E={E} is not a multiple of VPT={TOPK_SOFTMAX_VPT[dt]}"
    for nm, t, want in (("w", w, torch.float32), ("ids", ids, torch.int32), ("tei", tei, torch.int32)):
        if not isinstance(t, torch.fx.Node) or not isinstance(ctx.val(t), (Alloc, Coll)):
            return f"{nm} is not an allocation/value"
        if ctx.dtype_of(t) != want:
            return f"{nm} dtype {ctx.dtype_of(t)} is not {want}"
        ts = shape(ctx, t)
        if len(ts) != 2 or ts[0] != M:
            return f"{nm} shape {ts} vs logits [{M}, {E}]"
    if shape(ctx, w) != shape(ctx, ids):
        return f"w {shape(ctx, w)} vs ids {shape(ctx, ids)}"
    TOPK = shape(ctx, w)[1]
    if not 1 <= TOPK <= E:
        return f"TOPK={TOPK} vs E={E}"
    return None


class TopkSoftmaxRule(InPlaceRule):
    """`_moe_C.topk_softmax(w!, ids!, tei!, logits[M,E], renormalize, bias, is_padding)` -> `MoeRouterTopK{E,TOPK,VPT}` per token row."""
    modes = frozenset({"profile"})
    info = RuleInfo("TopkSoftmax[vllm-moe]", 1, "in-place-binding",
                    applies_to="_moe_C.topk_softmax(Tensor! w, Tensor! ids, Tensor! tei, logits, bool renormalize, Tensor? bias, Tensor? is_padding)",
                    preconditions=("logits is bf16 [M, E] with E in the difftested set (64, 128); renormalize a static bool (False -> MoeRouterTopK_v1, "
                                   "True -> MoeRouterTopKNorm_v1); bias None; is_padding None",
                                   "w is an unwritten allocation [M, TOPK] f32; ids [M, TOPK] i32; tei [M, TOPK] i32"),
                    evidence_domain="structural; numerics: MoeRouterTopK_v1 (vLLM _moe_C.topk_softmax fused kernel, cp-moe xcheck)")

    def targets(self):
        return _moe_C_op("topk_softmax")

    def applies(self, node, ctx):
        if (r := _not_owned(INDEXED_READ)):
            return r
        return _topk_softmax_checks(node, ctx) or True

    def emit(self, node, ctx):
        w, ids, tei, logits = node.args[:4]
        rows, lead = rows_of(ctx.coll(logits))
        M, E = shape(ctx, logits)
        TOPK = shape(ctx, w)[1]
        VPT = TOPK_SOFTMAX_VPT[ctx.dtype_of(logits)]
        # renormalize=True (Qwen3-MoE / Mixtral norm_topk_prob) is the same kernel with the lead thread's 1/sum epilogue:
        # MoeRouterTopKNorm_v1 (registry.moe); the ids are unchanged, the weights are the normalised probabilities.
        renorm = node.args[4]
        fn = ctx.kind("moe_router_topk" + ("_ordered" if active_moe_construction() == INDEXED_READ_ORDERED else "") + ("_norm" if renorm else ""), node, E=E, TOPK=TOPK, VPT=VPT)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        w_rows = [o[0] for o in outs]
        id_rows = [o[1] for o in outs]
        ctx.rebind(w, from_rows(w_rows, lead), self, what="topk_weights")
        ctx.rebind(ids, from_rows(id_rows, lead), self, what="topk_ids")
        ctx.rebind(tei, DeadOutput("token_expert_indices"), self, what="token_expert_indices")
        ctx.record(self, node, {"E": E, "TOPK": TOPK, "VPT": VPT, "rows": M, "renormalize": bool(renorm)}, emitted=f"{fn.id} x{M}")


class MoeAlignBlockSizeRule(InPlaceRule):
    """`_moe_C.moe_align_block_size(ids[M,TOPK], E, BLOCK_M, sorted_ids!, expert_ids!, num_post_padded!)` -> a `MoeSchedule` marker on
    `sorted_ids`; `expert_ids` and `num_post_padded` are dead outputs (the fused GEMM rule reads the schedule, not them)."""
    modes = frozenset({"profile"})
    info = RuleInfo("MoeAlignBlockSize[vllm-moe]", 1, "in-place-binding",
                    applies_to="_moe_C.moe_align_block_size(ids, int num_experts, int block_size, Tensor! sorted_ids, Tensor! expert_ids, Tensor! num_tokens_post_pad, ...)",
                    preconditions=("ids is an i32 [M, TOPK] value collection (the router's output)", "the three outputs are unwritten allocations"),
                    evidence_domain="structural only: a permutation for addressing; no numerics of its own")

    def targets(self):
        return _moe_C_op("moe_align_block_size")

    def applies(self, node, ctx):
        if (r := _not_owned(INDEXED_READ)):
            return r
        if len(node.args) < 6:
            return f"{len(node.args)} args"
        ids, E, bm, sorted_ids, expert_ids, npp = node.args[:6]
        if not isinstance(ids, torch.fx.Node) or not isinstance(ctx.val(ids), Coll):
            return "ids is not a value collection"
        if ctx.dtype_of(ids) != torch.int32:
            return f"ids dtype {ctx.dtype_of(ids)} is not int32"
        if len(shape(ctx, ids)) != 2:
            return f"ids shape {shape(ctx, ids)}"
        if not isinstance(E, int) or not isinstance(bm, int) or E <= 0 or bm <= 0:
            return f"num_experts={E!r} block_size={bm!r} are not static positive ints"
        for nm, t in (("sorted_ids", sorted_ids), ("expert_ids", expert_ids), ("num_tokens_post_pad", npp)):
            if not isinstance(t, torch.fx.Node) or not isinstance(ctx.val(t), Alloc):
                return f"{nm} is not an unwritten allocation"
        return True

    def emit(self, node, ctx):
        ids, E, bm, sorted_ids, expert_ids, npp = node.args[:6]
        M, TOPK = shape(ctx, ids)
        sched = MoeSchedule(ids, E, bm, M * TOPK)
        ctx.rebind(sorted_ids, sched, self, what="sorted_token_ids")
        ctx.rebind(expert_ids, DeadOutput("expert_ids (block schedule)"), self, what="expert_ids")
        ctx.rebind(npp, DeadOutput("num_tokens_post_padded"), self, what="num_tokens_post_padded")
        ctx.record(self, node, {"E": E, "BLOCK_M": bm, "pairs": M * TOPK}, emitted="schedule marker (no values)")


def _flat_pairs(c: Coll, n: int, elem) -> Coll:
    """A [M, TOPK] (or [M*TOPK]) collection as its `n` pair-ordered scalars."""
    return c.reshape(Array(n, elem))


class FusedMoeKernelRule(InPlaceRule):
    """`fused_moe_kernel` launch -> `MoeExpertGemm{E,K,N}` / `MoeExpertGemmW{E,K,N}` once per (token, slot) pair; `c` is rebound."""
    modes = frozenset({"profile"})
    info = RuleInfo("FusedMoeKernel[vllm-moe]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of fused_moe.fused_moe_kernel (kernel name + source + constexprs pinned)",
                    preconditions=("bf16 unquantized path: SPLIT_K 1, HAS_BIAS False, SWAP_AB False, USE_TD False, no fp8/int8/group quant, compute_type bf16",
                                   "a is bf16 [rows, K] row-major with rows*top_k == num_valid_tokens; b is bf16 [E, N, K] contiguous (stride_bk 1, stride_bn K, stride_be N*K)",
                                   "c is an unwritten allocation of num_valid_tokens rows of N; K % BLOCK_SIZE_K == 0, BLOCK_SIZE_K % 16 == 0",
                                   "launch geometry == the difftested one (BLOCK 64/64/32, GROUP_SIZE_M 8, 4 warps, 3 stages: vLLM's BI default for these stacks at every M)",
                                   "schedule: sorted_token_ids_ptr is a MoeSchedule marker over an i32 [M, TOPK] ids value, or naive_block_assignment with "
                                   "expert_ids_ptr the i32 pair ids value; topk_weights_ptr is the f32 [M, TOPK] routed weights value when MUL_ROUTED_WEIGHT"),
                    evidence_domain="structural; numerics: MoeExpertGemm_v1 / MoeExpertGemmW_v1 (the same Triton kernel; f32 accumulator, k16 chunks; cp-moe xcheck), _v2{DOT} where registry.targets.moe_expert_dot names a step")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != MOE_GEMM_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {MOE_GEMM_PINS['kernel']}"
        if (r := _not_owned(INDEXED_READ)) or (r := moe_expert_refusal()):
            return r
        sc = meta.get("scalars", {})
        if (r := _launch_pins(node, meta, sc)):
            return r
        mrw = sc.get("MUL_ROUTED_WEIGHT")
        if mrw not in (True, False):
            return f"MUL_ROUTED_WEIGHT={mrw!r}"
        top_k = sc.get("top_k")
        if not isinstance(top_k, int) or top_k < 1:
            return f"top_k={top_k!r}"
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        if a is None or b is None or c is None:
            return "a_ptr/b_ptr/c_ptr not all tensors"
        for nm, t in (("a", a), ("b", b)):
            if not isinstance(ctx.val(t), Coll):
                return f"{nm} is not a value collection ({type(ctx.val(t)).__name__})"
            if not dtype_is(ctx, t, torch.bfloat16):
                return f"numerics_unregistered: {nm} dtype {ctx.dtype_of(t)}"
        if not isinstance(ctx.val(c), (Alloc, Coll)):
            return "c is not an allocation/value"
        if ctx.dtype_of(c) != torch.bfloat16:
            return f"numerics_unregistered: c dtype {ctx.dtype_of(c)}"
        as_, bs, cs = shape(ctx, a), shape(ctx, b), shape(ctx, c)
        if len(as_) != 2 or len(bs) != 3 or len(cs) < 2:
            return f"ranks a{as_} b{bs} c{cs}"
        rows, K = as_
        E, N, K2 = bs
        nv = sc.get("num_valid_tokens")
        if K != K2 or (sc.get("N"), sc.get("K")) != (N, K):
            return f"shapes a{as_} b{bs} vs scalars N={sc.get('N')} K={sc.get('K')}"
        if rows * top_k != nv:
            return f"a rows {rows} x top_k {top_k} != num_valid_tokens {nv}"
        if math.prod(cs[:-1]) != nv or cs[-1] != N:
            return f"c {cs} is not {nv} rows of N={N}"
        if (sc.get("stride_am"), sc.get("stride_ak")) != (K, 1):
            return f"a strides {sc.get('stride_am')},{sc.get('stride_ak')} are not row-major [rows, K]"
        if (sc.get("stride_be"), sc.get("stride_bk"), sc.get("stride_bn")) != (N * K, 1, K):
            return f"b strides {sc.get('stride_be')},{sc.get('stride_bk')},{sc.get('stride_bn')} are not a contiguous [E, N, K] stack"
        if (sc.get("stride_cm"), sc.get("stride_cn")) != (N, 1):
            return f"c strides {sc.get('stride_cm')},{sc.get('stride_cn')} are not row-major rows of N"
        bk = sc.get("BLOCK_SIZE_K")
        if K % bk:
            return f"BLOCK_SIZE_K={bk!r} (need a multiple of 16 dividing K={K})"
        return self._schedule(node, ctx, meta, sc, nv, mrw) or True

    def _schedule(self, node, ctx, meta, sc, nv, mrw):
        """Where the pair -> (expert, weight) operands come from; a refusal string or None."""
        ids_t = launch_tensor(node, meta, "expert_ids_ptr")
        sorted_t = launch_tensor(node, meta, "sorted_token_ids_ptr")
        if sc.get("naive_block_assignment") is True:
            if sorted_t is not None:
                return "naive_block_assignment with a sorted_token_ids_ptr"
            if ids_t is None or not isinstance(ctx.val(ids_t), Coll) or ctx.dtype_of(ids_t) != torch.int32:
                return "naive schedule: expert_ids_ptr is not an i32 value collection"
            if math.prod(shape(ctx, ids_t)) != nv:
                return f"naive schedule: expert_ids {shape(ctx, ids_t)} is not {nv} pairs"
        else:
            if sorted_t is None or not isinstance(ctx.val(sorted_t), MoeSchedule):
                return f"sorted_token_ids_ptr is not a MoeSchedule marker ({type(ctx.val(sorted_t)).__name__ if sorted_t is not None else None})"
            sched = ctx.val(sorted_t)
            if sched.num_valid != nv or sched.block_m != sc.get("BLOCK_SIZE_M"):
                return f"schedule {sched} vs launch num_valid_tokens={nv} BLOCK_SIZE_M={sc.get('BLOCK_SIZE_M')}"
            if not isinstance(ctx.val(sched.ids_node), Coll):
                return "the schedule's ids are not a value collection"
        if mrw:
            w_t = launch_tensor(node, meta, "topk_weights_ptr")
            if w_t is None or not isinstance(ctx.val(w_t), Coll) or ctx.dtype_of(w_t) != torch.float32:
                return "MUL_ROUTED_WEIGHT: topk_weights_ptr is not an f32 value collection"
            if math.prod(shape(ctx, w_t)) != nv:
                return f"topk_weights {shape(ctx, w_t)} is not {nv} pairs"
        return None

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc = meta["scalars"]
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        mrw, top_k, nv = sc["MUL_ROUTED_WEIGHT"], sc["top_k"], sc["num_valid_tokens"]
        E, N, K = shape(ctx, b)
        a_rows, _ = rows_of(ctx.coll(a))
        bc = ctx.coll(b)
        if sc.get("naive_block_assignment") is True:
            ids_node = launch_tensor(node, meta, "expert_ids_ptr")
        else:
            ids_node = ctx.val(launch_tensor(node, meta, "sorted_token_ids_ptr")).ids_node
        from verity.ml.definitions.registry.prims import F32, I32
        ids = _flat_pairs(ctx.coll(ids_node), nv, I32)
        ws = _flat_pairs(ctx.coll(launch_tensor(node, meta, "topk_weights_ptr")), nv, F32) if mrw else None
        role, dot = moe_expert_role(mrw)
        fn = ctx.kind(role, node, E=E, K=K, N=N, **dot)
        hint = ctx.hint_name(node)
        outs = []
        for p in range(nv):
            x = a_rows[p // top_k]
            args = (x, ids[p]) + ((ws[p],) if mrw else ()) + (bc,)
            outs.append(ctx.call(fn, *args, node=node, rule=self, name=f"{hint}[{p}]"))
        cs = shape(ctx, c)
        ctx.rebind(c, from_rows(outs, tuple(cs[:-1])), self, what="c")
        ctx.bind_weight_role(b, self, "moe.expert_weight", E=E, K=K, N=N)
        wl = ctx.weight_of(b)
        ctx.record(self, node, {"E": E, "K": K, "N": N, "pairs": nv, "top_k": top_k, "MUL_ROUTED_WEIGHT": mrw,
                                "schedule": "naive" if sc.get("naive_block_assignment") is True else "sorted",
                                "tiles": {k: sc.get(k) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")},
                                "num_warps": (meta.get("geometry") or {}).get("num_warps"), "src_sha256": meta.get("src_sha256"),
                                "W_is_parameter": wl.name if wl else None}, emitted=f"{fn.id} x{nv}")


class MoeSumRule(InPlaceRule):
    """`_moe_C.moe_sum(y [M, TOPK, H] bf16, out! [M, H])` -> `MoeSum{TOPK,H}` per token."""
    modes = frozenset({"profile"})
    info = RuleInfo("MoeSum[vllm-moe]", 1, "in-place-binding", applies_to="_moe_C.moe_sum(Tensor input, Tensor! output)",
                    preconditions=("input is bf16 [M, TOPK, H]", "output is an unwritten allocation [M, H] bf16"),
                    evidence_domain="structural; numerics: MoeSum_v1 (vLLM _moe_C.moe_sum; cp-moe xcheck)")

    def targets(self):
        return _moe_C_op("moe_sum")

    def applies(self, node, ctx):
        if (r := _not_owned(INDEXED_READ)):
            return r
        if len(node.args) < 2:
            return f"{len(node.args)} args"
        y, out = node.args[:2]
        if not isinstance(y, torch.fx.Node) or not isinstance(ctx.val(y), Coll):
            return "input is not a value collection"
        if not dtype_is(ctx, y, torch.bfloat16):
            return f"numerics_unregistered: input dtype {ctx.dtype_of(y)}"
        ys = shape(ctx, y)
        if len(ys) != 3:
            return f"input shape {ys} is not [M, TOPK, H]"
        if not isinstance(out, torch.fx.Node) or not isinstance(ctx.val(out), (Alloc, Coll)):
            return "output is not an allocation/value"
        if ctx.dtype_of(out) != torch.bfloat16 or shape(ctx, out) != (ys[0], ys[2]):
            return f"output {shape(ctx, out)} {ctx.dtype_of(out)} vs input {ys}"
        return True

    def emit(self, node, ctx):
        y, out = node.args[:2]
        M, TOPK, H = shape(ctx, y)
        from verity.ml.definitions.registry.prims import BF16
        yc = ctx.coll(y).reshape(Array(M, Array(TOPK, Array(H, BF16))))
        fn = ctx.kind("moe_sum", node, TOPK=TOPK, H=H)
        outs = [ctx.call(fn, yc[t], node=node, rule=self, name=f"{ctx.hint_name(node)}[{t}]") for t in range(M)]
        ctx.rebind(out, from_rows(outs, (M,)), self, what="output")
        ctx.record(self, node, {"TOPK": TOPK, "H": H, "rows": M}, emitted=f"{fn.id} x{M}")


# ---------------------------------------------------------------------------------------------------------------------------------
# the PADDED family: the same six launches, ONE `MoeFfnFixedL_v1` per token, the interior as markers (module doc)
# ---------------------------------------------------------------------------------------------------------------------------------


class PaddedRoute:
    """`_moe_C.topk_softmax` under the padded construction: NOT the router's values -- the logits rows it was computed from and the
    geometry.  The block's Definition recomputes the route inside `MoeFfnFixedL_v1`; reading `topk_weights` / `topk_ids` as a value refuses."""

    def __init__(self, logits_node, rows: list, M: int, E: int, TOPK: int, VPT: int) -> None:
        self.logits_node, self.rows, self.M, self.E, self.TOPK, self.VPT = logits_node, rows, M, E, TOPK, VPT

    def __repr__(self) -> str:
        return f"PaddedRoute(M={self.M}, E={self.E}, TOPK={self.TOPK}, VPT={self.VPT}; interior of MoeFfnFixedL_v1, not a value)"


class PaddedStage:
    """An intermediate buffer of the padded block (`topk_weights`, `topk_ids`, the pair caches): the stage name and what it derives from."""

    def __init__(self, what: str, of) -> None:
        self.what, self.of = what, of

    def __repr__(self) -> str:
        return f"PaddedStage({self.what} of {self.of!r})"


class PaddedSchedule:
    """`moe_align_block_size` under the padded construction: the schedule marker over a `PaddedRoute` (N-pad-09: its shapes never enter the statics)."""

    def __init__(self, route: PaddedRoute, block_m: int, num_valid: int) -> None:
        self.route, self.block_m, self.num_valid = route, block_m, num_valid

    def __repr__(self) -> str:
        return f"PaddedSchedule(BLOCK_M={self.block_m}, pairs={self.num_valid} over {self.route!r})"


class PaddedUp:
    """The up/gate `fused_moe_kernel` (MUL_ROUTED_WEIGHT=False) under the padded construction: the hidden rows and W13 it read."""

    def __init__(self, route: PaddedRoute, x_rows: list, w13_node, w13: Coll, H: int, I: int, schedule: str) -> None:
        self.route, self.x_rows, self.w13_node, self.w13, self.H, self.I, self.schedule = route, x_rows, w13_node, w13, H, I, schedule

    def __repr__(self) -> str:
        return f"PaddedUp(H={self.H}, I={self.I}, {self.schedule} schedule, {self.route!r})"


class PaddedAct:
    """`_C.silu_and_mul` on the padded block's pair rows: the activation of a `PaddedUp`."""

    def __init__(self, up: PaddedUp) -> None:
        self.up = up

    def __repr__(self) -> str:
        return f"PaddedAct({self.up!r})"


class PaddedDown:
    """The down `fused_moe_kernel` (MUL_ROUTED_WEIGHT=True) under the padded construction: the activation and W2 it read."""

    def __init__(self, act: PaddedAct, w2_node, w2: Coll) -> None:
        self.act, self.w2_node, self.w2 = act, w2_node, w2

    def __repr__(self) -> str:
        return f"PaddedDown({self.act!r})"


class PaddedTopkSoftmaxRule(InPlaceRule):
    """`_moe_C.topk_softmax(...)` under `moe_construction = padded`: the route is the Definition's interior -- `w` / `ids` become
    `PaddedStage` markers over a `PaddedRoute` (the logits rows), `tei` is dead.  `renormalize=True` (Qwen3-MoE / Mixtral) has no fixed-slot
    Definition (`MoeFfnFixedL_v1` routes with `MoeRouterTopK_v1`): refused by name, those rows stay indexed-read."""
    modes = frozenset({"profile"})
    info = RuleInfo("PaddedTopkSoftmax[vllm-moe]", 1, "in-place-binding",
                    applies_to="_moe_C.topk_softmax(Tensor! w, Tensor! ids, Tensor! tei, logits, bool renormalize, Tensor? bias, Tensor? is_padding) "
                               "when the active TargetProfile declares moe_construction = padded",
                    preconditions=("TopkSoftmax[vllm-moe]'s shape / dtype / E preconditions", "renormalize False (MoeRouterTopK_v1 is the twin's router)"),
                    evidence_domain="structural; numerics: inside MoeFfnFixedL_v1 (MoeRouterTopK_v1, cp-moe xcheck)")

    def targets(self):
        return _moe_C_op("topk_softmax")

    def applies(self, node, ctx):
        if (r := _not_owned(PADDED)):
            return r
        if (r := _topk_softmax_checks(node, ctx)):
            return r
        if node.args[4]:
            return ("numerics_unregistered: renormalize=True has no fixed-slot Definition (MoeFfnFixedL_v1 routes with MoeRouterTopK_v1; "
                    "norm_topk_prob rows are indexed-read only)")
        return True

    def emit(self, node, ctx):
        w, ids, tei, logits = node.args[:4]
        rows, _lead = rows_of(ctx.coll(logits))
        M, E = shape(ctx, logits)
        TOPK = shape(ctx, w)[1]
        VPT = TOPK_SOFTMAX_VPT[ctx.dtype_of(logits)]
        route = PaddedRoute(logits, rows, M, E, TOPK, VPT)
        ctx.rebind(w, PaddedStage("topk_weights", route), self, what="topk_weights")
        ctx.rebind(ids, PaddedStage("topk_ids", route), self, what="topk_ids")
        ctx.rebind(tei, DeadOutput("token_expert_indices"), self, what="token_expert_indices")
        ctx.record(self, node, {"E": E, "TOPK": TOPK, "VPT": VPT, "rows": M, "renormalize": False, "moe_construction": PADDED},
                   emitted="PaddedRoute marker (the route is computed inside MoeFfnFixedL_v1 at moe_sum)")


class PaddedMoeAlignBlockSizeRule(InPlaceRule):
    """`_moe_C.moe_align_block_size(ids, E, BLOCK_M, sorted_ids!, expert_ids!, num_post_padded!)` under the padded construction: a
    `PaddedSchedule` marker over the route; the counters are dead."""
    modes = frozenset({"profile"})
    info = RuleInfo("PaddedMoeAlignBlockSize[vllm-moe]", 1, "in-place-binding",
                    applies_to="_moe_C.moe_align_block_size(ids, int num_experts, int block_size, Tensor! sorted_ids, Tensor! expert_ids, Tensor! num_tokens_post_pad, ...) "
                               "when moe_construction = padded",
                    preconditions=("ids is the PaddedRoute's topk_ids marker", "num_experts == the route's E", "the three outputs are unwritten allocations"),
                    evidence_domain="structural only: a permutation for addressing; no numerics of its own (N-pad-09: never in the statics)")

    def targets(self):
        return _moe_C_op("moe_align_block_size")

    def applies(self, node, ctx):
        if (r := _not_owned(PADDED)):
            return r
        if len(node.args) < 6:
            return f"{len(node.args)} args"
        ids, E, bm, sorted_ids, expert_ids, npp = node.args[:6]
        v = ctx.val(ids) if isinstance(ids, torch.fx.Node) else None
        if not (isinstance(v, PaddedStage) and v.what == "topk_ids"):
            return f"ids is not the padded route's topk_ids marker ({type(v).__name__})"
        if not isinstance(E, int) or not isinstance(bm, int) or E <= 0 or bm <= 0:
            return f"num_experts={E!r} block_size={bm!r} are not static positive ints"
        if E != v.of.E:
            return f"num_experts {E} is not the route's E {v.of.E}"
        for nm, t in (("sorted_ids", sorted_ids), ("expert_ids", expert_ids), ("num_tokens_post_pad", npp)):
            if not isinstance(t, torch.fx.Node) or not isinstance(ctx.val(t), Alloc):
                return f"{nm} is not an unwritten allocation"
        return True

    def emit(self, node, ctx):
        ids, E, bm, sorted_ids, expert_ids, npp = node.args[:6]
        route = ctx.val(ids).of
        ctx.rebind(sorted_ids, PaddedSchedule(route, bm, route.M * route.TOPK), self, what="sorted_token_ids")
        ctx.rebind(expert_ids, DeadOutput("expert_ids (block schedule)"), self, what="expert_ids")
        ctx.rebind(npp, DeadOutput("num_tokens_post_padded"), self, what="num_tokens_post_padded")
        ctx.record(self, node, {"E": E, "BLOCK_M": bm, "pairs": route.M * route.TOPK, "moe_construction": PADDED}, emitted="schedule marker (no values)")


class PaddedFusedMoeKernelRule(InPlaceRule):
    """`fused_moe_kernel` under the padded construction: the SAME pins as `FusedMoeKernel[vllm-moe]` (`_launch_pins`), the operands read
    through the markers -- up (MUL_ROUTED_WEIGHT=False, top_k=TOPK): `a` = the hidden rows [M,H] (a value), `b` = W13 [E,2I,H], the schedule
    over the route -> `c` := `PaddedUp`; down (MUL_ROUTED_WEIGHT=True, top_k=1): `a` = the `PaddedAct` of that up, `topk_weights_ptr` the
    route's weights marker, `b` = W2 [E,H,I] -> `c` := `PaddedDown`.  Both schedules (sorted / naive) accepted, as for MOE-01."""
    modes = frozenset({"profile"})
    info = RuleInfo("PaddedFusedMoeKernel[vllm-moe]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of fused_moe.fused_moe_kernel when moe_construction = padded (kernel name + source + constexprs pinned)",
                    preconditions=("FusedMoeKernel[vllm-moe]'s launch pins (bf16 unquantized path, difftested tiling)",
                                   "up: a is the bf16 [M,H] hidden value (M*top_k == num_valid_tokens, top_k == the route's TOPK); b bf16 [E,2I,H] contiguous; "
                                   "the schedule is the route's PaddedSchedule (sorted) or the route's topk_ids marker (naive)",
                                   "down: a is the PaddedAct of the up launch ([M*TOPK, I]); topk_weights_ptr the route's topk_weights marker; "
                                   "b bf16 [E,H,I] contiguous; top_k == 1",
                                   "c is an unwritten allocation of num_valid_tokens rows of N"),
                    evidence_domain="structural; numerics: inside MoeFfnFixedL_v1 (MoeExpertBody_v1 = the pair body without the expert-axis gather)")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != MOE_GEMM_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {MOE_GEMM_PINS['kernel']}"
        if (r := _not_owned(PADDED)) or (r := moe_expert_refusal(padded=True)):
            return r
        sc = meta.get("scalars", {})
        if (r := _launch_pins(node, meta, sc)):
            return r
        mrw = sc.get("MUL_ROUTED_WEIGHT")
        if mrw not in (True, False):
            return f"MUL_ROUTED_WEIGHT={mrw!r}"
        top_k = sc.get("top_k")
        if not isinstance(top_k, int) or top_k < 1:
            return f"top_k={top_k!r}"
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        if a is None or b is None or c is None:
            return "a_ptr/b_ptr/c_ptr not all tensors"
        if not isinstance(ctx.val(b), Coll) or not dtype_is(ctx, b, torch.bfloat16):
            return f"numerics_unregistered: b is not a bf16 value collection ({ctx.dtype_of(b)})"
        if not isinstance(ctx.val(c), (Alloc, Coll)) or ctx.dtype_of(c) != torch.bfloat16:
            return f"c is not a bf16 allocation/value ({ctx.dtype_of(c)})"
        bs, cs = shape(ctx, b), shape(ctx, c)
        if len(bs) != 3 or len(cs) < 2:
            return f"ranks b{bs} c{cs}"
        E, N, K = bs
        nv = sc.get("num_valid_tokens")
        if (sc.get("N"), sc.get("K")) != (N, K):
            return f"b{bs} vs scalars N={sc.get('N')} K={sc.get('K')}"
        if math.prod(cs[:-1]) != nv or cs[-1] != N:
            return f"c {cs} is not {nv} rows of N={N}"
        if (sc.get("stride_am"), sc.get("stride_ak")) != (K, 1):
            return f"a strides {sc.get('stride_am')},{sc.get('stride_ak')} are not row-major [rows, K]"
        if (sc.get("stride_be"), sc.get("stride_bk"), sc.get("stride_bn")) != (N * K, 1, K):
            return f"b strides are not a contiguous [E, N, K] stack"
        if (sc.get("stride_cm"), sc.get("stride_cn")) != (N, 1):
            return f"c strides are not row-major rows of N"
        if K % sc.get("BLOCK_SIZE_K"):
            return f"BLOCK_SIZE_K={sc.get('BLOCK_SIZE_K')!r} does not divide K={K}"
        route = self._route(node, ctx, meta, sc, nv)
        if isinstance(route, str):
            return route
        if E != route.E:
            return f"b's expert axis {E} is not the route's E {route.E}"
        if not mrw:                                                       # the up/gate launch
            if top_k != route.TOPK:
                return f"up launch top_k {top_k} != the route's TOPK {route.TOPK}"
            if not isinstance(ctx.val(a), Coll) or not dtype_is(ctx, a, torch.bfloat16):
                return f"up launch: a is not a bf16 value collection ({type(ctx.val(a)).__name__})"
            as_ = shape(ctx, a)
            if len(as_) != 2 or as_ != (route.M, K):
                return f"up launch: a{as_} is not the route's [M={route.M}, H={K}] hidden rows"
            if N % 2:
                return f"up launch: N={N} is not 2I"
            return True
        # the down launch
        if top_k != 1:
            return f"down launch top_k {top_k} != 1"
        act = ctx.val(a)
        if not isinstance(act, PaddedAct):
            return f"down launch: a is not the padded activation marker ({type(act).__name__})"
        if act.up.route is not route:
            return "down launch: the activation and the schedule belong to different routes"
        if K != act.up.I or N != act.up.H:
            return f"down launch: K={K} N={N} vs the up launch's I={act.up.I} H={act.up.H}"
        w_t = launch_tensor(node, meta, "topk_weights_ptr")
        wv = ctx.val(w_t) if w_t is not None else None
        if not (isinstance(wv, PaddedStage) and wv.what == "topk_weights" and wv.of is route):
            return f"down launch: topk_weights_ptr is not the route's topk_weights marker ({type(wv).__name__})"
        return True

    def _route(self, node, ctx, meta, sc, nv):
        """The `PaddedRoute` the launch's schedule addresses (sorted: a PaddedSchedule marker; naive: the topk_ids marker), or a refusal."""
        ids_t = launch_tensor(node, meta, "expert_ids_ptr")
        sorted_t = launch_tensor(node, meta, "sorted_token_ids_ptr")
        if sc.get("naive_block_assignment") is True:
            if sorted_t is not None:
                return "naive_block_assignment with a sorted_token_ids_ptr"
            v = ctx.val(ids_t) if ids_t is not None else None
            if not (isinstance(v, PaddedStage) and v.what == "topk_ids"):
                return f"naive schedule: expert_ids_ptr is not the route's topk_ids marker ({type(v).__name__})"
            route = v.of
        else:
            v = ctx.val(sorted_t) if sorted_t is not None else None
            if not isinstance(v, PaddedSchedule):
                return f"sorted_token_ids_ptr is not a PaddedSchedule marker ({type(v).__name__})"
            if v.num_valid != nv or v.block_m != sc.get("BLOCK_SIZE_M"):
                return f"schedule {v} vs launch num_valid_tokens={nv} BLOCK_SIZE_M={sc.get('BLOCK_SIZE_M')}"
            route = v.route
        if route.M * route.TOPK != nv:
            return f"route pairs {route.M}x{route.TOPK} != num_valid_tokens {nv}"
        return route

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc = meta["scalars"]
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        mrw, nv = sc["MUL_ROUTED_WEIGHT"], sc["num_valid_tokens"]
        E, N, K = shape(ctx, b)
        route = self._route(node, ctx, meta, sc, nv)
        sched = "naive" if sc.get("naive_block_assignment") is True else "sorted"
        ctx.bind_weight_role(b, self, "moe.expert_weight", E=E, K=K, N=N, construction=PADDED)
        wl = ctx.weight_of(b)
        if not mrw:
            x_rows, _lead = rows_of(ctx.coll(a))
            ctx.rebind(c, PaddedUp(route, x_rows, b, ctx.coll(b), H=K, I=N // 2, schedule=sched), self, what="c")
            emitted = "PaddedUp marker (W13 bound; the M calls are emitted at moe_sum)"
        else:
            ctx.rebind(c, PaddedDown(ctx.val(a), b, ctx.coll(b)), self, what="c")
            emitted = "PaddedDown marker (W2 bound; the M calls are emitted at moe_sum)"
        ctx.record(self, node, {"E": E, "K": K, "N": N, "pairs": nv, "top_k": sc["top_k"], "MUL_ROUTED_WEIGHT": mrw, "schedule": sched,
                                "tiles": {k: sc.get(k) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")},
                                "num_warps": (meta.get("geometry") or {}).get("num_warps"), "src_sha256": meta.get("src_sha256"),
                                "W_is_parameter": wl.name if wl else None, "moe_construction": PADDED}, emitted=emitted)


class PaddedViewRule(ViewRule):
    """A reshape (`aten.view / reshape / _unsafe_view / flatten / squeeze / unsqueeze`) of a padded-block MARKER (`PaddedUp` / `PaddedAct` /
    `PaddedDown` / `PaddedStage`): vLLM's `fused_experts` views its pair caches between the kernels (`intermediate_cache1.view(-1, N)` into
    `silu_and_mul`, `intermediate_cache3.view(*shape)` into `moe_sum`).  The marker is not a value collection (the profile's `Reshape` declines
    it: "operand is PaddedUp"), so a view of it is the SAME marker under the new torch shape -- nothing is read, nothing is emitted; the shape
    the consumer rule checks is torch's for the view node (`common.shape`), as it is for a collection.  Only under the padded construction."""
    modes = frozenset({"profile"})
    info = RuleInfo("PaddedView[vllm-moe]", 1, "view", applies_to="aten.view/reshape/_unsafe_view/flatten/squeeze/unsqueeze of a padded-block marker",
                    preconditions=("moe_construction = padded", "operand is a PaddedUp / PaddedAct / PaddedDown / PaddedStage marker",
                                   "element count is preserved (torch's own view check)"),
                    evidence_domain="structural (no gates): a marker's alias is the marker")

    def targets(self):
        from verity_vllm.program.frontend.rules.views import _RESHAPE
        return _RESHAPE

    def applies(self, node, ctx):
        if (r := _not_owned(PADDED)):
            return r
        v = ctx.val(node.args[0]) if node.args and isinstance(node.args[0], torch.fx.Node) else None
        if not isinstance(v, (PaddedUp, PaddedAct, PaddedDown, PaddedStage)):
            return f"operand is {type(v).__name__}, not a padded-block marker"
        import math
        a, b = shape(ctx, node.args[0]), shape(ctx, node)
        if a is None or b is None or math.prod(a) != math.prod(b):
            return f"view {a} -> {b} does not preserve the element count"
        return True

    def emit(self, node, ctx):
        ctx.define(node, ctx.val(node.args[0]), view_of=node.args[0])
        ctx.record(self, node, {"marker": type(ctx.val(node)).__name__, "from": list(shape(ctx, node.args[0])), "to": list(shape(ctx, node))},
                   emitted="(view of a padded-block marker; no gates)")


class PaddedSiluAndMulRule(InPlaceRule):
    """`_C.silu_and_mul(out!, x)` on the padded block's pair rows (`x` = a `PaddedUp` marker): `out` := `PaddedAct`.  The profile's
    `SiluAndMul[vllm]` declines the marker by itself ("x is not a value collection"); this rule owns it under the padded construction."""
    modes = frozenset({"profile"})
    info = RuleInfo("PaddedSiluAndMul[vllm-moe]", 1, "in-place-binding", applies_to="_C.silu_and_mul(Tensor! out, x) with x the padded up-GEMM marker",
                    preconditions=("x is the PaddedUp marker of this block", "out is an unwritten bf16 allocation [M*TOPK, I]"),
                    evidence_domain="structural; numerics: inside MoeFfnFixedL_v1 (SiluMul_v1{I} of MoeExpertBody_v1)")

    def targets(self):
        from verity_vllm.program.frontend.rules.vllm_iface import targets_for
        return targets_for("silu_and_mul")

    def applies(self, node, ctx):
        if (r := _not_owned(PADDED)):
            return r
        if len(node.args) < 2:
            return f"{len(node.args)} args"
        out, x = node.args[:2]
        up = ctx.val(x) if isinstance(x, torch.fx.Node) else None
        if not isinstance(up, PaddedUp):
            return f"x is not the padded up-GEMM marker ({type(up).__name__})"
        if not isinstance(out, torch.fx.Node) or not isinstance(ctx.val(out), (Alloc, Coll)):
            return "out is not an allocation/value"
        if ctx.dtype_of(out) != torch.bfloat16:
            return f"numerics_unregistered: out dtype {ctx.dtype_of(out)}"
        xs, os_ = shape(ctx, x), shape(ctx, out)
        if xs != (up.route.M * up.route.TOPK, 2 * up.I) or os_ != (up.route.M * up.route.TOPK, up.I):
            return f"x {xs} / out {os_} are not the block's [M*TOPK, 2I] / [M*TOPK, I] pair rows (M={up.route.M}, TOPK={up.route.TOPK}, I={up.I})"
        return True

    def emit(self, node, ctx):
        out, x = node.args[:2]
        up = ctx.val(x)
        ctx.rebind(out, PaddedAct(up), self, what="out")
        ctx.record(self, node, {"I": up.I, "rows": up.route.M * up.route.TOPK, "moe_construction": PADDED}, emitted="PaddedAct marker")


class PaddedMoeSumRule(InPlaceRule):
    """`_moe_C.moe_sum(y [M,TOPK,H], out! [M,H])` under the padded construction -- the block's LAST kernel, where every operand of the token's
    Definition is known: EMITS `MoeFfnFixedL_v1{E,TOPK,VPT,H,I}(x_t, logits_t, W13, W2)` for each of the M tokens and rebinds `out` to
    their rows.  The route (ids, weights), the pair caches and the activation are the Definition's interior; the ONLY values the block
    reads are the hidden rows, the router logits and the two weight stacks, the only value it produces the M output rows."""
    modes = frozenset({"profile"})
    info = RuleInfo("PaddedMoeSum[vllm-moe]", 1, "in-place-binding", applies_to="_moe_C.moe_sum(Tensor input, Tensor! output) with input the padded down-GEMM marker",
                    preconditions=("input is the PaddedDown marker of this block ([M, TOPK, H] pair rows)", "output is an unwritten bf16 allocation [M, H]"),
                    evidence_domain="structural; numerics: MoeFfnFixedL_v1 (registry.moe_pad: MoeRouterTopK_v1, MoeExpertBody_v1 x E, GatherBf16 select, MoeSum_v1; "
                                    "value-equal to the indexed-read reading of the same launches; lift of record LMoeFfnL_v1)")

    def targets(self):
        return _moe_C_op("moe_sum")

    def applies(self, node, ctx):
        if (r := _not_owned(PADDED)):
            return r
        if len(node.args) < 2:
            return f"{len(node.args)} args"
        y, out = node.args[:2]
        dn = ctx.val(y) if isinstance(y, torch.fx.Node) else None
        if not isinstance(dn, PaddedDown):
            return f"input is not the padded down-GEMM marker ({type(dn).__name__})"
        up = dn.act.up
        ys = shape(ctx, y)
        if ys != (up.route.M, up.route.TOPK, up.H):
            return f"input shape {ys} is not the block's [M={up.route.M}, TOPK={up.route.TOPK}, H={up.H}]"
        if not isinstance(out, torch.fx.Node) or not isinstance(ctx.val(out), (Alloc, Coll)):
            return "output is not an allocation/value"
        if ctx.dtype_of(out) != torch.bfloat16 or shape(ctx, out) != (up.route.M, up.H):
            return f"output {shape(ctx, out)} {ctx.dtype_of(out)} vs [M={up.route.M}, H={up.H}] bf16"
        return True

    def emit(self, node, ctx):
        y, out = node.args[:2]
        dn = ctx.val(y)
        up, route = dn.act.up, dn.act.up.route
        M, E, TOPK, VPT, H, I = route.M, route.E, route.TOPK, route.VPT, up.H, up.I
        fn = ctx.kind("moe_ffn_fixed", node, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)
        hint = ctx.hint_name(node)
        outs = [ctx.call(fn, up.x_rows[t], route.rows[t], up.w13, dn.w2, node=node, rule=self, name=f"{hint}[{t}]") for t in range(M)]
        ctx.rebind(out, from_rows(outs, (M,)), self, what="output")
        w13, w2 = ctx.weight_of(up.w13_node), ctx.weight_of(dn.w2_node)
        ctx.record(self, node, {"E": E, "TOPK": TOPK, "VPT": VPT, "H": H, "I": I, "rows": M, "schedule": up.schedule, "moe_construction": PADDED,
                                "W13_is_parameter": w13.name if w13 else None, "W2_is_parameter": w2.name if w2 else None,
                                "lift_of_record": "LMoeFfnL_v1 (hand-authored; registry.lifted.register_hand_lift)"},
                   emitted=f"{fn.id} x{M}")


#: both families in one list; each rule applies only under its own declared construction (`_not_owned`), so the block is bound by
#: exactly one family per Program.  The padded family is listed first: under `padded` the MOE-01 rules decline by name, under
#: `indexed-read` the padded rules do -- the order never decides.
VLLM_MOE_RULES = [PaddedTopkSoftmaxRule(), PaddedMoeAlignBlockSizeRule(), PaddedFusedMoeKernelRule(), PaddedViewRule(), PaddedSiluAndMulRule(),
                  PaddedMoeSumRule(),
                  TopkSoftmaxRule(), MoeAlignBlockSizeRule(), FusedMoeKernelRule(), MoeSumRule()]

__all__ = ["VLLM_MOE_RULES", "TopkSoftmaxRule", "MoeAlignBlockSizeRule", "FusedMoeKernelRule", "MoeSumRule", "MoeSchedule", "DeadOutput",
           "PaddedTopkSoftmaxRule", "PaddedMoeAlignBlockSizeRule", "PaddedFusedMoeKernelRule", "PaddedViewRule", "PaddedSiluAndMulRule", "PaddedMoeSumRule",
           "PaddedRoute", "PaddedStage", "PaddedSchedule", "PaddedUp", "PaddedAct", "PaddedDown", "active_moe_construction", "INDEXED_READ", "PADDED",
           "MOE_GEMM_PINS", "TOPK_SOFTMAX_VPT", "TOPK_SOFTMAX_E_TESTED"]
