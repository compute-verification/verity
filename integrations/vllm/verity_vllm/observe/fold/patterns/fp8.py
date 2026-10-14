"""[R15 fp8] Eager Match patterns of vLLM's BLOCK-SCALED FP8 linear (Hopper): the two custom ops `Fp8LinearMethod.apply` executes for a
pre-quantised checkpoint (`quantization_config`: fp8 / e4m3 / `weight_block_size [G, G]` / dynamic activations) when the kernel
selected is `CutlassFp8BlockScaledMMKernel` (`VLLM_USE_DEEP_GEMM=0`; the DeepGEMM JIT is not the panel's path):

    _C.per_token_group_fp8_quant(x[M,K] bf16, x_q![M,K] e4m3, x_s![M,K/G] f32, G, eps=1e-10, -448, 448, ue8m0=False, col_major, tma)
        -> Compute(Fp8GroupQuant_v1{K,G}, rows=M, (PerRow(x),), (PerRow(x_q), PerRow(x_s)))
    _C.cutlass_scaled_mm(out![M,N] bf16, x_q[M,K] e4m3, W.T [K,N] (view of the e4m3 [N,K] parameter), x_s[M,K/G] f32,
                         Ws.T [K/G,cdiv(N,G)] (view of the f32 [cdiv(N,G),K/G] `weight_scale_inv` parameter), bias=None)
        -> Compute(ScaledMmFp8BlockSharedScale_v1{K,N,G}, rows=M, (PerRow(x_q), PerRow(x_s), Shared(W), Shared(Ws)), (PerRow(out),))

The GEMM's Definition is the record target's `fp8_scale_construction`: `ScaledMmFp8BlockSharedScale_v1` by default (each weight
block's scale products once), `ScaledMmFp8Block_v1` under `per-coordinate`; one function, one signature.

Both registered Definitions (`verity_vllm.program.registry.fp8`) are pinned bit for bit on the H100 against these kernels (the module's
`_QUANT_CONF` / `_BLOCK_CONF` conformance notes; out/gen/sweep/evidence/fp8/).  The GEMM Definition is the blockwise CUTLASS kernel's
arithmetic (k32 steps, `s = FMUL(sfa, sfb)`, FFMA promotion per K-tile in ascending order) over the record target's e4m3 step
(`frontend.targets.scaled_mm_fp8_block_spec`: the v1 forms on hopper, the version 3 forms over `GemmTarget.fp8_block_dot` elsewhere).
A record whose GEMM target has no blockwise step is refused here fail-closed -- another GPU runs another kernel (Marlin /
torch._scaled_mm on Ada, DeepGEMM, its own blockwise CUTLASS kernel), which has no registered Definition; nothing is approximated by a
looser comparison (COORD 6).

Everything the kernels take as a knob that changes VALUES is pinned (`group_size == G`, `eps == 1e-10`, `fp8_min/max == -+448`,
no bias), except `scale_ue8m0`, which selects the quant Definition (`Fp8GroupQuantUe8m0_v1` under power-of-two scales, the
DeepGEMM-default branch); the layout knobs (`column_major_scales`, `tma_aligned`) only change the scale tensor's strides,
which the views carry.  The two weight operands of the GEMM are named per operand (`Compute.roots`) so the comparator can map each
weight range to its own field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import fp8

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_weight_view, is_row_major
from verity_vllm.observe.fold.patterns.ops import as_float, bound_args, first, op_name
from verity_vllm.observe.fold.patterns.resolutions import _fmt, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import _targets, target_of
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode

QUANT_EPS = 1e-10
FP8_MAX = 448.0
E4M3 = "float8_e4m3fn"
CUTLASS_KERNEL = "CutlassFp8BlockScaledMMKernel"   # the vLLM kernel class whose arithmetic ScaledMmFp8Block_v1 is


def header_fp8_kernel(ctx: StepContext) -> str | None:
    """The block-scaled FP8 linear kernel class the record's engine carried (`RunHeader.profile["backends"]["fp8_block_gemm_kernel"]`,
    written by `vllm_adapter.profile_manifest` from `fp8_block_gemm_kernel_of(model)`), None when the header does not say."""
    try:
        prof = ctx.header.profile or {}
        k = (prof.get("backends") or {}).get("fp8_block_gemm_kernel")
        return k if isinstance(k, str) else None
    except (AttributeError, TypeError):
        return None


def _arg(a: dict[str, Any], *names: str, pos: int) -> Any:
    v = first(*(a.get(n) for n in names))
    return v if v is not None else a.get(f"arg{pos}")


@dataclass
class PerTokenGroupFp8Quant:
    """`_C.per_token_group_fp8_quant` (csrc/quantization/fp8/per_token_group_quant.cu, `per_token_group_quant_8bit_kernel`) on a
    bf16 `[M, K]` activation: per row and per G-wide group, `s = fmaxf-fold(eps, |f32(x)|) / 448` -- under `scale_ue8m0` rounded up
    to a power of two (`SCALE_UE8M0=true`, the DeepGEMM-default branch `moe_kernel_quantize_input` takes on sm_90) -- and
    `q = e4m3(clamp(f32(x) / s))`; both outputs are materialised (`x_q` e4m3 `[M, K]`, `x_s` f32 `[M, K/G]`, column-major when vLLM
    asks for it).  Resolves `Fp8GroupQuant_v1{K,G}`, or `Fp8GroupQuantUe8m0_v1{K,G}` (registry.deepgemm) under `scale_ue8m0`.
    `group` is the G this instance resolves (the checkpoint's weight block for the dense linears, the refined block inside a
    fused-MoE scope), one of `fp8.QUANT_GROUPS`; a launch at another G is refused."""

    group: int
    name: str = "fp8.per_token_group_fp8_quant"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "per_token_group_fp8_quant":
            return None
        a = bound_args(node)
        x = _arg(a, "input", "x", pos=0)
        xq = _arg(a, "output_q", "output", "out_q", pos=1)
        xs = _arg(a, "output_s", "output_scale", "out_s", pos=2)
        G = _arg(a, "group_size", pos=3)
        eps = as_float(_arg(a, "eps", pos=4))
        lo, hi = as_float(_arg(a, "fp8_min", "min_8bit", pos=5)), as_float(_arg(a, "fp8_max", "max_8bit", pos=6))
        ue8m0 = _arg(a, "scale_ue8m0", "use_ue8m0", pos=7)
        if not all(isinstance(v, ViewDesc) for v in (x, xq, xs)) or not isinstance(G, int) or eps is None or lo is None or hi is None:
            return unsupported(node, "per_token_group_fp8_quant arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if G != self.group:
            return unsupported(node, "group size differs from the group this pattern resolves", group_size=G, pinned=self.group)
        if G not in fp8.QUANT_GROUPS:
            return unsupported(node, "group size is not one Fp8GroupQuant_v1 is registered at", group_size=G, registered=list(fp8.QUANT_GROUPS))
        if eps != QUANT_EPS or lo != -FP8_MAX or hi != FP8_MAX:
            return unsupported(node, "quantisation knobs differ from the registered Fp8GroupQuant_v1 (eps 1e-10, [-448, 448])",
                               eps=eps, fp8_min=lo, fp8_max=hi)
        if x.rank != 2 or not is_row_major(x) or xq.shape != x.shape or not is_row_major(xq):
            return unsupported(node, "input / output_q are not row-major [M,K] views of one shape", x=repr(x), xq=repr(xq))
        M, K = x.shape
        if K % G:
            return unsupported(node, "K is not a multiple of the group", K=K, G=G)
        if xs.shape != (M, K // G):
            return unsupported(node, "output_s is not a [M, K/G] view", xs=repr(xs), M=M, K=K, G=G)
        if not (x.dtype == "bfloat16" and xq.dtype == E4M3 and xs.dtype == "float32"):
            return unsupported(node, "Fp8GroupQuant_v1 is bf16 -> (e4m3, f32)", dtypes=[x.dtype, xq.dtype, xs.dtype])
        if bool(ue8m0):
            from verity_vllm.program.registry import deepgemm  # noqa: PLC0415  (loaded only on the ue8m0 branch)
            spec = bind(deepgemm.Fp8GroupQuantUe8m0, K=K, G=G)
        else:
            spec = bind(fp8.Fp8GroupQuant, K=K, G=G)
        return Compute(spec, M, (PerRow(x),), (PerRow(xq), PerRow(xs)), (node.id,),
                       hint=hint_of(node, "fp8_quant"), kernels=("per_token_group_quant_8bit_kernel",),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class CutlassScaledMmFp8Block:
    """`_C.cutlass_scaled_mm(out!, a, b, a_scales, b_scales, bias)` with a `[M, K/G]` x `[K/G, cdiv(N, G)]` scale pair = the BLOCKWISE
    dispatch (`cutlass_scaled_mm_blockwise_sm90_fp8` -> `cutlass_3x_gemm_fp8_blockwise<bf16, ...>`).  `b` is `weight.T` (a `[K, N]`
    strides `(1, K)` view of the e4m3 `[N, K]` parameter), `b_scales` is `weight_scale_inv.T` (a `[K/G, cdiv(N, G)]` view of the f32
    `[cdiv(N, G), K/G]` parameter; N may end mid-block, `fp8.weight_blocks`); both must be checkpoint parameters (weight Roots).
    Per-tensor / per-channel scale shapes are a different kernel (`cutlass_scaled_mm_sm90_fp8`, no block promotion) and are refused;
    so is a bias (the epilogue changes)."""

    group: int
    name: str = "fp8.cutlass_scaled_mm_blockwise"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "cutlass_scaled_mm":
            return None
        a = bound_args(node)
        out = _arg(a, "out", "c", pos=0)
        xq = _arg(a, "a", pos=1)
        b = _arg(a, "b", pos=2)
        xs = _arg(a, "a_scales", pos=3)
        bs = _arg(a, "b_scales", pos=4)
        bias = _arg(a, "bias", pos=5)
        if not all(isinstance(v, ViewDesc) for v in (out, xq, b, xs, bs)):
            return unsupported(node, "cutlass_scaled_mm arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if bias is not None:
            return unsupported(node, "bias epilogue: ScaledMmFp8Block_v1 has none", bias=repr(bias) if isinstance(bias, ViewDesc) else bias)
        tp, _recorded = target_of(ctx)
        T = _targets()
        try:
            T.fp8_block_dot(tp)
        except T.UnregisteredTarget as e:
            return unsupported(node, str(e), compute_capability=list(tp.compute_capability))
        # the record must ATTEST the kernel class its FP8 linears carried (`RunHeader.profile.backends.fp8_block_gemm_kernel`, read back
        # from `Fp8LinearMethod.fp8_linear` by the capture engine): `_C.cutlass_scaled_mm` is also reachable from other kernel classes'
        # fallbacks, and a header without the read-back (a pre-R15 capture) does not say which kernel ran -- refused, never assumed
        kernel = header_fp8_kernel(ctx)
        if kernel != CUTLASS_KERNEL:
            return unsupported(node, "the record does not attest CutlassFp8BlockScaledMMKernel for its FP8 linears (RunHeader.profile."
                                     "backends.fp8_block_gemm_kernel); ScaledMmFp8Block_v1 is that kernel's arithmetic", fp8_block_gemm_kernel=kernel)
        G = self.group
        if xq.rank != 2 or not is_row_major(xq) or out.rank != 2 or not is_row_major(out) or out.shape[0] != xq.shape[0]:
            return unsupported(node, "a / out are not row-major [M,K] / [M,N] views", a=repr(xq), out=repr(out))
        M, K = xq.shape
        N = out.shape[1]
        if K % G:
            return unsupported(node, "K is not a multiple of the block", K=K, N=N, G=G)
        if xs.shape != (M, K // G):
            return unsupported(node, "a_scales is not the per-token-group [M, K/G] scale (per-tensor / per-channel scaling is another kernel)",
                               a_scales=repr(xs), M=M, K=K, G=G)
        try:
            w = as_weight_view(b, K, N)
            ws = as_weight_view(bs, K // G, fp8.weight_blocks(N, G))
        except GeometryError as e:
            return unsupported(node, str(e))
        if not (xq.dtype == E4M3 and w.dtype == E4M3 and xs.dtype == "float32" and ws.dtype == "float32" and out.dtype == "bfloat16"):
            return unsupported(node, "ScaledMmFp8Block_v1 is (e4m3, f32 scales) -> bf16", dtypes=[xq.dtype, w.dtype, xs.dtype, ws.dtype, out.dtype])
        wroots, sroots = ctx.roots_of(w), ctx.roots_of(ws)
        if not any(r.role == "weight" for r in wroots) or not any(r.role == "weight" for r in sroots):
            return unsupported(node, "b / b_scales are not checkpoint parameters (weight Roots)", b=repr(w), b_scales=repr(ws))
        wname, sname = wroots[0].name, sroots[0].name
        return Compute(T.scaled_mm_fp8_block_spec(tp, K, N, G), M, (PerRow(xq), PerRow(xs), Shared(w), Shared(ws)), (PerRow(out),),
                       (node.id,), hint=hint_of(node, wname), kernels=("cutlass_3x_gemm_fp8_blockwise", "sm90_fp8_blockwise"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name, roots=(None, None, wname, sname))
