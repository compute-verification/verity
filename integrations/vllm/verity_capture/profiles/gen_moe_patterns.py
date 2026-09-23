"""Lane `moe` patterns: the OLMoE Mixture-of-Experts block as one *recognised* Unsupported scope.

`OlmoeMoE.forward` (vLLM d9105ea80, `vllm/model_executor/models/olmoe.py`) is `router_logits = gate(x)` (a dense
`ReplicatedLinear`, resolved by the base `GemmLaunch`) followed by the custom op `vllm.moe_forward(hidden_states,
router_logits, layer_name)`, which the observer re-enters (`observer.REENTER_PREFIXES`) so its body — `topk_softmax`,
`moe_align_block_size`, two `fused_moe_kernel` launches, `silu_and_mul`, `moe_sum` — appears as children of the op.

The block is resolved `Unsupported` BY DESIGN (RULES §6: a mechanism problem, not a missing pattern).  Which expert GEMM
an activation row flows through is decided at runtime by `topk` of the router logits, and the fused kernel computes all
selected (token, expert) GEMMs of a layer in one sorted / padded grouped launch — neither is a Compute the frozen Program
model has an instance for.  The pattern's only job is to *recognise* the scope and to put the complete tensor + kernel
metadata of its subtree into `Unsupported.detail`, so the resolution report is the evidence of the mechanism note
`out/gen/lanes/moe/MECHANISM_MOE.md`.  It claims no kernels (the census residue is the MoE kernel list, as expected) and
suppresses nothing: every child event is consumed by exactly this one Unsupported.

Nothing else is new here: OLMoE's q/k norms are ordinary `RMSNorm(eps=1e-5)` launches on a `.contiguous()` copy
(`_rms_norm_kernel`, resolved by the base `RMSNormTritonLaunch`), attention / RoPE / KV cache / dense GEMMs / embedding /
sampler are Llama-shaped and resolve with the base sm89 patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..patterns import GemmLaunch, base_op, bound_args, op_name, triton_name, unsupported
from ..resolver import Resolution, StepContext
from ..tree import TreeNode

# ---------------------------------------------------------------------------------------------------------------------
# the tuned batch-invariant GEMM tilings of sm89 (a specialization, not a widening of the base GemmLaunch)
# ---------------------------------------------------------------------------------------------------------------------

#: vLLM d9105ea80 `vllm/model_executor/determinism/batch_invariant_configs.py::_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"]`
#: (sm_89 = "ada"; `_get_matmul_config`): for a bf16 `matmul_persistent(a[M,K], b[K,N])` whose `(N, K)` is a key, the launch
#: uses the shape's `block_k` and the first M-bucket with `max_m >= M` (the last bucket above the range) instead of the
#: 128x128x64 / 8 warps / 3 stages default the base profile pins.  `{(N, K): (block_k, ((max_m, (block_m, block_n,
#: num_warps, num_stages)), ...))}`.  Copied verbatim from the pinned source (read on the pod, 2026-09-17); the table is
#: part of κ — another vLLM commit may retune it, which the pins then catch as `Unsupported`.
ADA_TUNED_MATMUL: dict[tuple[int, int], tuple[int, tuple[tuple[int, tuple[int, int, int, int]], ...]]] = {
    (12288, 2048): (64, ((1, (16, 128, 8, 2)), (4, (16, 64, 8, 5)), (8, (16, 128, 8, 4)), (16, (16, 128, 4, 5)),
                         (32, (32, 128, 4, 5)), (64, (32, 256, 4, 3)), (256, (128, 128, 8, 3)), (512, (128, 64, 4, 4)),
                         (1024, (128, 128, 8, 3)), (2048, (128, 128, 4, 3)))),
    (2048, 6144): (128, ((1, (16, 64, 4, 4)), (4, (16, 64, 8, 4)), (8, (16, 32, 8, 4)), (16, (16, 32, 8, 4)),
                         (32, (16, 64, 8, 3)), (64, (32, 64, 4, 4)), (256, (32, 64, 4, 4)), (512, (64, 64, 4, 3)),
                         (1024, (64, 64, 4, 3)), (2048, (64, 64, 4, 3)))),
    (4096, 2048): (64, ((1, (16, 256, 8, 3)), (4, (16, 64, 4, 3)), (8, (16, 128, 8, 2)), (16, (16, 32, 4, 5)),
                        (32, (16, 64, 4, 4)), (64, (64, 64, 8, 3)), (256, (64, 64, 4, 5)), (512, (64, 64, 4, 5)),
                        (1024, (128, 64, 4, 4)), (2048, (128, 64, 8, 4)))),
    (151936, 2048): (64, ((1, (16, 32, 8, 5)), (4, (16, 32, 8, 5)), (8, (16, 32, 8, 5)), (16, (16, 32, 4, 4)),
                          (32, (16, 64, 4, 4)), (64, (32, 64, 4, 4)), (256, (128, 128, 4, 3)), (512, (128, 128, 4, 3)),
                          (1024, (128, 128, 4, 3)), (2048, (128, 128, 4, 3)))),
    (2048, 2048): (64, ((1, (32, 128, 8, 5)), (4, (16, 256, 8, 3)), (8, (32, 128, 8, 3)), (16, (32, 128, 8, 3)),
                        (32, (16, 128, 8, 5)), (64, (16, 256, 4, 3)), (256, (64, 128, 8, 4)), (512, (64, 64, 4, 5)),
                        (1024, (64, 64, 4, 5)), (2048, (64, 64, 4, 5)))),
}


def tuned_matmul_config(M: int, N: int, K: int, table=ADA_TUNED_MATMUL) -> dict[str, int] | None:
    """`_get_matmul_config(M, N, K, bf16, default)` when `(N, K)` is tuned, else None (the default applies)."""
    shape = table.get((N, K))
    if shape is None:
        return None
    block_k, buckets = shape
    cfg = buckets[-1][1]
    for max_m, c in buckets:
        if max_m >= M:
            cfg = c
            break
    bm, bn, warps, stages = cfg
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": 8, "num_warps": warps,
            "num_stages": stages}


@dataclass
class GemmLaunchTunedAda:
    """The base `GemmLaunch` with the pins of the tuned sm89 table for the launch's `(N, K, M)` — a specialization for
    the launches the base instance reports as `launch constexpr/geometry differs from the pinned profile` (OLMoE's
    `o_proj`, N=K=2048: every H=2048 model's `o_proj` hits this key; its decode launch is 32x128x64 / 8 warps / 5 stages,
    its 32-row prefill launch 16x128x64 / 8 warps / 5 stages).

    Numerical claim: the SAME registered body `Gemm_v1{K,N}`.  `matmul_kernel_persistent` accumulates each output
    coordinate as the k-ascending chain of `tl.dot` k16 steps over BLOCK_SIZE_K-wide loads (`AmpereBF16TcDot16_v1`,
    bf16 RNE at the end); BLOCK_SIZE_M / BLOCK_SIZE_N / num_warps / num_stages / GROUP_SIZE_M schedule which output tile
    a program owns and how deep the load pipeline is, and do not enter the arithmetic of a coordinate.  `block_k` is 64
    for the (2048, 2048) key — the very value the base pins — so even the load-chunking is unchanged there.  The claim is
    CHECKED, not assumed: the tier-b replay (G5) evaluates `Gemm_v1{K=2048,N=2048}` on the captured operands of these
    launches (≈ 4.2 M leaves per row instance, under the 8 M budget) against the captured outputs.

    Declines (None) unless `(N, K)` is a tuned key AND the launch's constexprs / num_warps / num_stages equal that
    bucket exactly, so anything else still reaches the base instance and is reported by it."""

    base_constexprs: dict[str, Any]
    num_sms: int
    table: dict = field(default_factory=lambda: ADA_TUNED_MATMUL)
    name: str = "gemm.matmul_kernel_persistent.tuned_ada"
    kernel: str = "matmul_kernel_persistent"

    def _launch(self, node: TreeNode) -> TreeNode | None:
        if node.is_triton:
            return node if triton_name(node) == self.kernel else None
        if base_op(node.op) in GemmLaunch.wrapper_ops:
            launches = [n for n in node.walk() if n.is_triton and triton_name(n) == self.kernel]
            return launches[0] if len(launches) == 1 else None
        return None

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        launch = self._launch(node)
        if launch is None:
            return None
        a = bound_args(launch)
        M, N, K = a.get("M"), a.get("N"), a.get("K")
        if not all(isinstance(v, int) for v in (M, N, K)):
            return None
        cfg = tuned_matmul_config(M, N, K, self.table)
        if cfg is None:
            return None
        consts = dict(launch.enter.constexprs)
        pins = dict(self.base_constexprs)
        pins.update({k: cfg[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")})
        if any(consts.get(k, a.get(k)) != v for k, v in pins.items()):
            return None
        if launch.enter.num_warps != cfg["num_warps"] or launch.enter.num_stages != cfg["num_stages"]:
            return None
        delegate = GemmLaunch(pins, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"], num_sms=self.num_sms, name=self.name)
        return delegate.match(node, ctx)


# ---------------------------------------------------------------------------------------------------------------------
# the MoE block
# ---------------------------------------------------------------------------------------------------------------------

#: The op name prefix of the MoE custom op(s) (`vllm.moe_forward.default`, `vllm.moe_forward_shared.default`).
MOE_OP_PREFIX = "moe_forward"

#: Device kernels of the OLMoE MoE block on the pinned vLLM (Triton function names / CUDA kernel tokens).  Listed for
#: the mechanism note and the census reader; NOT declared as claimed (`Unsupported` claims nothing).
MOE_KERNELS_EXPECTED = ("topk_softmax", "moe_align_block_size", "fused_moe_kernel", "silu_and_mul", "moe_sum")


def _view(v: Any) -> dict[str, Any]:
    return {"shape": list(v.shape), "dtype": v.dtype, "strides": list(v.strides), "offset": v.offset, "alloc": v.alloc}


def _scalar(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    if hasattr(v, "shape") and hasattr(v, "strides"):
        return _view(v)
    return repr(v)


def moe_subtree_census(node: TreeNode) -> dict[str, Any]:
    """Every event of the scope with its tensor metadata: dispatcher ops with input/output views, Triton launches with
    grid / constexprs / num_warps / num_stages and their tensor arguments.  Serialisable (plain dicts) so it survives
    the resolution report's JSON."""
    ops: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    for n in node.walk():
        if n is node:
            continue
        if n.is_triton:
            e = n.enter
            launches.append({"id": n.id, "fn": triton_name(n), "grid": list(e.grid), "constexprs": dict(e.constexprs),
                             "num_warps": e.num_warps, "num_stages": e.num_stages,
                             "args": {k: _scalar(v) for k, v in e.args.items() if k not in e.constexprs}})
        else:
            ops.append({"id": n.id, "op": n.op, "depth": _depth(node, n),
                        "in": [_view(v) for v in n.input_views()], "out": [_view(v) for v in n.output_views()]})
    return {"dispatcher_ops": ops, "triton_launches": launches, "n_events": len(node.ids()) - 1}


def _depth(root: TreeNode, target: TreeNode) -> int:
    def go(n: TreeNode, d: int) -> int | None:
        if n is target:
            return d
        for c in n.children:
            r = go(c, d + 1)
            if r is not None:
                return r
        return None
    return go(root, 0) or 0


@dataclass
class MoEBlockScope:
    """`vllm.moe_forward*(hidden_states [M,H] bf16, router_logits [M,E] bf16, layer_name) -> [M,H] bf16`.

    Resolution: `Unsupported` with `detail = {phase, children, config: {E, TOPK, RENORMALIZE, I, H}, hidden_states,
    router_logits, output, census: moe_subtree_census(node)}`.  The decision that this is a mechanism problem (and the
    PROPOSED representation) is in `out/gen/lanes/moe/MECHANISM_MOE.md`."""

    name = "moe_block_scope"
    config: dict[str, Any]

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or not op_name(node.op).startswith(MOE_OP_PREFIX):
            return None
        C = self.config
        ins = node.input_views()
        outs = node.output_views()
        hidden = next((v for v in ins if len(v.shape) == 2 and v.shape[1] == C["H"]), None)
        logits = next((v for v in ins if len(v.shape) == 2 and v.shape[1] == C.get("NUM_EXPERTS")), None)
        return unsupported(
            node,
            "MoE block: the token->expert assignment is data-dependent (topk of the router logits) and the fused kernel "
            "computes the selected expert GEMMs of the layer in one grouped launch; no Program instance represents "
            "either (mechanism problem, RULES §6; see out/gen/lanes/moe/MECHANISM_MOE.md)",
            layer=node.enter.module,
            config={"E": C.get("NUM_EXPERTS"), "TOPK": C.get("TOPK"), "RENORMALIZE": C.get("RENORMALIZE"),
                    "NORM_TOPK_PROB": C.get("NORM_TOPK_PROB"), "I": C["I"], "H": C["H"]},
            rows=hidden.shape[0] if hidden is not None else None,
            hidden_states=_view(hidden) if hidden is not None else None,
            router_logits=_view(logits) if logits is not None else None,
            output=[_view(v) for v in outs],
            census=moe_subtree_census(node),
        )


def moe_patterns(config: dict[str, Any], *, gemm_constexprs: dict[str, Any], gemm_num_sms: int) -> list:
    """The lane's patterns, to be placed BEFORE the base list: `GemmLaunchTunedAda` must see a tuned launch before the
    base `GemmLaunch` reports its pins as differing (it declines everything else); `MoEBlockScope` recognises the
    custom op no base pattern matches (order only affects the reason text there)."""
    return [GemmLaunchTunedAda(base_constexprs=gemm_constexprs, num_sms=gemm_num_sms), MoEBlockScope(config)]
