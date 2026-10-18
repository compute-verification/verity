"""[R17 D90, COORD R17-4] The fold-side rule for a tensor-parallel rank's collectives: `vllm.all_reduce.default` / `vllm.all_gather.default`
(pynccl behind vLLM's custom op) resolve to ONE `AllReduce2_v1{N}` / `AllGather2_v1{N}` instance per row whose single `parts` operand is a
`resolver.Collective` -- this rank's own partial (a memory read of the collective INPUT storage, produced in this rank's log) and the PEER
rank's partial (the fold's `peers_<request>` root parameter, root class `collective_received`) in RANK order -- and whose output is the
RECEIVED storage (`node.outs`), written to memory so the rank's next kernel (RMSNorm / lm_head / sampler) reads a resolved value.

What this pattern decides from the rank's own log alone: kind, the issuing module (`node.enter.module` -- the site; a collective without a
module hint is REFUSED, fail closed: the cross-rank leg locates the peer by (step, k, site)), the partial's geometry (row-major bf16 [M, N]),
the received geometry ([M, N] for all_reduce; [M, world*N] for all_gather along the trailing axis), world size 2 and, for all_gather, `dim`
= -1 / last.  What it does NOT decide: the step/k numbering (the fold's, `fold._collective_prepare`), the rows' requests (the fold's
attribution), the peer's value (a declared input; its linkage to the other rank's committed partial is the cross-rank leg's and root
policy's, never the fold's assumption).  The Definitions are the registered TP2 composites (`verity_vllm.program.registry.b1_tp2`): the same ones the
rank Program binds through `vllm_bindings.AllReduce2Rule` / `AllGather2Rule`, so the two sides compare instance by instance.
"""

from __future__ import annotations

from verity_vllm.program.registry import b1_tp2
from verity_vllm.program.registry import tp_allreduce as T

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_row_major
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.resolver import Collective, Compute, PerRow, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode

ALL_REDUCE_OP = "vllm.all_reduce"
ALL_GATHER_OP = "vllm.all_gather"
KINDS = {ALL_REDUCE_OP: "all_reduce", ALL_GATHER_OP: "all_gather"}


class TpCollective:
    """`vllm.all_reduce(tensor [M,N] bf16, group) -> [M,N]`   -> `Compute(AllReduce2_v1{N}, rows=M, (Collective(own=tensor, "all_reduce", site),), (PerRow(out),))`
    `vllm.all_gather(tensor [M,N] bf16, dim, world, group) -> [M, world*N]` -> `Compute(AllGather2_v1{N}, rows=M, (Collective(own, "all_gather", site),), (PerRow(out),))`.
    [lane vllm-tp-n] `world` is the profile's communicator size: 2 binds the `*2` identities above, > 2 binds `AllReduce_v2{WORLD,N}` /
    `AllGather_v1{WORLD,N}` (`b1_tp2.all_reduce` / `all_gather`); the `parts` operand is `Array<world, Array<N, bf16>>` in rank order and
    the fold fills every peer rank's row from the request's `peers` root (one field per (collective, peer rank))."""

    name: str = "tp.collective"

    def __init__(self, world: int = 2) -> None:
        self.world = int(world)

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return None
        kind = KINDS.get(base_op(node.op))
        if kind is None:
            return None
        a = bound_args(node)
        x = a.get("tensor", a.get("arg0"))
        if not isinstance(x, ViewDesc):
            return unsupported(node, f"{kind}: input is not a tensor view", args={k: repr(v) for k, v in a.items()})
        outs = node.output_views()
        if len(outs) != 1:
            return unsupported(node, f"{kind}: expected one result view, got {len(outs)}", outs=[repr(o) for o in outs])
        out = outs[0]
        site = node.enter.module
        if not site:
            return unsupported(node, f"{kind}: no issuing module recorded on the collective (site unknown; fail closed)")
        if x.rank != 2 or x.dtype != "bfloat16" or not is_row_major(x):
            return unsupported(node, f"{kind}: partial is not a row-major bf16 [M,N] view", partial=repr(x))
        M, N = x.shape
        if kind == "all_gather":
            dim, world = a.get("dim", a.get("arg1")), a.get("world_size", a.get("arg2"))
            if not isinstance(dim, int) or not isinstance(world, int):
                return unsupported(node, "all_gather: dim/world_size not bound as ints", dim=repr(dim), world_size=repr(world))
            if world != self.world:
                return unsupported(node, f"all_gather: world_size {world} != the profile's world {self.world}", world_size=world)
            if dim not in (-1, 1):
                return unsupported(node, f"all_gather: gathers along dim {dim}; AllGather2_v1 states the trailing axis only", dim=dim)
            if out.shape != (M, self.world * N) or out.dtype != "bfloat16" or not is_row_major(out):
                return unsupported(node, f"all_gather: received view is not a row-major bf16 [{M},{self.world * N}] view", received=repr(out))
            spec = b1_tp2.all_gather(self.world, N)      # [lane vllm-tp-n] AllGather2_v1{N} at world 2, AllGather_v1{WORLD,N} beyond
        else:
            if out.shape != (M, N) or out.dtype != "bfloat16" or not is_row_major(out):
                return unsupported(node, f"all_reduce: received view is not a row-major bf16 [{M},{N}] view", received=repr(out))
            spec = b1_tp2.all_reduce(self.world, N)      # [lane vllm-tp-n] AllReduce2_v1{N} at world 2, AllReduce_v2{WORLD,N} beyond
        if out.alloc == x.alloc:
            return unsupported(node, f"{kind}: in-place collective (received storage is the partial's); the fold states a fresh received storage", partial=repr(x), received=repr(out))
        routed = served_route_spec(ctx.header, self.world, M, N) if kind == "all_reduce" and self.world > 2 else None
        if routed is not None:
            flat = ViewDesc(out.alloc, out.offset, (1, M * N), (M * N, 1), out.dtype, out.device)
            return Compute(routed, 1, (Collective(x, kind, site, flat=True),), (PerRow(flat),), tuple(sorted(node.ids())),
                           hint=hint_of(node, kind), kernels=(), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)
        return Compute(spec, M, (Collective(x, kind, site),), (PerRow(out),), tuple(sorted(node.ids())),
                       hint=hint_of(node, kind), kernels=(), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


def allreduce_settings(header) -> dict | None:
    """The all-reduce switches the capture ran under: its header's ``allreduce_settings`` when recorded, else the rank's recorded
    ``parallel_config.disable_custom_all_reduce`` (``notes.tp2.facts``) with the manifest's engine flags (`tp_allreduce.settings_of`);
    None when the capture recorded neither."""
    notes = getattr(header, "notes", None) or {}
    if notes.get("allreduce_settings"):
        return dict(notes["allreduce_settings"])
    pc = (((notes.get("tp2") or {}).get("facts") or {}).get("parallel_config") or {})
    if not isinstance(pc.get("disable_custom_all_reduce"), bool):
        return None
    return T.settings_of(pc["disable_custom_all_reduce"], (getattr(header, "profile", None) or {}).get("engine_flags") or {})


def served_route_spec(header, world: int, M: int, N: int):
    """The Build's Definition for one communicator all-reduce of [M,N] (`vllm_bindings.AllReduceRule`: one call over M * N words on the
    route vLLM takes at the capture's settings), or None when the capture's settings or `tp_allreduce.covered` name no route."""
    from verity.ml.circuits.defs import bind

    settings = allreduce_settings(header)
    if settings is None or world != 8:
        return None
    setting, route = T.served_route(settings, M, N)
    c = T.covered(setting, N, route, M) if setting is not None else None
    if c is None:
        return None
    defn, statics = T.route_binding(c, world, M, N)
    return bind(defn, **statics)


FUSED_AR_NORM_OP = "vllm.flashinfer_trtllm_fused_allreduce_norm"
FUSED_KIND = "fused_allreduce_rms_norm"
AR_RESIDUAL_RMS_NORM = 1          # FlashInfer `AllReduceFusionPattern.kARResidualRMSNorm`


class TpFusedAllReduceRmsNorm:
    """`vllm.flashinfer_trtllm_fused_allreduce_norm(allreduce_in! [M,N], residual [M,N], rms_gamma [N], eps, world, pdl, fp32_acc,
    max_token_num, pattern_code=kARResidualRMSNorm, norm_out! [M,N])` (`fused_allreduce_rms_norm`'s fused path, DeepSeek-V3.2's decoder
    layer) -> `Compute(AllReduceRmsNormMnnvl_v1{WORLD,N,SHOT,CLUSTER,EPS,FMA}, rows=M, (Collective(allreduce_in, kind, site),
    PerRow(residual), Shared(rms_gamma)), (PerRow(norm_out), PerRow(allreduce_in)))`: the Build's `FusedAllReduceRmsNormRule`, row for
    row.  The kernel writes the new residual into ``allreduce_in`` (read first, as every Compute reads before it writes); SHOT and CLUSTER
    are FlashInfer's for (M, N), FMA the captured entries' (`tp_allreduce.CAPTURED`, refused unless they agree)."""

    name: str = "tp.fused_allreduce_rms_norm"

    def __init__(self, world: int = 8) -> None:
        self.world = int(world)

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != FUSED_AR_NORM_OP:
            return None
        from verity.ml.circuits.defs import bind

        a = bound_args(node)
        x, res, w, out = (a.get(k) for k in ("allreduce_in", "residual", "rms_gamma", "norm_out"))
        eps, world, fp32, code = (a.get(k) for k in ("rms_eps", "world_size", "fp32_acc", "pattern_code"))
        if not all(isinstance(v, ViewDesc) for v in (x, res, w, out)):
            return unsupported(node, "fused all-reduce norm without allreduce_in / residual / rms_gamma / norm_out views", args={k: repr(v) for k, v in a.items()})
        if world != self.world or fp32 is not True or code != AR_RESIDUAL_RMS_NORM or a.get("quant_out") is not None \
                or a.get("scale_out") is not None or float(a.get("weight_bias", 0.0) or 0.0) != 0.0 or not isinstance(eps, float):
            return unsupported(node, "only kARResidualRMSNorm with fp32 accumulation, no quantisation and weight bias 0 at the profile's world",
                               world_size=world, fp32_acc=fp32, pattern_code=code)
        site = node.enter.module
        if not site:
            return unsupported(node, f"{FUSED_KIND}: no issuing module recorded (site unknown; fail closed)")
        M, N = x.shape if x.rank == 2 else (None, None)
        views_ok = all(v.rank == 2 and v.shape == (M, N) and v.dtype == "bfloat16" and is_row_major(v) for v in (x, res, out))
        if not views_ok or w.shape != (N,) or w.dtype != "bfloat16" or len({x.alloc, res.alloc, out.alloc}) != 3:
            return unsupported(node, "expected distinct row-major bf16 [M,N] allreduce_in / residual / norm_out and a [N] gamma",
                               allreduce_in=repr(x), residual=repr(res), norm_out=repr(out), gamma=repr(w))
        fma = {c.statics["FMA"] for c in T.CAPTURED if c.route == "fused" and c.N == N and c.lo <= M <= c.hi}
        if len(fma) != 1:
            return unsupported(node, "the captured fused entries do not name one FMA for this call", N=N, M=M, fma=sorted(fma))
        spec = bind(T.AllReduceRmsNormMnnvl, WORLD=self.world, N=N, SHOT=T.mnnvl_shot(M, N, self.world), CLUSTER=T.mnnvl_grid(M, N)[1],
                    EPS=eps, FMA=fma.pop())
        return Compute(spec, M, (Collective(x, FUSED_KIND, site), PerRow(res), Shared(w)), (PerRow(out), PerRow(x)), tuple(sorted(node.ids())),
                       hint=hint_of(node, FUSED_KIND), kernels=("allreduce_fusion", "oneshot", "twoshot", "rmsNormLamport"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name,
                       roots=(None, None, next((r.name for r in ctx.roots_of(w)), None)))
