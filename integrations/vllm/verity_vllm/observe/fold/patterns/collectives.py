"""The fold-side rule for a tensor-parallel rank's collectives: `vllm.all_reduce.default` / `vllm.all_gather.default`
(pynccl behind vLLM's custom op) resolve to ONE `AllReduce2_v1{N}` / `AllGather2_v1{N}` instance per row whose single `parts` operand is a
`resolver.Collective` -- this rank's own partial (a memory read of the collective INPUT storage, produced in this rank's log) and the PEER
rank's partial (the fold's `peers_<request>` root parameter, root class `collective_received`) in RANK order -- and whose output is the
RECEIVED storage (`node.outs`), written to memory so the rank's next kernel (RMSNorm / lm_head / sampler) reads a resolved value.

What this pattern decides from the rank's own log alone: kind, the issuing module (`node.enter.module` -- the site; a collective without a
module hint is REFUSED, fail closed: the cross-rank check locates the peer by (step, k, site)), the partial's geometry (row-major bf16 [M, N]),
the received geometry ([M, N] for all_reduce; [M, world*N] for all_gather along the trailing axis), world size 2 and, for all_gather, `dim`
= -1 / last.  What it does NOT decide: the step/k numbering (the fold's, `fold._collective_prepare`), the rows' requests (the fold's
attribution), the peer's value (a declared input; its linkage to the other rank's committed partial is the cross-rank check's and root
policy's, never the fold's assumption).  The Definitions are the registered TP2 composites (`verity.ml.definitions.registry.b1_tp2`): the same ones the
rank Program binds through `vllm_bindings.AllReduce2Rule` / `AllGather2Rule`, so the two sides compare instance by instance.
"""

from __future__ import annotations

from verity.ml.definitions.registry import b1_tp2
from verity.ml.definitions.registry import tp_allreduce as T

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.embedding import Embedding
from verity_vllm.observe.fold.patterns.geometry import is_int, is_row_major
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args
from verity_vllm.observe.fold.patterns.resolutions import attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.resolver import Collective, Compute, Group, PerRow, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode

ALL_REDUCE_OP = "vllm.all_reduce"
ALL_GATHER_OP = "vllm.all_gather"
KINDS = {ALL_REDUCE_OP: "all_reduce", ALL_GATHER_OP: "all_gather"}


class TpCollective:
    """`vllm.all_reduce(tensor [M,N] bf16, group) -> [M,N]`   -> `Compute(AllReduce2_v1{N}, rows=M, (Collective(own=tensor, "all_reduce", site),), (PerRow(out),))`
    `vllm.all_gather(tensor [M,N] bf16, dim, world, group) -> [M, world*N]` -> `Compute(AllGather2_v1{N}, rows=M, (Collective(own, "all_gather", site),), (PerRow(out),))`.
    `world` is the profile's communicator size: 2 binds the `*2` identities above, > 2 binds `AllReduce_v2{WORLD,N}` /
    `AllGather_v1{WORLD,N}` (`b1_tp2.all_reduce` / `all_gather`); the `parts` operand is `Array<world, Array<N, bf16>>` in rank order and
    the fold fills every peer rank's row from the request's `peers` root (one field per (collective, peer rank)).  An all-reduce on a
    route `tp_allreduce.CAPTURED` covers at the capture's settings, world, GPU and transport (`covered_route`) binds the Build's routed
    Definition instead: one flat call per request (`_per_request`), the parts of one call (step, k)."""

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
            spec = b1_tp2.all_gather(self.world, N)      # AllGather2_v1{N} at world 2, AllGather_v1{WORLD,N} beyond
        else:
            if out.shape != (M, N) or out.dtype != "bfloat16" or not is_row_major(out):
                return unsupported(node, f"all_reduce: received view is not a row-major bf16 [{M},{N}] view", received=repr(out))
            spec = b1_tp2.all_reduce(self.world, N)      # AllReduce2_v1{N} at world 2, AllReduce_v2{WORLD,N} beyond
        if out.alloc == x.alloc:
            return unsupported(node, f"{kind}: in-place collective (received storage is the partial's); the fold states a fresh received storage", partial=repr(x), received=repr(out))
        attribution = attribution_for(ctx, M, node.phase)
        routed = covered_route(ctx.header, self.world, M, N) if kind == "all_reduce" and self.world > 2 else None
        if routed is not None:
            return self._per_request(node, routed, x, out, site, attribution)
        return Compute(spec, M, (Collective(x, kind, site),), (PerRow(out),), tuple(sorted(node.ids())),
                       hint=hint_of(node, kind), kernels=(), attribution=attribution, pattern=self.name)

    def _per_request(self, node: TreeNode, routed: T.Captured, x: ViewDesc, out: ViewDesc, site: str, attribution) -> Resolution:
        """The routed all-reduce as the Build binds it: one call per request over its run of rows, each `rows * N` words on the route
        the launch's M picks (`route_binding`; refused where the route reduces an element by its offset in the launch)."""
        from verity.ml.circuits.defs import bind

        (M, N), parts = x.shape, []
        try:
            spans = request_spans(attribution) or [(0, M)]
            for lo, n in spans:
                defn, statics = T.route_binding(routed, self.world, M, N, n)
                own = ViewDesc(x.alloc, x.offset + lo * N, (n, N), (N, 1), x.dtype, x.device)
                flat = ViewDesc(out.alloc, out.offset + lo * N, (1, n * N), (n * N, 1), out.dtype, out.device)
                parts.append(Compute(bind(defn, **statics), 1, (Collective(own, "all_reduce", site, flat=True, call=x),), (PerRow(flat),),
                                     tuple(sorted(node.ids())), hint=hint_of(node, "all_reduce"), kernels=(),
                                     attribution=attribution[lo:lo + n], pattern=self.name))
        except ValueError as e:
            return unsupported(node, f"all_reduce: {e}", M=M, N=N)
        if len(parts) == 1:
            return parts[0]
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=hint_of(node, "all_reduce"), pattern=self.name)


def request_spans(attribution) -> list[tuple[int, int]]:
    """(first row, rows) of each request's run of rows in a launch's attribution, in row order; [] when the rows name no requests.
    A request whose rows are not one run is refused (`ValueError`): the Build reduces each request's rows as one call."""
    spans: list[tuple[str, int, int]] = []
    for i, (rid, _pos) in enumerate(attribution):
        if spans and spans[-1][0] == rid:
            spans[-1] = (rid, spans[-1][1], spans[-1][2] + 1)
        elif any(s[0] == rid for s in spans):
            raise ValueError(f"request {rid!r} owns rows apart from its run at {next(s[1] for s in spans if s[0] == rid)}")
        else:
            spans.append((rid, i, 1))
    return [(lo, n) for _rid, lo, n in spans]


def allreduce_settings(header) -> dict | None:
    """The all-reduce switches the capture ran under: its header's ``allreduce_settings`` when recorded, else the rank's recorded
    ``parallel_config.disable_custom_all_reduce`` (``notes.tp2.facts``) with the rank's NCCL environment (``nccl_env``) and the
    manifest's engine flags (`tp_allreduce.settings_of`); None when the capture recorded neither."""
    notes = getattr(header, "notes", None) or {}
    if notes.get("allreduce_settings"):
        return dict(notes["allreduce_settings"])
    facts = (notes.get("tp2") or {}).get("facts") or {}
    pc = facts.get("parallel_config") or {}
    if not isinstance(pc.get("disable_custom_all_reduce"), bool):
        return None
    env = {**(facts.get("nccl_env") or {}), **((getattr(header, "profile", None) or {}).get("engine_flags") or {})}
    return T.settings_of(pc["disable_custom_all_reduce"], env)


def capture_device(header) -> tuple[str, int] | None:
    """The capture's GPU as `tp_allreduce.device_of` names it (``RunHeader.profile["gpu"]``), or None when it recorded none."""
    gpu = (getattr(header, "profile", None) or {}).get("gpu") or {}
    cc, sms = gpu.get("compute_capability", gpu.get("cc")), gpu.get("multi_processor_count", gpu.get("sms"))
    return None if cc is None or sms is None else T.device_of(cc, sms)


def covered_route(header, world: int, M: int, N: int) -> T.Captured | None:
    """The captured route vLLM takes for one communicator all-reduce of [M,N] at the capture's settings, world, GPU and NCCL transport,
    which the Build binds as one call per request (`vllm_bindings.AllReduceRule`, `tp_allreduce.route_binding`), or None when the
    capture's settings or `tp_allreduce.covered` name no route."""
    settings = allreduce_settings(header)
    if settings is None:
        return None
    setting, route = T.served_route(settings, M, N)
    return (T.covered(setting, N, route, M, world=world, device=capture_device(header), transport=settings.get("nccl_transport"))
            if setting is not None else None)


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


class EmbeddingShardCuda:
    """The rank-local fused embedding lookup of a tensor-parallel rank: vLLM's `_C.vocab_parallel_embedding(out[M,H], ids[M],
    shard[VS,H], START, END, padding, added_start, added_end)` (the fused kernel of `VocabParallelEmbedding` at tp_size > 1:
    `out[i] = shard[ids[i]-START]` if START <= ids[i] < END else 0) IS `EmbeddingShard_v1{VS,H,START}(ids, shard)` (the registered TP2
    Definition, `verity.ml.definitions.registry.b1_tp2`) -- the rank-local partial the embedding all-reduce consumes; one operand set,
    entirely in this rank's log: `Compute(EmbeddingShard_v1{VS,H,START}, rows=M, (PerRow(input_ids), Shared(shard)), (PerRow(out),))`.
    Refuses padded shards and added-vocab ranges (not what `EmbeddingShard` states; derive_step's `bind_vocab_parallel_embeddings`
    refuses the same).  A `world > 1` fact of the architecture, not of any model: `profiles.generic.build_profile(role, world=N)`
    registers it for every rank profile."""

    name: str = "tp.embedding_shard_cuda"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        from verity.ml.circuits.defs import bind
        if node.is_triton or base_op(node.op) != "_C.vocab_parallel_embedding":
            return None
        a = bound_args(node)
        out, idx, w = a.get("out", a.get("arg0")), a.get("input_ids", a.get("arg1")), a.get("weight", a.get("arg2"))
        start, end = a.get("org_vocab_start_index", a.get("arg3")), a.get("org_vocab_end_index", a.get("arg4"))
        pad, add0, add1 = a.get("num_org_vocab_padding", a.get("arg5")), a.get("added_vocab_start_index", a.get("arg6")), a.get("added_vocab_end_index", a.get("arg7"))
        if not all(isinstance(v, ViewDesc) for v in (out, idx, w)) or not all(isinstance(v, int) for v in (start, end, pad, add0, add1)):
            return unsupported(node, "vocab_parallel_embedding arguments not bound as expected", args={k: repr(v) for k, v in a.items()})
        if pad != 0 or add1 != add0:
            return unsupported(node, "EmbeddingShard_v1 binds plain contiguous shards only (padded or added-vocab shard)", padding=pad, added=(add0, add1))
        if w.rank != 2 or not is_row_major(w) or w.dtype != "bfloat16":
            return unsupported(node, "embedding shard is not a row-major bf16 [VS,H] view", weight=repr(w))
        if not is_int(idx) or idx.rank != 1:
            return unsupported(node, "embedding indices are not an integer [M] view", indices=repr(idx))
        VS, H = w.shape
        if end - start != VS:
            return unsupported(node, "shard rows != END-START", VS=VS, START=start, END=end)
        M = idx.shape[0]
        if out.shape != (M, H) or not is_row_major(out) or out.dtype != "bfloat16":
            return unsupported(node, "embedding output is not a row-major bf16 [M,H] view", out=repr(out))
        tok = Embedding._token_view(idx, ctx)
        if isinstance(tok, str):
            return unsupported(node, tok, indices=repr(idx))
        roots = ctx.roots_of(w)
        return Compute(bind(b1_tp2.EmbeddingShard, VS=VS, H=H, START=start), M, (PerRow(tok), Shared(w)), (PerRow(out),), (node.id,),
                       hint=hint_of(node, roots[0].name if roots else "embed_tokens"), kernels=("vocab_parallel_embedding_kernel",),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)
