"""Tensor-parallel collectives: the `verity_tp::*` ops the construction's CollectiveBus emits."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll, array_of
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import StaticIndex
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_allreduce_settings
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _row_value
from verity_vllm.program.registry import tp_allreduce


# -- [R13 lane tp, P4] tensor-parallel collectives: `verity_tp::*` ops emitted by the construction's CollectiveBus (cb_a/tp_ops.py) ----------


def _tp_op(name: str) -> set:
    try:
        from verity_vllm.program.frontend.export_ops import ensure_ops
        ensure_ops()
        return {getattr(torch.ops.verity_tp, name).default}
    except Exception:  # noqa: BLE001 — construction scaffolding not importable: the rule has no targets
        return set()


def _hint_module(hint: str) -> str:
    """The module of a naming hint (`<module>/<node>[/<member>][i]` -> `<module>`; a bare node name -> ``)."""
    head = hint.split("/", 1)[0] if "/" in hint else ""
    return head.split("[", 1)[0]


def _collective_hint(node, ctx) -> str:
    """[R19 moe F-r19-moe-1, row #70] the naming hint of a TP collective row.  A collective whose rank-local operand is produced UNDER the
    collective's own module (a row-parallel linear's all-reduce of its Gemm, the MoE runner's all-reduce of its MoeSum, the logits
    processor's all-gather of its Gemm shard) is named under that module (`<module>/<node>`) as before: the Gemm / MoeSum row under the
    module IS the rank's partial (required_manifest TP-04 `members_for`).  A collective issued MID-module -- every rank-local operand row
    lives under ANOTHER module (OLMoE `OlmoeAttention._apply_qk_norm`: q and k all-gathered inside `self_attn`, each a slice of the
    `qkv_proj` Gemm's output) -- is named at its call SITE `<module>.<fx node>/<node>`, so the manifest (`collective_site`) requires it as
    the module's k-th collective identity WITH its own `shard_rank<r>` / `part_rank<r>` member, rather than folding both calls into one
    `<module>/out` of twice the width with no partial to bind (row #70: 6,944 unbound per rank).  The rank-local operand is the one that
    is NOT a Program input (the peer's partial arrives as a `peers_<k>` Input); when neither operand is an input (a TP1-style test
    Program) both are examined.  Its producing rows are read off the Program's own refs (`node_hints` of the rows' nodes: space "n")."""
    own_mod = ctx.stack_path(node) if node.meta.get("nn_module_stack") else ""
    base = ctx.hint_name(node)
    if not own_mod:
        return base
    producers: set[str] = set()
    operands = list(node.args[0]) if isinstance(node.args[0], (list, tuple)) else list(node.args[:2])   # WORLD-ary op: one list arg
    for a in operands:
        if not isinstance(a, torch.fx.Node) or ctx.is_input_storage(a) is not None:
            continue
        rows, _lead = rows_of(ctx.coll(a))
        for r in rows:
            ref = r.refs[0]
            if ref[0] == "n":
                producers.add(_hint_module(ctx.node_hints.get(int(ref[1]), "")))
            else:
                producers.add("")                                            # a parameter ref: no producing row (an input; not this module's)
    if not producers or own_mod in producers:
        return base
    return f"{own_mod}.{node.name}/{node.name}"


class AllReduce2Rule(OpRule):
    """`verity_tp::all_reduce2(part_rank0 [.., N] bf16, part_rank1 [.., N] bf16)` -> `all_reduce2{N}` (`AllReduce2_v1{N}`: one bf16 add per
    element = `ncclSum` bf16 of two terms) per row.  Operands are in RANK ORDER; one of them is this rank's partial (a value the rank
    computes: a K-sharded `Gemm_v1` output or an `EmbeddingShard` row), the other the PEER INPUT bound by `tp_links.json` to the other
    rank's committed partial — the received collective output is never a fresh trusted input.  Established on the pod (R13 tp step 1):
    the pynccl all-reduce result equals `bf16(f32(P_0) + f32(P_1))` bit-exactly on every recorded element."""
    modes = frozenset({"profile"})
    info = RuleInfo("AllReduce2[tp]", 1, "op-binding", applies_to="verity_tp::all_reduce2(a [.., N] bf16, b [.., N] bf16) (rank order)",
                    preconditions=("both operands are bf16 value collections of identical shape", "world size 2"),
                    evidence_domain="structural; numerics: AllReduce2_v1 (Bf16Add per element; exact-model-tested B1 TP2 + Llama TP2 step-1 probe)")

    def targets(self):
        return _tp_op("all_reduce2")

    def applies(self, node, ctx):
        for nm, a in (("a", node.args[0]), ("b", node.args[1])):
            r = _row_value(ctx, a, nm)
            if r:
                return r
        if shape(ctx, node.args[0]) != shape(ctx, node.args[1]):
            return f"partials differ in shape ({shape(ctx, node.args[0])} vs {shape(ctx, node.args[1])})"
        return True

    def emit(self, node, ctx):
        a, b = ctx.coll(node.args[0], "part_rank0"), ctx.coll(node.args[1], "part_rank1")
        ra, lead = rows_of(a)
        rb, _ = rows_of(b)
        N = ra[0].type.n
        fn = ctx.kind("all_reduce2", node, N=N)
        hint = _collective_hint(node, ctx)                                   # [R19 moe F-r19-moe-1] site-named when mid-module
        outs = [ctx.call(fn, array_of([x, y]), node=node, rule=self, name=f"{hint}[{i}]/all_reduce") for i, (x, y) in enumerate(zip(ra, rb))]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(ra), "world": 2}, emitted=f"{fn.id} x{len(ra)}", collective="all_reduce")


class AllGather2Rule(OpRule):
    """`verity_tp::all_gather2(part_rank0 [.., N] bf16, part_rank1 [.., N] bf16)` -> `all_gather2{N}` (`AllGather2_v1{N}`: the [2N] row IS
    the two shards side by side — a copy, no arithmetic) per row; rank order = column order (vLLM's `all_gather` along the last dim)."""
    modes = frozenset({"profile"})
    info = RuleInfo("AllGather2[tp]", 1, "op-binding", applies_to="verity_tp::all_gather2(a [.., N] bf16, b [.., N] bf16) (rank order, trailing axis)",
                    preconditions=("both operands are bf16 value collections of identical shape", "world size 2"),
                    evidence_domain="structural; numerics: AllGather2_v1 (concatenation; exact-model-tested)")

    def targets(self):
        return _tp_op("all_gather2")

    def applies(self, node, ctx):
        for nm, a in (("a", node.args[0]), ("b", node.args[1])):
            r = _row_value(ctx, a, nm)
            if r:
                return r
        if shape(ctx, node.args[0]) != shape(ctx, node.args[1]):
            return f"shards differ in shape ({shape(ctx, node.args[0])} vs {shape(ctx, node.args[1])})"
        return True

    def emit(self, node, ctx):
        a, b = ctx.coll(node.args[0], "part_rank0"), ctx.coll(node.args[1], "part_rank1")
        ra, lead = rows_of(a)
        rb, _ = rows_of(b)
        N = ra[0].type.n
        fn = ctx.kind("all_gather2", node, N=N)
        hint = _collective_hint(node, ctx)                                   # [R19 moe F-r19-moe-1] site-named when mid-module
        outs = [ctx.call(fn, array_of([x, y]), node=node, rule=self, name=f"{hint}[{i}]/all_gather") for i, (x, y) in enumerate(zip(ra, rb))]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(ra), "world": 2}, emitted=f"{fn.id} x{len(ra)}", collective="all_gather")


def _tp_parts(node) -> list:
    """The `parts` list argument of a WORLD-ary `verity_tp::all_reduce` / `all_gather` node (fx: a list of Nodes), in rank order."""
    a = node.args[0]
    return list(a) if isinstance(a, (list, tuple)) else [a]


def _served_route(settings: dict, M: int, N: int) -> tuple[str | None, str]:
    """The setting name and communicator route vLLM d9105ea80 takes for one bf16 [M,N] world-8 all-reduce."""
    custom = bool(settings.get("custom_all_reduce"))
    symm = bool(settings.get("symmetric_memory"))
    flashinfer = bool(settings.get("flashinfer"))
    setting = ("custom-off" if not custom and symm and flashinfer else
               "defaults" if custom and symm and flashinfer else
               "switch" if custom and not symm and flashinfer else None)
    nbytes = M * N * 2
    if flashinfer and nbytes <= tp_allreduce.FI_MAX_BYTES:
        route = "fi_ar_comm"
    elif custom and nbytes < (tp_allreduce.CUSTOM_SYMM_MAX_BYTES if symm else tp_allreduce.CUSTOM_MAX_BYTES):
        route = "ca_comm"
    elif symm and nbytes <= tp_allreduce.SYMM_MEM_MAX_BYTES:
        route = "symm_mem_comm"
    else:
        route = "pynccl_comm"
    return setting, route


def _coverage(parts, node, ctx, *, fused: bool = False):
    settings = active_allreduce_settings()
    if settings is None:
        return None, "all-reduce settings were not observed from the workload"
    if len(parts) != 8:
        return None, f"world {len(parts)}: tp_allreduce.CAPTURED covers world 8 only"
    rows = [rows_of(ctx.coll(a))[0] for a in parts]
    M, N = len(rows[0]), rows[0][0].type.n
    setting, route = _served_route(settings, M, N)
    if fused:
        route = "fused" if bool(settings.get("flashinfer")) and M <= tp_allreduce.fused_max_tokens(N, len(parts)) else "unfused"
    if setting is None:
        return None, f"tp_allreduce.covered(): switches {settings!r} do not name a captured setting"
    captured = tp_allreduce.covered(setting, N, route, M)
    if captured is None:
        return None, (f"tp_allreduce.covered() has no entry for setting={setting!r}, N={N}, route={route!r}, M={M}; "
                      "the route is refused until a capture records it")
    return {"captured": captured, "rows": rows, "M": M, "N": N, "setting": setting, "route": route}, None


def _allreduce_kind(ctx, node, plan, world: int):
    c, M, N, route = plan["captured"], plan["M"], plan["N"], plan["route"]
    size = M * N
    if c.definition == "AllReduceMnnvl_v1":
        return ctx.kind("all_reduce_mnnvl", node, WORLD=world, N=size, SHOT=tp_allreduce.mnnvl_shot(M, N, world))
    if c.definition == "AllReduceMultimem_v1":
        return ctx.kind("all_reduce_multimem", node, WORLD=world, SIZE=size, **c.statics)
    if c.definition == "AllReduceCustom_v1":
        return ctx.kind("all_reduce_custom", node, WORLD=world, SIZE=size, **c.statics)
    raise ValueError(f"covered route {route!r} names unsupported definition {c.definition!r}")


class AllReduceRule(OpRule):
    """[lane vllm-tp-n] `verity_tp::all_reduce([part_rank0, .., part_rank{W-1}] each [.., N] bf16)` -> `all_reduce{WORLD=W, N}` (`AllReduce_v2{WORLD,N}`:
    the captured world-8 route selected by the call's M and the workload's custom-all-reduce, symmetric-memory and FlashInfer settings.
    A route absent from `tp_allreduce.covered()` refuses by name."""
    modes = frozenset({"profile"})
    info = RuleInfo("AllReduce[tp]", 1, "op-binding", applies_to="verity_tp::all_reduce([a_0 .. a_{W-1}] each [.., N] bf16) (rank order), W > 2",
                    preconditions=("every operand is a bf16 value collection of identical shape", "world size 8",
                                   "tp_allreduce.covered(setting,N,route,M) returns a captured entry"),
                    evidence_domain="structural; numerics: registry.tp_allreduce.CAPTURED")

    def targets(self):
        return _tp_op("all_reduce")

    def applies(self, node, ctx):
        parts = _tp_parts(node)
        if len(parts) <= 2:
            return f"world {len(parts)}: the two-rank collective is AllReduce2 (recorded identity)"
        for i, a in enumerate(parts):
            r = _row_value(ctx, a, f"part_rank{i}")
            if r:
                return r
        shapes = {shape(ctx, a) for a in parts}
        if len(shapes) != 1:
            return f"partials differ in shape ({sorted(map(str, shapes))})"
        _plan, why = _coverage(parts, node, ctx)
        if why:
            return f"numerics_unregistered: {why}"
        return True

    def emit(self, node, ctx):
        parts = _tp_parts(node)
        W = len(parts)
        colls = [ctx.coll(a, f"part_rank{i}") for i, a in enumerate(parts)]
        rows, lead = zip(*(rows_of(c) for c in colls))
        rows = list(rows)
        plan, why = _coverage(parts, node, ctx)
        assert plan is not None and why is None
        M, N = plan["M"], plan["N"]
        fn = _allreduce_kind(ctx, node, plan, W)
        flat_parts = array_of([array_of(list(rs)).reshape(Array(M * N, Value(16))) for rs in rows])
        hint = _collective_hint(node, ctx)
        out = ctx.call(fn, flat_parts, node=node, rule=self, name=f"{hint}/all_reduce")
        ctx.define(node, from_rows([out[i * N:(i + 1) * N] for i in range(M)], lead[0]))
        ctx.record(self, node, {"N": N, "rows": M, "world": W, "setting": plan["setting"], "route": plan["route"],
                               "coverage": plan["captured"]._asdict()}, emitted=fn.id, collective="all_reduce")


class FusedAllReduceRmsNormRule(OpRule):
    """FlashInfer's fused world-8 all-reduce + residual add + RMSNorm leaf authored by the export shim."""
    modes = frozenset({"profile"})
    info = RuleInfo("AllReduceRmsNorm[tp]", 1, "op-binding",
                    applies_to="verity_tp::fused_allreduce_rms_norm([part_rank0..7], residual, weight, eps)",
                    preconditions=("bf16 parts/residual [M,N], bf16 weight [N]", "tp_allreduce.covered(..., route='fused')"),
                    evidence_domain="registry.tp_allreduce.CAPTURED")

    def targets(self):
        return _tp_op("fused_allreduce_rms_norm")

    def applies(self, node, ctx):
        parts = _tp_parts(node)
        for i, a in enumerate(parts):
            if (r := _row_value(ctx, a, f"part_rank{i}")):
                return r
        residual, weight, eps = node.args[1:4]
        if (r := _row_value(ctx, residual, "residual")) or (r := _row_value(ctx, weight, "weight")):
            return r
        shapes = {shape(ctx, a) for a in parts}
        if len(shapes) != 1 or shape(ctx, residual) not in shapes:
            return "parts and residual differ in shape"
        shp = next(iter(shapes))
        if shp is None or len(shp) != 2 or shape(ctx, weight) != (shp[1],):
            return f"expected [M,N] parts/residual and [N] weight, got {shp}, {shape(ctx, weight)}"
        if not isinstance(eps, (int, float)) or isinstance(eps, bool):
            return "eps is not a static float"
        _plan, why = _coverage(parts, node, ctx, fused=True)
        return f"numerics_unregistered: {why}" if why else True

    def emit(self, node, ctx):
        parts = _tp_parts(node)
        colls = [ctx.coll(a, f"part_rank{i}") for i, a in enumerate(parts)]
        rows, leads = zip(*(rows_of(c) for c in colls))
        residual_rows, lead = rows_of(ctx.coll(node.args[1], "residual"))
        weight, eps = ctx.coll(node.args[2], "weight"), float(node.args[3])
        plan, why = _coverage(parts, node, ctx, fused=True)
        assert plan is not None and why is None
        M, N, W = plan["M"], plan["N"], len(parts)
        shot = tp_allreduce.mnnvl_shot(M, N, W)
        _block, cluster = tp_allreduce.mnnvl_grid(M, N)
        fn = ctx.kind("all_reduce_rmsnorm_mnnvl", node, WORLD=W, N=N, SHOT=shot, CLUSTER=cluster, EPS=eps,
                      FMA=bool(plan["captured"].statics["FMA"]))
        hint = _collective_hint(node, ctx)
        outs = [ctx.call(fn, array_of([rows[r][i] for r in range(W)]), residual_rows[i], weight,
                         node=node, rule=self, name=f"{hint}[{i}]/fused_all_reduce_rmsnorm") for i in range(M)]
        ctx.define(node, [from_rows([o[0] for o in outs], lead), from_rows([o[1] for o in outs], lead)])
        ctx.record(self, node, {"N": N, "rows": M, "world": W, "setting": plan["setting"], "route": "fused",
                               "coverage": plan["captured"]._asdict()}, emitted=f"{fn.id} x{M}", collective="all_reduce")


class AllGatherRule(OpRule):
    """[lane vllm-tp-n] `verity_tp::all_gather([part_rank0, .., part_rank{W-1}] each [.., N] bf16)` -> `all_gather{WORLD=W, N}` (`AllGather_v1{WORLD,N}`:
    the [W*N] row IS the W shards side by side, a copy) per row, W > 2 (world 2 is `AllGather2Rule`)."""
    modes = frozenset({"profile"})
    info = RuleInfo("AllGather[tp]", 1, "op-binding", applies_to="verity_tp::all_gather([a_0 .. a_{W-1}] each [.., N] bf16) (rank order, trailing axis), W > 2",
                    preconditions=("every operand is a bf16 value collection of identical shape", "world size > 2"),
                    evidence_domain="structural; numerics: AllGather_v1{WORLD,N} (concatenation; no gates)")

    def targets(self):
        return _tp_op("all_gather")

    def applies(self, node, ctx):
        parts = _tp_parts(node)
        if len(parts) <= 2:
            return f"world {len(parts)}: the two-rank collective is AllGather2 (recorded identity)"
        for i, a in enumerate(parts):
            r = _row_value(ctx, a, f"part_rank{i}")
            if r:
                return r
        shapes = {shape(ctx, a) for a in parts}
        if len(shapes) != 1:
            return f"shards differ in shape ({sorted(map(str, shapes))})"
        return True

    def emit(self, node, ctx):
        parts = _tp_parts(node)
        W = len(parts)
        colls = [ctx.coll(a, f"part_rank{i}") for i, a in enumerate(parts)]
        rows, lead = zip(*(rows_of(c) for c in colls))
        rows = list(rows)
        N = rows[0][0].type.n
        fn = ctx.kind("all_gather", node, WORLD=W, N=N)
        hint = _collective_hint(node, ctx)
        outs = [ctx.call(fn, array_of(list(xs)), node=node, rule=self, name=f"{hint}[{i}]/all_gather") for i, xs in enumerate(zip(*rows))]
        ctx.define(node, from_rows(outs, lead[0]))
        ctx.record(self, node, {"N": N, "rows": len(rows[0]), "world": W}, emitted=f"{fn.id} x{len(rows[0])}", collective="all_gather")


class EmbeddingShardRule(OpRule):
    """`verity_tp::embedding_shard(ids, shard [VS, H] bf16, START, END)` -> `embedding_shard{VS,H,START}` (`EmbeddingShard_v1`: the row
    `shard[id - START]` if START <= id < END else zeros) per token — one rank of vLLM's `VocabParallelEmbedding` (`_C.vocab_parallel_embedding`);
    the shard is this rank's committed weight input (role `embedding.table` at [VS, H])."""
    modes = frozenset({"profile"})
    info = RuleInfo("EmbeddingShard[tp]", 1, "op-binding", applies_to="verity_tp::embedding_shard(ids I32 value, shard [VS,H] bf16, START, END)",
                    preconditions=("shard is a bf16 value collection [VS, H] with VS == END - START", "ids is a value collection of 32-bit token ids"),
                    evidence_domain="structural; numerics: EmbeddingShard_v1 (masked shard gather; exact-model-tested)")

    def targets(self):
        return _tp_op("embedding_shard")

    def applies(self, node, ctx):
        ids, table, start, end = node.args[:4]
        r = _row_value(ctx, table, "shard")
        if r:
            return r
        shp = shape(ctx, table)
        if shp is None or len(shp) != 2:
            return "shard is not 2-D"
        if not isinstance(start, int) or not isinstance(end, int) or end - start != shp[0]:
            return f"START/END ({start}, {end}) do not span the shard's VS={shp[0] if shp else None}"
        v = ctx.val(ids)
        if isinstance(v, StaticIndex):
            return "ids are static structure"
        if not isinstance(v, Coll):
            return "ids is not a value collection"
        if shape_of(v.type)[1] != 32:
            return f"numerics_unregistered: ids carry {shape_of(v.type)[1]}-bit leaves; EmbeddingShard_v1 takes I32 token ids"
        return True

    def emit(self, node, ctx):
        ids, table = ctx.coll(node.args[0], "ids"), ctx.coll(node.args[1], "shard")
        start = int(node.args[2])
        VS, H = shape_of(table.type)[0]
        shp = shape_of(ids.type)[0]
        n = 1
        for s in shp:
            n *= s
        flat = ids.reshape(Array(n, Value(32)))
        fn = ctx.kind("embedding_shard", node, VS=VS, H=H, START=start)
        outs = [ctx.call(fn, flat[i], table, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]/shard") for i in range(n)]
        ctx.define(node, array_of(outs).reshape(_tensor_type(tuple(shp) + (H,), 16)))
        ctx.bind_weight_role(node.args[1], self, "embedding.table", V=VS, H=H)
        ctx.record(self, node, {"VS": VS, "H": H, "START": start, "tokens": n}, emitted=f"{fn.id} x{n}")
