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

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.patterns import PerRow, attribution_for, base_op, bound_args, hint_of, is_row_major, unsupported
from verity_vllm.observe.resolver import Collective, Compute, Resolution, StepContext
from verity_vllm.observe.tree import TreeNode

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
        return Compute(spec, M, (Collective(x, kind, site),), (PerRow(out),), tuple(sorted(node.ids())),
                       hint=hint_of(node, kind), kernels=(), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)
