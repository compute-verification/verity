"""[R15 tp, moved here by P6 from `profiles/gen_llama_llama32_1b_tp2rank.py`] The rank-local fused embedding lookup of a tensor-parallel
rank: vLLM's `_C.vocab_parallel_embedding(out[M,H], ids[M], shard[VS,H], START, END, padding, added_start, added_end)` (the fused kernel of
`VocabParallelEmbedding` at tp_size > 1: `out[i] = shard[ids[i]-START]` if START <= ids[i] < END else 0) IS
`EmbeddingShard_v1{VS,H,START}(ids, shard)` (the registered TP2 Definition, `verity_vllm.program.registry.b1_tp2`) -- the rank-local partial the
embedding all-reduce consumes; one operand set, entirely in this rank's log.  Refuses padded shards and added-vocab ranges (not what
`EmbeddingShard` states; derive_step's `bind_vocab_parallel_embeddings` refuses the same).  A `world > 1` fact of the architecture, not of
any model: `profiles.generic.build_profile(role, world=N)` registers it for every rank profile."""

from __future__ import annotations

from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import b1_tp2

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import Embedding, PerRow, Shared, attribution_for, base_op, bound_args, hint_of, is_int, is_row_major, unsupported
from verity_vllm.observe.fold.resolver import Compute, Resolution, StepContext
from verity_vllm.observe.fold.tree import TreeNode


class EmbeddingShardCuda:
    """`_C.vocab_parallel_embedding(...)` -> `Compute(EmbeddingShard_v1{VS,H,START}, rows=M, (PerRow(input_ids), Shared(shard)), (PerRow(out),))`."""

    name: str = "tp.embedding_shard_cuda"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
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
