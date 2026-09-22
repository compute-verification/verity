"""[R15 tp, M-0204 (a)/(b) DIAGNOSTIC] the rank-LOCAL configuration of one rank of Llama-3.2-1B at tensor_parallel_size=2 -- what ONE
rank's aten stream computes between the collectives -- as a κ profile for the protected fold / resolver, so the per-rank raw log
(`tp2_capture --tp 2` -> `log.rank<r>.jsonl.gz`) can be resolved and folded as far as the TP1 machinery reaches, and the exact gap
for the (b) fold rule (M-0346) is MEASURED rather than argued.

The rank holds the shards vLLM's parallel layers give it: heads NH/2 + KVH/2 (QKVParallelLinear rows, o_proj's K), I/2 (gate_up rows,
down_proj's K), V/2 (VocabParallelEmbedding / ParallelLMHead rows); H, the norms, the rotary table and NL are replicated.  Those are
exactly the shapes `analytic.config_of` derives from the checkpoint config with `num_attention_heads`, `num_key_value_heads`,
`intermediate_size`, `vocab_size` divided by the world -- so the rank profile is the base module's `build_profile` over that derived
configuration (the weight map, the pattern list and the κ pins are the shared Llama ones; the o_proj tuned-GEMM specialization is keyed
(N,K)=(2048,2048) and does not apply at the shard shape (2048,1024) -- left in the list, it simply never matches).

Two rank facts the shard configuration alone does not state, added here:
  * the rank's own patterns: `EmbeddingShardCuda` -- vLLM's fused `_C.vocab_parallel_embedding(out, ids, shard, START, END, 0, V, V)` IS
    `EmbeddingShard_v1{VS=END-START, H, START}(ids, shard)` (the registered TP2 Definition, `verity_vllm.registry.b1_tp2`; padded / added-vocab
    shards refused) -- the rank-local partial the embedding all-reduce consumes; one operand set, entirely in this rank's log;
  * the sampler runs over the FULL vocabulary (the all-gathered logits): `TokenSelect*` patterns take `vocab = V`, not V/2.

[R17 D90, COORD R17-4] The collectives (`vllm.all_reduce.default` / `vllm.all_gather.default`, pynccl behind vLLM's custom op) resolve
through `tp.collective_pattern.TpCollective`: `AllReduce2_v1{N}(parts)` / `AllGather2_v1{N}(parts)` per row with `parts` = (own partial,
peer partial) in RANK order, the peer partial being the fold's `peers_<request>` ROOT PARAMETER (root class `collective_received`,
D90 -- the other rank's committed partial, linked by the rank Program's `tp_links.json` and checked by the cross-rank leg), and the received
storage the instance's OUTPUT.  Before R17 this profile had no pattern for them and a rank log yielded Unsupported exactly at the
collective sites (33 all_reduce + 1 all_gather per step for Llama-3.2-1B; the R15/R16 record `out/gen/sweep/lanes/tp-STATUS.md` cites).
"""

from __future__ import annotations

import json
from pathlib import Path

from verity.ir.defs import bind
from verity_vllm.registry import b1_tp2

from .. import analytic
from ..patterns import Embedding, PerRow, Shared, TokenSelectArgmax, TokenSelectGumbelTwoStage, attribution_for, base_op, bound_args, hint_of, is_int, is_row_major, unsupported
from ..resolver import Compute, Resolution, StepContext
from ..events import ViewDesc
from ..tree import TreeNode
from .gen_llama_llama32_1b import CONFIG_JSON, REPO_ID, REVISION
from .vllm_d9105ea80_sm89_eager import build_profile, make_check_header, make_root_class, make_weight_field, with_geometry  # noqa: F401


class EmbeddingShardCuda:
    """`_C.vocab_parallel_embedding(out[M,H], ids[M], shard[VS,H], START, END, padding, added_start, added_end)` (vLLM's fused kernel for
    `VocabParallelEmbedding` at tp_size > 1: `out[i] = shard[ids[i]-START]` if START <= ids[i] < END else 0) ->
    `Compute(EmbeddingShard_v1{VS,H,START}, rows=M, (PerRow(input_ids), Shared(shard)), (PerRow(out),))`.  Refuses padded shards and
    added-vocab ranges (not what `EmbeddingShard` states; derive_step's `bind_vocab_parallel_embeddings` refuses the same)."""

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

CASE = "LLAMA32_1B"
WORLD = 2
PROFILE_ID = f"vllm-d9105ea80-sm89-eager-bf16-llama3.2-1b-tp{WORLD}-rank"

HF_CONFIG: dict = json.loads(Path(CONFIG_JSON).read_text())
#: the rank-local shard configuration: heads, kv heads, intermediate and vocab rows divided by the world (H, NL, norms replicated)
RANK_CONFIG: dict = dict(HF_CONFIG)
for _k in ("num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"):
    assert HF_CONFIG[_k] % WORLD == 0, (_k, HF_CONFIG[_k], WORLD)
    RANK_CONFIG[_k] = HF_CONFIG[_k] // WORLD
C, PROVENANCE = analytic.config_of(RANK_CONFIG, model=REPO_ID, revision=REVISION)
C = with_geometry(C)
assert C["family"] == "llama" and C["D"] == 64 and C["BN"] == 128, (C["family"], C["D"], C["BN"])
assert (C["H"], C["NH"], C["KVH"], C["I"], C["V"]) == (2048, 16, 4, 4096, 64128), C

weight_field = make_weight_field(C)
root_class = make_root_class(weight_field)
check_header = make_check_header(C)

PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note=f"Llama-3.2-1B @ {REVISION[:8]} ONE RANK of TP{WORLD} (rank-local shard configuration): H={C['H']} NL={C['NL']} "
                                   f"NH={C['NH']} KVH={C['KVH']} D={C['D']} I={C['I']} V={C['V']} -- diagnostic profile for the per-rank fold gap "
                                   f"(collectives / masked embedding / logits all-gather deliberately unmatched, M-0346)")
# the rank's fused embedding shard lookup (before the base Embedding pattern, which does not match the fused op anyway) and, [R17 D90,
# COORD R17-4], the rank's collectives: `vllm.all_reduce` / `vllm.all_gather` -> AllReduce2_v1 / AllGather2_v1 per row over
# (own partial, peer partial = the `peers_<request>` root parameter) in rank order; the received storage is the instance's OUTPUT
from ..tp.collective_pattern import TpCollective  # noqa: E402

PROFILE.patterns = [EmbeddingShardCuda(), TpCollective(world=WORLD)] + list(PROFILE.patterns)
# the sampler sees the all-gathered logits: the full vocabulary, not the rank's V/2 shard
V_FULL = int(HF_CONFIG["vocab_size"])
for _p in PROFILE.patterns:
    if isinstance(_p, (TokenSelectGumbelTwoStage, TokenSelectArgmax)):
        _p.vocab = V_FULL
PROFILE.notes["tp_rank_diagnostic"] = {"world": WORLD, "rank_config_divided": ["num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"],
                                       "sampler_vocab": V_FULL, "own_patterns": [EmbeddingShardCuda.name, TpCollective.name],
                                       "collectives": "[R17 D90] vllm.all_reduce / vllm.all_gather -> AllReduce2_v1{N} / AllGather2_v1{N} per row over "
                                                      "(own partial, peer partial) in rank order; the peer partial is the fold's `peers_<request>` root "
                                                      "parameter (root class collective_received), linked to the other rank's committed partial by the "
                                                      "cross-rank leg (tp2_rank_match) -- never a free prescribed input"}
