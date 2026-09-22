"""[R17 tp, COORD R17-46 (Daniel's cp-tp queue: llama TP2 B8 -> qwen TP2 B8 -> Qwen3-MoE TP2 B2)] the rank-LOCAL configuration of ONE
rank of Qwen2.5-1.5B at tensor_parallel_size=2 -- the κ profile the D90 per-rank fold Match (`tp2_fold_match`) folds a rank log
(`tp2_capture --tp 2 --record-collectives` -> `log.rank<r>.jsonl.gz`) with, exactly as `gen_llama_llama32_1b_tp2rank` does for Llama.

The rank holds the shards vLLM's parallel layers give it: NH/2 = 6 heads + KVH/2 = 1 kv head (QKVParallelLinear rows, o_proj's K), I/2 = 4480
(gate_up rows, down_proj's K), V/2 = 75,968 (VocabParallelEmbedding / the TIED ParallelLMHead rows); H = 1536, the norms, the rotary table and
NL = 28 are replicated.  Everything model-level is `b1.B1_CONFIG` (the checkpoint pins of `vllm_d9105ea80_sm89_eager_qwen15`) with those four
divided and the two derived widths recomputed: QKV = (NH + 2 KVH) D = 1024 (so the fused projection's separate `aten.add` is
`BiasAdd_v1{N=1024}` -- Qwen2's QKV bias is sharded with the rows), GU = 2 I = 8960.  FA2 static BN = fa2_kblock_n(128) = 64 (head dim 128,
unchanged by the shard).

Rank facts the shard configuration alone does not state (shared with the Llama rank profile, imported from it):
  * `EmbeddingShardCuda`: vLLM's fused `_C.vocab_parallel_embedding` IS `EmbeddingShard_v1{VS, H, START}(ids, shard)` -- the rank-local
    partial the embedding all-reduce consumes;
  * `TpCollective(world=2)` [D90, COORD R17-4]: `vllm.all_reduce.default` / `vllm.all_gather.default` -> `AllReduce2_v1{N}` /
    `AllGather2_v1{N}` per row over (own partial, peer partial) in RANK order, the peer partial the fold's `peers_<request>` ROOT PARAMETER
    (root class `collective_received`, the other rank's committed partial, linked by `tp_links.json`, checked by the cross-rank leg) and the
    received storage the instance's OUTPUT -- never a free prescribed input;
  * the sampler runs over the FULL vocabulary (the all-gathered logits): `TokenSelect*` patterns take `vocab = V`, not V/2.
"""

from __future__ import annotations

from verity_vllm.registry import b1

from ..patterns import TokenSelectArgmax, TokenSelectGumbelTwoStage
from ..tp.collective_pattern import TpCollective
from .gen_llama_llama32_1b_tp2rank import EmbeddingShardCuda
from .vllm_d9105ea80_sm89_eager import build_profile, make_check_header, make_root_class, make_weight_field, with_geometry  # noqa: F401

CASE = "B1"
WORLD = 2
PROFILE_ID = f"vllm-d9105ea80-sm89-eager-bf16-qwen2.5-1.5b-tp{WORLD}-rank"

FULL: dict = dict(b1.B1_CONFIG)
#: the rank-local shard configuration: heads, kv heads, intermediate and vocab rows divided by the world; QKV / GU recomputed from them
RANK_CONFIG: dict = dict(FULL)
for _k in ("NH", "KVH", "I", "V"):
    assert FULL[_k] % WORLD == 0, (_k, FULL[_k], WORLD)
    RANK_CONFIG[_k] = FULL[_k] // WORLD
RANK_CONFIG["QKV"] = (RANK_CONFIG["NH"] + 2 * RANK_CONFIG["KVH"]) * RANK_CONFIG["D"]
RANK_CONFIG["GU"] = 2 * RANK_CONFIG["I"]
C = with_geometry(RANK_CONFIG)      # BN = 64 (head dim 128)
assert (C["H"], C["NL"], C["NH"], C["KVH"], C["D"], C["I"], C["V"], C["QKV"], C["GU"]) == (1536, 28, 6, 1, 128, 4480, 75968, 1024, 8960), C
assert C["BN"] == 64 and C["BIAS"] is True and C["tie_word_embeddings"] is True, C

weight_field = make_weight_field(C)
root_class = make_root_class(weight_field)
check_header = make_check_header(C)

PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note=f"Qwen2.5-1.5B @ {FULL['revision'][:8]} ONE RANK of TP{WORLD} (rank-local shard configuration): H={C['H']} NL={C['NL']} "
                                   f"NH={C['NH']} KVH={C['KVH']} D={C['D']} I={C['I']} V={C['V']} QKV={C['QKV']} (sharded QKV bias -> BiasAdd_v1{{{C['QKV']}}}) "
                                   f"GU={C['GU']}, FA2 kBlockN 64; collectives via TpCollective (D90)")
PROFILE.patterns = [EmbeddingShardCuda(), TpCollective(world=WORLD)] + list(PROFILE.patterns)
# the sampler sees the all-gathered logits: the full vocabulary, not the rank's V/2 shard
V_FULL = int(FULL["V"])
for _p in PROFILE.patterns:
    if isinstance(_p, (TokenSelectGumbelTwoStage, TokenSelectArgmax)):
        _p.vocab = V_FULL
PROFILE.notes["tp_rank"] = {"world": WORLD, "rank_config_divided": ["NH", "KVH", "I", "V"], "recomputed": {"QKV": C["QKV"], "GU": C["GU"]},
                            "sampler_vocab": V_FULL, "own_patterns": [EmbeddingShardCuda.name, TpCollective.name],
                            "base": "vllm_d9105ea80_sm89_eager_qwen15 (b1.B1_CONFIG pins) / gen_llama_llama32_1b_tp2rank (the rank facts)",
                            "collectives": "[R17 D90] vllm.all_reduce / vllm.all_gather -> AllReduce2_v1{N} / AllGather2_v1{N} per row over (own partial, "
                                           "peer partial) in rank order; the peer partial is the fold's `peers_<request>` root parameter (root class "
                                           "collective_received), linked to the other rank's committed partial by the cross-rank leg (tp2_rank_match)"}
