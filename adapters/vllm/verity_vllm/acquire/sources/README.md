# verity_vllm.acquire.sources

The acquisition sources. A source is an object a Commit attaches to its committer: it takes values from the engine
while it runs and hands them to the committer during a committed step. Each module's docstring says which values its
source holds and where it takes them; `taps` attaches the sources a Commit's manifest names.

## Interface

A module's `__all__` lists the names code outside the package may import from it. What the integration's other
packages and the benchmarks import, module by module: from `taps`, `TapSources`, `attach`, `attached`, `warmup`,
`for_ranks` and `require_on_every_rank`; from `hidden_source`, `HiddenM1Source`, `HiddenM1SourceFA3`,
`make_hidden_source`, `attention_hidden`, `with_hidden_class`, `CHUNK_WORDS`, `KBLOCK_M` and the stream's geometry
(`fa2_kblock_n`, `fa2_swapped`, `fa2_tap_buffer_words`, `fa2_tap_words_per_seq`, `fa3_tile`, `mat_total_words`,
`plan_fa2_tap_groups`); from `moe_source`, `MoeBlockSource` and `plane_layer`; from `moe_align_source`, `arm_replay`,
`capture_witness` and `replay_stats`; from `partial_source`, `TPPartialSource` and `late_read_check`; from
`call_boundary_plan`, `Plan`, `to_json` and `of_json`; from `call_boundary_source`, `FAMILY`, `manifest_sites`,
`plan_of` and `unattached`; from `int8_source`, `Int8Source`; from `attention_source`, `AttentionSource`, `MLA_ROWS`,
`INDEX_ROWS` and `INDEX_SCALE`; from `served_compiled_source`, `ServedCompiledSource` and `linkage`; from
`compiled_source`, `compile_settings`; from `inductor_hooks`, `FLUSH`; from `oracle_witness`, `SUMMARY_KEYS`; from
`allreduce_norm`, `FUSED` and `REDUCED_MEMBER`. One private name is read from outside the package and is not
declared: `_committed`, of `partial_source` and of `served_compiled_source`. No other package imports the remaining
modules.
