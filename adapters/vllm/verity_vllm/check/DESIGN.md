# verity_vllm.check: design

## What other code imports

A module's `__all__` lists the names that the adapter's other packages and code outside the adapter import from it. A
module without one is the package's own, but for `c2_rules`: other code reads two names of it, both with a leading
underscore, so it states none. The Commit's side: `verdict` gives
`commit_verdict` and `from_record`; `commit_rules` the required classes and their coverage; `commit_tokens`
`account_links`; `cublas_route_pin` the route pin's gate and its stamp on the Commit record; `oracle_compare` the
compare, the Match record's reader (`MatchOracle`), the producers of the members it reads, and the reader of opened
values that it hands on from `commit.opened`; `value_check` the value check of the native rows. The weights:
`weights_of_record` gives the record's derivation, its `check` against the served parameters, a field's composition
and the `weights-of-record` command; `checkpoint_index` the shards of a checkpoint of record and a shard's header;
`derived_weights` and `fp8_ue8m0` the weights vLLM makes or rewrites at load; `fp8_moe_scales` and `tp_shards` what a
TP rank holds of a tensor; `component_programs` a row's Programs on disk, by shape.
The Match's side: `gates` gives the gates, their ids
and the gates document; `verdict_names` gives the ids alone, each by what its gate checks, and imports nothing, for a
reader of a gates document that must not load the gates; `batch` the batch invariance of a template; `program_ops`
the operations the Match's checks run on Programs; `codes` the verdict codes; `fold_compare` the `fold-compare` command,
the loaders of a profile and of a folded log, and a capture's snapshots with the values read through them
(`SnapshotStore`, `Values`). `oracle_compare`, `weights_of_record`,
`gates` and `fold_compare` hand on names whose code is in `match_oracle`, `producers`, `composed_weights`,
`engine_constants`, `gate_base`, `gate_rules` and `fold_sources`; other code reads them through the first four.
Six names with a leading underscore are read by `pipeline` and `program.predict` and are not declared:
`verdict._c1_verdict` and `_after_release_verdict`, `c2_rules._query_population_scope` and
`_complete_replay_population_gap`, `weights_of_record._check_analytic` and `_default_manifest`. `benchmarks/one_stage`
reads four more, of `fold_compare`: `_NP`, `_attach_specs`, `_component_ranges` and `_in_place`.
