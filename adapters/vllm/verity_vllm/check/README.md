# verity_vllm.check

The checks of a row and the verdicts they give. `gates` evaluates a Match's gates over the records of one run
directory. `verdict` writes a Commit's verdict, one document beside the files it is derived from, by folding the rules
of `commit_rules` and `c2_rules`, each a `result.CheckResult`; `executed_prefix` and `commit_tokens` link a request's
executed prefix to the Match account and to the Commit's token record. `weights_of_record` derives a digest of every
weight field of a Program from the bytes of the checkpoint of record and compares the served parameters with it, and
`oracle_compare` compares a Commit's committed values with the Match record's. `match/` holds the checks of a Match
and `replay/` the sampled replay of a Commit. Each module's docstring says what it holds.

## What other code imports

A module's `__all__` lists the names that the adapter's other packages and code outside the adapter import from it. A
module without one is the package's own, but for two that other code reads and that state none: `c2_rules`, and
`fold_compare`, which the `fold-compare` command runs. The Commit's side: `verdict` gives
`commit_verdict` and `from_record`; `commit_rules` the required classes and their coverage; `commit_tokens`
`account_links`; `cublas_route_pin` the route pin's gate and its stamp on the Commit record; `oracle_compare` the
compare, the Match record's reader (`MatchOracle`), the producers of the members it reads, and the reader of opened
values that it hands on from `commit.opened`; `value_check` the value check of the native rows. The weights:
`weights_of_record` gives the record's derivation, its `check` against the served parameters, a field's composition
and the `weights-of-record` command; `checkpoint_index` the shards of a checkpoint of record; `derived_weights` and
`fp8_ue8m0` the weights vLLM makes or rewrites at load; `fp8_moe_scales` and `tp_shards` what a TP rank holds of a
tensor; `component_programs` a row's component Programs on disk. The Match's side: `gates` gives the gates, their ids
and the gates document; `verdict_names` gives the ids alone, each by what its gate checks, and imports nothing, for a
reader of a gates document that must not load the gates; `batch` the batch invariance of a template; `program_ops`
the operations the Match's checks run on Programs; `codes` the verdict codes. `oracle_compare`, `weights_of_record` and `gates`
hand on names whose code is in `match_oracle`, `producers`, `composed_weights`, `engine_constants`, `gate_base` and
`gate_rules`; other code reads them through the first three.
Six names with a leading underscore are read by `pipeline` and `program.predict` and are not declared:
`verdict._c1_verdict` and `_after_release_verdict`, `c2_rules._query_population_scope` and
`_complete_replay_population_gap`, `weights_of_record._check_analytic` and `_default_manifest`.
