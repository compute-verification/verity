# verity_vllm.check.match

The Match stage's checks: a recorded execution's fold against the workload Program a Build declared. Each module's
docstring says what it holds.

## Interface

- **Provides:** the commands `verity-vllm global-match`, `batch-decomp`, `program-compare` and `program-dag`; the
  cross-rank checks of a tensor-parallel Match, that the ranks' folds state the same collectives, and the binding of a
  rank's Commit to them (`rank_match`); the sampled replay of the collectives' outputs from the ranks' committed
  partials (`xrank_collectives`); and the shape a derived directory states (`derived_index`).
- **Loaded by:** the integration's `pipeline`: its command line, the Build (`components`, `global_program`) and the
  tensor-parallel stages (`tp`). Otherwise it is run, as those commands.
- **Loads:** `verity.circuits` and `verity.definitions`; the integration's `program`, `records`, `observe`, `query`,
  `engine`, `config`, `settings` and `log`, and `check.program_ops` and `check.replay`; numpy, and torch inside the
  sampled replay. Two commands' `main` still read `pipeline.gc_tuning.tune` inside the function; the layering list holds
  them.

What the integration's other packages import, module by module; each module lists these names in its `__all__`.
`Options` and `main` of `global_match`, `batch_decomp` and `program_dag`, which the command line takes from a
command's module. `derived_shape` of `derived_index`, which `batch_decomp` hands on under the same name.
`cross_rank_check`, `commit_binding`, `load_recorder`, `own_producers_from_record` and `same_module_path` of
`rank_match`; the last says when two module paths name one module, and `pipeline.tp.links` hands it on under the same
name. `ROUTED_FAMILIES`, `pick_strata`, `check_picks`, `fold_pairs`, `coverage_of_unreplayable`, `routed_expected`,
`is_embedding_site`, `spec_key` and `words` of `xrank_collectives`. `MODELLED_ENGINE` and `compilation_facts` of
`declaration`: the engine conditions a workload Program models and the facts of a declared compilation block, which
`pipeline.global_program` hands on under the same names. `program_compare` is a command too and declares no `__all__`
yet. No file outside the integration imports this package.
