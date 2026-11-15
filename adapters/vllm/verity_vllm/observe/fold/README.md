# verity_vllm.observe.fold

The fold: a raw observation log and a profile become a record Program. Its modules are the resolver, the versioned
memory, the kernel patterns (`patterns/`) and the profiles (`profiles/`); `fold.py`'s docstring has the contract.

## Interface

What the integration's other packages and the benchmarks import from this package, module by module. From `fold`:
`fold`, `FoldResult` and `summarize`. From `resolver`: `Compute`, `Transparent`, `Profile` and `parts_of`. From
`resolve_log`: `resolve_log` and `load_profile`, and the `resolve-log` command runs its `main` over its `Options`. From
`tree`: `step_events`. From `supplied`: `program_registered_values`. From `patterns.collectives`: `allreduce_settings`;
from `patterns.sampling`: `with_sampler_construction`; from `patterns.pouw`: `with_protocols`. From `profiles`:
`profile_for_case` and `profile_module_for_case`, and a profile is also imported by its module's name,
`verity_vllm.observe.fold.profiles.<name>`. From `profiles.generic`: `derive_config`, `effective_sliding_window`,
`module_name`, `rank_hf_config`, `router_vpt_bf16` and `structural_refusal`, and `program.predict.dsa` also reads its
private `_MLA_SCALES` and `_mla_weight_field`. Each of these modules but `profiles.generic` declares its names in
`__all__`. The other modules that declare one (`patterns` and five of its modules, two of `profiles`) declare it for
the fold's own files and the tests: no library module outside the fold imports a name from them.
