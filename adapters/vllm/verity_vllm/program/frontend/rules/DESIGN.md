# verity_vllm.program.frontend.rules: design

What someone working on vLLM's rules needs beyond the [README](README.md) and each module's docstring.

## What other code imports

A module's `__all__` lists the names it gives to library code outside the package. The package itself gives the rule
sets (`DEFAULT_RULES` and the lists it is made of, which load at first use) and `vocabulary_for`; `vllm_bindings` gives
`vllm_ruleset`, `VLLM_BINDING_RULES`, the pins of the admitted kernels and what a Build observes on its host
(`observe_*`, `clear_observations`, `declared_derived_accounting`); the modules of one op family each give that
family's rules with the pins, geometry tables and helpers other packages read. A module that states no `__all__` is
the package's own. Four private names are read from outside the package all the same, and are not declared:
`family._field_intervals`, which a benchmark reads too, `vllm_bindings.bf16_gemm._f32out_phase`,
`vllm_bindings.collectives._ROUTE_KINDS` and `vocab._dense_kinds`.
