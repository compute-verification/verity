# verity_vllm.program.frontend.rules

The lowering rules of the Build phase for vLLM's own ops. A rule takes one node of a model's exported ATen graph and
emits one Call of a registered Definition, or refuses the node with a named reason. `DEFAULT_RULES` is the versioned
default rule set: the members of the PyTorch adapter's `TORCH_RULES` (`verity_torch.frontend.rules`), whose rules
target ATen ops only, and of `VLLM_RULES`, whose rules each target an op of vLLM's. `vllm_bindings.vllm_ruleset()` is
the default with the bindings of vLLM's custom ops, and `vocab` fills the PyTorch adapter's table of the kinds a rule
may emit with vLLM's profiles. Each module's docstring says what it holds.

## What other code imports

A module's `__all__` lists the names it gives to library code outside the package. The package itself gives the rule
sets (`DEFAULT_RULES` and the lists it is made of, which load at first use) and `vocabulary_for`; `vllm_bindings` gives
`vllm_ruleset`, `VLLM_BINDING_RULES`, the pins of the admitted kernels and what a Build observes on its host
(`observe_*`, `clear_observations`, `declared_derived_accounting`); the modules of one op family each give that
family's rules with the pins, geometry tables and helpers other packages read. A module that states no `__all__` is
the package's own. Four private names are read from outside the package all the same, and are not declared:
`family._field_intervals`, which a benchmark reads too, `vllm_bindings.bf16_gemm._f32out_phase`,
`vllm_bindings.collectives._ROUTE_KINDS` and `vocab._dense_kinds`.
