# verity_vllm.program.frontend.rules

The lowering rules of the Build phase. A rule takes one node of a model's exported ATen graph and emits one Call of a
registered Definition, or refuses the node with a named reason. `DEFAULT_RULES` is the versioned default rule set: the
members of `TORCH_RULES`, whose rules target ATen ops only, and of `VLLM_RULES`, whose rules each target an op of
vLLM's. `vllm_bindings.vllm_ruleset()` is the default with the bindings of vLLM's custom ops, and `vocab` holds the
kinds a rule may emit under each profile. Each module's docstring says what it holds.

## What other code imports

A module's `__all__` lists the names it gives to library code outside the package, and across one line inside it: the
reference, pattern and training rules (`reference`, `ref_kinds`, `patterns` and the `training` modules) read the
package's other modules through these lists, and are read through their own. On either side of that line a module
reads its neighbours freely. The package itself gives the rule
sets (`DEFAULT_RULES` and the lists it is made of, which load at first use) and `vocabulary_for`; `base` gives the
types a rule is written in (`RuleSet`, `RuleInfo` and the rule classes); `vllm_bindings` gives `vllm_ruleset`,
`VLLM_BINDING_RULES`, the pins of the admitted kernels and what a Build observes on its host (`observe_*`,
`clear_observations`, `declared_derived_accounting`); the modules of one op family each give that family's rules with
the pins, geometry tables and helpers other packages read. A module that states no `__all__` is the package's own. Six
private names are read from outside the package or across that line all the same, and are not declared. Four are read
from outside: `family._field_intervals`, which a benchmark reads too, `vllm_bindings.bf16_gemm._f32out_phase`,
`vllm_bindings.collectives._ROUTE_KINDS` and `vocab._dense_kinds`. Two are read by `patterns`: `views._STATIC_ARITH`
and `views._STATIC_MAKERS`.
