# verity_torch.frontend: design

What someone working on the translator needs beyond the [README](README.md) and each module's docstring.

## What other packages import

A module of this directory that another package imports from declares those names in its `__all__`; a name outside
that list is the translator's own. The package itself gives no name: each is imported from its module, and importing a
module loads only what that module imports. From the vLLM adapter: `derive.derive_report`, the entry point, which
its `vllm_derive` calls with vLLM's rules, vocabulary and served rules, and `boolean.derived` (`pipeline`); `report`'s `DeriveReport` and `DerivationTail`, and the walk's types and refusals in
`derive`, `outcomes` and `value_types` (`program`). From outside the adapters: `benchmarks/training_slowdown` reads `train_step`'s `step_program`, `export_steps`,
`primitives` and `AdamW`, and `train_trace`'s `CaptureRefused`, `ATTENTION` and `BACKWARD`. The rule files of both adapters and the tests also read private names of `derive`, `strided_views` and
`outcomes`; those are not part of what the package offers. `rules/` is imported from outside the package too and is
not listed here.
