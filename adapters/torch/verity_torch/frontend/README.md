# verity_torch.frontend

The Torch-to-circuit translator: a traced torch model becomes a Program in which every Call names a registered
Definition. `derive` exports the model and walks its graph (the walk's state is `derive_context`; a functional ATen
graph, the one `torch.func` and `make_fx` produce, is `derive_graph`'s), `rules/` holds the rules on ATen ops the walk
applies, `report` is what a derivation produced, `outcomes` and `errors` are its refusals, `boolean` gives its Program
on bits, and `emit` writes a finished derivation's runtime correspondence. `train_step` derives one training step as a
Program, and consecutive steps with torch's own optimizers, which `train_trace` traces as one graph. Each module's docstring says what it holds.

## What other packages import

A module of this directory that another package imports from declares those names in its `__all__`; a name outside
that list is the translator's own. The package itself gives no name: each is imported from its module, and importing a
module loads only what that module imports. From the vLLM adapter: `derive.derive_report`, the entry point, which
its `vllm_derive` calls with vLLM's rules, vocabulary and served rules, and `boolean.derived` (`pipeline`); `report`'s `DeriveReport` and `DerivationTail`, and the walk's types and refusals in
`derive`, `outcomes` and `value_types` (`program`). From outside the adapters: `calibration/training` reads `train_step`'s `step_program`, `export_steps`,
`primitives` and `AdamW`, and `train_trace`'s `CaptureRefused`, `ATTENTION` and `BACKWARD`. The rule files of both adapters and the tests also read private names of `derive`, `strided_views` and
`outcomes`; those are not part of what the package offers. `rules/` is imported from outside the package too and is
not listed here.
