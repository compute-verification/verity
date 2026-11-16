# verity_vllm.protocols

The protocol options of a vLLM run, each one adapter module written against `interface.py` (`spot.py` for SPOT,
`pous.py`, and `pouw/`), the linear site their executors share (`linear/`), and the node's registered store that PoUB
audits, with the hooks that keep it current while vLLM serves. The package's docstring has the order the options
install in, their phases and what a run is refused for.

## Interface

What the integration's other packages, the benchmarks and `verity`'s own tests import from this package, module by
module. From the package itself: `Context`, `Refused`, `WeightsView`, `adapter`, `driver`, `parse`, `refusals`,
`refused`, `requires_build`, `install`, `engine_fills`, `from_engine`, `load_settings`, `at_row`, `at_capture`,
`at_commit`, `commit_guard`, `own_auditor`, `weights_view`, `register_weights` and `into_verdict`. From `interface`:
`Context`, `Option` and `Target`. From `linear.api`: `Arm`, `Control`, `LinearSite`, `PassSpec` and `Stack`; from
`linear.vllm_site`: `attach` and `discover`. From `poub`, which attaches PoUB's node store
(`verity.protocols.resource_accounting.memory.poub.prover.node_store`) to the engine: `Binder`, `Every`, `WeightRegion` and
`model_weights` of `binder`, and `CommitSession` of `remote`, which `pipeline.remote_session` reaches through `driver`.
From
`spot`: `NAME` and `DEFAULT_SCHEME`, by which serve-openai's Pearl+ Commit windows name its entry
(`pipeline.serve_pearl`). No other name of `spot.py` and none of `pous.py` is imported from outside: each is reached as
its option, through the package. The package, `interface`, `spot`, both modules of `linear/`, and `binder` and `remote`
of `poub` declare these names in `__all__`.
