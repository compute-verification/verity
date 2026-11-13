# verity_vllm.protocols

The protocol options of a vLLM run, each one adapter module written against `interface.py` (`spot.py` for SPOT,
`pous.py`, and `pouw/`), the linear site their executors share (`linear/`), and the node's registered store that PoUB
audits, with the hooks that keep it current while vLLM serves. The package's docstring has the order the options
install in, their phases and what a run is refused for.

## Interface

What the integration's other packages, the benchmarks and `verity`'s own tests import from this package, module by
module. From the package itself: `Context`, `Refused`, `WeightsView`, `adapter`, `driver`, `parse`, `refusals`,
`requires_build`, `install`, `engine_fills`, `from_engine`, `load_settings`, `at_row`, `at_capture`, `at_commit`,
`commit_guard`, `own_auditor`, `weights_view`, `register_weights` and `into_verdict`. From `interface`: `Context`,
`Option` and `Target`. From `linear.api`: `Arm`, `Control`, `LinearSite`, `PassSpec` and `Stack`; from
`linear.vllm_site`: `attach` and `discover`. From the store's modules: `NodeMemory` of `registry`, `PageLayout` of its
own `interface`, `Lines` of `memory_map`, `Binder`, `Every`, `WeightRegion` and `model_weights` of `binder`, and
`StoreBackend` and `prove` of `remote`; one test of `verity` also reads `registry`'s private `_page_blocks`. No name of
`pous.py` or `spot.py` is imported from outside: each is reached as its option, through the package. The package,
`interface`, both modules of `linear/` and the store's `registry`, `interface`, `memory_map` and `binder` declare these
names in `__all__`.
