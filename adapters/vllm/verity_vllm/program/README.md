# verity_vllm.program

The vLLM integration's Program layer: a traced model and an engine's settings become a Program in which every Call names
a registered Definition. The package's docstring and each module's say what it holds; `predict/` has a README of its own.

## Interface

- **Provides:** the frontends that lower a traced model onto the Definitions (`frontend`), the workload Program over a
  run's requests (`workload`), the lifted Program (`lifting`), and the provisional Definitions awaiting admission
  (`quarantine`).
- **Loaded by:** the integration's `pipeline`, `observe`, `check`, `engine`, `acquire`, `query`, `properties` and
  `protocols` packages and its tools, `circuit-check`, and the benchmarks that build or read a Program.
- **Loads:** `verity.circuits` and `verity.definitions`; torch, and vLLM inside functions; the integration's
  `engine`, `config` and `correspondence`; and, inside functions, its `check`, `observe` and `settings` and
  `verity.protocols.pouw.prover`.
- **Names other packages read:** a top-level module that states `__all__` lists there every name another package or
  a benchmark reads from it; its other names are its own. The modules read that way: `model` (a checkpoint's
  configuration, the type of its weights Record and its rotary table), `compact` (operand references as run
  progressions), `instances_file` (the Build's instance sequence, its reader and its writers), `prune` (a body's dead
  units and its dead-free form), `prefix_plan` (the cached prefix of each request of a prefix-caching workload),
  `sampling_event` (the families of Calls that end an engine step, and the sampler construction a Program declares),
  `tp_binding` and `tp_workload` (a tensor-parallel request's binding of its rank Programs, and the workload over
  them) and `compiled_relations` (the compiled kernels' argument roles, and MKL's one-thread warm-up). `workload`,
  `inputs` and `lifting.continuation` are read too and state none yet. One name
  outside a list is read by other modules: `model._inv_freq`, by `check.weights_of_record` and `predict.build`.
