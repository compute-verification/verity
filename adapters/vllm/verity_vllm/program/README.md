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
