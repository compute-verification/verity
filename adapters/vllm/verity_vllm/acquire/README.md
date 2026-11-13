# verity_vllm.acquire

Value acquisition: how a served run's values reach the Commit. The package's docstring lists its modules.

## Interface

- **Provides:** the acquisition plan, derived from the required-value manifest and the runtime correspondence (`plan`);
  the hooks installed from it (`install`); the sources those hooks feed (`sources`); and the CPU gate of a row
  (`verity-vllm gate`).
- **Loaded by:** the integration's `pipeline`, `commit`, `program.predict`, `properties` and `protocols` packages,
  `benchmarks/served_commit` and `benchmarks/harness`.
- **Loads:** torch, and vLLM inside functions; the integration's `engine`, `query`, `commit`, `config`,
  `settings`, `program`, `protocols`, `correspondence` and `observe`; and `verity.definitions`, `verity.circuits`
  and `verity.hardware`.
