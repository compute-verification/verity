# verity_vllm.query

The query over a derived Program: which of its values a Commit must commit. The package's docstring lists its modules.

## Interface

- **Provides:** a derived Program as a call graph (`program_view`), the module-body query (`module_body`), the required
  values under the query of record (`required`), and the required-value manifest: its format, its binding to what the
  runtime commits, and the check that a Commit used it (`manifest`).
- **Loaded by:** the integration's `pipeline`, `acquire`, `check`, `commit`, `program.predict`, `correspondence`,
  `properties` and `protocols` packages, `circuit-check`, and the benchmarks that read a Program's view.
- **Loads:** `verity.circuits` and `verity.definitions`, and the integration's `program` and `correspondence`.
