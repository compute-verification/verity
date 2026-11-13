# verity_vllm.correspondence

Where each Call of a derived Program lives at runtime.

## Interface

- **Provides:** the `runtime-correspondence/v1` record (`runtime`) and its two readers: `reader_for_query` for the query
  and `reader_for_acquire` for the acquisition plan.
- **Loaded by:** the integration's `check`, `pipeline`, `acquire`, `program` and `query` packages, `circuit-check` and
  `benchmarks/served_commit`.
- **Loads:** `verity.circuits` and `verity.definitions`, the integration's `observe`, `query` and `config`, and
  torch inside one function.
