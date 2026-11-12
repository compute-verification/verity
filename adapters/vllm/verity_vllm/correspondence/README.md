# verity_vllm.correspondence

Where each Call of a derived Program lives at runtime.

## Interface

- **Provides:** the `runtime-correspondence/v1` record (`runtime`) and its two readers: `reader_for_query` for the query
  and `reader_for_acquire` for the acquisition plan.
- **Loaded by:** the integration's `check`, `pipeline`, `acquire`, `program` and `query` packages, `circuit-check` and
  `benchmarks/served_commit`.
- **Loads:** `verity.circuits`, and the integration's `query` and `records`.

What the integration's other packages and the benchmarks import, module by module; each module lists these names in
its `__all__`. From `reader_for_query`: `Correspondence`, `CallName`, `runtime_rows`, `parse_call_name` and
`call_name_of`. From `reader_for_acquire`: `AcquireCorrespondence`, `ReturnSlot`, `RuntimeTree` and `RUNNER_PATH`.
From `runtime`: `RuntimeCorrespondence`, `NAMESPACE`, `SCHEMA` and `correspondence_digest`.
