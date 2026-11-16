# verity_vllm.correspondence: design

What someone working on the correspondence needs beyond the [README](README.md) and each module's docstring.

## Interface

- **Provides:** the `runtime-correspondence/v1` record (`verity_torch.correspondence.runtime`) and its two readers:
  `reader_for_query` for the query and `reader_for_acquire` for the acquisition plan.
- **Loaded by:** the integration's `check`, `pipeline`, `acquire`, `program` and `query` packages, `circuit-check` and
  `benchmarks/served_commit`.
- **Loads:** `verity.circuits`, and the integration's `query` and `records`.

What the integration's other packages and the benchmarks import, module by module; each module lists these names in
its `__all__`. From `reader_for_query`: `Correspondence`, `CallName`, `runtime_rows`, `parse_call_name` and
`call_name_of`. From `reader_for_acquire`: `AcquireCorrespondence`, `ReturnSlot`, `RuntimeTree` and `RUNNER_PATH`.
From `verity_torch.correspondence.runtime`: `RuntimeCorrespondence`, `NAMESPACE`, `SCHEMA` and `correspondence_digest`.
