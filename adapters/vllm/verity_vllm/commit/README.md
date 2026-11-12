# verity_vllm.commit

The Commit: it commits a served run's values while it serves. The package's docstring lists its modules.

## Interface

- **Provides:** the GPU committer (`committer`); the `rows-sha512` scheme every Commit commits under (`step_rows`); the
  capture map, the binding map with its coverage check, and the partition of record; the challenge's samples and the
  opened values.
- **Loaded by:** the integration's `pipeline`, `check`, `acquire`, `protocols`, `properties`, `engine` and
  `program.predict` packages and its `glue`, and `benchmarks/served_commit` and the other benchmarks that commit or open
  a run's values.
- **Loads:** `verity.commitments`, `verity.circuits`, `verity.definitions.kernels`, `verity.commitments.windows`,
  `verity.commitments.randomness`, `verity.definitions` and `verity.errors`; `verity.protocols.spot.verifier`, in
  `registered_rows`; torch; and the integration's `acquire`, `query`, `engine`, `pipeline`, `records`, `config` and
  `settings`.
