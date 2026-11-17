# verity_vllm.commit

The Commit: it commits a workload's values while it serves. The package's docstring lists its modules.

## Interface

- **Provides:** the GPU committer (`committer`); the `rows-sha512` scheme every Commit commits under (`step_rows`); the
  capture map, the binding map with its coverage check, and the partition of record; the challenge's samples and the
  opened values. A module's `__all__` lists the names code outside the package may import from it, and every module
  that another package of the integration or code outside it imports states one. Code outside the integration
  imports from the committers (`committer.native_host`, `committer.native_collect`, `committer.native_jit`,
  `committer.host_retention`), the scheme and its row tables (`step_rows`, `row_tables`, `row_tables_device`,
  `history_tables`, `history_tables_device`, `scheme`) and the opened values (`opened`, `index`, `memory_map`,
  `store_dump`). The integration's other packages import from those and from `challenge`, `capture_map`, `binding`,
  `partition`, `executed_prefix`, `served_record`, `registered_rows`, `run_roots`, `native_ranges`, `json_chunks` and
  five more of the committer's modules (`device_pool`, `flush_points`, `plan_cache`, `run_values`, `topp_splits`).
  `row_tables_device` and `history_tables_device` take the row-table kernels' build from PoUW's serving package
  (`row_table_host_lib`, `hm_tables`, …) and a device's copy of them from the PyTorch adapter (`verity_torch.pouw`'s
  `RowTableLibrary`), and re-export none of it. Not yet behind an `__all__`: one
  module of another package reads `store_dump._as_bytes`.
- **Loaded by:** the integration's `pipeline`, `check`, `acquire`, `protocols`, `engine` and
  `program.predict` packages and its `glue`, and `benchmarks/served_commit` and the other benchmarks that commit or open
  a run's values.
- **Loads:** `verity.commitments`, `verity.circuits`, `verity.commitments.cuda`, `verity.commitments.periods`,
  `verity.commitments.randomness`, `verity_vllm.definitions` and `verity.errors`; `verity.protocols.proofs.spot.verifier`, in
  `registered_rows`; torch; and the integration's `acquire`, `query`, `engine`, `pipeline`, `records`, `config` and
  `settings`.
