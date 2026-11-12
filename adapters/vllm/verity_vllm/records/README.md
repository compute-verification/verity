# verity_vllm.records

The records one stage writes and another reads, typed: one dataclass a record kind, whose `load` refuses a missing
field by name. The package's docstring lists the record kinds.

## Interface

- **Provides:** a module a record kind, the field readers they share (`fields`) and the schema ids of the records
  that have no type yet (`schema_ids`).
- **Loaded by:** the integration's `check`, `pipeline`, `program`, `commit`, `engine`, `query`, `observe`,
  `correspondence` and `config`.
- **Loads:** nothing else of the integration, so every stage may import it; torch inside one function.

What the integration's other packages import, module by module: each record's class, with its `SCHEMA` or its `FILE`
where another package reads one. `BlockClass`, `BlockJudgment`, `RULES` and `shape_checks` of `block_counts`;
`BoundaryCommit`, `FILE` and `SCHEMA` of `boundary_commit`; `ASSIGN`, `ROOTS`, `TableAssignment` and `TableRoots` of
`boundary_tables`; `BuildSummary` of `build_summary`; `CommitBinding` of
`commit_binding`; `CommitRun`, `SCHEMA` and `TP_SCHEMA` of `commit_run`; `DeclaredStop` of `declared_stop`;
`ExpertTables` and `SCHEMA` of `expert_tables`; `GlobalMatch` and `SCHEMA` of `global_match`; `InferenceBlocks` and
`SCHEMA` of `inference_blocks`; `LogitsCapture` of `logits_capture`; `FILE` and `PredictedRow` of `predicted_row`;
`QueryChecks` of `query_checks`; `RequiredManifest` and `SCHEMA` of `required_manifest`; `RuntimeTree`, `SCHEMA`,
`RUNNER_PATH`, `mro_names`, `runtime_tree_of` and `runtime_tree_from_paths` of `runtime_tree`; `ServedRequest` and
`ServedTokens` of `served_tokens`; `WeightsRegistration` of `weights_registration`; `SCHEMA`, `Workload` and
`WorkloadSampling` of `workload`; `RecordError` of `fields`; and every constant of `schema_ids`. Each module but
`boundary_commit`, `required_manifest` and `schema_ids` lists its names in `__all__`.
