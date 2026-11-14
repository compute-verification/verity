# verity_vllm.query

The query over a derived Program: which of its values a Commit must commit. The package's docstring lists its modules.

## Interface

- **Provides:** a derived Program as a call graph (`program_view`), the module-body query (`module_body`), the required
  values under the query of record (`required`), and the required-value manifest: its format, its binding to what the
  runtime commits, and the check that a Commit used it (`manifest`).
- **Loaded by:** the integration's `pipeline`, `acquire`, `check`, `commit`, `program.predict`, `correspondence`,
  `properties` and `protocols` packages, `circuit-check`, and the benchmarks that read a Program's view.
- **Loads:** `verity.circuits` and `verity_vllm.definitions`, and the integration's `program`, `correspondence` and
  `records`.

What the integration's other packages, the benchmarks and `verity`'s own tests import, module by module; each module
lists these names in its `__all__`. From `program_view`: `ProgramView`, `from_instances`, `iter_instance_rows`,
`param_reads`, `ports_of_type`, `spec_base` and `spec_statics`. From `required`: `QUERIES`, `QUERY_OF_RECORD`,
`Q_WORD_ID`, `SAMPLING_EVENTS`, `population`, `request_manifest` and `under`. From `module_body`: `literal_calls`.
From `compose`: `compose_global` and `merge_ranks`. From `manifest.bind`: `Binder`, `HIDDEN_FAMILIES`, `Port`,
`coverage_check`, `runtime_place`, `source_ports` and `split_name`. From `manifest.verify`: `verify`. From
`manifest.format`: the manifest's schema id and its digests, the names of the families and members a manifest holds,
and the functions that load a request's Programs and read a spec's family and statics. From each opt-in policy
(`norm_scales`, `router_softmax`, `scale_products`, `vocab_range`): its `FAMILY`, `MEMBER` and `POLICY`, and the
functions that place and size its values. Two private names are imported as well and are in no `__all__`:
`program_view._read_instance_header`, by `pipeline` and `check`, and `vocab_range._s32`, by `acquire` and `properties`.
