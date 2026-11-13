# verity_vllm.pipeline

The adapter's top layer: the Build, Match and Commit stage drivers, the row driver over them, the committing endpoint
and the command line (`cli.py`). Each section says what its modules are and which of their names other code imports.

## The row driver

`verity-vllm row` takes one row of a workload file through its stages (Build, Match, Commit) and writes the row's
records beside it. Its modules are the driver (`row`, `row_driver`, `row_stages`, `row_build`, `row_plan`,
`row_config`, `row_records`), the records written beside a row or a run (`config_record`, `verdict_record`,
`registration_log`, `experiment`, `report`, `timeline`), what `build --launch-context` and the re-baseline epoch's row
job ask (`launch_context`, `epoch`) and the stages as `research` Tools (`research`, `research_tools`,
`research_outputs`, `research_result`, `store_data`). No file outside `adapters/vllm/` imports them; the `research`
tool registry names the five Tools of `research_tools` by module path (`VLLM_BUILD`, `VLLM_PLAN`, `VLLM_MATCH`,
`VLLM_COMMIT`, `VLLM_REPLAY`). What the adapter's other modules import is what each module lists in `__all__`:
`config_record` (`INDEX`, `KIND`, `NAME`, `TIERS`, `aot_of`, `index_refusal`, `load_index`, `record`, `write_index`),
`epoch` (`build_programs`, `resolve`, `row_digests`, and `COMMANDS` and `main` for the command line), `experiment`
(`build_spec`, `code_version`, `write_experiment`, `write_spec`), `launch_context` (`bad_chunks`, `batch_dispatch`,
`batch_dispatch_of`, `missing_batch`, `rows_at`, `shape_contexts`, and `Options` and `main` for the command line),
`registration_log` (`lower`, `record`, `word_bits`), `report` (the configuration card's names), `row_config`
(`REPLAY_TIMEOUT`, `commit_verdict`, `config_record`, `word_check`), `store_data` (`data`, `fetch`, `listing`,
`selected_from`), `timeline` (`Timeline`, and `COMMANDS` and `main`), `verdict_record` (`COMMANDS` and `main`) and
`research_tools` (the Tools, their closure and what a stage's command line is parsed with). Five modules that others
import from state no `__all__`. `row_records` is imported for two constants (`COMPOSE_PEAK_RATIO`, and the least limit
of one part of a rank's fold Match), its readers and choices (`applicable_taps`, `build_jobs_auto`, `compose_bytes`,
`composed_ok`, `declared_target`, `decomp_rc`, `derived_ok`, `fold_phase_s`, `fold_plan`, `load_json`,
`longest_first`, `manifest_timeout`, `missing_tap_build`, `model_layers`, `tap_policy_flags`, `tap_source_args`) and
the TP row's summaries and lines (`tp_build_digests`, `tp_build_summary`, `tp_check_line`, `tp_commit_line`,
`tp_commit_passed`, `tp_fold_line`, `tp_manifest_complete`, `tp_manifest_line`, `tp_manifest_not_of_record`,
`tp_match_summary`, `tp_passed`, `tp_rank_workload_digest`, `tp_served_shapes`, `tp_shape`). `row_driver` is imported
for `Row`, `Stop` and `stage_on`, and `row`, `research_outputs` and `research_result` only by the command line, for
their commands. One private name is imported from another module, `experiment._digest_doc`.

## The Build and the Match

The Build derives a row's Programs and the Match checks them against a live serve. `build` derives one request
wrapper's Program from the pinned vLLM model class and writes it with its instance sequence (`build_instances`).
`global_program` composes the derived request Programs into a workload's Program. `manifest` builds the
required-value manifest under the query of record, and `manifest_verify` rebuilds one and compares digests. `match`
takes one configuration through every Match stage into one run directory, and `served_match` is the Match of a
served-compiled row on live data. Beside them are the workload generators (`workload`, `coverage_workloads`) and
`workload_file`, which makes a workload file present and checked; `program_cache`, a Build's derives reused on one
host; `circuit_db`, the circuit database behind `verity-vllm records`; `neutrality`, which rebuilds the rows of
record's Programs and fails if a Program digest moved; `program_graph`, a row's Program as a graph; the store and
the export of a row's sampled proof blocks (`vu_store`, `vu_export`); and the top-p split's GPU check and its
fixture (`topp_pin_check`, `topp_split_probe`).

What code outside these modules takes from one of them is what its `__all__` lists. For a command that is its options
(`Options`, or `COMMANDS`) and `main`, which the command line reads, and beside them: `build` (`read_tp_links`,
`same_module_path`, `sampler_geometry`, `sampling_policy`, `write_instances`), `build_instances` (`write_instances`,
which `benchmarks/prove_path` imports too), `circuit_db` (`STAGED`, `checkpoint`, `data`), `global_program`
(`MODELLED_ENGINE`, `compilation_facts`, `request_id_of`, `stop_rule`, `workload_target`), `manifest` (`build`,
`build_global`, `cpus`, `incomplete_rc`, `taps_of`), `program_cache` (`key`, `lookup`, `publish`, `restore`),
`served_match` (`ServedCommit`), `vu_store` (`after_replay`) and `workload_file` (`workload_file`, which the
benchmarks import). Three modules state no `__all__`: `match`, from which the TP row's fold Match imports
`fold_construction_args` beside the command; `topp_pin_check`, a command and nothing else; and `vu_export`, which
only `program_graph` and `vu_store` read.
