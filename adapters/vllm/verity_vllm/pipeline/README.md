# verity_vllm.pipeline

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
