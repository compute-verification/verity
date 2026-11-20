# verity_vllm.acquire

Value acquisition: how a served run's values reach the Commit. The package's docstring lists its modules.

## Interface

- **Provides:** the acquisition plan, derived from the required-value manifest and the runtime correspondence (`plan`);
  the hooks installed from it (`install`); and the sources those hooks feed (`sources`). The names the adapter's other
  packages read, module by module: `committer_api` gives `Opening`, `RunCommitment`, `StepCommitment` and `declaration`. `compiled` gives
  `schedule_of_workload` and `main` (`verify`, run under `cli.run`). `fa2_plane_classes` gives `LayoutPolicy`,
  `StepSequences`, `attention_impls_of_record`, `check_step`, `counts_by_layer`, `declarations_from_attention_impls`
  and `uniform_rule`. `inductor_templates` gives
  `TEMPLATE`, `Template`, `Unmodelled`, `holes`, `instantiate`, `kernel_sources` and `source_sha256`. `install` gives
  `IncompletePlan`, `OccurrenceSink`, `apply_plan_to_committer` and `install_from_plan`. `plan` gives
  `analytic_table_of_buffer`. `served` gives `assemble`, `build`, `probe_routes` and its constant for the static
  split count of a step a full graph may hold. `served_artifact` gives `PARTS`,
  `differences`, `pinned`, `seed`, `winners` and the command `verity-vllm served-pin`. `served_device` gives `card_of`
  and `device_of`. `served_moe` gives `REQUIRED` and `expand`. `stage` gives `plan_at_stage` and `run_plan`. Each
  of these modules states an `__all__`, which holds its names here, a command's `COMMANDS` and `main`, and what the
  package's own modules read from it by a public name. The names
  of `sources` are not listed here.
- **Loaded by:** the integration's `pipeline`, `commit`, `program.predict` and `protocols` packages,
  `benchmarks/served_commit` and `benchmarks/harness`.
- **Loads:** torch and numpy, and vLLM inside functions; the integration's `engine`, `query`, `commit`, `config`, `settings`, `program`,
  `protocols`, `correspondence`, `definitions`, `records` and `log`; and `verity.circuits` and `verity.device`.
