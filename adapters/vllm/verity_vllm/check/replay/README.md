# verity_vllm.check.replay

The sampled replay of a Commit: it samples rows of each request's Program, reads their operands from the committed
values and the registered weights, evaluates each by its family's row kernel, and compares the result with the
committed words bit for bit. Each module's docstring says what it holds.

## What other code imports

A module's `__all__` lists the names code outside the package may import from it. A module without one is the
package's own. `driver` is the replay: `sampled_replay`, the population it enumerates (`populate`), the facts of the
engine it records and the boundary linkage. `definitions.of_record` refuses a replay whose tree does not evaluate the
Definitions the Build of record bound, and `definitions.premises` names what a replay of a Program's rows does not
hold, for its record. A caller hands the replay each request's Program rows as an `index.ProgramIndex` and the
committed words as an `opening.CommittedStore`, and `request_replay.Replay` answers the same reads for one request
replayed on the CPU from its Program. `evaluate` evaluates one row, `population` gives a request's population and its
split at the executed prefix, `checks.check_rows` names the rows that compute a Program's checks,
`coverage.form_b_families` names the families the Commit value-checks by form (B) only, `cached_prefix.layout_of`
reads a warm Program's prefix layout, and `replay_codes` holds the codes of the replay's record that a Commit's
verdict reads. `linkage.LINKAGE_SCHEMA` is the id of the boundary linkage's record. Beside the replay of rows,
`served_check` checks a served-compiled row, `pouw_circuit` replays PoUW's `ncp-v2`, and `stoch_recompute` is the
value check of a stochastic row's record: the `stoch-recompute` command (`Options`, `main`) and the `recompute_run`
it runs. `fork` gives a worker pool that dies with its parent.

Six names that no `__all__` lists have a reader outside the package all the same: the pipeline reads
`driver._malloc_trim`, `driver._prewarm_evaluators` and `driver._prewarm_weights`, and `benchmarks/one_stage/a10.py`
reads `evaluate._evaluate_row`, `evaluate._weight_runs` and `evaluate._Unresolved`.
