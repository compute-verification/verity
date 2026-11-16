[2026-11-16T21:02Z] 57e0b6dddedf
PAGE, 1:02 PM PST. PR 5's list, before I delete anything. Python of `benchmarks/slowdown`: what goes, what moves, and what each kept command still prints. Nothing is deleted yet.

The module graph is from AST imports [read]. The kept callers' closure is `hardware`, `instances`, `lean`, `min_cut`, `params`, plus `theorem.multiple_bound` [read].

**Removed: whole files** [read: no kept caller imports them]
- Modules: `slowdown/{analysis,attacks,circuits,classes,envelopes,families,figures,operating,profile,scenario}.py`, 3,394 lines.
- Commands: `run.py` (`verity bench slowdown`, 445 lines) and `periods.py` (457 lines).
- `metrics.toml`: the metric `training/slowdown-onsite-h200`, whose command is `periods.py --out`, under goal `training/one-slowdown-calculator`. That removes a metric of record. Your call.
- Data: `data/{envelopes,exclusion,families,circuits,scenarios,scenarios_algorithms,scenarios_datacenters}.toml`.
- Tests: `tests/benchmarks/slowdown/test_{analysis,attacks,circuits,classes,envelopes,families,figures,operating,scenario,held_to_core}.py`.

**Removed: in part**
- `theorem.py` keeps only `multiple_bound`, because compute-accounting's `benchmarks/nci/{fit,statement}.py` import it [read]. Their files stay untouched.
- `test_theorem.py` keeps `multiple_bound`'s test.
- `test_min_cut.py` drops the 7 of its 36 tests that read `classes`, `families`, `analysis`, `theorem` or `periods` [read, AST]; 29 stay.
- `params.py` drops the derived `regime.remote.*` params, and `test_params.py` drops their expectations.
- `data/premises.toml` keeps the two rows `min_cut` reads (`training-cut-hardness`, `training-cut-exports`) [read]. The other 16 go, among them the cover rows #10564's stop named.
- `data/reference.toml` loses its `scenario.*` tables. It keeps `fleet.*`, `policy.v4-pro-session`, `model.*` and the `units`, `location` and `fleet` tables of `hardware.toml`. `spot_figures/figure1.py` (proofs), `network_traces/leak_bounds.py` (network-accounting) and certifier_term's tests read those as files [read].
- READMEs: `benchmarks/slowdown/README.md` is rewritten down to the kept modules. Also the `slowdown/` row of `benchmarks/README.md`, and `circuit_properties/README.md`'s sentence on `circuits.toml`.
- Allowlist rows of deleted files, in `repo_retired_terms`, `repo_env_reads` (`run.py`) and `repo_ruff` (`test_theorem.py`). The `inputs` of `tests/benchmarks/slowdown/pyproject.toml`.

**Moved: the only additions, all training's files**
1. `periods.FLEETS` (one tuple) goes to `params.py`. `params.PERIODS` already exists. `training_slowdown/calculator.py` and `inference_only/policy.py` read them from `params` instead of importing `periods` [read: both import `periods` only for `FLEETS` and `PERIODS`].
2. `envelopes.Policy` and `policy()` (about 15 lines) go to `params.py`, and `calculator.policy` reads `params.policy`.
3. `[policy.kimi-k2-1m-raw]`, one row, goes from `envelopes.toml` to `reference.toml`. This keeps `calculator.policies()` the same list, because a training config's `policy = "kimi-k2-1m-raw"` is checked by name [read].

**What each kept command prints**
- `verity bench training_slowdown`: the same. `calculator.fleets()`, `periods()`, `policy()` and `policies()` read the same keys through `params` [inferred; I'll run its tests].
- `verity bench inference_only`: one line changes. `report.py` prints "A deployment's slowdown, over what it registers and what it leaves free, is the calculator's: uv run python benchmarks/slowdown/periods.py" (its `CALCULATOR`), which names a deleted command. That line goes, and so does `inference_only/README.md`'s mention. Its fleet-by-period rows are the same numbers, from the same `FLEETS` and `PERIODS` [inferred].
- `verity bench circuit_properties`: the same. It reads `instances` (`models.toml`) and `Registry` [inferred].
- compute-accounting's `benchmarks/nci` scripts: the same. `multiple_bound` and `instances` stay [read].
- `tests/test_one_slowdown.py`: stays green [inferred]. `min_cut.py` is still the one tracked copy, and every file in its `ALLOWED` still exists and reaches it.
- `test_lean_names`: the calculator cites fewer names. The `ServingBlocks` and `Slowdown.Cover` citations go with `envelopes.py` and `classes.py`, which resolves #10564's and #10570's Python stops [inferred].

**Outside text that goes stale, not mine. I'd list these in the body, not edit them**
- `benchmarks/nci/README.md` line 21: "the slowdown calculator's remote scenario at the H200 gives 3.04" (compute-accounting's). That scenario goes.
- `GLOSSARY.md` lines 222–233: retired-term rows naming `benchmarks/slowdown` items.
- `ARCHITECTURE.md` lines 98 and 280.

**#10564 and #10570: stack, don't fold.** PR 5 deletes `classes.py`'s cover formula and the cover rows of `premises.toml` in any case. So both drafts stay as they are and land after PR 5, with no change to either. Folding would make PR 5 a mixed Lean and Python change that also waits on compute-accounting's answer.

Size, estimated from the file counts: about −8,000 lines against about +25 moved.

I'll start on your go-ahead, or with your changes to the list.

