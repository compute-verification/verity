[2026-11-12T22:52Z] c1eca64a776b
METRIC, Thu 12 Nov 2:53 PM PST. **Draft #9532 is ready for you to read at `1e858ab44b1251a709153fcf355d3f37ed0f27b7`. Both suites pass, and one run of the row's command recorded `training/inference-only-slowdown` = 1.0, equal to its JSON's `value`. Nothing is in flight.**

- **The PR** [RAN: `research queue open --draft`]: https://github.com/compute-verification/verity/pull/9532. Its body ends `Lane training` / `Refs CON-17` / `Goal accounting/inference-only-small-model`. `open` linked it to CON-17 through the Linear app, and it warned that no label of CON-17 names that goal: CON-17 carries only `from:top`.
- **The row** [READ: the branch's `benchmarks/metrics.toml`]: the id `training/inference-only-slowdown` with your title. `unit` is the JSON's `unit` word for word, `better = "higher"`, `every = "7d"`, the goal above and no `floor`. `command = research run --on auto --mem-gb 16 --cwd clone -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run python benchmarks/inference_only/run.py --out "$OUT"'`.
- **The recording** [READ]: `report.main` calls `kit.writer(args.out).measurement(METRIC, doc["value"], UNIT)` after `inference_only.json`, only for `served.toml`. Under a run that writes `$RESEARCH_RUN_DIR/result.json`. The console reads it by name from the run's `bench-result/v1` artifact. The kit's `measurement` has no free field, so the JSON's `check` doesn't go there.
- **Suites** [RAN `r20261112-222736-5aac`]:
  - `passed  benchmarks/inference_only    10 passed in 10 s; 10 outcomes recorded`
  - `passed  repository                   356 passed in 393 s; 356 outcomes recorded`
  - `2 suites: 2 passed (0 cached), 0 failed; 6.7 min of test time`
- **The proving run** [RAN `r20261112-223954-72cf`, vy-nebius-cpu-14]:
  - rc 0, `result=valid`.
  - Its `result.json` measurement has `name` `training/inference-only-slowdown`, `value` 1.0, `commit` `1e858ab44b12` and `run` `r20261112-223954-72cf`. Its value and unit equal `out/inference_only.json`'s, and that JSON's `check` is `not run`.
  - Full form: 355 s wall with `uv sync` and `lake build`; the Lean count took 55 s and 6.9 GiB; the run peaked at 8.9 GiB PSS, inside its 16 GB.
  - `--stated-only` [RAN `r20261112-222738-ba62`]: 1.9 s wall, 81 MiB, the same 1.0. It also recorded under the id.
- **`research metrics show`**: left out, as you said. The site reads `metrics.toml` at `main` only [READ: infra `registry.ts`].
- **When the row lands** [INFERRED from infra `values.ts`]: the tab takes the newer of `r20261112-223954-72cf` and `r20261112-222738-ba62` as its latest value until your run from `main`.
- An earlier launch, `r20261112-222630-193e`, waited 13 minutes for a slot with 40 GB free on vy-nebius-cpu-8. I cancelled it before it started.
- **#9515**: still open. Its edit is to the first row and this PR appends at the end, so the two merge cleanly [INFERRED from the diffs].
- **Handoff note**: `training-20261112T2252Z-handoff-METRIC`, synced.

Done: the row, the recording, the tests, the README item, both suites, the proving run and the draft. Left, in order: your read and mark of #9532; after it lands, your run of the command from `main`; merging `main` once if #9515 lands first; a goal label on CON-17 if you want `open`'s warning gone.

