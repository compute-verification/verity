[2026-11-17T00:19Z] 9d26bbf31e24
model: claude-opus-5-5-high; subscriptions removed: none held

STEP to the training lead, Mon 16 Nov 4:19 PM PST. Wiring items 1 to 3 are in one PR, marked: https://github.com/compute-verification/verity/pull/10675 at `d76e8be55a6`, branch `cursor/training-benchmarks-run-from-clone-4207`, issue TRN-177 [ran].

- **Issue.** The goals page links no issue to `training/benchmarks-on-dashboard` [ran: `research goals show --json`], so the PR takes TRN-177. `queue open` warned that TRN-177 has no label naming a live goal (its only label is `from:top`). I posted nothing to Linear, so it wants the label `training/benchmarks-on-dashboard`, which is yours to add. The body names `Goal training/benchmarks-on-dashboard`.
- **Item 1** [ran].
  - Above the index's Entry points table, one sentence: `uv sync --all-packages --extra torch-cpu` makes every runner's environment, and `uv run verity bench NAME` runs one. That is the convention exfiltration, memory, kernels, bool_build_cost, torch_gpu and inference_only's own first block use.
  - `inference_only`'s second block now syncs first. training_slowdown's lines sync first and say where each runs from.
  - `core = "verity bench inference_only"` and its interface test are unchanged.
- **Item 2: `verity bench training_slowdown ENTRY --library [--root DIR]`** [ran].
  - It is a third option in the runner's existing exclusive group, beside `--local` and `--on`. It hands the one config to `library.main` and makes no store call.
  - Why: the runner already resolves ENTRY, and this adds no machinery. The four-job path is unchanged; a test holds both.
  - The README says its number is the bound on one traced step at the entry's policy, with no entry trained.
  - The point is `training/library-least-slowdown`, `partial`, with `member` set to the entry and `model` set to its start.
  - Index line: `uv run verity bench training_slowdown l16-t20-1.1x-fullmuon-fe-wsd --library --root DIR`. It took 157 s locally on CPU.
- **Item 3: `llama32-1b` in `references.configs()`** [ran].
  - Its configuration is the adapter's of record, read by the adapter's own reader from the pinned config.json of `unsloth/Llama-3.2-1B` (bf16). No file was missing.
  - `--models llama32-1b --out DIR` writes `training/smallest-product-subcircuit` as `partial`, with `extra["model"]`, and `card` from the host stamp, as #10635 does.
  - The layer has 9.6e10 gates. Its fan-out, fan-in and products agree with the configuration's shapes. It took 17 s locally.
- **Printed lines, fresh clone, CPU** (all pasted in the body) [ran]:
  - `{"name": "training/library-least-slowdown", "value": 4237319.259488665, "unit": "slowdown factor", "mark": "partial", "member": "l16-t20-1.1x-fullmuon-fe-wsd", "model": "l16-t20", "card": "none"}`
  - `{"name": "training/smallest-product-subcircuit", "value": 681, "unit": "Boolean gates", "mark": "partial", "model": "unsloth/Llama-3.2-1B", "card": "none"}`
  - `inference_only --stated-only`: value 1.0, `modeled`.
- **Runs** (id and state) [ran]:
  - `r20261116-234027-cfb1`, failed. It ran the lines in one fresh clone and every line printed its number. Its last line wrote `training/inference-only-slowdown` a second time into the same result, which the kit refuses. That was my chaining, not the line.
  - `r20261116-234819-cc6f`, done. It ran that last line alone.
  - `r20261116-234116-2d66`, failed. Four suites passed (verity-graph 248, verity-bench 841, verity-circuit-properties 157, the grader 36 with 3 skipped), and training_slowdown failed 2. My new test pinned the `OUT` setting for later tests; I reproduced it locally and fixed it in `d76e8be55a6`.
  - `r20261117-000054-a008`, done. training_slowdown at the fix, fresh: 804 passed.
  - `r20261116-234155-3eae`, done. The repository suite, fresh: 376 passed.
- **Conflict.** The mark's preflight says #10675 doesn't merge with top's marked #10518 in `benchmarks/circuit_properties/README.md`, `benchmarks/inference_only/README.md` and `benchmarks/training_slowdown/README.md` [ran]. #10518 is likely the rename pass [inferred]. Whichever lands second must merge `main` and be marked again. I'll do that on #10675 once #10518 is on `main`, unless you say otherwise.
- **Reader.** A reader is wanted on #10675, from whoever you staff; I asked no one. It touches no Lean, lock, verifier or soundness-path file, so the low-risk lane may land it on the mark alone [inferred].
- **Item 4.** The list of 8 went to you at 3:28 PM. Its tiny PR waits for the 6 PM map to be on `main`.

No event, PR comment or run output addressed to @training was seen.

