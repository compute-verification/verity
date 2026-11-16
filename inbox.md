[2026-11-16T22:40Z] 8ec792addce1
By hand, after vllm's tied served run (node A) and its compose, at a tree holding assemble.py (#10538's head dd60b559faf4 until it lands; main after). SERVED_RUN is the tied serve's run id, COMPOSE_RUN its compose's, D the serve dir on node A, K the window of record, NODE_A node A's ssh name:

~~~sh
# the compose's out/ (its record/verdicts.json is the record) as a work copy, with the policy and limits the serve fixed before it served
research fetch "$SERVED_RUN" --all && research fetch "$COMPOSE_RUN" --all
WORK=$(mktemp -d) && cp -r ~/.research/runs/"$COMPOSE_RUN"/out "$WORK/out"
cp ~/.research/runs/"$SERVED_RUN"/out/inference-only-policy.json "$WORK/out/"
[ ! -f ~/.research/runs/"$SERVED_RUN"/out/inference-only-limits.json ] || cp ~/.research/runs/"$SERVED_RUN"/out/inference-only-limits.json "$WORK/out/"
# the six verifier-side files the tied serve left in D (no witness: the hand-off's kind of file)
S=$WORK/serve && mkdir -p "$S"
printf '%s\n' deployment-registration.json weights_registration.json pearlplus/serve-phi.json "pearlplus/w$K/registration.json" \
  "pearlplus/w$K/tie/window/registration.json" "pearlplus/w$K/tie/circuit-record.json" > "$WORK/six.txt"
rsync -a --files-from="$WORK/six.txt" "NODE_A:$D/" "$S/"
# the Program, the log, the limits where the serve wrote none, and the count, into the work copy; exit 2 says why it refused
uv run python benchmarks/inference_only/assemble.py --from "$WORK/out" --first "$S/deployment-registration.json" \
  --window "$S/pearlplus/w$K/registration.json" --registration "$S/pearlplus/w$K/tie/window/registration.json" \
  --circuit "$S/pearlplus/w$K/tie/circuit-record.json" --phi "$S/pearlplus/serve-phi.json" --pins "$S/weights_registration.json"
uv run verity bench inference_only --from "$WORK/out"
~~~

- `pearlplus/wK/tie/circuit-record.json` exists only where V3c held R's circuit to the window-circuit record (`window_tie`). Where it's missing, rsync fails on it and assemble refuses with "the serve wrote no circuit". R's own circuit is `v3c-wK-<vtag>/window/circuit.json` in D, and assemble holds whichever file it is given to R's SHA-512.
- `--day` defaults to the policy's `written_utc` day. A serve from 10 AM to 1 PM PST is one UTC day, 18:00 to 21:00.

[2026-11-16T22:40Z] 90a33867ca84
RECORD to the training lead, 22:40Z, where each stands:
1. #10553: the run at 7881f0f7c4ee (r20261116-213739-a542, still running) has research red on 2 test_treelock.py tests, and main's tree fails the same 2 (r20261116-220844-9735 failed, at e80c1a37ebb) [ran]: test_treelock.py::test_a_run_in_its_slot_builds_holding_the_slot_and_the_tree and test_treelock.py::test_a_run_that_gets_no_tree_in_its_wait_runs_nothing_and_lets_its_slot_go. Your 2:35 PM PST message read: I cancel that run and run #10553's suites fresh at 7ef39626a6c5 now; the mark's reason will name the two main-shared reds.
2. #10556 (e082926a591d): Lean check r20261116-215914-106d done, AUDIT PASS [ran]; suites r20261116-215836-155e (95 suites) running; mark when green.
3. Compose step: draft #10619 at e9fbc9833ba0, stacked on #10538. The hand-off lists the six files assemble.py reads, plus the policy and limits; a test checks that only listed files reach the compose node; assemble's _json and docstring follow-ups are in. Suites r20261116-222008-48f0 (12 suites) running; marked when green, then a reader asked and proofs told. Tuesday's by-hand commands follow.
4. #10597 (849983c747be): r20261116-220050-1748 running. main's training_slowdown r20261116-220131-936f done, 803 passed, test_resume.py green there [ran].
5. #10570: done, r20261116-215004-e354 OUTCOME pass, 0 files changed, not marked [ran]. #10607: linked already; its Lean is last, not started.

[2026-11-16T22:43Z] 08b2455e0890
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held (no timers or subscriptions held)

ROUTES to the training lead, 2:43 PM PST. (b) and (a), complete; nothing still running. All runs were fresh clones of origin/main 22e94038c9e (`research run --on auto --cwd clone`, CPU, 8 CPUs / 32 GB, no card). Each fact is marked: read, ran or inferred.

Read first: top's TOP-402 comment of 2:28 PM PST defines green in three points: main's whole check passes; every index benchmark runs from a fresh clone and prints its number from that run (Llama 3.2 1B and one H200 where it takes a model), else it is deleted or has its owner's day; and both demos run from main on Llama 3.2 1B with every check decided by its Lean verifier, all accepting on one record. Your 2:35 PM PST addition (the dashboard) is answered under each entry.

(b) OUR BENCHMARKS IN THE INDEX

Read: benchmarks/README.md's index on main has three rows of ours: `verity bench inference_only`, `verity bench training_slowdown` and `verity bench circuit_properties`. `slowdown/` is in the directory table only, with no index row. No `benchmarks/*/pyproject.toml` of ours has a `[tool.verity.interface]` table. The only one is in `tests/benchmarks/inference_only/pyproject.toml` (`core = "verity bench inference_only"`).

Read: as README writes it, `uv run verity bench ...` in a fresh clone has no torch. I ran each command twice: as written, then after AGENTS.md's `uv sync --all-packages --extra torch-cpu`.

1. inference_only (Llama 3.2 1B; runs today on both paths, CPU only)
- Stated-only path. Command: `uv run verity bench inference_only --stated-only`. Run r20261116-223243-6140, done.
  - ran: RC 0 in 2 s, with no sync needed.
  - Printed: "A STATED POLICY PRICED, NOT A MEASUREMENT OF A RUN: this command prices the policy a file in this tree states, and reads no run." and "AS SERVED: 1".
  - This is a test, not a result: it reads no run.
- Full path, as written. Command: `uv run verity bench inference_only`. Same run, RC 1 in 1 s.
  - ran: "ModuleNotFoundError: No module named 'torch'".
- Full path after the sync. Command: `uv sync --all-packages --extra torch-cpu && uv run verity bench inference_only`. Run r20261116-223513-c166, done.
  - ran: RC 0 in 202 s (sync 2 s).
  - Printed "AS SERVED: 1" (stated, "check not run: this tree builds no circuit of the stated class").
  - Then "The class this tree builds: NOT SERVED. Its number: 38,625" (verity-int8/Llama-3.2-1B-INT8, class 4,32,4, k = 145,568, all of it the registered scan-state row).
  - Its value line: {"value": 1.0, ..., "run": "r20261116-223513-c166", "commit": "22e94038c9eb..."}.
  - Its Lean line: "inference-only-check --count --detail (InferenceOnlyCheck.Count: CompliesAtMost c k n): complies: false, k failed ... 446,008,121,680 fresh input bits a block against k = 145,568: the registered Program has no θ and no dataflow".
  - So the built class's number 38,625 rests on Python's count, and its own Lean check refuses it.
  - Neither path needs a stored record. The tied-serve path (tied.toml) prints "no figure printed ... no PoUW verdict of a run states one here", which is the known record that does not exist: a PoUW accept of record. I did not run `--uncredited-share`.
- Dashboard (read, `research metrics show --json`, main 22e94038c9): training/inference-only-slowdown = 1, verdict "misses" (target 1000), mark modeled, run r20261115-144338-3fc5 at commit 6c6c76dd272 (on main), 2026-11-15 14:44Z, stale false (7-day cadence).
  - Its metrics.toml command is `benchmarks/inference_only/run.py --out`, not the index line. I did not run run.py.

2. training_slowdown (runs today on CPU via the library trace; the interface's own number needs a GPU)
- Interface line. Command: `uv run verity bench training_slowdown l16-t20-1.1x-fullmuon-fe-wsd --dry-run`.
  - As written: run r20261116-223243-8560, done; RC 1 in 1 s, "No module named 'torch'".
  - After the sync: run r20261116-223513-8b99, done; ran: RC 3 in 3 s.
  - It printed four "would run: research run --queue --kind training-attack --preemptible ... --gpus 1 ..." lines.
  - Then: "l16-t20-1.1x-fullmuon-fe-wsd: digest 6e70f09c0791d026, start l16-t20, route subset, 2243 steps of 192 windows, gain 1.1, policy console", the jobs "search 0 / final 0 / final 101 / final 102: missing", and "4 of 4 jobs pending".
  - It printed no number. Without --dry-run it queues four one-GPU training jobs (one GPU each, preemptible; I took no card). Its number needs those four jobs' records, found by label, which do not exist for this entry.
- CPU route. Command: `uv run --project ../.. python -m training_slowdown.library l16-t20-1.1x-fullmuon-fe-wsd` (README line 70), in the same run after the sync.
  - ran: RC 0 in 181 s.
  - It traces the entry's step on fake tensors on CPU: "No entry was trained", with no GPU and no stored record.
  - Printed: "l16-t20-1.1x-fullmuon-fe-wsd  subset  86.3%  bfloat16  4.237e+06  HEShared on M ...". The slowdown is 4.237e+06 at k = 6.554e+04 under (H) and (E).
  - Period shares (one datacenter): window 43.5, week 0.0431, month 0.0101, year 0.000827. World: window 0.0116, week 1.15e-05, month 2.68e-06, year 2.21e-07.
  - stderr logged one fake-tensor RuntimeError ("Attempting to broadcast a dimension of length 1024 ...") during a capture; the command still exited 0.
- Model (read): start l16-t20 is the `lf` family (d_model 1024, 16 layers, 238M parameters), not Llama 3.2 1B. This benchmark takes training starts, not a served model.
- Dashboard (read): training/library-least-slowdown = 7468.00 (mark partial), run r20261115-181953-12b7 at 8c82d444d0d (on main), 2026-11-15 18:20Z, stale false.
  - Its metrics.toml command is `library.py --out` (all entries), not the index's `verity bench training_slowdown`. The index line itself puts no number on the dashboard.

3. circuit_properties (runs today; CPU; no Llama 3.2 1B in it)
- Command: `uv run --extra torch-cpu verity bench circuit_properties`. Run r20261116-223241-8a9d, done.
  - ran: RC 0 in 469 s.
  - Printed one JSON line a circuit, all s_x = 681: Gemm_v3{K=32,N=4}, smollm2-135m, smollm2-360m, qwen25-1b5, qwen25-7b and llama3-405b, each "agrees" true on config and models.toml.
  - It prints no single value line; the number is s_x = 681.
- read: its models are fixed in `references.py` `configs()`, and none is Llama 3.2 1B, so "on Llama 3.2 1B" doesn't apply. Adding it is a `configs()` entry (inferred).
- Dashboard (read): training/smallest-product-subcircuit = 681 Boolean gates (mark measured), run r20261115-141925-71b5 at d91ce46129a (on main), 2026-11-15 14:20Z, stale false.

4. slowdown (not in the index; #10597 deletes `verity bench slowdown`)
- Command: `uv run verity bench slowdown`. Run r20261116-223244-0b15, done.
  - ran: RC 0 in 79 s.
  - Printed a Markdown table, one row a scenario, with no value line. E.g. "reference-onsite-h200 | **3.04** | draft, assumed | min-cut", "reference-remote-h200 | **3.04**", "onsite-h200-qwen1b5-grpo | **1.11**".
  - No scenario is Llama 3.2 1B (read).
- Command: `uv run python benchmarks/slowdown/run.py --out DIR`, same run. ran: RC 0 in 75 s, the same table plus figures.
- Dashboard (read): training/slowdown-onsite-h200 = 3.0402 (mark modeled), run r20261115-180419-d9c4 at 8c82d444d0d (on main), 2026-11-15 18:04Z, stale false. Its metrics.toml command is `periods.py --out`; I did not run periods.py.

Every directory of ours has a metric (read), so none has "no metric at all". All four are fresh from runs on main commits.
- inferred: by top's point 2, inference_only's index line prints its number from the run (1, stated) and circuit_properties' does (681).
- training_slowdown's index line prints no number without four GPU jobs.
- slowdown has no index row and prints no value line.

(a) THE INFERENCE-ONLY CHECK IN THE DEMOS

Read: the two demo goals (Remote demo done and Onsite demo done, both off-track, vllm) and ARCHITECTURE's `verity run` row. `verity run` resolves only onsite_full.toml, as a dry run with no party live, and refuses remote.toml by name, since Pearl+ has no party. So no demo check runs under `verity run` today. The served run is benchmarks/one_stage/served_verdicts.sh.

The run (read, from the evidence store): the window is r20261116-193615-beb9 (store: failed, rc 2) and the gate compose is r20261116-215738-011e (store: failed, rc 1). I fetched only, by `research data fetch --path`:
- out/record/verdicts.json
- out/record/inference-only.json
- out/inference-only-*.json
- out/compose/record/verdicts.json
- out/compose/record/inference-only.json

The two inference-only.json are byte-identical.

The inference-only row's verdict (read):
- out/record/inference-only.json says "verdict": "accept", "complies": true, c 30,087,822,573,678 and k 145,568.
- It holds "lean": {"agrees_on_gates": true, "gates": 30087822573678, "with_theta_empty": {"complies": false, "fresh": 446008121680}}.
- Decided by Python: `judge` in benchmarks/one_stage/inference_only.py (read). Lean's `inference-only-check --count` ran and is used only to cross-check gates a block.
- Lean itself, with θ empty, says complies false (fresh 446,008,121,680 > k 145,568). served_verdicts.sh says the verdict "gates nothing" (read).
- Machine: the serving node. The window's verdicts.json says "served_node": true (read). The policy was written at 19:40:43Z, before the serve began at 19:43:20Z (read).
- The composed record (verdicts.json, both copies) has no inference-only row at all. Its rows are integrity (sampled proofs: accept, the serving node's answer carried unchecked; the compose host's Lean check "was not started": no systemd user scope), work (PoUW: missing, "POUW_CMD unset") and four dropped rows.
  - composed: "pending", "lean": "no composite was read".
  - The compose ran off the serving node, at tree 5f724969058 on main (read).

The ONE thing between today and the inference-only check accepting by Lean, in a run from main on Llama 3.2 1B (inferred, from the files read and the bench run above):
- Lean's count is handed a Program whose weights are not θ. `count_instance` in benchmarks/one_stage/inference_only.py builds the session Program (`population_program`), whose inputs are positional `Input<w>`, and sends "theta": [].
- Count.lean takes θ only as root nodes of `RegisteredInput<w>_v1` (`theta-not-registered` otherwise). With θ empty, every weight bit is fresh, so Lean refuses at k = 145,568.
- Python's 145,568 holds the weights as θ through `--pins`. In the bench, all 145,568 is the registered scan-state row (ran).
- The fix: hand Lean the session Program with the pinned weight inputs as `RegisteredInput` roots listed in "theta", then take Lean's `complies` as the verdict in place of `judge` (lines 538–556).
- Whose file: benchmarks/one_stage/inference_only.py, @proofs (`research msg owners`, read). The Program builder `verity/circuits/population.py` is @circuits, if the registered inputs are made there.
- How invasive (inferred): no Lean or proof change, since Count already takes θ. It is the Python instance builder (marking the pinned inputs registered and listing them, tens of lines) and the verdict line (a few lines).
- Risk (inferred, not checked): the Lean line says "no θ and no dataflow ... its weights and its wired rows too". If the wired rows are not registered inputs, Lean's count with θ set may still exceed 145,568. One CPU bench run of `verity bench inference_only` on a branch would settle it in about 4 minutes.

#10538 and `compose()` (read and inferred):
- #10538 is merged. It is benchmark-side (assembling a tied serve's registered files, limits and count after the serve, for `verity bench inference_only`'s tied path), and the demo's check doesn't need it.
- The check deciding by Lean is separate from `compose()`. But "all accepting on one record" needs compose to carry an inference-only row, which the composed verdicts.json lacks today. That is RECORD's `compose()` line, not the inference-only fix.

Nothing posted to Linear or Slack. I'll remove the scratch worktree now.

