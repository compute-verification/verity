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

