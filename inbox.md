[2026-11-14T08:57Z] 3e2d0c690b6d
NAMES2, Sat 14 Nov 12:56 AM PST. First message: set up, note read, where the sequence stands.

**Answer one question by 1:27 AM PST or I go on with my recommendation: may I run section 6's steps 1 to 4 now (build the chain on the VM, case B, push it and open the draft PR), and launch step 5's records runs myself with `--on auto` the minute the chain is pushed, as the note says, unless ci says it launches them in its stage?**

Setup and the note
- RAN: section 2 line by line. uv 0.12.20 at `$HOME/uv-0.12.20` (pod_setup.sh pins 0.12.20); scripts at `/agent/repos/wt/work`; refs fetched: maps `fb8db33486f8`, r8 `016c1313e173`, r4 `7e983d74c25b`, the kept commit's branch. `RESEARCH_AGENT_URL` is mine. The note and its section 6 read; nothing failed to reproduce.
- RAN: `origin/main` f28bdad24636; the maps' head is not an ancestor: **case B** (the chain merges the maps). The kept commit 218d81c74d is not on `main`: `KEPT_BEFORE` stays default.

#8449 and the step
- RAN: `research queue status 8449 --json` (97 s): state `ready`, head fb8db33486f8, `draft` false, `position` null, `commit` null ("admitted: … once it has a land verdict"). gh: OPEN, not merged.
- INFERRED: step 0 is done (READY, mark, ci's land verdict). Next is step 1 then 2 (VM only, pushes nothing, about 30 min), 3 (reading), 4 (push, draft PR). Step 5 is the first that needs ci: the two lock runs in its stage.
- What I need from you or ci: (1) the question in bold; (2) from ci, only if it wants it: whether it launches step 5's runs itself or wants them in a named slot. If #8449 lands before step 2 starts, I run case A (no `MAPS=`) and tell you.

Runs (named keys, last lines)
- RAN: c7e8 done, rc 1, 6666 s, tree b4da7f28da3c: "88 suites: 82 passed (0 cached), 6 failed; 267.7 min of test time". The six failing suites and 25 tests are exactly the note's table; the eight suites unread at 08:43Z all passed.
- RAN: 5879 (base, check's Lean package) done rc 0: "AUDIT: PASS", 1730 declarations in 17 modules, 0 guarantees. 953e (r8) still waiting.
- RAN: 1961, 0b93, 1349, 182b, 953e: waiting. No failure outside the table so far.

Cure of 42ac6068 (task B), what I will do unless you say otherwise
- READ: the reader's two findings hold in r8's tree (`Check.lean:149-150`, `inference_only_reference.py:141-144`; "baseline" at `Check.lean:29`, `:214`, `Theta.lean:6`, `lakefile.toml:1`). Also RAN: "novel reads" stays in the check's lock reasons (`lean-audit.json` lines 70, 78) and the liveness difftest's message (`DescLiveness.lean:58`); I add it to the paragraph so no placement is partial again.
- Way: `hand_edits.py older_check` cannot run on r8's head (its old string is gone there), so I run the script's own line on 42ac6068's parent b9fb40b9 and cherry-pick the five commits above it, on a new branch `cursor/training-rename-rehearsal-6b-a32b` (no force on NAMES' branch). Not a ninth rehearsal: no map reruns; the tree differs from r8 in the README alone.
- Question, recommended answer "leave it": `desc_difftest_vectors.py:305`'s docstring says B's gates are "novel to" the first inference block, plain English; I leave it out of the paragraph.

Chain branches will be `cursor/training-rename-a32b` and `cursor/training-rename-moves-a32b`. Events: your 12:50 AM PST inbox message reached me through the shell hook (acted on: this message).

