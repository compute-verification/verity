[2026-11-14T16:07Z] 3eaf6ad68f6e
training to every worker and reader, Sat 14 Nov 8:06 AM PST. No reply needed.

**The model rule changed (Daniel, TOP-423): only a lead runs Fable 5.2. You run `claude-opus-5-5-high` and stay on it.**

- Any subagent you launch names its model in the call: `claude-opus-5-5-high`, or `claude-sonnet-5-5-high` for a really well-specified task. Never Fable, never `inherit`, never a launch with no model.
- If a picker offers neither model, do not launch: do the work yourself or send it to me.
- A plan, a design or a judgment no check catches comes to me, in your next message, with what you recommend.

[2026-11-14T16:19Z] 1f6283cc0c45
READMAPS to RECORD, Sat 14 Nov 8:20 AM PST: #9968's delta 548936694a7b..6ea6039e29d3 (one commit).

**Both cures are right, not just green. The suite does build Lean, `verity_difftest` was its only missing input, the old test still catches what it is for, and the body is true at this head.**

Q1, the two cures:
- RAN: I ran `suites.py benchmarks/inference_only verity-check --fresh` at 6ea6039e29d3 in a worktree with its own `uv sync` env and `lake` on PATH. Result: "2 suites: 2 passed (0 cached), 0 failed". benchmarks/inference_only had 45 passed, and verity-check had 1014 passed. The runner did the flock `lake build` first, as it does for an after-Lean suite.
- READ: the suite really is after-Lean. `test_the_compiled_check_gives_the_stand_ins_public_verdicts` calls `B.lake_built(CHECK)`, which is `lake build inference-only-record-check` in `verity/protocols/policy/inference_only/verifier/nci/lean`. That package requires the circuits, interface, spot and flock-verifier packages by path. The suite's inputs list that package's `lakefile.toml`, and `suites.py` lists the suite in `AFTER_LEAN`.
- RAN, mutation G1 (`tools/difftest/python/verity_difftest` dropped from the inputs): the suite failed. The guard named only `verity_difftest/__init__.py`, `protocol.py` and their `.pyc` files, read by `test_served_record.py`'s child processes. So `verity_difftest` was the one missing input. Reverted; the tree is clean.
- READ: the read path is `served_record_case.py`, which loads `nci_record_reference`. That loads `period_case.py`, which imports `verity_difftest`. Everything else it imports is under the `verity` input already listed.

Q2, the deleted `READS_ONLY` entry (RAN, each reverted, the tree clean):
- At the head, `test_after_lean.py` has 3 passed.
- m1, `"benchmarks/inference_only"` taken out of `AFTER_LEAN`: `test_every_suite_that_builds_a_lake_package_runs_after_the_lean_steps` fails, naming the suite and its `…nci/lean/lakefile.toml`. So a Lean-building suite outside both lists is still caught.
- m2, the deleted entry restored: `test_after_lean_and_reads_only_name_only_suites_they_hold` fails with "READS_ONLY … is in AFTER_LEAN: remove ['benchmarks/inference_only']". This is the quick run's failure, so the deletion was required, not optional.

Q3, the body:
- READ: the suite joins `AFTER_LEAN`, leaves `READS_ONLY`, and its inputs gain the difftest harness, which the record check's case script reads. All three are true of the diff and of the head's tree.
- RAN `gh`: "Builds on #9502…" holds, since #9502 is OPEN and its head is in this branch. #10012 is OPEN, which fits "arrives here from main when this branch next merges it".
- READ: the body names no machine.
- RAN, run keys only: r20261114-085547-1c4c is state `failed`, rc 1, at tree_sha 548936694a7b…, which fits the 2-suite failure you report; I did not read its output. r20261114-154622-0004 is state `running`, rc None, at tree_sha 6ea6039e29d3…, so it has no verdict yet.

Q4: READ, nothing else changed. The commit's parent is 548936694a7b and it is the only commit in the range: 2 files, 3 lines added and 4 removed, every line one of the two cures.

READY 6ea6039e29d3ac07314bab3ebde7a8f311f3f010

MUST FIX
- none.

SHOULD FIX
- none.

NOTE
- RAN: before this turn, my #9968 runs in worktrees imported `verity`, `verity_difftest` and the other workspace packages from the main checkout, through `/workspace/.venv`'s editable installs. This turn's runs used the worktree's own env, so they test the head's own code.
- INFERRED: the quick run r20261114-154622-0004 had not finished when I read it. Its verdict is the one that gates; my uncached pass of the same two suites is evidence, not a replacement.

For the lead to read
- READ: an event addressed to @training, the broadcast at 8:06 AM PST on the model rule (TOP-423). Not acted on beyond following it.
- READ: an event addressed to me, training's 8:16 AM PST brief (read b285f3e7e464 on `cursor/training-rename-2-a32b`). Queued behind this verdict.

