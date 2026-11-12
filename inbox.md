[2026-11-12T11:20Z] 56e79f29baf0
grade to training (coordinator), TRN-117, #9029 on main. Thu 12 Nov 3:19 AM PST. Each claim is RAN (a run or command whose output I read), READ or INFERRED.

## Done-when
- RAN: `cursor/training-trace-job-dtype-0cd4` holds `main` at `33faab2be81c`. Head **a6f8e21fb0b02ed214c9a1ceaf62aa75a327c187** (parents cfb7a3ae5c38 and 33faab2be81c5940213f544c22995ebe6ac700ea; a merge commit, no rebase, no force push; no conflict; remerge diff empty; tree 1563cd9deff2, the one your `git merge-tree` gives). I read the branch's head on origin before each push and each body edit. I hold no timer (RAN: the subscriptions list is empty).
- RAN: the gate, the suites and the port checks pass at that head (below). The body is rewritten for it and read back. The PR is a draft, unmarked, no label, base `main` (RAN: `gh pr view 9029`).
- Its change over `main`: 22 files, +489 −185, all under `benchmarks/training_slowdown/` (RAN: `git diff --shortstat`).

## The gate (RAN on the VM, at a6f8e21fb0b0)
"ratchet gate 33faab2be81c..a6f8e21fb0b0: 1 change(s), 0 failed"
Its second line: "passed: ruff finds nothing new in the 21 Python file(s) it changes; no direct environment read is new …; no silent except handler is new …; no lane-text site is new in the 22 file(s) it changes; no retired-term site is new in the 22 file(s) it changes". No list entry added.

## Runs (every tree is a6f8e21fb0b0 but c0c4's; RAN and READ each)
| run | what | tree | last line |
| --- | --- | --- | --- |
| r20261112-093845-48e7 | `suites.py --changed origin/main --fresh` with `VERITY_GRADER_SLOWDOWN_CODE`, then `pins.py --stale`, then ruff | a6f8e21fb0b0 | "ruff exit 0" |
| r20261112-093917-113a | port check, 22 entries | a6f8e21fb0b0 | "port check exit 0" |
| r20261112-093815-ccb2 | port check, the six l16 | a6f8e21fb0b0 | "port check exit 0" |
| r20261112-093751-4057 | port check, xl (waited 31 min for its slot) | a6f8e21fb0b0 | "port check exit 0" |
| r20261112-093942-1474 | port check, the three qat | a6f8e21fb0b0 | "port check exit 0" |
| r20261112-095346-c0c4 | the stored point's grade, by main | 33faab2be81c | "d92d4873…  own.txt" |
| r20261112-095355-b8b9 | the stored point's grade, by the head | a6f8e21fb0b0 | "f99e5a32…  own.txt" |
| r20261112-104315-a08c | bounds with the what-if on the 32 traces the four port checks kept | a6f8e21fb0b0 | "bounds exit 0" |
| r20261112-104902-3e58 | the 32 runs of record read by the head's code (waited 26 min for its slot) | a6f8e21fb0b0 | "records exit 0" |

- 48e7: "origin/main here: 33faab2be81c…; merge-base with HEAD: 33faab2be81c…"; "suites: the change from origin/main touches 4": repository 349 passed, benchmarks/training_slowdown 563 passed, the grader 31 passed (none skipped), verity-graph 234 passed; "4 suites: 4 passed (0 cached), 0 failed"; "suites --changed origin/main --fresh exit 0"; "pins --stale exit 0"; "All checks passed!".
- READ, its logs: the grader's, the harness's and verity-graph's whole (one warning in the harness's: torch's UserWarning in main's test of a refused step, in every earlier run too, a test this PR does not change). The repository suite's log is 17,964 lines: I read its test lines, the heads of its 21 warnings and every line naming `training_slowdown` (7, all in files this PR does not change), not every line.
- I added `--fresh`, the pins' stale check and ruff to your `suites.py --changed origin/main`: a superset of what you listed.

## The port checks: all 32 rerun, none kept
- RAN, what each check reads (an audit hook that only records the process's opens, listed directories and imported modules under the tree; `reads.json` in each run's output): all four runs read the same 47 files, 35 of the package (33 modules, with `of_record.toml` and `starts.toml`) and 12 under `verity/protocols/inference_only/rho` (`rho/min_cut.py`, 11 of `data/`). Beyond those: each entry's `attacks/ENTRY.toml`, and `routes/ema.py` (the 22 and the six l16) or `routes/lora.py` and `routes/qat.py` (the qat run). 70, 54, 48 and 52 files. Outside the tree: the records, the wrapper and the entries' list (the same sha256 as the earlier runs) and the environment (`uv.lock`, equal).
- RAN (`git diff --quiet` per file): against d47f246d956 (the six l16 and xl), two of the files read differ: `model.py` (main's rotary tables, #9070) and `work.py` (the docstring). Against cfb7a3ae5c38 (the other 25): `model.py`.
- So by your ruling no verdict is kept: every check reads `model.py`. All 32 rerun, all 32 equal: "22 entries: 22 equal, 0 differ, 0 refused.", "6 entries: 6 equal, 0 differ, 0 refused.", "1 entry: 1 equal, 0 differ, 0 refused.", "3 entries: 3 equal, 0 differ, 0 refused.". Expected differences: norm 32, loss 12, average 6; no `width` line.
- RAN (diff against the earlier runs 96ad, 9570, 480e, 0e2e): the output is line for line theirs but the lines of the head, node and thread count and the lines the recorder's script adds.

## Your two questions: neither changed, and the merge touches neither
RAN: `git diff cfb7a3ae5c38 a6f8e21fb0b0 -- benchmarks/training_slowdown` is `model.py` and `tests/test_model.py` alone. The grader's circuit and test, the capture, and the qat step test are as you read them.

## The body (sent with `research queue edit`, read back; 35,939 bytes)
- The top: head a6f8e21fb0b; "It stands on `main`, merged into the branch at `33faab2be81c`"; the two things left to your review, neither changed.
- No line names #8910, #9003 or #9187 (RAN: no `#NNNN` in the body); "carrier" is gone; the merged sides are named by commit.
- Section 6 is now "How the branch holds `main`": four merge commits, what the last brings, the eight-file table, the gate's two lines at this head.
- Section 8: the size over `main`.
- Section 9: the runs at this head, the paragraph on which port checks were rerun and why, and the earlier heads' runs cut to two lines.
- Sections 3, 4 and 5 cite this head's runs where a run was replaced: 48e7, 1474, c0c4, b8b9, a08c, 3e58. Section 4's what-if now covers 32 entries (it was 29): the slowdown with attention at float32 is 0.432 to 0.591 of the slowdown as traced, N, O, b_y, β_w, β_x, F, subset and what binds equal at all 32 (RAN, a08c; its 29 common lines equal r20261112-063335-2666's).
- Still cited at earlier heads, each named with its head: the code before 4.23.0 (r20261112-053627-c5bd at 034f76f61368; r20261112-060031-77ea and r20261112-064615-38ab at 4e221ac275bc), the six tiny jobs at 85e11b42d8c, r20261112-070234-e939 (the qat test without the switch, a scratch commit at d47f246d956), the published-table trace r20261111-170620-6a0e, and section 9's two lines on what the earlier heads' runs showed.

## Not checked
- Any GPU job (CPU nodes only). The published tables' control build. `verity check` as a whole. The suites the change does not touch. `te_core.gates_main` at this head (both files it compares are main's here, RAN: `git diff origin/main HEAD` empty).
- The repository suite's log line by line (see above).
- The two grade runs, the bounds run and the records run were not asked for; nothing was republished and no stored record changed.

## One thing you did not ask about: main has moved
- READ: `main` is at 33866d93e66e now (#9204, #9192, #9191, #9077, #9196, #9152, #9044, #9163, #9183 since). It changes one file under the harness, `benchmarks/training_slowdown/pyproject.toml` (the read guard's `untraced` list gains seven test files, `tests/test_grade.py`, `tests/test_routes.py` and `tests/test_run.py` among them), the graph's `test_file` family, and the ratchet gate's docstring (#9044: the ready mark's preflight now runs the gate on the PR's merge, and only warns).
- RAN on the VM, nothing pushed: a trial merge of that main into the head (commit 51bc32e3a589, on no branch) has no conflict, and main's gate on it prints "ratchet gate 33866d93e66e..51bc32e3a589: 1 change(s), 0 failed".
- RAN: none of the 52 files that moved is one a port check reads, and `uv.lock` is equal. So at a second merge the port checks' verdicts would be kept and the suites rerun.
- I did not merge it: the done-when holds at a6f8e21fb0b0 and a second merge moves the head under your review. Say the word and I merge it and rerun the gate and the suites.

Branch cursor/training-trace-job-dtype-0cd4, head a6f8e21fb0b02ed214c9a1ceaf62aa75a327c187, pushed, nothing unpushed, no run of mine out.

