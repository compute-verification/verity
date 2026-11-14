[2026-11-14T06:44Z] cf36cc7768d7

READCALC, Fri 13 Nov, 10:44 PM PST.

**#9877 is NOT READY on three must-fixes. `circuit_properties.work.bound` loses its every-contraction-length family and its bound can fall (23.948 to 3.70127 on a Work I built), which the body reads as "no figure moves". And two body sentences are false: "Main since then touches none of these files", and "Every slowdown the tree computes now comes from one function". The training figures and the inference-only report hold, equal by the author's runs and by my reading, and the root test catches a direct `bound` call.**

a95a595c639f2728ae1a8b70e0d327e31a0dbc67 NOT READY, 3 must-fixes.

What I read:
- Whole: the diff of `min_cut.py` (`Product`, `Chosen`, `SUBWORKS`, `clique`, `slowdown_of`), `calculator.py`'s `_figure`, `work.py`'s `products` and `bound`, `policy.py` and `tied.py`, `tests/test_one_slowdown.py`, the `tests/conftest.py` diff, the `test_work.py` and `test_program_reading.py` diffs, and the body.
- The author's inputs: `cap.py`, `replay.py`, `compare.py` and `job.sh` in r20261113-214216-4796, and `job2.sh` in r20261114-011106-92b7.
- Sampled: the README diffs, and the `test_min_cut.py` and `test_formalism.py` diffs.
- Not opened: `test_grade.py`'s diff beyond its stub change, and the rest of `work.py`'s reader.

## MUST FIX

1. **`circuit_properties.work.bound` is narrowed, and its bound falls on some Works, without the body saying so.**
   - What changed: the base's `bound` (`work.py` at 56f6c267563) tried every contraction length as a threshold (its K0 loop) × the kinds, and took the largest. The head's `bound` (`work.py:1435-1445`) hands the products to `slowdown_of`. That function tries only the β-product thresholds × `SUBWORKS`, with `K_thr` taken on the whole work's choice.
   - The body says the opposite. "The family is the training benchmark's, moved and not widened" (What moved) holds for training, but for this caller the family is narrowed. "On the circuit-properties bounds, dropping the every-length loop lowered none of the 12 that exist" (Item 8) covers only the bounds the tests ask for.
   - RAN on my VM, base worktree against head worktree, `work.bound` on one hand-built Work (`/tmp/ex.json`):
     - The Work: k = 65536, c = 1e12, 2e15 gates, and two groups.
     - Group 1: k_dim 4, 1e12 MACs, 4e12 wires, β (4, 8), weight (T, F), out 16, edge A–F.
     - Group 2: k_dim 256, 1e12 MACs, 3.125e10 wires, β (1, 2), weight (F, T), out 8, edge F–D.
     - Base: 47896 blocks, slowdown 23.948, sub-work [k_dim 256].
     - Head: 7402.54 blocks, slowdown 3.70127, sub-work [k_dim 4]. `K_thr` ≈ 1231 on the whole work's choice leaves the second sub-work empty.
   - RAN on 3000 random Works of the same shape (`/tmp/cmp_cp.py`): 857 lower in blocks, 265 lower in slowdown, 0 higher.
   - INFERRED, not shown: whether a Program `work.read` reads gives such a Work. Every Work above is built by hand.
   - Fix, either of:
     - (a) Put the every-contraction-length thresholds into `slowdown_of`'s family. The body's own Item 8(a) measured 0 rises on all 92 training figures and 37 library rows, so training's figures stay as they are.
     - (b) Keep the narrowing and say so in the body: "`work.bound` loses its every-length thresholds; on a Work like this one its bound falls from 23.9 to 3.7. On the tests' 12 bounds nothing falls."

2. **The body, line 2 of its head paragraph: "Main since then touches none of these files (READ: `git diff --stat`)." This is false.**
   - READ (`git diff --stat 56f6c267563 origin/main` over the change's files, main at 512644d14fb): main has changed `benchmarks/rho/README.md`, `benchmarks/training_slowdown/README.md`, `test_formalism.py`, `test_grade.py` and `calculator.py`. RAN `git merge-tree` with 512644d14fb: clean.
   - RAN: the merge with main at 8e018c9331f is clean (local commit c120af56fa, not pushed). On it these pass on my VM: `test_one_calculator.py` 11, `test_formalism.py` 51, `test_grade.py` 40 (quick), `test_program_reading.py` 1 (4 slow deselected), `test_min_cut.py` 23, `test_work.py` 99 and `tests/test_one_slowdown.py` 3.
   - Fix: "Main since then touches five of these files; the merge is clean and the touched tests pass on it."

3. **"Every slowdown the tree computes now comes from one function, `min_cut.slowdown_of`" is false.** It is the body's bold first sentence, and also `tests/test_one_slowdown.py` line 1 and `min_cut.py:301` ("The one way from a work to a slowdown").
   - READ: `benchmarks/nci/slowdown.py` computes slowdowns without `min_cut`, a thread-theorem lower bound and a construction's upper bound. The body's "Not in this change" lists it as untouched.
   - Fix: "every slowdown the calculator's callers compute" (the body and the docstring), or name `nci/slowdown.py` as a separate calculation the test does not cover.

## SHOULD FIX

1. **The F fallback rises when a group states no edge.**
   - `products` (`work.py:1432`) passes `pairs=g.edges or None`, so one edgeless group sends the whole work to the kinds' rule.
   - RAN on a hand-built Work (`/tmp/ex2.json`): base ω = 4, F 1.225, slowdown 525.057; head ω None, F 1.155, slowdown 563.426.
   - With `pairs=g.edges` the head gives 525.057 again, and `test_work.py` still passes (99).
   - INFERRED from `_read`: unreachable from a Program, since every charged group there has an edge.
   - Fix: `pairs=g.edges`.
2. **A door taken under another name escapes the root test.**
   - RAN: in `benchmarks/training_slowdown/training_slowdown/library.py`, `door = C._min_cut; door().bound(...)` passes `tests/test_one_slowdown.py`.
   - Fix: also count an assignment whose value names a door, as `_assigned` does for paths.
3. **The body names a team machine** (`vy-nebius-cpu-11`, Checks) in a public pull request. Drop it; the run id carries it.

## NOTE

1. The tie-break changed. The old chooser took the smallest P on a tie, the new one takes the largest q. On ties the blocks are equal, and only `Priced.groups`, `N` and `O` can differ.
2. Of the 64 calls recorded for `circuit_properties` (capR), 50 pass `F` and were skipped, so 14 were compared. The body's "7 bounds" and "14" are right.
3. "`test_work.py` holds 99 tests, all unchanged": the base had 100. `test_clique` moved to `benchmarks/rho/tests/test_min_cut.py`, and one assertion now reads `MC.clique`. Item 7's "unchanged but for the calls the move forces" is the accurate one.
4. "`training_slowdown/README.md` grows by 5 bytes net": RAN `wc -c`, it is 6.
5. "Untouched: `work.py`": there are two `work.py` files and this change edits `circuit_properties`' one. Name `training_slowdown/work.py`.
6. The body's "Item 7" and "Item 8" are labels from the brief. A reader of the pull request has no brief: name the sections by what they show.
7. The body says "Refs TRN-101"; the brief names no issue for #9877.

## For the lead to read

1. **Whether `circuit_properties`' bound may narrow.** This is must-fix 1. Option (a) keeps every caller's figure at least where it was. Option (b) is the smaller change, but the bound then falls on some Works.
2. **`F` given to `slowdown_of` overrides the rule on every sub-work.** Tests use it (`test_program_reading.blocks`), and the root test forbids it outside tests. With that rule, `test_program_reading`'s inequality compares both readings on the narrowed family.
3. **The root test's scope.** It holds `min_cut`'s callers and leaves out `nci/slowdown.py`, which is why must-fix 3 matters for what it claims.

## Merge with #9730 (head 201484307189db1455a6f3713cc39a874d844fbc)

RAN `git merge-tree --write-tree 2014843 a95a595`: two files conflict and the rest merge cleanly. Pushed nothing.
- `benchmarks/circuit_properties/README.md`, one hunk:
  - #9730 rewrites the premise paragraph (`points.check`, `work.RepeatedRows`).
  - #9877 rewrites the `work.bound(work, c)` sentence that follows it.
  - The right result is #9730's paragraph, then #9877's sentence ("hands the charged products (`work.products`…) to the calculator's `min_cut.slowdown_of`, which chooses the sub-work, none from `HEAD_K` on").
- `benchmarks/circuit_properties/tests/test_work.py`, one hunk:
  - #9730 adds `GUARDED`, `_guarded_attention` and its two tests just above `test_clique`.
  - #9877 deletes `test_clique`, which moved to `test_min_cut.py` as `min_cut.clique`; `WK.clique` no longer exists.
  - The right result is #9730's additions without `test_clique`.
- Clean: `work.py`, `benchmarks/rho/README.md` and `test_program_reading.py`.
- RAN on that resolution (scratch worktree `/tmp/m9730x77`, uncommitted): `test_work.py` 120 passed, `test_min_cut.py` 23, `test_program_reading.py` 5 passed with 4 slow deselected, and `tests/test_one_slowdown.py` 3. Neither the 4 slow tests nor any whole suite was run.

## #9889's `ALLOWED` and `min_cut.HE_CHARGED`

#9877 changes nothing about who deletes the entry: it touches neither `HE_CHARGED` nor the lock.
- RAN: the #9877 × #9889 merge passes `test_lean_names.py`, with 35 cited.
- RAN earlier: #9519 merged with #9889 fails `test_allowed_only_shrinks` on HECharged. So whichever of #9519 and #9889 lands second deletes the entry, as before.

## Mutations (each RAN on my VM in my worktree, undone, `git status` clean after)

- A direct `min_cut.bound(...)` call added to `benchmarks/inference_only/policy.py` fails `tests/test_one_slowdown.py::test_every_slowdown_comes_from_the_one_function`.
- `getattr(C._min_cut(), "bound")` in `library.py` fails, as "not in ALLOWED".
- The door under another name (should-fix 2) passes, which is the escape.
- The stale-entry test, `test_allowed_only_shrinks_to_files_that_still_reach_it`, is in `BOOKKEEPING` (`tests/conftest.py`), so a stale entry warns, as AGENTS.md lets bookkeeping warn.

## The body's claims

1. Records equal as JSON text, training: READ, r20261113-214216-4796, capT "177 equal, 0 differ", calls recorded on the base suite and replayed in two trees with their own venvs.
   - Could it pass with a figure changed? No: the whole record is compared as `json.dumps` text.
   - It covers only the recorded calls. Must-fix 1 is a caller whose figures were compared only on its tests' calls.
2. Library: READ, the same run, 37/37 equal, and stdout and `library.json` byte-equal.
3. Inference-only: READ, r20261114-011106-92b7, the text equal excluding the printed JSON line and the JSON equal but for `commit` and `run`. Also READ by me: `policy.bound` is one forward product at F = 1, with `bound`'s old defaults for β_w, β_x and b_y.
4. "Both full runs exited 1 … because the job's shared `result.json`": READ in job.sh, true. The stated run, with its own file, has rc 0.
5. `circuit_properties`, 0 fall and 0 rise: READ, true on the 14 compared calls; RAN false in general (must-fix 1).
6. "`Priced.O` is now a float … Nothing reads `Priced.O` outside the tests": READ, true, and stronger: nothing reads it at all (the tests read `Work.O`).
7. The training `_figure`: READ, equivalent to the old chooser in thresholds, pairing, tie-break and key order.
8. Item 8's table: READ in r20261113-214216-4796; the last lines match the body.
9. "The one commit between them changes only the root test": RAN `git diff --stat 72029c5 a95a595`, true (`tests/test_one_slowdown.py`, +2 −8).
10. Checks, r20261113-222645-95bf: READ `stdout.log`, last line "7 suites: 7 passed (0 cached), 0 failed; 32.6 min of test time". State done, rc 0, `tree_sha` a95a595c639f2728ae1a8b70e0d327e31a0dbc67. The counts match (training 696, circuit-properties 118, rho 617, inference_only 21, grader 28 with 3 skipped, repository 365, verity-graph 246).
11. r20261113-214300-ae4f, "the same but for one silent `except`": READ, failed with rc 1 at 72029c5, and the commit after removes the `except`.
12. "`ruff check` … nothing new; `ratchet_gate` 1 change(s), 0 failed": not checked.
13. "Main since then touches none of these files": false (must-fix 2).

## Tree rules, paths, time

- Paths are in today's layout. There is no run id, PR number or name in tracked files, and no wall-clock read in the new tests. Only the body names a machine (should-fix 3).
- Suite time, RAN on my VM, base against head:
  - `test_min_cut.py`: 18 tests in 1.6 s at the base, 23 in 1.8 s at the head.
  - `test_work.py`: 100 in 6.6 s, then 99 in 6.3 s.
  - The new root test: 3 in about 2 s.
- Runs I launched: none. Everything I ran was local named files, `-k` selections and `git merge-tree`, all unpushed.

The whole report is note:20261114T0642Z-report-readcalc-9877 (synced).

