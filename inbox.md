[2026-11-14T04:04Z] 55d1c57091bc
STEP, Fri 13 Nov 8:04 PM PST. The port of #9514 onto main is done and checked. All three by-name suites passed on their own lines.

- [RAN] Branch `cursor/trn144-step-fused-attention-main-28a9`, head `dbf5e361776219b013a62c27d26baa2cb0b5ac08`, cut from origin/main 56f6c267563. Draft #9863 is on base main and unmarked. Its body is updated with every last line below. Nothing was pushed while the three runs ran.
- [RAN] How the commits came over, old → new:
  - Cherry-picked with -x and applied clean: 3f83527→738f1b5, 1ad556c→03eade2, 5ea9f49→19c382f, 229dfd4→8671d9a, 4c9fd1e→59dfc88, bf3fb08→4824fd5, c14944e→19255c1, dafee6c→21a2f57, 860fab8→da242b6, bbc4af4→b9cdabb.
  - fd1046e→21e4214, cherry-picked with two hunks settled by hand:
    - test_grader.py imports: main's `verity.circuits.*` imports stand, with GRADER's `fa2_train` and `fa2_train_bwd` registry imports on top.
    - test_slowdown_code.py, `test_the_tiny_steps_beside_the_slowdown_code`: main's non-cc lines stand. At a cc, GRADER's assertion now reads four columns: O differs at every fused product, and at the products other than the scores the circuit's O is exactly 2× the trace's (32-bit accumulators against the 16-bit query pricing of capture 4.26.0).
  - Two commits by hand:
    - 163a12f: `grader.Cells` imports `prims` from `verity.circuits.ops`. dafee's clean pick carried the retired `verity.definitions.ops` and failed at import.
    - dbf5e36: `grader.read`'s docstring now says `grader-products/v2`.
- [RAN] Item 3: zero retired names in my added lines, under each of the 29 move maps. The one word settled by hand is that `prims` import, under definitions_kept_map.
- [RAN] Item 4: `git diff --stat origin/main...HEAD` lists your ten files and no others, 1053 insertions and 227 deletions, nothing under `verity/`.
- [RAN] Item 5: the import-list adapter test failure is gone on main (3 passed on the VM).
- [RAN] Item 6: the README is 46,424 bytes at my head, +108 net.
- [READ] Item 7: no VERSION moved. training_slowdown's VERSION is 4.26.0. The grader has no VERSION. `grader.READ` is `grader-products/v2`, from GRADER's dafee. I took no number.
- [RAN] Checks:
  - VM:
    - The grader's two test files, with the slowdown code set: `39 passed in 51.59s`.
    - test_train_sdpa.py: 26 passed.
  - r20261113-213708-84e1, the `--changed` suites: cancelled at its 5-hour limit, so no summary line. Its `out/suites.log` has 21 `passed` lines of 24 suites and no `failed` line. The 21 are in #9863's body.
  - r20261114-030744-b391, verity-numerical, tree dbf5e36: `1 suites: 1 passed (0 cached), 0 failed; 0.8 min of test time` (662 passed, 7 skipped).
  - r20261114-030744-242b, verity-circuit-check, tree dbf5e36: `1 suites: 1 passed (0 cached), 0 failed; 37.2 min of test time` (448 passed, 5 skipped).
  - r20261114-030744-49da, verity-flock, tree dbf5e36: `1 suites: 1 passed (0 cached), 0 failed; 48.9 min of test time` (2102 passed, 69 skipped). suites.py ran the `lake build` of Flock's verifier first.
  - r20261113-214203-39ed, ruff and the ratchet: `passed: ruff finds nothing new`. The three I001 findings in the adapter are main's own.
  - r20261113-213726-4c30, circuit-check: `34 target(s), 0 failure(s)`. No Definition, template or lowering changed.
  - Card question: the sm_120 tests trace on fake tensors and read no card. They ran and passed on a CPU node on Thursday (r20261112-194903-bdcc). No GPU run.
- [RAN] c303, relaunched with env -u as r20261114-012108-c619, which passed. Relaunched unchanged as r20261114-021650-54bd, cancelled after one line (verity-numerical passed). Replaced by the three runs above.
- [INFERRED] Friction for infra:
  - The slot env's `CHECK_SLOT_BOOK_GB` broke trees older than 70cd2660ede.
  - The 5-hour limit cuts off a full `--changed` run.
  - An `--on auto` twin exits 75, which reads as a failure.
- Handoff note: training-20261114T010107Z-handoff-STEP, status done, synced.

