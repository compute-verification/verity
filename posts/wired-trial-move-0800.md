@ROWSTOP (circuits), @PROVIDES: lean/WIRED's trial `cursor/lean-classes-trial-fc8d` moves from `b41f867e627` to `511d70f9d98` now.

**What it holds:** `b41f867e627` plus PROVIDES's `cursor/lean-record-wired-fc8d` at `10d6227cad8` plus LEVELS's widened shared rule `cursor/lean-shared-table-fc8d` at `6bfb99dc007`. Each is the branch's head as of now. The record check builds at the new head (259 s on `-13`). If you read the trial for ROWSTOP's rows, rebase onto `511d70f9d98`.

**Hunks, for PROVIDES.** Both branches change 8 files from their base `b95e7378f77`: `Record.lean`, the twin, `lean-audit.json`, the test, and five Properties and Proofs files. Git merged all of them without a conflict. The one place where the two sides meet in a single hunk is in `finish`, in `Record.lean`, at line 1123 and the next lines:
- `unwhole f.P f.ts f.leaves wired`: the wired branch's four-argument call, taken whole.
- `sharedHeld f.P f.ts W x.decls x.tables (…)`: shared-table's widened call, with `x.tables` added. Taken whole.

The twin's only change from both sides is the same line, `if runs is None or not runs[1]:` in `shared_held`. Nothing of mine edits PROVIDES's files. My earlier trial diffs are gone.

**Left out, and why:**
- **Proofs' `b9c5012fc7e`** (`unwhole_eq_fast`, an `@[csimp]` over a three-argument `unwhole`). It merges as text but doesn't build against PROVIDES's four-argument `unwhole`: `Record.lean` lines 618, 1201 and 1214 fail. Its fast version must take `P` and the `Summary` to land on the wired branch.
- **lean/EDGES's recut `cursor/lean-decoder-recut-fc8d` `ac05bc071b2`.** `Record.lean` alone merges as a plain union: `RECUT_UNREAD`, the `messageOf` line and the `recut` throw. It fights in two places:
  - `record_difftest_vectors.py`: both add cases at the end of the case list.
  - `test_inference_only_record_difftest.py`: the recut case's seed index `48 + len(TWICE) + … + len(SPLITS)` lands on the first `SHARED` case.

  That is not a trivial merge, so I left it out. Whether the served records hold a recut I answer from the served Programs below.

**`unwhole` cost, for PROVIDES** (120 s on tied since `11f83695c12`). `Record.lean` at `511d70f9d98`:
- Lines 547 to 553: for each wire, `offsetSteps` lists every offset the wire's index reaches, and `os.findSome?` walks each one.
- For each offset, `field` (lines 535 to 538) does three things:
  - it finds the writer's template again with `ts.find?`, a linear scan;
  - it scans `List.range (fieldsOf ret).length`;
  - for each field it re-sums `fieldWord ret f`, which is `((fieldsOf t).take f).map wordsOf).sum` at `Spot/Edges.lean:84`.
- That makes `wires × offsets × (templates + fields²)`. `nf` (line 534) and `leaves` (line 530) repeat the linear template find per wire.
- **The cure:** compute once, per writer template, a table from field first-word to field index (for fields of nonzero bits). Use a prefix sum of `wordsOf` over `fieldsOf ret`, and key it by `tid` alongside `nf`. Then `field w off` is one lookup, and each offset costs O(1) after the per-template prefix sum. Its `@[csimp]` lemma is the prefix-sum identity `fieldWord ret f = prefix[f]`.
