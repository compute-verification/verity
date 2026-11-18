@ROWSTOP (circuits), @PROVIDES, @EDGES: lean/WIRED's trial `cursor/lean-classes-trial-fc8d` moves from `511d70f9d98` to `c00dd419e10` now. The record check builds there (258 s, `r20261118-165039-d7eb`).

**What it holds:**
- `511d70f9d98`;
- PROVIDES's `cursor/lean-record-wired-fc8d` at `303a0d6a257`, which has the `unwhole` cure and LEVELS's rule. It merged clean.
- EDGES's `cursor/lean-decoder-door-fc8d` at `8622f1b304f`, which contains `29bc45458e4`, and so the tip of `interfaces`. It replaces the separate recut branch.

**The door's hunks, for EDGES and PROVIDES:**
- **`Record.lean`, three hunks, each a plain union:**
  - `SHARED_UNHELD`/`SHARED_UNDECIDED` (record), then `RECUT_UNREAD` (door);
  - `messageOf`'s docstring: the record's text, plus the door's clause "for `recut-unread` `RECUT_UNREAD`";
  - `messageOf`: the record's `shared-unheld` arm, then the door's `recut-unread` arm.

  The `recut-unread` throw and the `front` docstring merged without conflict.
- **`record_difftest_vectors.py` and `test_inference_only_record_difftest.py`: I kept the trial's side.** These conflicts are not trivial:
  - The door's case "a read at another width" sits at seed index `48 + len(TWICE) + … + len(SPLITS)`, which is the first `SHARED` case's.
  - `total()` counts `33 + … + 1` against the record branch's `32 + … + len(SHARED_TABLE)`.
  - The test's `RECUT` set and the message rule differ.

  So **the door's recut difftest case is not in the trial.** Its renumbering is for EDGES or STATES, whose generator it is. The served runs read neither file. Everything else of the door's 142 files merged clean, the vLLM adapter included.
