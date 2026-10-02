import FlockSoundness.Discharge.ZkExec.Verify

/-!
# flock-verify `--zk`'s acceptance sets up each table with `Stmt.setupH`

`AcceptsZK I t` sets up the session with `Stmt.setupTables` at `zkInputs I`, whose tags differ from `--statement`'s
only in the identity (`zkTags_eq`), so `Exec`'s facts about `setupTables` apply at `zkInputs I` (under `hh`, as
there):

* `acceptsZK_setupH : AcceptsZKSetupH`: every table is `setupH`'s statement under `tagsAt (zkTags I) J j` and the coin
  tree, and passes `Zk.shapeOk`.
* `acceptsZK_one : AcceptsZKOne`: for one table, at the record's draw, with `Zk.verify` on `#[Setup.ofCircuit st]`.
* `acceptsZK_retained : AcceptsZKRetained`: under tags that retain rounds, every stream of the record retained every
  round's bytes.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec

theorem acceptsZK_setupH : AcceptsZKSetupH := by
  sorry

theorem acceptsZK_one : AcceptsZKOne := by
  sorry

theorem acceptsZK_retained : AcceptsZKRetained := by
  sorry

end FlockSoundness.Discharge.ZkExec
