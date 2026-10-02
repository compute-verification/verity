import FlockSoundness.Discharge.ZkExec.Statements
import FlockSoundness.Discharge.Exec.Tables

/-!
# flock-verify `--zk` accepts, table by table and rep by rep

* `zk_verify_ok : ZkVerifyOk`: `Zk.verify`'s session checks (the coin tree, the record, the tables, the proof files),
  then every table's two reps with one cap.
* `zk_verifyRep_ok : ZkRepOk`: one rep's checks: its stream and proof file, the PCS parameters, the pads' root, the
  transcript program `repProg` on the stream's rounds consuming all of them, and the cap `root_B`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec

theorem zk_verify_ok : ZkVerifyOk := by
  sorry

theorem zk_verifyRep_ok : ZkRepOk := by
  sorry

end FlockSoundness.Discharge.ZkExec
