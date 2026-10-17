import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Complete.Zk.Level0Premises
import Proofs.Flock.Claims.Complete.Zk.Level0Verifier

/-!
# Masked completeness from `ShapeRest`

Level 0's four premises are proved (`proverRuns`, `proverShape`, `lanes`, `verifier`), so `complete_jsonT_of_level0`
leaves only `ShapeRest` (`complete_jsonT_of_shapeRest`).
-/

namespace FlockComplete.L0

open Flock FlockComplete FlockDeploy ZkTables

theorem complete_jsonT_of_shapeRest (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (gS : ShapeRest H) : Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_of_level0 H S proverRuns proverShape lanes verifier gS

end FlockComplete.L0
