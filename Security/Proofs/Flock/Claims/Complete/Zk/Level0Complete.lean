import Proofs.Flock.Claims.Complete.ZkTables.ShapeGo
import Proofs.Flock.Claims.Complete.Zk.Level0Premises
import Proofs.Flock.Claims.Complete.Zk.Level0Verifier

/-!
# Masked completeness

Level 0's four premises are proved (`proverRuns`, `proverShape`, `lanes`, `verifier`), so `complete_jsonT_of_level0`
leaves only the JSON specification (`complete_jsonT_closed`).
-/

namespace FlockComplete.L0

open Flock FlockComplete FlockDeploy ZkTables

theorem complete_jsonT_closed (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_of_level0 H S proverRuns proverShape lanes verifier

end FlockComplete.L0
