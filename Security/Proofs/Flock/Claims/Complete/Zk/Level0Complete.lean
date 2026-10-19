import Proofs.Flock.Claims.Complete.ZkTables.ShapeGo
import Proofs.Flock.Claims.Complete.Zk.Level0Premises
import Proofs.Flock.Claims.Complete.Zk.Level0Verifier
import Proofs.Flock.Claims.Complete.Zk.RankDeployed

/-!
# Masked completeness

Level 0's four premises are proved (`proverRuns`, `proverShape`, `lanes`, `verifier`), so `complete_jsonT_of_level0`
leaves the JSON specification and the mask-rank bound at the verifier's coins (`complete_jsonT_closed`), which
`MaskRank.maskRankBound_of` gives from `DeployedRankBound` and `SmallSlotBound` (`complete_jsonT_of_rank`). Both are
proved (`MaskRank.maskRankBound`), so masked completeness rests on the JSON specification alone
(`complete_jsonT_of_spec`).
-/

namespace FlockComplete.L0

open Flock FlockComplete FlockDeploy ZkTables

theorem complete_jsonT_closed (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (hMR : MaskRank.MaskRankBound) : Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_of_level0 H S proverRuns proverShape lanes verifier hMR

theorem complete_jsonT_of_rank (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (hD : MaskRank.DeployedRankBound) (hS : MaskRank.SmallSlotBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_closed H S (MaskRank.maskRankBound_of hD hS)

/-- **Masked completeness at the JSON specification alone**: `complete_jsonT_closed` with its rank premise
discharged (`MaskRank.maskRankBound`). -/
theorem complete_jsonT_of_spec (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_closed H S MaskRank.maskRankBound

end FlockComplete.L0
