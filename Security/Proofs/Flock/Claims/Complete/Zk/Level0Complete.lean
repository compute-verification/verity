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

The record's prover is `flock-circuit --zk`'s, `ZkTables.honestPR` (`(deployZk rel).P`), and its error `ZkTables.δc`:
`complete_of` takes `honestP`'s acceptance outside `BadZk` to `honestPR`'s through `run_honestPR`, at the witness with
the tape's mask words, which `maskFree_relation` keeps in the relation. Its refusals (an outer coin `1`, a degenerate
mask rank) are `BadZk`'s, so the error is `εZk`'s with no further term. `complete_jsonT_honestPR` and
`complete_jsonT_honestPR_of_rank` state it with the prover and the error written out.
-/

namespace FlockComplete.L0

open Flock FlockComplete FlockDeploy ZkTables

theorem complete_jsonT_closed (H : Claims.Msg → Claims.Msg) 
    (hMR : MaskRank.MaskRankBound) : Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_of_level0 H proverRuns proverShape lanes verifier hMR

theorem complete_jsonT_of_rank (H : Claims.Msg → Claims.Msg) 
    (hD : MaskRank.DeployedRankBound) (hS : MaskRank.SmallSlotBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_closed H (MaskRank.maskRankBound_of hD hS)

/-- **Masked completeness at the JSON specification alone**: `complete_jsonT_closed` with its rank premise
discharged (`MaskRank.maskRankBound`). -/
theorem complete_jsonT_of_spec (H : Claims.Msg → Claims.Msg) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_closed H MaskRank.maskRankBound

/-- **`flock-circuit --zk`'s prover is complete** at the deployed masked verifier and relation, with error
`ZkTables.δc H`. -/
theorem complete_jsonT_honestPR (H : Claims.Msg → Claims.Msg) 
    (hMR : MaskRank.MaskRankBound) :
    CompleteWith H (deployZk Readable.relation) ZkTables.honestPR (ZkTables.δc H) :=
  complete_jsonT_closed H hMR

theorem complete_jsonT_honestPR_of_rank (H : Claims.Msg → Claims.Msg) 
    (hD : MaskRank.DeployedRankBound) (hS : MaskRank.SmallSlotBound) :
    CompleteWith H (deployZk Readable.relation) ZkTables.honestPR (ZkTables.δc H) :=
  complete_jsonT_of_rank H hD hS

end FlockComplete.L0
