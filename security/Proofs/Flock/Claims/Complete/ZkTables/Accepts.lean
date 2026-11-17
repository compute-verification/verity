module

public import Proofs.Flock.Claims.Complete.ZkTables.ShapeGo
public import Proofs.Flock.Claims.Complete.Zk.Level0Premises
public import Proofs.Flock.Claims.Complete.Zk.RankDeployed
public import Proofs.Flock.Claims.Complete.Coins
public import Proofs.Flock.Claims.Complete.Zk.Level0Verifier
import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Recursion.DeployChecked

/-!
# Masked completeness and an accepted masked run

`complete_jsonT`'s four table-level premises reduce to level 0's four (`Level0`), of which `Zk.Level0Premises` proves
three, `Zk.Level0Verifier` the fourth (`L0.verifier`), and `RankDeployed` the mask-rank bound. So the four goals
`complete_zk` reads (`RepShapedZk`, `ProverRunsZk`, `LigeritoZk`, `Inner`) and the masked completeness claim hold:

* `repShapedZk`: `RepShapedZk` at the deployed relation holds.
* `proverRunsZk_of_verifier`, `lgInnerZk_of_verifier`: the other three from `Level0.Verifier`.
* `complete_zk_of_verifier`: `Claims.Flock.Complete H (deployZk Readable.relation)` from `Level0.Verifier`.
* `accepts_zk_of_verifier`: at a statement the masked verifier can accept (`readyZk H`) in the session the numbers are
  stated at (`ZkIn`), in scope and with a witness, the honest masked prover is accepted at some coins of every tape,
  since the error there is below `1` (`Claims.Flock.Complete_meets_zk`).
* `complete_masked`, `accepts_masked`: the two at `L0.verifier`.
-/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock FlockDeploy FlockComplete

variable (H : Claims.Msg → Claims.Msg)

/-- **`RepShapedZk` holds at the deployed relation**: `RepShapedT` beyond level 0's shape (`shapeRest`) at level 0's
shape (`L0.proverShape`). -/
theorem repShapedZk (h64 : Claims.Out64 H) : JsonRT.RepShapedZk H Readable.relation :=
  repShapedZk_of (shapeRest H L0.proverShape) h64

/-- **`ProverRunsZk` at the deployed relation from level 0's verifier premise** (`runsRest`). -/
theorem proverRunsZk_of_verifier (hV : Level0.Verifier) (h64 : Claims.Out64 H) :
    JsonRT.ProverRunsZk H Readable.relation :=
  proverRunsZk_of (runsRest H L0.proverRuns L0.lanes hV) h64

/-- **`LigeritoZk` and `Inner` at the deployed relation from level 0's verifier premise** (`ligeritoRest`,
`innerRest`). -/
theorem lgInnerZk_of_verifier (hV : Level0.Verifier) (h64 : Claims.Out64 H) :
    FlockComplete.LigeritoZk H Readable.relation ∧ FlockComplete.Inner H Readable.relation :=
  lgInnerZk_of h64 (shapeRest H L0.proverShape) (runsRest H L0.proverRuns L0.lanes hV) (ligeritoRest H hV)
    (innerRest H hV)

/-- **Flock's masked verifier is complete at the deployed relation from level 0's verifier premise**:
`complete_jsonT_of_level0` at level 0's other three (`L0.proverRuns`, `L0.proverShape`, `L0.lanes`) and the mask-rank
bound (`MaskRank.maskRankBound`). -/
theorem complete_zk_of_verifier (hV : Level0.Verifier) : Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_of_level0 H L0.proverRuns L0.proverShape L0.lanes hV MaskRank.maskRankBound

/-- **The honest masked prover is accepted at some coins**, from level 0's verifier premise: at a statement the masked
verifier can accept (`readyZk H`), in the session the numbers are stated at (`ZkIn`), in scope and with a witness, for
every tape, masked completeness (`complete_zk_of_verifier`) bounds the probability of acceptance below by `1 - δc`, which
is positive there (`Claims.Flock.Complete_meets_zk`). -/
theorem accepts_zk_of_verifier (hV : Level0.Verifier) (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins}
    {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} (hin : ZkIn x s spec sts) (hready : readyZk H x = true)
    {w : Claims.Flock.Wit} (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    ∃ r, (Claims.run H ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω r).out = .acc :=
  RecDeploy.exists_of_pr_pos (lt_of_lt_of_le
    (by linarith [(Claims.Flock.Complete_meets_zk Readable.relation H hin hready).2])
    (complete_zk_of_verifier H hV h64 x w hS hR ω))

/-- **Flock's masked verifier is complete at the deployed relation**: `complete_zk_of_verifier` at level 0's verifier
premise (`L0.verifier`). -/
theorem complete_masked : Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_zk_of_verifier H L0.verifier

/-- **The honest masked prover is accepted at some coins**: at a statement the masked verifier can accept (`readyZk H`),
in the session the numbers are stated at (`ZkIn`), in scope and with a witness, for every tape (`accepts_zk_of_verifier`
at `L0.verifier`). -/
theorem accepts_masked (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hin : ZkIn x s spec sts) (hready : readyZk H x = true) {w : Claims.Flock.Wit}
    (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    ∃ r, (Claims.run H ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω r).out = .acc :=
  accepts_zk_of_verifier H L0.verifier h64 hin hready hS hR ω

end FlockComplete.ZkTables
