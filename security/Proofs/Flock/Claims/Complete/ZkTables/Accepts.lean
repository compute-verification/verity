module

public import Proofs.Flock.Claims.Complete.ZkTables.ShapeGo
public import Proofs.Flock.Claims.Complete.Zk.Level0Premises
public import Proofs.Flock.Claims.Complete.Zk.RankDeployed
public import Proofs.Flock.Claims.Complete.Coins
public import Proofs.Flock.Claims.Complete.Zk.Level0Verifier
import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Recursion.DeployChecked

/-! Masked completeness at the deployed relation (`complete_masked`) and an accepted run of the honest masked prover
(`accepts_masked`). -/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock FlockDeploy FlockComplete

variable (H : Claims.Msg → Claims.Msg)

theorem repShapedZk (h64 : Claims.Out64 H) : JsonRT.RepShapedZk H Readable.relation :=
  repShapedZk_of (shapeRest H L0.proverShape) h64

theorem proverRunsZk_of_verifier (hV : Level0.Verifier) (h64 : Claims.Out64 H) :
    JsonRT.ProverRunsZk H Readable.relation :=
  proverRunsZk_of (runsRest H L0.proverRuns L0.lanes hV) h64

theorem lgInnerZk_of_verifier (hV : Level0.Verifier) (h64 : Claims.Out64 H) :
    FlockComplete.LigeritoZk H Readable.relation ∧ FlockComplete.Inner H Readable.relation :=
  lgInnerZk_of h64 (shapeRest H L0.proverShape) (runsRest H L0.proverRuns L0.lanes hV) (ligeritoRest H hV)
    (innerRest H hV)

theorem complete_zk_of_verifier (hV : Level0.Verifier) : Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT_of_level0 H L0.proverRuns L0.proverShape L0.lanes hV MaskRank.maskRankBound

theorem accepts_zk_of_verifier (hV : Level0.Verifier) (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins}
    {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} (hin : ZkIn x s spec sts) (hready : readyZk H x = true)
    {w : Claims.Flock.Wit} (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    ∃ r, (Claims.run H ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω r).out = .acc :=
  RecDeploy.exists_of_pr_pos (lt_of_lt_of_le
    (by linarith [(Claims.Flock.Complete_meets_zk Readable.relation H hin hready).2])
    (complete_zk_of_verifier H hV h64 x w hS hR ω))

theorem complete_masked : Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_zk_of_verifier H L0.verifier

theorem accepts_masked (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hin : ZkIn x s spec sts) (hready : readyZk H x = true) {w : Claims.Flock.Wit}
    (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    ∃ r, (Claims.run H ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω r).out = .acc :=
  accepts_zk_of_verifier H L0.verifier h64 hin hready hS hR ω

end FlockComplete.ZkTables
