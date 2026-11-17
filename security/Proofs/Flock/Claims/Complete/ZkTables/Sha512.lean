module

public import Proofs.Flock.Claims.Complete.ZkTables.Accepts
import Proofs.Flock.Claims.Recursion.HonestVerdictFinal

/-! Masked completeness and an accepted masked run at SHA-512, where `Claims.Out64` holds (`complete_masked_sha512`,
`accepts_masked_sha512`). -/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock FlockDeploy FlockComplete

theorem complete_masked_sha512 (x : Claims.Msg) (w : Claims.Flock.Wit)
    (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    1 - (deployZk Readable.relation).δc FlockDeploy.sha512 x ≤
      Claims.Pr fun r : Fin ((deployZk Readable.relation).V.k x) → FlockDeploy.Coin =>
        (Claims.run FlockDeploy.sha512 ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω
          r).out = .acc :=
  complete_masked FlockDeploy.sha512 out64_sha512 x w hS hR ω

theorem accepts_masked_sha512 {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : ZkIn x s spec sts) (hready : readyZk FlockDeploy.sha512 x = true) {w : Claims.Flock.Wit}
    (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    ∃ r, (Claims.run FlockDeploy.sha512 ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω
      r).out = .acc :=
  accepts_masked FlockDeploy.sha512 out64_sha512 hin hready hS hR ω

end FlockComplete.ZkTables
