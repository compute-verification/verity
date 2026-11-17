module

public import Proofs.Flock.Claims.Complete.ZkTables.Accepts
import Proofs.Flock.Claims.Recursion.HonestVerdictFinal

/-!
# Masked completeness at SHA-512

`complete_masked` and `accepts_masked` at the deployed hash, whose outputs are 64 bytes (`out64_sha512`), so neither
reads `Claims.Out64`: `complete_masked_sha512` is `Claims.Complete`'s body at SHA-512, `accepts_masked_sha512` an
accepted masked run at SHA-512.
-/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock FlockDeploy FlockComplete

/-- **Flock's masked verifier is complete at SHA-512**: on every statement in scope and witness in the relation, the
honest masked prover is accepted with probability at least `1 − δ_c(x)` over the verifier's coins, at every tape. -/
theorem complete_masked_sha512 (x : Claims.Msg) (w : Claims.Flock.Wit)
    (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    1 - (deployZk Readable.relation).δc FlockDeploy.sha512 x ≤
      Claims.Pr fun r : Fin ((deployZk Readable.relation).V.k x) → FlockDeploy.Coin =>
        (Claims.run FlockDeploy.sha512 ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω
          r).out = .acc :=
  complete_masked FlockDeploy.sha512 out64_sha512 x w hS hR ω

/-- **The honest masked prover is accepted at some coins at SHA-512**: at a statement the masked verifier can accept
(`readyZk sha512`), in the session the numbers are stated at (`ZkIn`), in scope and with a witness, at every tape. -/
theorem accepts_masked_sha512 {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : ZkIn x s spec sts) (hready : readyZk FlockDeploy.sha512 x = true) {w : Claims.Flock.Wit}
    (hS : Claims.Flock.Scope (deployZk Readable.relation) x)
    (hR : Claims.Flock.R (deployZk Readable.relation).parse x w) (ω : (deployZk Readable.relation).Tape) :
    ∃ r, (Claims.run FlockDeploy.sha512 ((deployZk Readable.relation).P w) (deployZk Readable.relation).V x ω
      r).out = .acc :=
  accepts_masked FlockDeploy.sha512 out64_sha512 hin hready hS hR ω

end FlockComplete.ZkTables
