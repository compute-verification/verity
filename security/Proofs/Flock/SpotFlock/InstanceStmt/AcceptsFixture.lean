module

public import Proofs.Flock.SpotFlock.InstanceStmt.RecordFits
public import Proofs.Flock.SpotFlock.InstanceStmt.Rel
public import Proofs.Flock.SpotFlock.InstanceStmt.ScopeZk
public import Proofs.Properties.Flock.Nets
public import Proofs.Flock.Claims.Complete.ZkTables.Sha512

/-! # The verifier accepts the honest run at the two-instance statement

`accepts_fixture`: at the statement `x r` of any plain run `r`, the masked Flock verifier accepts the honest masked prover
at some coins, whatever the prover's and the verifier's tapes: its witness is the zero trace `Rel.wX r`, which satisfies
the statement's relation (`Rel.rel_x`), the statement is in the masked scope (`scope_x`), its honest record fits its
length word (`recordFits_x`), and the fixed nets build (`fixedNetsBuild`); `accepts_x_of_scope` puts them together. `complete_fixture` is the probability form (`complete_masked_sha512`).
-/

@[expose] public section

namespace SpotFlock.InstanceStmt

/-- **The honest masked run at `x r` is accepted.** For every plain run `r` and every tape `ω` of the masked deployment, at
some coins `c` the verifier's verdict on the honest prover's run on the statement `x r`, with the zero trace as its
witness, is `acc`. -/
theorem accepts_fixture {r : Run} (hr : r.Plain) (ω : (FlockDeploy.deployZk Flock.Readable.relation).Tape) :
    ∃ c, (Claims.run FlockDeploy.sha512
        ((FlockDeploy.deployZk Flock.Readable.relation).P (Rel.wX r))
        (FlockDeploy.deployZk Flock.Readable.relation).V (x r) ω c).out = .acc :=
  accepts_x_of_scope Properties.Flock.Nets.fixedNetsBuild hr (w := Rel.wX r)
    (scope_x Properties.Flock.Nets.fixedNetsBuild hr) (Rel.rel_x Properties.Flock.Nets.fixedNetsBuild hr) ω

/-- **The honest masked run at `x r` is accepted with probability at least `1 − δ_c`** over the verifier's coins, at every
tape: `complete_masked_sha512` at the statement, the zero trace, `scope_x` and `rel_x`. -/
theorem complete_fixture {r : Run} (hr : r.Plain) (ω : (FlockDeploy.deployZk Flock.Readable.relation).Tape) :
    1 - (FlockDeploy.deployZk Flock.Readable.relation).δc FlockDeploy.sha512 (x r) ≤
      Claims.Pr fun c : Fin ((FlockDeploy.deployZk Flock.Readable.relation).V.k (x r)) → FlockDeploy.Coin =>
        (Claims.run FlockDeploy.sha512 ((FlockDeploy.deployZk Flock.Readable.relation).P (Rel.wX r))
          (FlockDeploy.deployZk Flock.Readable.relation).V (x r) ω c).out = .acc :=
  FlockComplete.ZkTables.complete_masked_sha512 (x r) (Rel.wX r) (scope_x Properties.Flock.Nets.fixedNetsBuild hr)
    (Rel.rel_x Properties.Flock.Nets.fixedNetsBuild hr) ω

end SpotFlock.InstanceStmt
