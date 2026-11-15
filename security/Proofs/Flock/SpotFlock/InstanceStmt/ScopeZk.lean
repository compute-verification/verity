module

public import Proofs.Flock.SpotFlock.InstanceStmt.Scope
public import Proofs.Flock.Claims.DeployCode

/-! # The two-instance statement in Flock's masked scope, as the claims read it

`scope_x` is `Claims.Flock.Scope` at the masked deployment and the statement `x r`: the deployment's scope check passes
there (`scopeCheck_x`), and that check is the scope (`FlockDeploy.scopeMatches_zk`).
-/

@[expose] public section

open Properties.Flock.Nets (FixedNetsBuild)

namespace SpotFlock.InstanceStmt

/-- **The statement is in Flock's masked scope**, where the verifier's two fixed nets build. -/
theorem scope_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    Claims.Flock.Scope (FlockDeploy.deployZk Flock.Readable.relation) (x r) :=
  (FlockDeploy.scopeMatches_zk Flock.Readable.relation (x r)).2 (scopeCheck_x hn hr)

end SpotFlock.InstanceStmt
