module

public import Proofs.Flock.SpotFlock.InstanceStmt.Scope
public import Proofs.Flock.Claims.DeployCode

/-! # The two-instance statement in Flock's masked scope, as the claims read it

`scope_x` is `Claims.Flock.Scope` at the masked deployment and the statement `x r`: the deployment's scope check passes
there (`scopeCheck_x`), and that check is the scope (`FlockDeploy.scopeMatches_zk`, through `scope_of_check`).
-/

@[expose] public section

open Properties.Flock.Nets (FixedNetsBuild)

namespace SpotFlock.InstanceStmt

/-- A statement the deployed `--zk` scope check passes is in the masked deployment's scope. Stated at any statement, so
matching the check against the deployment's field never evaluates a concrete one. -/
theorem scope_of_check {m} (h : FlockDeploy.scopeCheck true m = true) :
    Claims.Flock.Scope (FlockDeploy.deployZk Flock.Readable.relation) m :=
  (FlockDeploy.scopeMatches_zk Flock.Readable.relation m).2 h

/-- **The statement is in Flock's masked scope**, where the verifier's two fixed nets build. -/
theorem scope_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    Claims.Flock.Scope (FlockDeploy.deployZk Flock.Readable.relation) (x r) :=
  scope_of_check (scopeCheck_x hn hr)

end SpotFlock.InstanceStmt
