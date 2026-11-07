module

public import Proofs.Flock.Claims.ZK.FrameZK
public import Proofs.Flock.SessionLoop.Concrete.Stmt

/-!
# The session loop serves the concrete statement (`ConcreteServes`)

The one fact `SessionLoopZKAnyDeployed`'s honest witness takes that nothing in Lean proves yet: the session loop's
`Hello` leaves at the concrete `--zk` statement (`Loop.Serves FlockDeploy.concreteStmt`). The setup parses it
(`FlockDeploy.concreteStmt_setupOf`); what is left is that its sizes are within the frames' (`Session.sizesOk` at
`FlockDeploy.concreteSetup`) and that the verifier's `Hello` there is JSON (`Hello.msg`), two computations on one
statement.
-/

@[expose] public section

namespace Claims.ZKServer.Loop

/-- **The session loop serves the concrete statement**: `Serves` at `FlockDeploy.concreteStmt`. -/
def ConcreteServes : Prop := Serves FlockDeploy.concreteStmt

end Claims.ZKServer.Loop
