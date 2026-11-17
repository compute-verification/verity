module

public import Proofs.Flock.SessionLoop.Concrete.Live
public import Proofs.Properties.Flock.Nets

/-! The concrete statement is `Live`, unconditionally, in a module of its own so that what imports `Live.lean` doesn't build the nets' proof. -/

@[expose] public section

namespace FlockDeploy

theorem concreteStmt_rel :
    _root_.Claims.Flock.R (deployZk _root_.Flock.Readable.relation).parse concreteStmt wC :=
  rel_concreteStmt Properties.Flock.Nets.fixedNetsBuild

theorem concreteStmt_live : _root_.Claims.ZKMal.Masked.Live concreteStmt :=
  live_concreteStmt Properties.Flock.Nets.fixedNetsBuild

theorem concreteStmt_jOf : _root_.Claims.ZKMal.Masked.jOf concreteStmt = 1 :=
  jOf_concreteStmt Properties.Flock.Nets.fixedNetsBuild

end FlockDeploy
