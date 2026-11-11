module

public import Proofs.Flock.SessionLoop.Concrete.Live
public import Proofs.Properties.Flock.Nets

/-! # The concrete statement is `Live`, unconditionally

`Live.lean` proves the concrete statement live where the verifier's two fixed nets build (`FixedNetsBuild`), and
`Properties.Flock.Nets.fixedNetsBuild` proves that they do. Applied here, in a module of its own so that what imports
`Live.lean` doesn't build the nets' proof: the concrete statement is in the deployed relation with the trace `wC` as its
witness (`concreteStmt_rel`), it is `Live` (`concreteStmt_live`), and its masked session has one table
(`concreteStmt_jOf`).
-/

@[expose] public section

namespace FlockDeploy

/-- **The concrete statement is in the deployed relation** with the trace `wC` as its witness. -/
theorem concreteStmt_rel :
    _root_.Claims.Flock.R (deployZk _root_.Flock.Readable.relation).parse concreteStmt wC :=
  rel_concreteStmt Properties.Flock.Nets.fixedNetsBuild

/-- **The concrete statement is `Live`.** -/
theorem concreteStmt_live : _root_.Claims.ZKMal.Masked.Live concreteStmt :=
  live_concreteStmt Properties.Flock.Nets.fixedNetsBuild

/-- **The concrete statement's masked session has one table.** -/
theorem concreteStmt_jOf : _root_.Claims.ZKMal.Masked.jOf concreteStmt = 1 :=
  jOf_concreteStmt Properties.Flock.Nets.fixedNetsBuild

end FlockDeploy
