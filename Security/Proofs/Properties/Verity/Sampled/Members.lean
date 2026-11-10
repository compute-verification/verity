module

public import Proofs.Properties.Verity.Sampled.Members.Draw
public import Proofs.Properties.Verity.Sampled.Members.Landing
public import Proofs.Properties.Verity.Sampled.Members.Project
public import Proofs.Properties.Verity.Sampled.Members.Session
public import Proofs.Properties.Verity.Sampled.Members.Widen

/-!
# The sampled leg at a window of member sessions

The period's one sample read member by member, behind the window's audit (`Flock.Guarantees.JointAuditV5`) under its
registered member layout (`JointAuditV5.Laid`):

* `Members.Draw`: the window's sample at the period's draw, and each member's sample and sample file from it;
* `Members.Widen`: the window's audit with its sample's coins widened by the period's other coins, and its bound in the
  registered audit's terms;
* `Members.Landing`: a strategy that leaves a state where its game ends, and such strategies one after another;
* `Members.Session`: the session's projection, leaving the position of the period's play it ends at;
* `Members.Project`: a strategy of the period's game as a strategy of the window's audit, the members' sessions'
  projections one after another on one play, with the joint law it simulates and its bound.
-/
