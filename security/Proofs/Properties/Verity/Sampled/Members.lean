module

public import Proofs.Properties.Verity.Sampled.Members.Draw
public import Proofs.Properties.Verity.Sampled.Members.Widen

/-!
# The sampled leg at a window of member sessions

The period's one sample read member by member, behind the window's audit (`Flock.Guarantees.JointAuditV5`) under its
registered member layout (`JointAuditV5.Laid`):

* `Members.Draw`: the window's sample at the period's draw, and each member's sample and sample file from it;
* `Members.Widen`: the window's audit with its sample's coins widened by the period's other coins, and its bound in the
  registered audit's terms.

The projection through the members' sessions and the coin map over them are `Window.Walk` and `Window.Draw`.
-/
