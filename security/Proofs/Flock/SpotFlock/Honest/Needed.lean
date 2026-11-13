module

public import Proofs.Flock.SpotFlock.Honest.Once
public import Proofs.Flock.SpotFlock.NeededBitsHeld

/-!
# The address premise at the instance without `NeededBits`

Every gate a checked unit needs is one of its own unit's bits at every held code (`neededBits_held`), so the width,
the five facts and the address premise at the instance drop that hypothesis (`widthFact_of_portsOnce`,
`five_of_portsOnce`, `addressesConnected_of_portsOnce`). What they still take: the row goal, `RefPorts R`, each member
naming its ports once (`PortsOnce`) and `CopiesHeld`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The width at the instance**, where its member names its ports once. -/
theorem widthFact_of_portsOnce (hO : PortsOnce (held R B desc q c₀)) :
    WidthFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) :=
  widthFact_of_once R B desc q c₀ (neededBits_held _) hO

/-- **The five address facts at the instance**, without `NeededBits`. -/
theorem five_of_portsOnce
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hO : PortsOnce (held R B desc q c₀)) (hC : CopiesHeld R B desc q c₀) :
    Five R B desc q c₀ :=
  five_of_once R B desc q c₀ hG hR hO hC (neededBits_held _)

/-- **The address premise at the instance**, without `NeededBits`. -/
theorem addressesConnected_of_portsOnce
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hO : PortsOnce (held R B desc q c₀)) (hC : CopiesHeld R B desc q c₀) :
    AddressesConnected (held R B desc q c₀) :=
  addressesConnected_of_once R B desc q c₀ hG hR hO hC (neededBits_held _)

end SpotFlock.Honest
