module

public import Proofs.Flock.SpotFlock.Honest.Five
public import Proofs.Flock.SpotFlock.AddressFacts.WrittenOf

/-!
# The width at the instance from the template's port names

Where the design evaluates and each member's circuit names each of its ports once (`PortsOnce`), each commitment
position's bit is inside its port's row (`bitsInPorts_of_once`); at the instance that is `RefWidths R`
(`refWidths_of_bits`). An accepted run's design evaluates (`two_of_ok`), so the width fact, the five facts and the
address premise at the instance take `PortsOnce` where they took `RefWidths` (`widthFact_of_once`, `five_of_once`,
`addressesConnected_of_once`): a condition on the template's port names, where `RefWidths` asked each bit's
position.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **Each bit's position is inside its port's row at `R`**, where each commitment position's bit is inside its
port's row at the instance: its one member's circuit is `R`'s. -/
theorem refWidths_of_bits (hB : BitsInPorts (held R B desc q c₀)) : RefWidths R := by
  intro y hy
  rcases List.mem_append.1 hy with hs | hw
  · exact hB.1 y hs
  · exact hB.2 y hw

/-- **An accepted run's design evaluates.** -/
theorem two_of_ok {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) :
    ∃ s d, two (held R B desc q c₀) s = some d := by
  obtain ⟨Pt, d, spec, hx, hP, h2, -⟩ := hr
  exact ⟨_, d, h2⟩

/-- **The width at the instance, from the template's port names**: at an accepted run the design evaluates, so
with each port named once every bit's position is inside its port's row. -/
theorem widthFact_of_once (hN : NeededBits (held R B desc q c₀)) (hO : PortsOnce (held R B desc q c₀)) :
    WidthFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) := by
  intro r hr
  obtain ⟨s, d, h2⟩ := two_of_ok R B desc q c₀ hr
  exact widthFact_held R B desc q c₀ hN (refWidths_of_bits R B desc q c₀ (bitsInPorts_of_once h2 hO)) r hr

/-- **The five address facts at the instance**, the width from the template's port names. -/
theorem five_of_once
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hO : PortsOnce (held R B desc q c₀)) (hC : CopiesHeld R B desc q c₀)
    (hN : NeededBits (held R B desc q c₀)) : Five R B desc q c₀ :=
  ⟨widthFact_of_once R B desc q c₀ hN hO, copyFact_held R B desc q c₀ hG hR hN hC, rowFact_held R B desc q c₀ hG hR hN,
    entryFact_held R B desc q c₀ hR hN, oneTreeFact_held R B desc q c₀ hR hN⟩

/-- **The address premise at the instance**, the width from the template's port names. -/
theorem addressesConnected_of_once
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hO : PortsOnce (held R B desc q c₀)) (hC : CopiesHeld R B desc q c₀)
    (hN : NeededBits (held R B desc q c₀)) : AddressesConnected (held R B desc q c₀) :=
  addressesConnected_held R B desc q c₀ (five_of_once R B desc q c₀ hG hR hO hC hN)

end SpotFlock.Honest
