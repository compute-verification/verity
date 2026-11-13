module

public import Proofs.Flock.SpotFlock.Honest.Needed
public import Proofs.Flock.SpotFlock.AddressFacts.PortsOnceOf

/-!
# The address premise at the instance, from what an accepted design checks

Beside the row goal and `RefPorts R`, the five facts at the instance took two hypotheses (`five_of_portsOnce`). Both
are read off the design an accepted run evaluates (`two_of_ok`), so they ask nothing of `R`:

* each member's circuit names its ports once where the design evaluates (`portsOnce_of_rows`), so at the instance
  `PortsOnce` holds at an accepted run (`portsOnce_of_ok`);
* each unit's bit is copied at the column the layout reads it at where the design evaluates (`copiesAt_of_two`), and at
  the instance that is `CopiesHeld`, each bit's position being inside its port's row (`copiesHeld_of_at`,
  `copiesHeld_of_ok`).

The width then has no hypothesis (`widthFact_instance`), and the copy, the five facts, the address premise and the
guarantees' premises take the row goal and `RefPorts R` alone (`copyFact_of_goal`, `five_of_goal`,
`addressesConnected_of_goal`, `premises_of_goal`). `HonestExists` lacks those two and an accepted run
(`honest_of_goal`).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **`R`'s circuit names its ports once**, at an accepted run of the instance: its design evaluates, and there each
member's parse checks its rows' names. -/
theorem portsOnce_of_ok {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) :
    PortsOnce (held R B desc q c₀) := by
  obtain ⟨s, d, h2⟩ := two_of_ok R B desc q c₀ hr
  exact portsOnce_of_rows h2

/-- **`CopiesHeld` from the copy at each unit's bit**, where each commitment position's bit is inside its port's row:
the instance's one member is `R`'s circuit, and inside the row the place of a row position is its bit. -/
theorem copiesHeld_of_at (hC : CopiesAt (held R B desc q c₀)) (hB : BitsInPorts (held R B desc q c₀)) :
    CopiesHeld R B desc q c₀ := by
  intro k i rp hb
  obtain ⟨m, hm, hx⟩ := hC 0 k i rp hb
  have h0 : (held R B desc q c₀).ms[0]? = some (stmtOf R.x, #[0, 1]) := rfl
  obtain rfl : m = (stmtOf R.x, #[0, 1]) := Option.some.inj (hm.symm.trans h0)
  have hbit : rp.bit <
      16 * ((memberCircuit (held R B desc q c₀) 0).ports[portNamed (held R B desc q c₀) 0 rp.row]!).words := by
    rcases hb.2 with ⟨hs, -⟩ | ⟨-, hw, -, -⟩
    · exact hB.1 _ (List.mem_of_find?_eq_some hs)
    · exact hB.2 _ (List.mem_of_find?_eq_some hw)
  have e : ((placeAt (held R B desc q c₀) 0 rp).2 : ℕ) = rp.bit := by
    rewrite [placeAt, dite_eq_left hbit]
    rfl
  rw [e]
  exact hx

/-- **`CopiesHeld` at an accepted run of the instance.** -/
theorem copiesHeld_of_ok {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) :
    CopiesHeld R B desc q c₀ := by
  obtain ⟨s, d, h2⟩ := two_of_ok R B desc q c₀ hr
  have hO : PortsOnce (held R B desc q c₀) := portsOnce_of_rows h2
  exact copiesHeld_of_at R B desc q c₀ (copiesAt_of_two h2 hO) (bitsInPorts_of_once h2 hO)

/-- **The width at the instance**, with no hypothesis. -/
theorem widthFact_instance : WidthFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) := by
  intro r hr
  exact widthFact_of_portsOnce R B desc q c₀ (portsOnce_of_ok R B desc q c₀ hr) r hr

/-- **The copy at the instance**, from the row goal and `R`'s ports. -/
theorem copyFact_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) : CopyFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) := by
  intro r hr
  exact copyFact_held R B desc q c₀ hG hR (neededBits_held _) (copiesHeld_of_ok R B desc q c₀ hr) r hr

/-- **The five address facts at the instance**, from the row goal and `R`'s ports. -/
theorem five_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) : Five R B desc q c₀ :=
  ⟨widthFact_instance R B desc q c₀, copyFact_of_goal R B desc q c₀ hG hR,
    rowFact_held R B desc q c₀ hG hR (neededBits_held _), entryFact_held R B desc q c₀ hR (neededBits_held _),
    oneTreeFact_held R B desc q c₀ hR (neededBits_held _)⟩

/-- **The address premise at the instance**, from the row goal and `R`'s ports. -/
theorem addressesConnected_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) : AddressesConnected (held R B desc q c₀) :=
  addressesConnected_held R B desc q c₀ (five_of_goal R B desc q c₀ hG hR)

/-- **The two-stage guarantees' premises at the instance**, from the row goal and `R`'s ports. -/
theorem premises_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) : Premises (held R B desc q c₀) :=
  ⟨builtWired_held R B desc q c₀, addressesConnected_of_goal R B desc q c₀ hG hR⟩

/-- **What `HonestExists` lacks at the instance**: the row goal, the roots' conditions on `R`'s ports and an accepted
run. -/
theorem honest_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hA : Accepts (held R B desc q c₀)) : HonestExists :=
  honest_of_owed ⟨R, B, desc, q, c₀, addressesConnected_of_goal R B desc q c₀ hG hR, hA⟩

end SpotFlock.Honest
