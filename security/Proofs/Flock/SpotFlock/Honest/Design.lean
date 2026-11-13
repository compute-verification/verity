module

public import Proofs.Flock.SpotFlock.Honest.Needed
public import Proofs.Flock.SpotFlock.Honest.Written
public import Proofs.Flock.SpotFlock.AddressFacts.PortsOnceOf

/-!
# The address premise at the instance, from what an accepted design checks

Beside the row goal, the five facts at the instance took `RefPorts R` and two hypotheses more
(`five_of_portsOnce`). Three of these are read off the design an accepted run evaluates (`two_of_ok`), so they ask
nothing of `R`:

* each member's circuit names its ports once where the design evaluates (`portsOnce_of_rows`), so at the instance
  `PortsOnce` holds at an accepted run (`portsOnce_of_ok`);
* each unit's bit is copied at the column the layout reads it at where the design evaluates (`copiesAt_of_two`), and at
  the instance that is `CopiesHeld`, each bit's position being inside its port's row (`copiesHeld_of_at`,
  `copiesHeld_of_ok`);
* every written bit's row is one of `R`'s ports where the design evaluates (`written_of_two`), the first half of
  `RefPorts R` (`refPorts_of_ok`).

What is left of `RefPorts R` is that no port of `R` is named like an opening row. The width has no hypothesis
(`widthFact_instance`); the entry and the one tree take the ports' names (`entryFact_of_ports`,
`oneTreeFact_of_ports`); the copy, the row, the five facts, the address premise and the guarantees' premises take the
row goal and the ports' names (`copyFact_of_goal`, `rowFact_of_goal`, `five_of_goal`, `addressesConnected_of_goal`,
`premises_of_goal`). `HonestExists` lacks those two and an accepted run (`honest_of_goal`).
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

/-- **What the roots ask of `R`'s ports, at an accepted run of the instance**: its design evaluates, so the wiring's
check put every written bit's row among `R`'s ports, and what is left is that none is named like an opening row. -/
theorem refPorts_of_ok (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false)
    {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) : RefPorts R := by
  obtain ⟨s, d, h2⟩ := two_of_ok R B desc q c₀ hr
  exact refPorts_of_two R B desc q c₀ hP h2

/-- **The copy at the instance**, from the row goal and the ports' names. -/
theorem copyFact_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false) :
    CopyFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) := by
  intro r hr
  exact copyFact_held R B desc q c₀ hG (refPorts_of_ok R B desc q c₀ hP hr) (neededBits_held _)
    (copiesHeld_of_ok R B desc q c₀ hr) r hr

/-- **The row at the instance**, from the row goal and the ports' names. -/
theorem rowFact_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false) :
    RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) := by
  intro r hr
  exact rowFact_held R B desc q c₀ hG (refPorts_of_ok R B desc q c₀ hP hr) (neededBits_held _) r hr

/-- **The entry at the instance**, from the ports' names. -/
theorem entryFact_of_ports (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false) :
    EntryFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) := by
  intro r hr
  exact entryFact_held R B desc q c₀ (refPorts_of_ok R B desc q c₀ hP hr) (neededBits_held _) r hr

/-- **One tree a gate at the instance**, from the ports' names: the wiring map moves nothing, and the roots hold at
each accepted run. -/
theorem oneTreeFact_of_ports (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false) :
    OneTreeFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) :=
  ⟨fun i => by rw [wireMap_held]; exact ⟨rfl, rfl⟩,
    fun r hr => rootsHold_held R B desc q c₀ (refPorts_of_ok R B desc q c₀ hP hr) (neededBits_held _) r hr⟩

/-- **The five address facts at the instance**, from the row goal and the ports' names. -/
theorem five_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false) : Five R B desc q c₀ :=
  ⟨widthFact_instance R B desc q c₀, copyFact_of_goal R B desc q c₀ hG hP, rowFact_of_goal R B desc q c₀ hG hP,
    entryFact_of_ports R B desc q c₀ hP, oneTreeFact_of_ports R B desc q c₀ hP⟩

/-- **The address premise at the instance**, from the row goal and the ports' names. -/
theorem addressesConnected_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false) :
    AddressesConnected (held R B desc q c₀) :=
  addressesConnected_held R B desc q c₀ (five_of_goal R B desc q c₀ hG hP)

/-- **The two-stage guarantees' premises at the instance**, from the row goal and the ports' names. -/
theorem premises_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false) :
    Premises (held R B desc q c₀) :=
  ⟨builtWired_held R B desc q c₀, addressesConnected_of_goal R B desc q c₀ hG hP⟩

/-- **What `HonestExists` lacks at the instance**: the row goal, that no port of `R` is named like an opening row,
and an accepted run. -/
theorem honest_of_goal
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hP : ∀ p ∈ R.ports, Spot.SecondStage.OPENING_ROW.contains p = false)
    (hA : Accepts (held R B desc q c₀)) : HonestExists :=
  honest_of_owed ⟨R, B, desc, q, c₀, addressesConnected_of_goal R B desc q c₀ hG hP, hA⟩

end SpotFlock.Honest
