module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressesGive
public import Proofs.Flock.SpotFlock.NeededBitsHeld
public import Proofs.Flock.SpotFlock.AddressFacts.OfRuns

/-!
# The address premise from the row and the own reads

Every fact the address premise asks is stated over accepted runs, and an accepted run's design evaluates
(`CheckedRun.design_of_ok`). Where the design evaluates, the registered wiring passes its check
(`CheckedRun.wiring_of_ok`), every member's bits are rowed (`membersRowed_of_two`) and each commitment position's row
is a port of its member's circuit (`rowsArePorts_of_two`); at every held code, every gate a proof block needs is one
of its own bits (`neededBits_held`) and a gate is at one written position at most (`writtenOnce_held`). So the wiring
facts hold at each accepted run of any held code where a registered read at a port no link or wire names reads a port
of the interior record (`OwnReads`): `wiring_of_ownReads`.

With the row at the address map they give the width, the copy, the entry and one tree a gate, so the address premise
(`addressesConnected_of_row`): of `addressesConnected_of_runs`' eight premises, `OwnReads` and the row are left. At
the instance that is the design premises (`premises_of_row`) and, with an accepted run, `HonestExists`
(`honest_of_row`).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

/-- **The wiring facts at an accepted run**, where a registered read at a port no link or wire names reads a port of
the interior record: the run's design evaluates, and the rest holds wherever it does. -/
theorem wiring_of_ownReads {h : Held} (hOwn : OwnReads h) {r : CheckedRun h.stage} (hr : r.Ok h) : Wiring h := by
  obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
  have hO : WrittenOnce h := writtenOnce_held h
  exact wiring_of_wiringOk (CheckedRun.wiring_of_ok hr).2 (neededBits_held h)
    (inside_of_rowed (membersRowed_of_two h h2) hO) hO (rowsArePorts_of_two h h2) hOwn

/-- **The address premise, from the row at the address map and the own reads.** -/
theorem addressesConnected_of_row {h : Held} (hOwn : OwnReads h)
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h := by
  have hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h := fun _ hr => wiring_of_ownReads hOwn hr
  have hE : EntryFact h (wireMap h) (returned h) (addressOf h) := fun r hr i hi =>
    entryFact_of_wiring (hW r hr) r hr i hi
  have hT : OneTreeFact h (wireMap h) (returned h) (addressOf h) :=
    ⟨addressOf_shares h, fun r hr i hi => (oneTreeFact_of_wiring (hW r hr)).2 r hr i hi⟩
  exact addressesConnected_of_facts h fun _ =>
    ⟨addressOf h, widthFact_of_runs hW, copyFact_of_runs hW hRow, hRow, hE, hT⟩

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The design premises at the instance**, from the row at the address map and the own reads. -/
theorem premises_of_row (hOwn : OwnReads (held R B desc q c₀))
    (hRow : RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀))) : Premises (held R B desc q c₀) :=
  ⟨builtWired_held R B desc q c₀, addressesConnected_of_row hOwn hRow⟩

/-- **`HonestExists`**, from the row at the instance's address map, its own reads and a run it accepts. -/
theorem honest_of_row (hOwn : OwnReads (held R B desc q c₀))
    (hRow : RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀))) (hA : Accepts (held R B desc q c₀)) : HonestExists :=
  honest_of_owed ⟨R, B, desc, q, c₀, addressesConnected_of_row hOwn hRow, hA⟩

end SpotFlock.Honest
