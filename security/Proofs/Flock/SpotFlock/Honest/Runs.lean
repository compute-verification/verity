module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressesGive
public import Proofs.Flock.SpotFlock.NeededBitsHeld
public import Proofs.Flock.SpotFlock.AddressFacts.OfRuns

/-!
# The address premise from the row and well-formed reference reads

Every fact the address premise asks is stated over accepted runs, and an accepted run's design evaluates
(`CheckedRun.design_of_ok`). Where the design evaluates, the registered wiring passes its check
(`CheckedRun.wiring_of_ok`), every wired source bit has a gate to go to (`CheckedRun.wiresMove_of_ok`), every
member's bits are rowed (`membersRowed_of_two`) and each commitment position's row is a port of its member's circuit
(`rowsArePorts_of_two`); at every held code, every gate a proof block needs is one of its own bits
(`neededBits_held`) and a gate is at one written position at most (`writtenOnce_held`). So the wiring facts hold at
each accepted run of any held code whose reference reads are well formed (`RefReadsWf`): `wiring_of_refReads`.

With the row at the address map they give the width and the copy (`widthFact_of_runs`, `copyFact_of_runs`), so the
address premise (`addressesConnected_of_row`): of `addressesConnected_of_runs`' eight premises, `RefReadsWf` and the
row are left. At the instance the reference reads are `R`'s (`refReadsWf_held`), so that is the design premises
(`premises_of_row`) and, with an accepted run, `HonestExists` (`honest_of_row`).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

/-- **The wiring facts at an accepted run**, where the reference reads are well formed: the run's design evaluates,
and the rest holds wherever it does. -/
theorem wiring_of_refReads {h : Held} (hWf : RefReadsWf h) {r : CheckedRun h.stage} (hr : r.Ok h) : Wiring h := by
  obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
  have hO : WrittenOnce h := writtenOnce_held h
  exact wiring_of_wiringOk (CheckedRun.wiring_of_ok hr).2 (CheckedRun.wiresMove_of_ok hr) (neededBits_held h)
    (inside_of_rowed (membersRowed_of_two h h2) hO) hO (rowsArePorts_of_two h h2) hWf

/-- **The address premise, from the row at the address map and well-formed reference reads.** -/
theorem addressesConnected_of_row {h : Held} (hWf : RefReadsWf h)
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h := by
  have hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h := fun _ hr => wiring_of_refReads hWf hr
  exact addressesConnected_of_wiring h hW (widthFact_of_runs hW) (copyFact_of_runs hW hRow) hRow

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The instance's reference reads are well formed**, where `R`'s are. -/
theorem refReadsWf_held (hWf : Claims.Json.Wf R.reads) : RefReadsWf (held R B desc q c₀) := fun _ _ => hWf

/-- **The design premises at the instance**, from the row at the address map and `R`'s reference reads being well
formed. -/
theorem premises_of_row (hWf : Claims.Json.Wf R.reads)
    (hRow : RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀))) : Premises (held R B desc q c₀) :=
  ⟨builtWired_held R B desc q c₀, addressesConnected_of_row (refReadsWf_held R B desc q c₀ hWf) hRow⟩

/-- **`HonestExists`**, from the row at the instance's address map, `R`'s reference reads being well formed and a
run the instance accepts. -/
theorem honest_of_row (hWf : Claims.Json.Wf R.reads)
    (hRow : RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀))) (hA : Accepts (held R B desc q c₀)) : HonestExists :=
  honest_of_owed ⟨R, B, desc, q, c₀, addressesConnected_of_row (refReadsWf_held R B desc q c₀ hWf) hRow, hA⟩

end SpotFlock.Honest
