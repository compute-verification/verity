module

public import Proofs.Flock.SpotFlock.AddressFacts.RowWires
public import Proofs.Flock.SpotFlock.AddressFacts.RowFirst

/-!
# The condition on the wires, from the design check

`AddressRow.wiresChecked h` (`RowLeaves`) is a premise of the row fact at the address map. At copies whose wires the
design derives it is `wiresUnlinked h` alone (`RowWires`): no link names a wire's reader port.

The second stage checks the design at each session it reads (`SecondStage.designOk`, through `readsOk`), over what the
verifier holds of the run (`ownOf`), whose links and wires are the auditor's copies. So a property of every design the
check passes is a property of the copies wherever a run is accepted.

`ReadersUnlinked` is that property of the check: it passes no design in which a link names a wire's reader port.
`SecondStage.wiresOk` asks of a wire's writer port that no link names it and asks nothing of its reader port, so
`ReadersUnlinked` is false of the check as it stands and is a premise here; it is deleted once the check asks the same
of the reader port, where it is a lemma of the check. Under it an accepted run gives `wiresUnlinked`
(`CheckedRun.wiresUnlinked_of_ok`), and the row fact and the addresses premise take the equation "the wires are the
derived ones" in place of `wiresChecked` (`rowFact_of_derived`, `addressesConnected_of_derived`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

namespace AddressRow

/-- **The design check passes no design in which a link names a wire's reader port**: wherever `SecondStage.wiresOk`
passes, no link of the design names the template and port a wire is read at. A premise on the check's code: it is
false of `SecondStage.wiresOk` as it stands, which asks this of a wire's writer port only, and this definition is
deleted once the check asks it of the reader port too. -/
def ReadersUnlinked : Prop :=
  ∀ (ps : Array Spot.SecondStage.Part) (own : Spot.SecondStage.Own), Spot.SecondStage.wiresOk ps own = true →
    ∀ w ∈ own.wires.getD #[], (Spot.SecondStage.linkOf own w.template w.port).isNone = true

end AddressRow

namespace CheckedRun

open AddressRow

variable {h : Held} {r : CheckedRun h.stage}

/-- **No link names a wire's reader port**, at an accepted run, under `ReadersUnlinked`: the second stage checks the
design over what the verifier holds of the run (`ownOf`), whose links and wires are the auditor's copies, so the
check's lookup of a link (`SecondStage.linkOf`) is the copies' (`heldLink`). -/
theorem wiresUnlinked_of_ok (hcl : ReadersUnlinked) (hr : r.Ok h) : wiresUnlinked h = true := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb0, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, -, -, -, hdes⟩ := design_of_readsOk hat.2
  simp only [Spot.SecondStage.designOk, Bool.and_eq_true] at hdes
  unfold wiresUnlinked
  refine Array.all_eq_true_iff_forall_mem.2 fun w hmem => ?_
  exact hcl _ _ hdes.2 w hmem

/-- **The condition on the auditor's copies of the wires**, at an accepted run whose wires are those of a registered
Definition (`SecondStage.replayWires`), under `ReadersUnlinked`. -/
theorem wiresChecked_of_ok (hcl : ReadersUnlinked) {P : _root_.Flock.Program.Prog} {key : String}
    (hw : h.wires = Spot.SecondStage.replayWires P key) (hr : r.Ok h) : wiresChecked h = true :=
  wiresChecked_of_unlinked hw (wiresUnlinked_of_ok hcl hr)

end CheckedRun

/-- **The second premise of the row fact**, from the wiring facts of each accepted run at derived wires, under
`ReadersUnlinked`: the premise speaks of accepted runs only, and each gives the condition on the wires. -/
theorem AddressRow.ownLeaves_of_derived {h : Held} (hcl : AddressRow.ReadersUnlinked)
    (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h) {P : _root_.Flock.Program.Prog} {key : String}
    (hw : h.wires = Spot.SecondStage.replayWires P key) : AddressRow.OwnLeaves h :=
  fun r hr _ _ hown hmem ⟨_, _, hy, hrow⟩ hl =>
    CheckedRun.own_leaves (hW r hr) (CheckedRun.wiresChecked_of_ok hcl hw hr) hr hown hmem hy hrow hl

/-- **The row fact at the address map, for a first stage of count `K`, at derived wires**: `rowFact_of_firstK` with
the condition on the auditor's copies of the wires replaced by the equation that they are the derived ones, under
`ReadersUnlinked`. -/
theorem rowFact_of_derived {h : Held} (hcl : AddressRow.ReadersUnlinked)
    (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h) (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) {P : _root_.Flock.Program.Prog} {key : String}
    (hw : h.wires = Spot.SecondStage.replayWires P key) : RowFact h (wireMap h) (returned h) (addressOf h) :=
  rowFact_of_runs hW (AddressRow.sits_of_slots hna (AddressRow.sampleSlots_of_firstK hp hK))
    (AddressRow.ownLeaves_of_derived hcl hW hw) (AddressRow.publishedRows_of_runs hW)

/-- **The addresses premise, for a first stage of count `K`, at derived wires**: from well-formed reference reads, no
inner block always checked, the run's population check and the equation that the auditor's copies of the wires are the
derived ones, under `ReadersUnlinked`. -/
theorem addressesConnected_of_derived {h : Held} (hcl : AddressRow.ReadersUnlinked) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) {P : _root_.Flock.Program.Prog} {key : String}
    (hw : h.wires = Spot.SecondStage.replayWires P key) : AddressesConnected h :=
  addressesConnected_of_row hWf
    (rowFact_of_derived hcl (fun _ hr => wiring_of_refReads hWf hr) hna hp hK hw)

end SpotFlock
