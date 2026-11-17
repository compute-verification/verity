module

public import Proofs.Flock.SpotFlock.EntryRoots
public import Proofs.Flock.SpotFlock.PortsMember
public import Proofs.Flock.SpotFlock.AddressFactsGive
public import Proofs.Flock.SpotFlock.BitsInUnitsOf
public import Proofs.Flock.SpotFlock.OwnReads
public import Proofs.Flock.SpotFlock.WiresMoveRows

/-! The addresses premise `AddressesConnected h`, from the facts of an accepted run's checked unit alone
(`addressesConnected_of_runs`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

theorem CheckedRun.wiring_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) :
    membersScoped h = true ∧ wiringOk h = true := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact scoped_wiring_of_two h h2

theorem CheckedRun.wiresMove_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) : WiresMove h := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact wiresMove_of_two h h2

/-- The check decides the wired source bits once per template, which gives `WiresMove h` only where the design evaluates
(`wiresMove_of_two`), so it is taken beside the check. -/
theorem wiring_of_wiringOk (hw : wiringOk h = true) (hM : WiresMove h) (hN : NeededBits h) (hI : Inside h)
    (hO : WrittenOnce h) (hR : RowsArePorts h) (hWf : RefReadsWf h) : Wiring h := by
  obtain ⟨-, hfirst, hnamed, -, halike, hunwired, -, hown, -, -⟩ := (wiringOk_iff h).1 hw
  exact ⟨hN, hI, hO, hR, portsMember h, ownReads_of hown hWf, sourcesNamed_of_rows h hnamed, hM, hfirst, halike,
    returnedWhole_of_once hO, writtenUnwired_of_rows h hunwired⟩

variable (h) in
theorem addressesConnected_of_wiring (hWr : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h := by
  have hE : EntryFact h (wireMap h) (returned h) (addressOf h) := fun r hr i hi =>
    entryFact_of_wiring (hWr r hr) r hr i hi
  have hT : OneTreeFact h (wireMap h) (returned h) (addressOf h) :=
    ⟨addressOf_shares h, fun r hr i hi => (oneTreeFact_of_wiring (hWr r hr)).2 r hr i hi⟩
  exact addressesConnected_of_facts h fun _ => ⟨addressOf h, hW, hC, hRow, hE, hT⟩

variable (h) in
/-- No fact of the registered wiring is a premise: an accepted run's held code passes the check of it. -/
theorem addressesConnected_of_runs (hN : NeededBits h) (hI : Inside h) (hO : WrittenOnce h) (hR : RowsArePorts h)
    (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h :=
  addressesConnected_of_wiring h
    (fun _ hr => wiring_of_wiringOk (CheckedRun.wiring_of_ok hr).2 (CheckedRun.wiresMove_of_ok hr) hN hI hO hR hWf)
    hW hC hRow

variable (h) in
theorem addressesConnected_of_blocks (hN : NeededBits h) (hcl : SideBySide h) (hU : BitsInUnits h)
    (hO : WrittenOnce h) (hR : RowsArePorts h) (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h))
    (hC : CopyFact h (addressOf h)) (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    AddressesConnected h :=
  addressesConnected_of_runs h hN (inside_of hN hcl hU hO) hO hR hWf hW hC hRow

variable (h) in
theorem addressesConnected_of_rowed (hM : MembersRowed h) (hO : WrittenOnce h) (hR : RowsArePorts h)
    (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h :=
  addressesConnected_of_blocks h (neededBits_of_rowed hM) (sideBySide_of_held h) (bitsInUnits h) hO hR hWf hW hC hRow

end SpotFlock
