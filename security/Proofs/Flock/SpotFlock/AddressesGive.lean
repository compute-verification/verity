module

public import Proofs.Flock.SpotFlock.EntryRoots
public import Proofs.Flock.SpotFlock.PortsMember
public import Proofs.Flock.SpotFlock.AddressFactsGive
public import Proofs.Flock.SpotFlock.BitsInUnitsOf
public import Proofs.Flock.SpotFlock.OwnReads
public import Proofs.Flock.SpotFlock.WiresMoveRows

/-!
# The addresses premise, from the facts of an accepted run

`AddressesConnected h` is the premise the two-stage guarantees take at the code's own wiring. It follows, at the
address map `addressOf h`, from facts of an accepted run's checked unit alone (`addressesConnected_of_runs`):

* the width, the copy and the row at the address map (`WidthFact`, `CopyFact`, `RowFact`);
* every gate the unit needs is one of its own bits (`NeededBits`), a needed gate the wiring sends inside its replay
  block stays inside (`Inside`), a gate is at one written position at most (`WrittenOnce`), a commitment position's
  row is a port of its member's circuit (`RowsArePorts`), and each template's reference reads are well formed
  (`RefReadsWf`).

The entry and one tree a gate are proved from those (`entryFact_of_wiring`, `oneTreeFact_of_wiring`), and what they
ask of the registered wiring is no premise: a run is accepted only where the design evaluates, and the design
evaluates only where the registered wiring passes its check (`CheckedRun.wiring_of_ok`, `wiringOk`). The check
decides its clauses on the rows once per template; the statements the entry reads bit by bit follow
(`sourcesNamed_of_rows`, `writtenUnwired_of_rows`, and `wiresMove_of_rows` where the design evaluates:
`CheckedRun.wiresMove_of_ok`), as does a registered read at a port no link or wire names reading a port of the
interior record (`OwnReads`, `ownReads_of`), and the boundary record holds no written bit at a port no link names
because a gate is at one written position at most (`returnedWhole_of_once`).

Each of those facts is stated over accepted runs, so its own proof may read the design's conditions the same way.
`Inside` is one of them proved (`inside_of`), from the members' proof blocks standing side by side (`SideBySide`) and
each bit being a gate of its unit's proof block (`BitsInUnits`): `addressesConnected_of_blocks`. Both of those hold at
every held code (`sideBySide_of_held`, `bitsInUnits`), and `NeededBits` where every member's bits are rowed
(`MembersRowed`): `addressesConnected_of_rowed`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

/-- **An accepted run's held code passes the design's conditions**: every member's circuit is in the scope Flock's
hidden-output proof covers and the registered wiring passes its check. -/
theorem CheckedRun.wiring_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) :
    membersScoped h = true ∧ wiringOk h = true := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact scoped_wiring_of_two h h2

/-- **At an accepted run, every wired source bit has a gate to go to**: the run's design evaluates
(`wiresMove_of_two`). -/
theorem CheckedRun.wiresMove_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) : WiresMove h := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact wiresMove_of_two h h2

/-- **What the entry and one tree a gate ask of the held code**, where the registered wiring passes its check and
every wired source bit has a gate to go to: the facts of an accepted run, and the check's own clauses. The check
decides the wired source bits once per template (`wiresMoveRows`), which gives `WiresMove h` only where the design
evaluates (`wiresMove_of_two`), so it is taken beside the check. -/
theorem wiring_of_wiringOk (hw : wiringOk h = true) (hM : WiresMove h) (hN : NeededBits h) (hI : Inside h)
    (hO : WrittenOnce h) (hR : RowsArePorts h) (hWf : RefReadsWf h) : Wiring h := by
  obtain ⟨-, hfirst, hnamed, -, halike, hunwired, -, hown, -, -⟩ := (wiringOk_iff h).1 hw
  exact ⟨hN, hI, hO, hR, portsMember h, ownReads_of hown hWf, sourcesNamed_of_rows h hnamed, hM, hfirst, halike,
    returnedWhole_of_once hO, writtenUnwired_of_rows h hunwired⟩

variable (h) in
/-- **The addresses premise, from what the entry and one tree a gate ask at each accepted run**: the width, the copy
and the row at the address map, with `Wiring h` at every accepted run, give `AddressesConnected h`. A fact that
holds only where the design evaluates is read off the run there. -/
theorem addressesConnected_of_wiring (hWr : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h := by
  have hE : EntryFact h (wireMap h) (returned h) (addressOf h) := fun r hr i hi =>
    entryFact_of_wiring (hWr r hr) r hr i hi
  have hT : OneTreeFact h (wireMap h) (returned h) (addressOf h) :=
    ⟨addressOf_shares h, fun r hr i hi => (oneTreeFact_of_wiring (hWr r hr)).2 r hr i hi⟩
  exact addressesConnected_of_facts h fun _ => ⟨addressOf h, hW, hC, hRow, hE, hT⟩

variable (h) in
/-- **The addresses premise, from the facts of an accepted run**: the width, the copy and the row at the address
map, with the facts the entry and one tree a gate read of an accepted run's checked unit, give
`AddressesConnected h`. No fact of the registered wiring is a premise: an accepted run's held code passes the check
of it. -/
theorem addressesConnected_of_runs (hN : NeededBits h) (hI : Inside h) (hO : WrittenOnce h) (hR : RowsArePorts h)
    (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h :=
  addressesConnected_of_wiring h
    (fun _ hr => wiring_of_wiringOk (CheckedRun.wiring_of_ok hr).2 (CheckedRun.wiresMove_of_ok hr) hN hI hO hR hWf)
    hW hC hRow

variable (h) in
/-- **The addresses premise, from the facts of the members' programs and of an accepted run**:
`addressesConnected_of_runs`, with a needed gate staying inside its replay block (`Inside`) from the members' proof
blocks standing side by side and each bit being a gate of its unit's proof block (`inside_of`). -/
theorem addressesConnected_of_blocks (hN : NeededBits h) (hcl : SideBySide h) (hU : BitsInUnits h)
    (hO : WrittenOnce h) (hR : RowsArePorts h) (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h))
    (hC : CopyFact h (addressOf h)) (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    AddressesConnected h :=
  addressesConnected_of_runs h hN (inside_of hN hcl hU hO) hO hR hWf hW hC hRow

variable (h) in
/-- **The addresses premise, where every member's bits are rowed**: `addressesConnected_of_blocks`, with every needed
gate one of its unit's bits from the members' bits being rowed (`neededBits_of_rowed`), and the members' proof blocks
side by side and each bit a gate of its unit's proof block at every held code. -/
theorem addressesConnected_of_rowed (hM : MembersRowed h) (hO : WrittenOnce h) (hR : RowsArePorts h)
    (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h :=
  addressesConnected_of_blocks h (neededBits_of_rowed hM) (sideBySide_of_held h) (bitsInUnits h) hO hR hWf hW hC hRow

end SpotFlock
