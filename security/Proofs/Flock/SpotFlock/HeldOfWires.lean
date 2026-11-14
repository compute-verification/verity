module

public import Proofs.Flock.SpotFlock.HeldOf
public import Proofs.Flock.SpotFlock.AddressFacts.RowWires

/-!
# The held copies' wires are the derived ones

`AddressRow.wiresChecked h` is a premise of the row fact at the address map. At held copies whose wires are those of
a registered Definition (`SecondStage.replayWires`) it is `AddressRow.wiresUnlinked h` alone: no link names a wire's
reader port (`AddressRow.wiresChecked_of_unlinked`).

A held copy `heldOf` builds has those wires by construction, and its links are the derived ones too
(`wires_of_heldOf`, `links_of_heldOf`), with no premise on the parts. A held copy that agrees with what the run reads
has them as `HeldMatches.wires` says. At either, `wiresChecked` follows from `wiresUnlinked`
(`wiresChecked_of_heldOf`, `wiresChecked_of_matches`).
-/

@[expose] public section

namespace SpotFlock.HeldOf

/-- **The held wires are the derived ones**, at a held copy `heldOf` builds: the wires of the outer block's registered
Definition (`SecondStage.replayWires`). -/
theorem wires_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) :
    h.wires = Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  rfl

/-- **The held links are the derived ones**, at a held copy `heldOf` builds: the links of the outer block's registered
Definition at the run's record (`SecondStage.replayLinks`). -/
theorem links_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) :
    h.links =
      Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  rfl

/-- **The condition on the held wires, at a held copy `heldOf` builds**, is that no link names a wire's reader port
(`AddressRow.wiresUnlinked`): the wires are derived, so each is the first at its reader port. -/
theorem wiresChecked_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hu : AddressRow.wiresUnlinked h = true) :
    AddressRow.wiresChecked h = true :=
  AddressRow.wiresChecked_of_unlinked (wires_of_heldOf hh) hu

/-- **The condition on the held wires, at a held copy that agrees with what the run reads** (`HeldMatches`), is that
no link names a wire's reader port (`AddressRow.wiresUnlinked`). -/
theorem wiresChecked_of_matches {i : Spot.Run.Input} {it : Spot.Run.Interior} {h : Held}
    (hm : RunsAs.HeldMatches i it h) (hu : AddressRow.wiresUnlinked h = true) :
    AddressRow.wiresChecked h = true :=
  AddressRow.wiresChecked_of_unlinked hm.wires hu

end SpotFlock.HeldOf
