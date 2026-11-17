module

public import Proofs.Flock.SpotFlock.HeldOf
public import Proofs.Flock.SpotFlock.AddressFacts.RowWires

/-! The held copies' wires are the derived ones, so `wiresChecked` follows from `wiresUnlinked`
(`wiresChecked_of_heldOf`, `wiresChecked_of_matches`). -/

@[expose] public section

namespace SpotFlock.HeldOf

theorem wires_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) :
    h.wires = Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  rfl

theorem links_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) :
    h.links =
      Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  rfl

theorem wiresChecked_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hu : AddressRow.wiresUnlinked h = true) :
    AddressRow.wiresChecked h = true :=
  AddressRow.wiresChecked_of_unlinked (wires_of_heldOf hh) hu

theorem wiresChecked_of_matches {i : Spot.Run.Input} {it : Spot.Run.Interior} {h : Held}
    (hm : RunsAs.HeldMatches i it h) (hu : AddressRow.wiresUnlinked h = true) :
    AddressRow.wiresChecked h = true :=
  AddressRow.wiresChecked_of_unlinked hm.wires hu

end SpotFlock.HeldOf
