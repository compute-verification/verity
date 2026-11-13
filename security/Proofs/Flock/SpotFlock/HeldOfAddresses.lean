module

public import Proofs.Flock.SpotFlock.HeldOfWires
public import Proofs.Flock.SpotFlock.AddressFacts.RowUnlinked

/-!
# The addresses premise at the held copies the run reads

`addressesConnected_of_derived` (`AddressFacts/RowUnlinked`) proves the addresses premise at held copies whose wires
are those of a registered Definition (`SecondStage.replayWires`). Held copies that agree with what the run reads
(`RunsAs.HeldMatches`) have those wires (`HeldMatches.wires`), and so do the held copies `heldOf` builds
(`HeldOf.wires_of_heldOf`). So at either the addresses premise takes no premise about the wires
(`addressesConnected_of_matches`, `addressesConnected_of_heldOf`): well-formed reference reads, no inner block always
checked, the run's population check and a first stage of count `K`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **The addresses premise, at held copies that agree with what the run reads** (`RunsAs.HeldMatches`), for a first
stage of count `K`: from well-formed reference reads, no inner block always checked and the run's population check.
The copies' wires are the derived ones (`HeldMatches.wires`), and the design check holds each accepted run to no link
at a wire's reader port (`addressesConnected_of_derived`). -/
theorem addressesConnected_of_matches {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior} {h : Held}
    (hm : SpotFlock.RunsAs.HeldMatches i it h) (hWf : RefReadsWf h) (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) : AddressesConnected h :=
  addressesConnected_of_derived hWf hna hp hK hm.wires

/-- **The addresses premise, at a held copy `heldOf` builds**, for a first stage of count `K`: as
`addressesConnected_of_matches`, with the copies' wires the derived ones by construction (`HeldOf.wires_of_heldOf`),
so with no premise on the parts. -/
theorem addressesConnected_of_heldOf {i : _root_.Spot.Run.Input} {it : _root_.Spot.Run.Interior}
    {u : SpotFlock.HeldOf.Unread} {h : Held} (hh : SpotFlock.HeldOf.heldOf i it u = some h) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) : AddressesConnected h :=
  addressesConnected_of_derived hWf hna hp hK (SpotFlock.HeldOf.wires_of_heldOf hh)

end SpotFlock
