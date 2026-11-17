module

public import Proofs.Flock.SpotFlock.AddressFacts.RowSample
public import Proofs.Flock.SpotFlock.AddressFacts.RowPublished
public import Proofs.Flock.SpotFlock.AddressFacts.Connected
public import Proofs.Flock.SpotFlock.SampleBelow

/-! The row fact and the addresses premise for a first stage of count `K` (`rowFact_of_firstK`,
`addressesConnected_of_firstK`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

theorem sampleSlots_of_firstK {h : Held} {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) : SampleSlots h :=
  sampleSlots_of_units hp fun c => _root_.SamplingDeploy.RegTwo.sample_lt h.reg hK c

end AddressRow

theorem rowFact_of_firstK {h : Held} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) :
    RowFact h (wireMap h) (returned h) (addressOf h) :=
  rowFact_of_checks hW hna (AddressRow.sampleSlots_of_firstK hp hK) hwc

theorem addressesConnected_of_firstK {h : Held} (hWf : RefReadsWf h) (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) : AddressesConnected h :=
  addressesConnected_of_row hWf (rowFact_of_firstK (fun _ hr => wiring_of_refReads hWf hr) hna hp hK hwc)

end SpotFlock
