module

public import Proofs.Flock.SpotFlock.AddressFacts.RowSample
public import Proofs.Flock.SpotFlock.AddressFacts.RowPublished
public import Proofs.Flock.SpotFlock.AddressFacts.Connected
public import Proofs.Flock.SpotFlock.SampleBelow

/-!
# The row fact and the addresses premise, for a first stage of count `K`

A first stage that samples a uniform `K`-subset of the outer blocks (`firstK`) samples below the population
`Partition.units` returns (`RegTwo.sample_lt`), and the run's population check holds that population to the record's
(`AddressRow.units_of_popOk`). So such a first stage samples slots of the layout that name themselves
(`sampleSlots_of_firstK`), and the row fact at the address map takes four premises about the auditor's copies and
none about an accepted run's session (`rowFact_of_firstK`): the wiring facts at each accepted run, no inner block
always checked, the population check, and the condition on the auditor's copies of the wires. With the reference
reads well formed, which gives the wiring facts, that is the addresses premise (`addressesConnected_of_firstK`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

/-- **A first stage of count `K` samples slots that name themselves** (`SampleSlots`), under the run's population
check: it samples below the population (`RegTwo.sample_lt`), which the check holds to `n`. -/
theorem sampleSlots_of_firstK {h : Held} {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) : SampleSlots h :=
  sampleSlots_of_units hp fun c => _root_.SamplingDeploy.RegTwo.sample_lt h.reg hK c

end AddressRow

/-- **The row fact at the address map, for a first stage of count `K`**: from the wiring facts of each accepted run,
no inner block always checked, the run's population check and the condition on the auditor's copies of the wires. -/
theorem rowFact_of_firstK {h : Held} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) :
    RowFact h (wireMap h) (returned h) (addressOf h) :=
  rowFact_of_checks hW hna (AddressRow.sampleSlots_of_firstK hp hK) hwc

/-- **The addresses premise, for a first stage of count `K`**: from well-formed reference reads, no inner block
always checked, the run's population check and the condition on the auditor's copies of the wires. -/
theorem addressesConnected_of_firstK {h : Held} (hWf : RefReadsWf h) (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) : AddressesConnected h :=
  addressesConnected_of_row hWf (rowFact_of_firstK (fun _ hr => wiring_of_refReads hWf hr) hna hp hK hwc)

end SpotFlock
