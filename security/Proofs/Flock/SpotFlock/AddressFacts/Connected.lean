module

public import Proofs.Flock.SpotFlock.AddressFacts.OfRuns
public import Proofs.Flock.SpotFlock.AddressesHeld

/-! The addresses premise from the row at the address map and well-formed reference reads
(`addressesConnected_of_row`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

theorem wiring_of_refReads (hWf : RefReadsWf h) {r : CheckedRun h.stage} (hr : r.Ok h) : Wiring h :=
  wiring_of_wiringOk (CheckedRun.wiring_of_ok hr).2 (CheckedRun.wiresMove_of_ok hr) (neededBits_held h)
    (inside_of (neededBits_held h) (sideBySide_of_held h) (bitsInUnits h) (writtenOnce_held h))
    (writtenOnce_held h) (CheckedRun.rows_of_ok hr) hWf

theorem addressesConnected_of_row (hWf : RefReadsWf h)
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h :=
  have hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h := fun _ hr => wiring_of_refReads hWf hr
  addressesConnected_of_wiring h hW (widthFact_of_runs hW) (copyFact_of_runs hW hRow) hRow

end SpotFlock
