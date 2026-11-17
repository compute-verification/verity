module

public import Proofs.Flock.SpotFlock.AddressesGive
public import Proofs.Flock.SpotFlock.NeededBitsHeld
public import Proofs.Flock.SpotFlock.WiredKS

/-! SPOT[Flock]'s two-stage guarantees at the connected program, from the width, the copy, the row and well-formed
reference reads (`ksTwo_of_facts`, `exfilTwo_of_facts`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

theorem CheckedRun.rows_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) : RowsArePorts h := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact rowsArePorts_of_two h h2

variable (h) in
theorem addressesConnected_of_held (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h))
    (hC : CopyFact h (addressOf h)) (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    AddressesConnected h :=
  addressesConnected_of_wiring h
    (fun _ hr => wiring_of_wiringOk (CheckedRun.wiring_of_ok hr).2 (CheckedRun.wiresMove_of_ok hr)
      (neededBits_held h)
      (inside_of (neededBits_held h) (sideBySide_of_held h) (bitsInUnits h) (writtenOnce_held h))
      (writtenOnce_held h) (CheckedRun.rows_of_ok hr) hWf) hW hC hRow

variable (h) in
theorem ksTwo_of_facts (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).KS FlockDeploy.sha512 :=
  ksTwo_connected h hB (addressesConnected_of_held h hWf hW hC hRow)

variable (h) in
theorem exfilTwo_of_facts (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).Exfil FlockDeploy.sha512 :=
  exfilTwo_connected h hB (addressesConnected_of_held h hWf hW hC hRow)

end SpotFlock
