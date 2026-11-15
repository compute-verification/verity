module

public import Proofs.Flock.SpotFlock.AddressesGive
public import Proofs.Flock.SpotFlock.NeededBitsHeld
public import Proofs.Flock.SpotFlock.WiredKS

/-!
# The two-stage guarantees at the connected program, from the width, the copy and the row

`AddressesConnected h` is the premise the two-stage guarantees take at the code's own wiring. Of the facts the
assembly reads at an accepted run (`addressesConnected_of_wiring`), every one but four holds at every held code or
is read off the accepted run's own design:

* every gate a checked unit needs is one of its own unit's bits (`neededBits_held`), the members' proof blocks stand
  side by side (`sideBySide_of_held`), each bit is a gate of its unit's proof block (`bitsInUnits`) and a gate is at
  one written position at most (`writtenOnce_held`): at every held code;
* a commitment position's row is a port of its member's circuit (`rowsArePorts_of_two`), the registered wiring
  passes its check (`CheckedRun.wiring_of_ok`) and every wired source bit has a gate to go to
  (`CheckedRun.wiresMove_of_ok`): where the design evaluates, which an accepted run's does.

What remains is the width, the copy and the row at the address map (`WidthFact`, `CopyFact`, `RowFact`) and the
reference reads being well formed (`RefReadsWf`): `addressesConnected_of_held`. With the session frame built
(`BuiltWired`), those four give SPOT[Flock]'s two-stage knowledge soundness and its exfiltration bound at the
connected program (`ksTwo_of_facts`, `exfilTwo_of_facts`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

/-- **At an accepted run, a commitment position's row is a port of its member's circuit**: the run's design
evaluates. -/
theorem CheckedRun.rows_of_ok {r : CheckedRun h.stage} (hr : r.Ok h) : RowsArePorts h := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hrest⟩ := hr
  exact rowsArePorts_of_two h h2

variable (h) in
/-- **The addresses premise, from the width, the copy, the row and well-formed reference reads.** -/
theorem addressesConnected_of_held (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h))
    (hC : CopyFact h (addressOf h)) (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    AddressesConnected h :=
  addressesConnected_of_wiring h
    (fun _ hr => wiring_of_wiringOk (CheckedRun.wiring_of_ok hr).2 (CheckedRun.wiresMove_of_ok hr)
      (neededBits_held h)
      (inside_of (neededBits_held h) (sideBySide_of_held h) (bitsInUnits h) (writtenOnce_held h))
      (writtenOnce_held h) (CheckedRun.rows_of_ok hr) hWf) hW hC hRow

variable (h) in
/-- **SPOT[Flock]'s two-stage knowledge soundness at the connected program, from the built frame and four facts**:
`ksTwo_connected` with its addresses premise from the width, the copy, the row and well-formed reference reads. -/
theorem ksTwo_of_facts (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).KS FlockDeploy.sha512 :=
  ksTwo_connected h hB (addressesConnected_of_held h hWf hW hC hRow)

variable (h) in
/-- **SPOT[Flock]'s two-stage exfiltration bound at the connected program, from the built frame and four facts.** -/
theorem exfilTwo_of_facts (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).Exfil FlockDeploy.sha512 :=
  exfilTwo_connected h hB (addressesConnected_of_held h hWf hW hC hRow)

end SpotFlock
