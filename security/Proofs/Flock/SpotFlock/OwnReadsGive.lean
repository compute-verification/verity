module

public import Proofs.Flock.SpotFlock.AddressesGive
public import Proofs.Flock.SpotFlock.OwnReads

/-!
# `OwnReads` at the assembly, from well-formed reference reads

The assembly of the addresses premise (`AddressesGive`) takes `OwnReads h` as a fact of an accepted run. `OwnReads h`
speaks only of accepted runs, and an accepted run's held code passes the registered wiring's check
(`CheckedRun.wiring_of_ok`), one of whose clauses is the decided condition `refReadsOwn h`. So where the held
reference reads are well-formed trees (`RefReadsWf h`), `OwnReads h` is no premise:

* `ownReads_of_wf`: `RefReadsWf h` gives `OwnReads h`;
* `addressesConnected_of_wf`: the assembly where every member's bits are rowed (`addressesConnected_of_rowed`), with
  `RefReadsWf h` in the place of `OwnReads h`.

## What this does not show

`RefReadsWf h` at any held code: it is a fact of how the reference reads were built (`HeldOf.refReadsWf_of`, from the
parameters' registered reads being well-formed trees).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

/-- **`OwnReads`, from well-formed reference reads**: at an accepted run the registered wiring passes its check, so
each reference read at a port no link or wire names reads a port of the interior registration (`refReadsOwn`), and
with the reference reads well-formed that is what `OwnReads` asks (`ownReads_of`). -/
theorem ownReads_of_wf (hWf : RefReadsWf h) : OwnReads h := by
  intro r hr
  obtain ⟨-, -, -, -, -, -, -, -, hown, -⟩ := (wiringOk_iff h).1 (CheckedRun.wiring_of_ok hr).2
  exact ownReads_of hown hWf r hr

variable (h) in
/-- **The addresses premise, where every member's bits are rowed and the reference reads are well-formed**:
`addressesConnected_of_rowed`, with `OwnReads` from `ownReads_of_wf`. -/
theorem addressesConnected_of_wf (hM : MembersRowed h) (hO : WrittenOnce h) (hR : RowsArePorts h)
    (hWf : RefReadsWf h) (hW : WidthFact h (addressOf h)) (hC : CopyFact h (addressOf h))
    (hRow : RowFact h (wireMap h) (returned h) (addressOf h)) : AddressesConnected h :=
  addressesConnected_of_rowed h hM hO hR (ownReads_of_wf hWf) hW hC hRow

end SpotFlock
