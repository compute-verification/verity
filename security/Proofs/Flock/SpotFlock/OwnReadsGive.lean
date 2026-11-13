module

public import Proofs.Flock.SpotFlock.AddressesGive
public import Proofs.Flock.SpotFlock.OwnReads

/-!
# `OwnReads` at the assembly, from well-formed reference reads

The assembly of the addresses premise (`AddressesGive`) takes `OwnReads h` as a fact of an accepted run. `OwnReads h`
speaks only of accepted runs, and an accepted run's held code passes the registered wiring's check
(`CheckedRun.wiring_of_ok`), one of whose clauses is the decided condition `refReadsOwn h`. So where the held
reference reads are well-formed trees (`RefReadsWf h`), `OwnReads h` is no premise:

* `refReadsOwn_of_wiringOk`: the registered wiring's check has the clause `refReadsOwn`, wherever it stands among the
  check's clauses;
* `ownReads_of_wf`: `RefReadsWf h` gives `OwnReads h`. The assembly passes it where it takes `OwnReads h`.

## What this does not show

`RefReadsWf h` at any held code: it is a fact of how the reference reads were built.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

/-- The registered wiring's check has the clause `refReadsOwn`. -/
theorem refReadsOwn_of_wiringOk (hw : wiringOk h = true) : refReadsOwn h = true := by
  have hc := (wiringOk_iff h).1 hw
  simp only [hc]

/-- **`OwnReads`, from well-formed reference reads**: at an accepted run the registered wiring passes its check, so
each reference read at a port no link or wire names reads a port of the interior registration (`refReadsOwn`), and
with the reference reads well-formed that is what `OwnReads` asks (`ownReads_of`). -/
theorem ownReads_of_wf (hWf : RefReadsWf h) : OwnReads h := by
  intro r hr
  exact ownReads_of (refReadsOwn_of_wiringOk (CheckedRun.wiring_of_ok hr).2) hWf r hr

end SpotFlock
