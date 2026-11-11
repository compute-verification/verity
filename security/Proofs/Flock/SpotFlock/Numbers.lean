module

public import Proofs.Flock.SpotFlock.KS

/-!
# SPOT[Flock]'s inner numbers

`SpotFlock.innerNumbersAt_code`: at SPOT[Flock]'s two-stage code (`SpotFlock.code h`), the inner numbers are Flock's at
every unit's session statement at that unit's `N₀` (`Claims.Sampling.InnerNumbersAt` at `Code.unitN₀`, its template's
reference statement's shapes'): the waits and truncation by definition, and `ε_table` and the weight at most the code's
`tables` tables' bounds. It is `innerNumbersAt_deploy` at `sessionsIn_code`, which reads each template's reference
statement `h.xS₀ t` in Flock's masked scope (`Held.xS₀_ok`). One `N₀` per statement (`InnerNumbers`) would hold only
where every template's is the same.
-/

@[expose] public section

namespace SpotFlock

variable (h : Held)

/-- **SPOT[Flock]'s inner numbers are Flock's** at every unit's session statement at its template's `N₀`, at every
relation. -/
theorem innerNumbersAt_code (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Sampling.InnerNumbersAt (SamplingDeploy.deploy rel (code h)) (code h).unitN₀ :=
  SamplingDeploy.innerNumbersAt_deploy rel (code h) (sessionsIn_code h)

end SpotFlock
