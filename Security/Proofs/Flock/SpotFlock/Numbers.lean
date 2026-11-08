module

public import Proofs.Flock.SpotFlock.KS

/-!
# SPOT[Flock]'s inner numbers

`SpotFlock.innerNumbers_code`: at SPOT[Flock]'s two-stage code (`SpotFlock.code h`), from the sessions' scope
(`SessionsIn`), the inner numbers are Flock's at every session's statement (`Claims.Sampling.InnerNumbers`): the waits
and truncation by definition, `N₀` the code's, and `ε_table` and the weight at most the code's `tables` tables'
bounds. It is `innerNumbers_deploy`. The scope is a premise: `SessionsIn` reads one `N₀` per statement, and a unit's
session has its own template's (`sessionsIn_at`).
-/

@[expose] public section

namespace SpotFlock

variable (h : Held)

/-- **SPOT[Flock]'s inner numbers are Flock's** at every session's statement, at every relation, from the sessions'
scope. -/
theorem innerNumbers_code (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (hin : SamplingDeploy.SessionsIn (code h)) : Claims.Sampling.InnerNumbers (SamplingDeploy.deploy rel (code h)) :=
  SamplingDeploy.innerNumbers_deploy rel (code h) hin

end SpotFlock
