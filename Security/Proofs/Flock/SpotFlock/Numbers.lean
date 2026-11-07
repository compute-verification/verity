module

public import Proofs.Flock.SpotFlock.KS

/-!
# SPOT-Flock's inner numbers

`SpotFlock.innerNumbers_code`: at the deployed two-stage code (`SpotFlock.code h`), the inner numbers are Flock's at
every session's statement (`Claims.Sampling.InnerNumbers`): the waits and truncation by definition, `N₀` the reference
statement's shapes', and `ε_table` and the weight at most the code's `tables` tables' bounds. It is
`innerNumbers_deploy` at `sessionsIn_code`, which reads the reference statement `h.xS₀` in Flock's masked scope
(`Held.xS₀_ok`): at a transcript refused before its session, the session is `h.xS₀`'s, and Flock's `ε_table` and
weight at `h.xS₀` are within the code's only where its tables are in range.
-/

@[expose] public section

namespace SpotFlock

variable (h : Held)

/-- **SPOT-Flock's inner numbers are Flock's** at every session's statement, at every relation. -/
theorem innerNumbers_code (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Sampling.InnerNumbers (SamplingDeploy.deploy rel (code h)) :=
  SamplingDeploy.innerNumbers_deploy rel (code h) (sessionsIn_code h)

end SpotFlock
