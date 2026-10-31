import Definitions.Nci.Support
import Definitions.Nci.Threads
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# The inference-only policy's reads counted by information: the vocabulary

`novelReads` counts what a set of gates reads by width: the bits on its wires from outside it and `θ`. A wire
can be wide and carry little: a copy of a value read on another wire, a value another read determines, or a state rebuilt
from a short seed and registered data. A memory cap needs reads counted by information: the bits a set's reads carry over
the runs the committed registered information allows, whatever their width.

* `novelInfo d θv R`: `log₂` of the number of distinct values `R`'s fresh input reads (`in(R) \ θ`) take over the runs
  with the registered information at `θv`. It is at most `novelReads d R`, and equal to it when those reads are fresh
  input gates.
* `SmallInfo r θv c S R`: the thread draft's small subcircuit with its reads counted by information: at most `c`
  computed gates, reading at most `S` bits of information. Every `Small` subcircuit is `SmallInfo`.
* `Threading.ThreadedInfo θv c S d e`: `(d, e)`-threaded over the subcircuits small by information. Like `Threaded`, it
  is a condition on wiring, over more subcircuits.
-/

namespace Nci

open Verity Finset

/-- **The information in `R`'s fresh input reads**: `log₂` of the number of distinct values the gates of `in(R) \ θ`
take over the runs of `d` with the registered information at `θv`. -/
noncomputable def novelInfo (d : RegisteredCircuit) (θv : d.D.Transcript) (R : Finset (Fin d.D.N)) : ℝ :=
  Real.logb 2 (support d θv (d.D.into R \ d.θ))

/-- **A small subcircuit, by information**: at most `c` computed gates, whose fresh input reads carry at most `S` bits
over the runs with the registered information at `θv`. -/
def SmallInfo (r : RegisteredCircuit) (θv : r.D.Transcript) (c : ℕ) (S : ℝ) (R : Finset (Fin r.D.N)) : Prop :=
  R ⊆ r.D.computed ∧ r.D.size R ≤ c ∧ novelInfo r θv R ≤ S

namespace Threading

variable {r : RegisteredCircuit} (T : Threading r)

/-- **`(d, e)`-threaded by information** at `(c, S)`: every subcircuit small by information that reads fewer than `d`
fresh input values of a thread computes at most `e` of that thread's counted multiplications per value it reads from the
thread. -/
def ThreadedInfo (θv : r.D.Transcript) (c : ℕ) (S : ℝ) (d e : ℕ) : Prop :=
  ∀ R, SmallInfo r θv c S R → ∀ t, (T.threadReads R t).card < d →
    (T.threadMults R t).card ≤ e * (T.threadReads R t).card

end Threading

end Nci
