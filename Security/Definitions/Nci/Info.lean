import Definitions.Nci.Support
import Definitions.Nci.Threads
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# NCI's reads counted by information: the vocabulary

`novelReads` counts what a set of gates reads by width: the bits on its wires from outside it and the baseline. A wire
can be wide and carry little: a copy of a value read on another wire, a value another read determines, or a state rebuilt
from a short seed and registered data. A memory cap needs reads counted by information: the bits a set's reads carry over
the runs the committed baseline allows, whatever their width.

* `novelInfo d θv R`: `log₂` of the number of distinct values `R`'s novel reads (`in(R) \ θ`) take over the runs with
  the baseline at `θv`. It is at most `novelReads d R`, and equal to it when those reads are novel input gates.
* `SmallInfo c θv Γ S R`: the thread draft's small subcircuit with its reads counted by information: at most `Γ`
  computed gates, reading at most `S` bits of information. Every `Small` subcircuit is `SmallInfo`.
* `Threading.ThreadedInfo θv Γ S d e`: `(d, e)`-threaded over the subcircuits small by information. Like `Threaded`, it
  is a condition on wiring, over more subcircuits.
-/

namespace Nci

open Verity Finset

/-- **The information in `R`'s novel reads**: `log₂` of the number of distinct values the gates of `in(R) \ θ` take over
the runs of `d` with the baseline at `θv`. -/
noncomputable def novelInfo (d : RegisteredCircuit) (θv : d.D.Transcript) (R : Finset (Fin d.D.N)) : ℝ :=
  Real.logb 2 (support d θv (d.D.into R \ d.θ))

/-- **A small subcircuit, by information**: at most `Γ` computed gates, whose novel reads carry at most `S` bits over
the runs with the baseline at `θv`. -/
def SmallInfo (c : RegisteredCircuit) (θv : c.D.Transcript) (Γ : ℕ) (S : ℝ) (R : Finset (Fin c.D.N)) : Prop :=
  R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ novelInfo c θv R ≤ S

namespace Threading

variable {c : RegisteredCircuit} (T : Threading c)

/-- **`(d, e)`-threaded by information** at `(Γ, S)`: every subcircuit small by information that reads fewer than `d`
novel values of a thread computes at most `e` of that thread's counted multiplications per value it reads from the
thread. -/
def ThreadedInfo (θv : c.D.Transcript) (Γ : ℕ) (S : ℝ) (d e : ℕ) : Prop :=
  ∀ R, SmallInfo c θv Γ S R → ∀ t, (T.threadReads R t).card < d →
    (T.threadMults R t).card ≤ e * (T.threadReads R t).card

end Threading

end Nci
