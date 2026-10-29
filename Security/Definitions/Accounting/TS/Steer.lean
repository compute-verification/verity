import Definitions.Accounting.TS.Slots

/-!
# Steering a segmented consumer (trusted)

STEER-PROOF's route B: the consumer rows SAMPLING's LIN-STEER is about
(`Consumer`), with their words at a stride. A consumer of a row of `m` coordinates commits one word a segment (`seg i`
the segment of coordinate `i`, `G` segments, a partial sum at stride `L`), so zeroing coordinates of its noised operand
spares a segment's work only once the segment has nothing left to sum.

* `segActive seg act Z`: the segments holding a noise-active coordinate outside `Z`.
* `segCount seg act g`: segment `g`'s noise-active coordinates.
* `segConsumer seg act c`: the consumer that pays `c` a segment with an active coordinate left, its true least work
  once `Z` is zeroed.
* `segUntouched seg act Z`, `adaptConsumer seg act c`: the same row under E1(b)'s accounting, which counts a segment's
  words as saved (adaptive) once any coordinate of the segment is zeroed.
-/

namespace Accounting.TS

open Finset

variable {m G : ℕ}

/-- The segments holding a noise-active coordinate outside `Z`. -/
def segActive (seg : Fin m → Fin G) (act Z : Finset (Fin m)) : Finset (Fin G) :=
  univ.filter fun g => ∃ i ∈ act, seg i = g ∧ i ∉ Z

/-- Segment `g`'s noise-active coordinates. -/
def segCount (seg : Fin m → Fin G) (act : Finset (Fin m)) (g : Fin G) : ℕ :=
  (act.filter fun i => seg i = g).card

/-- **A segmented consumer**: its least work once `Z` is zeroed is `c` for each segment with a noise-active coordinate
left (`segActive`). -/
noncomputable def segConsumer (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) : Consumer m where
  act := act
  cost Z := c * (segActive seg act Z).card

/-- The segments holding a noise-active coordinate and no coordinate of `Z`. -/
def segUntouched (seg : Fin m → Fin G) (act Z : Finset (Fin m)) : Finset (Fin G) :=
  univ.filter fun g => (∃ i ∈ act, seg i = g) ∧ ∀ i ∈ Z, seg i ≠ g

/-- **A segmented consumer under E1(b)'s accounting**: `c` for each active segment none of whose coordinates is zeroed;
a segment touched by `Z` counts as saved. -/
noncomputable def adaptConsumer (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) : Consumer m where
  act := act
  cost Z := c * (segUntouched seg act Z).card

end Accounting.TS
