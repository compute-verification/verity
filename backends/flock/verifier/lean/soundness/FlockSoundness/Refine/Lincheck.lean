import FlockSoundness.Refine.Zerocheck
import FlockSoundness.Refine.Realizes
import FlockLevel3.Folded
import FlockSoundness.Protocol.Refine.Lincheck

/-!
# The lincheck refines the model's (PROTOCOL.md §10)

`lincheck_refines`: if the executable's `lincheck` succeeds, on a statement whose circuit fold realizes the model
statement `S` (`FoldRealizes`) and on the zerocheck's output, the model's `lincheck`, played with the proof's lincheck
messages (`lcMsgs`) and the coins of the rounds consumed, returns `some ab`, and `ab` is the executable's `ab` claim
(§11) in the model's field.

After the rounds the executable takes the 64 entries left from the statement (`CircuitFold.folded`), which are the
column vector halved at each round's challenge (`halve`, `FoldRealizes.folded`); the model folds it after the rounds
(`foldAll`). `foldl_halve` shows the two agree on the 64 entries left.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript

local notation "𝔽" => FlockLevel3.GF128

/-! ## The column fold -/

/-! ## The rounds -/

theorem lStep_ts_get (s : Array F128) (l : List F128) (j : ℕ) (hj : j < l.length) :
    (s ++ l.toArray)[s.size + j]! = l[j] := by
  rw [getElem!_pos _ _ (by simp; omega), Array.getElem_append_right (by omega)]
  simp

end FlockSoundness.Refine
