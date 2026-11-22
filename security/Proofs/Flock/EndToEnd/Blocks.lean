module

public import Proofs.Flock.EndToEnd.Blocks.Statement
public import Proofs.Flock.EndToEnd.Period
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Game

/-!
# A period's incorrect proof blocks on a set of its sessions

`Flock.SecurityProofs.sum_filter_incorrect`: the incorrect proof blocks a run records on a set `I` of its sessions, as a
sum over the records.
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- The incorrect proof blocks of the sampling blocks of `I`, as a sum over every sampling block weighing `0` outside `I`. -/
theorem sum_filter_incorrect (I : Finset ℕ) (l : List SamplingBlockRec) :
    ((l.filter fun r => r.idx ∈ I).map SamplingBlockRec.incorrect).sum =
      (l.map fun r => if r.idx ∈ I then r.incorrect else 0).sum := by
  induction l with
  | nil => rfl
  | cons r l ih =>
      by_cases h : r.idx ∈ I
      · simp [h, ih]
      · simp [h, ih]

end Flock.SecurityProofs
