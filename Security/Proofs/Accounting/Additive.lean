import Definitions.Accounting.Guarantees.Additive
import Proofs.Core.Accel.Model
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.NNRat

/-!
# Compute additivity's capacity from the machine model 𝓜: proofs

`Accounting.Guarantees.AccelRate` and `AccelAdditive`. The one step that reads 𝓜 is that units add over runs back to
back: a period's stream is its runs' instructions one after another (`stream_cons`, `Machine.units_append`).
-/

open scoped NNRat

namespace Accounting.SecurityProofs.Additivity

open Finset Verity.Accel

variable {m : Machine}

theorem stream_cons (e : AccelRun m) (es : List (AccelRun m)) : stream (e :: es) = e.instrs ++ stream es := by
  simp [stream]

theorem periodSeconds_cons (e : AccelRun m) (es : List (AccelRun m)) :
    periodSeconds (e :: es) = e.T + periodSeconds es := by
  simp [periodSeconds]

theorem sum_range_getElem? {α : Type*} (l : List α) (f : α → ℚ≥0) :
    ∑ k ∈ range l.length, ((l[k]?).map f).getD 0 = (l.map f).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    rw [List.length_cons, Finset.sum_range_succ']
    simp [ih, add_comm]

/-- The occurrences issued, summed, are the instructions' units. -/
theorem sum_issued (iss : List m.Instr) (r : Res) : ∑ p ∈ issued iss, occUse iss r p = Machine.units iss r :=
  sum_range_getElem? iss _

end Accounting.SecurityProofs.Additivity

namespace Accounting.SecurityProofs

open Finset Verity.Accel Additivity

theorem AccelRate : Guarantees.AccelRate := by
  intro m es h r
  induction es with
  | nil => simp [stream, periodSeconds, Machine.units]
  | cons e es ih =>
    rw [stream_cons, Machine.units_append, periodSeconds_cons, mul_add]
    exact add_le_add (h e (by simp) r) (ih fun e' he' => h e' (by simp [he']))

theorem AccelAdditive : Guarantees.AccelAdditive := by
  intro m iss T hrate k S hS hdisj r
  calc ∑ j, ∑ p ∈ S j, occUse iss r p
      = ∑ p ∈ univ.biUnion S, occUse iss r p :=
        (Finset.sum_biUnion fun i _ j _ hij => hdisj i j hij).symm
    _ ≤ ∑ p ∈ issued iss, occUse iss r p :=
        Finset.sum_le_sum_of_subset (Finset.biUnion_subset.2 fun j _ => hS j)
    _ = Machine.units iss r := sum_issued _ _
    _ ≤ _ := hrate r

end Accounting.SecurityProofs
