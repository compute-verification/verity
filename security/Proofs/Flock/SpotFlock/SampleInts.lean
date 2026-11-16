module

public import Properties.Spot
public import Proofs.Core.QSort

/-!
# A sample's integers read back, and an ascending array is its own sort

What `SpotFlock.RunsAs` reads of a checked sample's units:

* `Properties.Spot.Credited.filterMap_int_arr`: an array whose every entry `Spot.int?` reads (`mapM`) is read whole by
  `filterMap int?`;
* `Verity.Sampled.qsort_ascending`: an array of integers that ascends (`Properties.Spot.Ascending`) is its own `qsort`.

They are declared in the namespaces `Properties.Spot.Credited` and `Verity.Sampled`, where their other readers are:
`Proofs.Properties.Spot.Credited` imports this file and reads the first, and `Proofs.Properties.Verity.Sampled.Held`,
through it, the second.
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot.Credited

theorem filterMap_int : ∀ {l : List Json} {u : List Int}, l.mapM int? = some u → l.filterMap int? = u
  | [], u, h => by
    simp only [List.mapM_nil] at h
    exact Option.some.inj h
  | a :: l, u, h => by
    simp only [List.mapM_cons] at h
    cases ha : int? a with
    | none => simp [ha] at h
    | some x =>
      cases hl : l.mapM int? with
      | none => simp [ha, hl] at h
      | some xs =>
        simp only [ha, hl, Option.bind_eq_bind, Option.bind_some] at h
        obtain rfl := Option.some.inj h
        simp only [List.filterMap_cons, ha, filterMap_int hl]

theorem toList_of_mapM {us : Array Json} {xs : Array Int} (h : us.mapM int? = some xs) :
    us.toList.mapM int? = some xs.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : us.toList.mapM int? with
  | none => rw [hl] at h; cases h
  | some l =>
    rw [hl] at h
    obtain rfl := Option.some.inj h
    rfl

/-- Integers read with `int?` throughout are read by `filterMap int?`. -/
theorem filterMap_int_arr {us : Array Json} {xs : Array Int} (h : us.mapM int? = some xs) :
    us.filterMap int? = xs :=
  Array.toList_inj.1 (by rw [Array.toList_filterMap]; exact filterMap_int (toList_of_mapM h))

end Properties.Spot.Credited

namespace Verity.Sampled

theorem ascending_lt {xs : Array Int} (h : _root_.Properties.Spot.Ascending xs) :
    ∀ i j (hi : i < xs.size) (hj : j < xs.size), i < j → xs[i] < xs[j] := by
  intro i j hi hj hij
  induction j with
  | zero => omega
  | succ j ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 hij with hlt | rfl
    · exact Int.lt_trans (ih (by omega) hlt) (h j hj)
    · exact h i hj

theorem qsort_ascending {xs : Array Int} (h : _root_.Properties.Spot.Ascending xs) :
    xs.qsort (· < ·) = xs := by
  refine Verity.QSort.qsort_sorted _ _ fun i hi j hj => ?_
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · simp only [ascending_lt h i j hi hj hij, hij, decide_true]
  · simp only [Int.lt_irrefl, Nat.lt_irrefl, decide_false]
  · have := ascending_lt h j i hj hi hij
    simp only [show ¬ xs[i] < xs[j] from Int.not_lt.2 (Int.le_of_lt this), show ¬ i < j by omega, decide_false]

end Verity.Sampled
