import Batteries.Tactic.OpenPrivate

/-!
# The coin tree's four names sort to themselves

`CoinTree.checkRecord` sorts the record's `coin_tree` keys with `Array.qsort` and compares them with
`key, nonce, root, spec`. `qsort_coinTree` runs that sort on the four names: three partitions (`qpart03`, `qpart01`,
`qpart23`) and the one-element sorts. It sits in a module of its own, on core's `simp` set, where string comparisons
reduce by `decide`.
-/

open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic

namespace FlockComplete.JsonRT

private theorem qpart03 : Array.qpartition #v["key", "nonce", "root", "spec"] (fun x1 x2 => decide (x1 < x2)) 0 3
    (by decide) (by decide) (by decide) = (⟨1, by decide⟩, #v["key", "nonce", "root", "spec"]) := by
  unfold Array.qpartition
  simp
  repeat (rw [Array.qpartition.loop]; simp)

private theorem qpart01 : Array.qpartition #v["key", "nonce", "root", "spec"] (fun x1 x2 => decide (x1 < x2)) 0 1
    (by decide) (by decide) (by decide) = (⟨0, by decide⟩, #v["key", "nonce", "root", "spec"]) := by
  unfold Array.qpartition
  simp
  repeat (rw [Array.qpartition.loop]; simp)

private theorem qpart23 : Array.qpartition #v["key", "nonce", "root", "spec"] (fun x1 x2 => decide (x1 < x2)) 2 3
    (by decide) (by decide) (by decide) = (⟨2, by decide⟩, #v["key", "nonce", "root", "spec"]) := by
  unfold Array.qpartition
  simp
  repeat (rw [Array.qpartition.loop]; simp)

private theorem sort_one (n : Nat) (v : Vector String 4) (h0 : n < 4) :
    Array.qsort.sort (fun x1 x2 => decide (x1 < x2)) v n n (by omega) h0 h0 = v := by
  rw [Array.qsort.sort]
  simp

/-- **The coin tree's four names sort to themselves**, as `checkRecord` sorts them. -/
theorem qsort_coinTree :
    #["key", "nonce", "root", "spec"].qsort (· < ·) = #["key", "nonce", "root", "spec"] := by
  unfold Array.qsort
  simp only [List.size_toArray, List.length_cons, List.length_nil, Nat.reduceAdd, Nat.reduceSub, Nat.reduceEqDiff,
    dite_false, Nat.min_def, Nat.max_def, Nat.reduceLeDiff, ite_true, Nat.zero_le]
  rw [Array.qsort.sort]
  simp only [Nat.reduceLT, dite_true, Array.toVector]
  rw [qpart03]
  simp only [dite_false, show ¬ (1 ≥ 3) by decide, Nat.reduceAdd]
  have hin : Array.qsort.sort (fun x1 x2 => decide (x1 < x2)) #v["key", "nonce", "root", "spec"] 0 1 (by decide)
      (by decide) (by decide) = #v["key", "nonce", "root", "spec"] := by
    rw [Array.qsort.sort]
    simp only [Nat.reduceLT, dite_true]
    rw [qpart01]
    simp only [show ¬ (0 ≥ 1) by decide, dite_false, Nat.reduceAdd, sort_one]
  rw [hin, Array.qsort.sort]
  simp only [Nat.reduceLT, dite_true]
  rw [qpart23]
  simp only [show ¬ (2 ≥ 3) by decide, dite_false, Nat.reduceAdd, sort_one]

end FlockComplete.JsonRT
