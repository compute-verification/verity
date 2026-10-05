import Proofs.Flock.Soundness.Refine.Tapes
import Proofs.Flock.Soundness.Refine.Exec
import Proofs.Flock.Soundness.Refine.Plays

/-!
# Element seams: the executable's arithmetic is the model's

The model computes in the fields of `execArith` (`GF128`, `GF256`); the executable computes on their words (`F128`,
`F256`) and arrays. `fOf` (`GF128.of`) maps a word to its field element, and it is a ring homomorphism on the operations
the verifier uses (`fOf_add`, `fOf_mul`, `fOf_inv`). This file adds what the phases need on top: the executable's `dot`
is a sum (`fOf_dot`), the weight arrays have 64 entries, reading a coin as `getD` or `!` agrees, and the zerocheck
vector `pinned ++ r_outer` is the model's `zcVector`.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

theorem fOf_add (a b : F128) : fOf (a + b) = fOf a + fOf b := of_add a b
theorem fOf_mul (a b : F128) : fOf (a * b) = fOf a * fOf b := of_mul a b
theorem fOf_inv (a : F128) : fOf a.inv = (fOf a)⁻¹ := of_inv a
theorem fOf_one : fOf 1 = 1 := GF128.one_eq.symm
theorem fOf_zero : fOf 0 = 0 := GF128.zero_eq.symm

theorem fOf_injective : Function.Injective fOf := fun _ _ h => h

/-- An out-of-range read is the default word, zero. -/
theorem getElem!_eq_getD (c : Array F128) (i : ℕ) : c[i]! = c.getD i 0 := by
  by_cases hi : i < c.size
  · rw [getElem!_pos c i hi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]; rfl
  · rw [getElem!_neg c i hi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl

theorem dot_eq_foldl (a b : Array F128) :
    dot a b = (List.range (min a.size b.size)).foldl (fun acc i => acc + a[i]! * b[i]!) 0 := by
  unfold dot
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  simp [Std.Legacy.Range.size, List.range_eq_range']
  rfl

theorem fOf_foldl_dot (a b : Array F128) :
    ∀ (k : ℕ) (acc : F128), fOf ((List.range k).foldl (fun acc i => acc + a[i]! * b[i]!) acc) =
      fOf acc + ∑ i ∈ Finset.range k, fOf a[i]! * fOf b[i]! := by
  intro k
  induction k with
  | zero => intro acc; simp
  | succ k ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, fOf_add, ih, fOf_mul,
        Finset.sum_range_succ, add_assoc]

/-- **`dot` is a sum** over the first array's entries; the second array's missing entries read as zero. -/
theorem fOf_dot (a b : Array F128) (n : ℕ) (ha : a.size = n) :
    fOf (dot a b) = ∑ i : Fin n, fOf a[i.val]! * fOf b[i.val]! := by
  rw [dot_eq_foldl, fOf_foldl_dot, fOf_zero, zero_add, Fin.sum_univ_eq_sum_range (fun i => fOf a[i]! * fOf b[i]!) n]
  rw [ha]
  by_cases hb : n ≤ b.size
  · rw [min_eq_left hb]
  · rw [min_eq_right (by omega)]
    refine Finset.sum_subset (Finset.range_mono (by omega)) fun i hi hni => ?_
    simp only [Finset.mem_range, not_lt] at hi hni
    rw [getElem!_neg b i (by omega)]
    show fOf a[i]! * fOf 0 = 0
    rw [fOf_zero, mul_zero]

theorem lagrange_size (nodes : Array F128) (d z : F128) : (lagrange nodes d z).size = nodes.size := by
  unfold lagrange
  split <;> simp

theorem nodesL_size : nodesL.size = 64 := by simp [nodesL]
theorem nodesS_size : nodesS.size = 64 := by simp [nodesS]

theorem skipWeights_size (z : F128) : (skipWeights z).size = 64 := by
  rw [skipWeights, lagrange_size, nodesS_size]

theorem combWeights_size (z : F128) : (combWeights z).size = 64 := by
  unfold combWeights
  split <;> simp

/-- The zerocheck vector (§9.2): the executable's `pinned ++ r_outer` is the model's `zcVector` on the coins
`r_outer`. -/
theorem zcVector_eq {m : ℕ} (c : Array F128) (hc : c.size = m - 13) (hm : 13 ≤ m) (j : Fin (m - 6)) :
    Model.zcVector execArith (vecOf (m - 13) c) j = fOf ((pinned ++ c)[j.val]!) := by
  unfold Model.zcVector
  have hp : pinned.size = 7 := pinned_size
  have hj : j.val < (pinned ++ c).size := by rw [Array.size_append, hp, hc]; omega
  rw [getElem!_pos (pinned ++ c) j.val hj, Array.getElem_append]
  by_cases h7 : j.val < 7
  · have h7' : j.val < pinned.size := by omega
    simp only [h7, h7', dite_true]
    show pinnedF _ = _
    unfold pinnedF
    rw [getElem!_pos pinned j.val h7']
  · have h7' : ¬ j.val < pinned.size := by omega
    simp only [h7, h7', dite_false]
    show fOf c[j.val - 7]! = _
    rw [getElem!_pos c (j.val - 7) (by omega)]
    simp only [hp]

end FlockSoundness.Refine
