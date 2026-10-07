import Definitions.Pous.Accounting.Rebuild
import Proofs.Pous.Chain.Flat

/-!
# The rebuild cost and the graded window

`R(d) = rebuildCalls (2^9) d` is `Σ_{v < 512} (min(v, d) + 1)`: `3,057` at `d = 5`, `6,578` at `d = 12`, and
non-decreasing in `d`. The rounds of every row of `Pous.Guarantees.BandGradedMeetsFamily` last under
`240·6 + 3·13 = 1,479` call-times.
-/

namespace PousRebuild

open Pous Pous.Sponge Pous.BandChain Pous.Rebuild

/-- Label `v` of the band absorbs `min(v, d)` parents. -/
theorem length_parentsDesc_band (B d : ℕ) (v : Fin B) :
    (parentsDesc (bandDAG B d) v).length = min v.val d := by
  unfold parentsDesc
  rw [Finset.length_sort, card_par_band]

/-- `R(d) = Σ_{v < B} (min(v, d) + 1)`. -/
theorem rebuildCalls_eq (B d : ℕ) : rebuildCalls B d = ∑ v ∈ Finset.range B, (min v d + 1) := by
  unfold rebuildCalls
  rw [Finset.sum_range (fun v => min v d + 1)]
  exact Finset.sum_congr rfl fun v _ => by rw [length_parentsDesc_band]

/-- With `d < B`: the first `d` labels cost `1, …, d`, every later one `d + 1`. -/
theorem rebuildCalls_split (B d : ℕ) (hd : d ≤ B) :
    rebuildCalls B d = ∑ v ∈ Finset.range d, (v + 1) + (B - d) * (d + 1) := by
  rw [rebuildCalls_eq, show B = d + (B - d) by omega, Finset.sum_range_add, Nat.add_sub_cancel_left]
  congr 1
  · exact Finset.sum_congr rfl fun v hv => by rw [min_eq_left (Finset.mem_range.1 hv).le]
  · rw [Finset.sum_congr rfl fun v _ => by rw [min_eq_right (Nat.le_add_right d v)], Finset.sum_const,
      Finset.card_range, smul_eq_mul]

theorem rebuildCalls_512_5 : rebuildCalls (2 ^ 9) 5 = 3057 := by
  rw [rebuildCalls_split _ _ (by norm_num)]
  simp [Finset.sum_range_succ]

theorem rebuildCalls_512_12 : rebuildCalls (2 ^ 9) 12 = 6578 := by
  rw [rebuildCalls_split _ _ (by norm_num)]
  simp [Finset.sum_range_succ]

/-- `R` is non-decreasing in the in-degree. -/
theorem rebuildCalls_mono (B : ℕ) {d d' : ℕ} (h : d ≤ d') : rebuildCalls B d ≤ rebuildCalls B d' := by
  rw [rebuildCalls_eq, rebuildCalls_eq]
  exact Finset.sum_le_sum fun v _ => Nat.add_le_add_right (min_le_min_left v h) 1

theorem rebuildCalls_ge (d : ℕ) (hd : 5 ≤ d) : 3057 ≤ rebuildCalls (2 ^ 9) d :=
  rebuildCalls_512_5 ▸ rebuildCalls_mono _ hd

/-- **Every graded row's rounds last under `1,479` call-times** when `D_late ≤ 12`; the worst row is `(3, 243)`. -/
theorem gradedRoundSpan_le (D_late : ℕ) (h : D_late ≤ 12) :
    ∀ tk ∈ ([(0, 111), (1, 160), (2, 203), (3, 243)] : List (ℕ × ℕ)),
      gradedRoundSpan 5 D_late tk.2 tk.1 ≤ 1479 := by
  intro tk htk
  simp only [List.mem_cons, List.not_mem_nil, or_false] at htk
  rcases htk with rfl | rfl | rfl | rfl <;> simp only [gradedRoundSpan] <;> omega

end PousRebuild
