import Proofs.Pous.P3.Column.BandComb
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# Weighted starts and ends in the band

Two peeling lemmas with weights, and the window/block counting they are used with.

* `weighted_starts`: if ranks `t` increase along band edges of `R`, then `Σ_{w ∈ R} φ(t w) ≤ Σ_{s ∈ starts R} Ψ(t s)`
  whenever `φ x + Ψ (x + 1) ≤ Ψ x` and `Ψ` is non-increasing. With `φ x = #{u : x < r u}` and `Ψ x = Σ_u (r u − x)`,
  each rebuild segment is charged to its start.
* `weighted_ends`: dually, `Σ_{u ∈ U} ψ(r u) ≤ Σ_{x ∈ ends U} Θ(r x)` whenever `ψ x + Θ (x − 1) ≤ Θ x` and `Θ` is
  non-decreasing. With `ψ x = x − t` and `Θ x = tri (x − t)`, each dropped segment is charged to its end.
-/

namespace PousP3Column

open Pous Pous.ColumnGame

variable {n : ℕ}

/-! ### Windows of starts -/

theorem starts_window_disjoint (k : ℕ) (S : Finset (Fin n)) {a b : Fin n} (ha : a ∈ starts k S)
    (hb : b ∈ starts k S) (hab : a ≠ b) : Disjoint (window k a) (window k b) := by
  rw [Finset.disjoint_left]
  intro u hua hub
  simp only [window, Finset.mem_filter, Finset.mem_univ, true_and] at hua hub
  unfold Band at hua hub
  rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hab) with hlt | hlt
  · exact (Finset.mem_filter.1 hb).2 a (starts_subset k S ha) ⟨hlt, by omega⟩
  · exact (Finset.mem_filter.1 ha).2 b (starts_subset k S hb) ⟨hlt, by omega⟩

theorem starts_small_card_le_one (k : ℕ) (S : Finset (Fin n)) :
    ((starts k S).filter fun v => ¬ k ≤ v.val).card ≤ 1 := by
  rw [Finset.card_le_one]
  intro a ha b hb
  rw [Finset.mem_filter] at ha hb
  by_contra hne
  rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne) with hlt | hlt
  · exact (Finset.mem_filter.1 hb.1).2 a (starts_subset k S ha.1) ⟨hlt, by omega⟩
  · exact (Finset.mem_filter.1 ha.1).2 b (starts_subset k S hb.1) ⟨hlt, by omega⟩

/-- The full starts, each with its window, occupy disjoint `k + 1` nodes. -/
theorem full_starts_mul_le (k : ℕ) (S : Finset (Fin n)) :
    ((starts k S).filter fun v => k ≤ v.val).card * (k + 1) ≤ n := by
  classical
  set F := (starts k S).filter fun v => k ≤ v.val
  let blk : Fin n → Finset (Fin n) := fun v => insert v (window k v)
  have hnot : ∀ v : Fin n, v ∉ window k v := fun v h => by
    have := (Finset.mem_filter.1 h).2
    unfold Band at this
    omega
  have hcard : ∀ v ∈ F, k + 1 ≤ (blk v).card := by
    intro v hv
    rw [Finset.card_insert_of_notMem (hnot v)]
    have := window_card k v (Finset.mem_filter.1 hv).2
    omega
  have hdisj : (F : Set (Fin n)).PairwiseDisjoint blk := by
    intro a ha b hb hab
    rw [Finset.coe_filter] at ha hb
    rw [Function.onFun, Finset.disjoint_left]
    intro u hua hub
    rw [Finset.mem_insert] at hua hub
    have hsa := starts_subset k S ha.1
    have hsb := starts_subset k S hb.1
    rcases hua with rfl | hua <;> rcases hub with rfl | hub
    · exact hab rfl
    · exact (Finset.mem_filter.1 hb.1).2 u hsa (Finset.mem_filter.1 hub).2
    · exact (Finset.mem_filter.1 ha.1).2 u hsb (Finset.mem_filter.1 hua).2
    · exact Finset.disjoint_left.1 (starts_window_disjoint k S ha.1 hb.1 hab) hua hub
  calc F.card * (k + 1) = ∑ _v ∈ F, (k + 1) := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ ∑ v ∈ F, (blk v).card := Finset.sum_le_sum hcard
    _ = (F.biUnion blk).card := (Finset.card_biUnion hdisj).symm
    _ ≤ (Finset.univ : Finset (Fin n)).card := Finset.card_le_card (Finset.subset_univ _)
    _ = n := by rw [Finset.card_univ, Fintype.card_fin]

/-! ### Weighted starts -/

/-- **Weighted starts.** Charge each node of `R` to the start of its segment. -/
theorem weighted_starts (k : ℕ) (t : Fin n → ℕ) (φ Ψ : ℕ → ℕ) (hΨ : ∀ x, φ x + Ψ (x + 1) ≤ Ψ x)
    (hmono : ∀ x y, x ≤ y → Ψ y ≤ Ψ x) :
    ∀ R : Finset (Fin n), (∀ a b, Band k a b → a ∈ R → b ∈ R → t a < t b) →
      ∑ w ∈ R, φ (t w) ≤ ∑ s ∈ starts k R, Ψ (t s) := by
  classical
  intro R
  induction R using Finset.strongInduction with
  | H R ih =>
    intro ht
    rcases R.eq_empty_or_nonempty with h0 | hne
    · simp [h0]
    set T := starts k R with hTdef
    set R' := R \ T with hR'def
    have hTR : T ⊆ R := starts_subset k R
    have hTne : T.Nonempty := by
      refine ⟨R.min' hne, Finset.mem_filter.2 ⟨R.min'_mem hne, fun u hu huv => ?_⟩⟩
      have := R.min'_le u hu
      exact absurd huv.1 (not_lt.2 (Fin.le_def.1 this))
    have hlt : R' ⊂ R := by
      refine Finset.ssubset_iff_subset_ne.2 ⟨Finset.sdiff_subset, fun h => ?_⟩
      obtain ⟨x, hx⟩ := hTne
      have : x ∈ R' := h ▸ hTR hx
      exact (Finset.mem_sdiff.1 this).2 hx
    have hIH := ih R' hlt fun a b h ha hb => ht a b h (Finset.mem_sdiff.1 ha).1 (Finset.mem_sdiff.1 hb).1
    -- each start of `R'` has a start of `R` among its predecessors, with a smaller rank
    let g : Fin n → Fin n := fun v => if hv : ∃ u ∈ R, Band k u v then Classical.choose hv else v
    have hg : ∀ v ∈ starts k R', g v ∈ T ∧ Band k (g v) v ∧ t (g v) < t v := by
      intro v hv
      have hv' := Finset.mem_sdiff.1 (starts_subset k R' hv)
      have hex := exists_band_of_not_start hv'.1 hv'.2
      have hgv : g v = Classical.choose hex := by simp only [g, hex, ↓reduceDIte]
      obtain ⟨hS, hB⟩ := Classical.choose_spec hex
      rw [← hgv] at hS hB
      refine ⟨?_, hB, ht _ _ hB hS hv'.1⟩
      by_contra hT
      exact (Finset.mem_filter.1 hv).2 (g v) (Finset.mem_sdiff.2 ⟨hS, hT⟩) hB
    have hginj : Set.InjOn g (starts k R') := by
      intro a ha b hb hab
      obtain ⟨-, hBa, -⟩ := hg a ha
      obtain ⟨-, hBb, -⟩ := hg b hb
      rw [hab] at hBa
      unfold Band at hBa hBb
      by_contra hne'
      rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne') with h | h
      · exact (Finset.mem_filter.1 hb).2 a (starts_subset k R' ha) ⟨h, by omega⟩
      · exact (Finset.mem_filter.1 ha).2 b (starts_subset k R' hb) ⟨h, by omega⟩
    have hsplit : ∑ w ∈ R, φ (t w) = ∑ w ∈ T, φ (t w) + ∑ w ∈ R', φ (t w) := by
      rw [hR'def, ← Finset.sum_sdiff hTR, add_comm]
    have h2 : ∑ s ∈ starts k R', Ψ (t s) ≤ ∑ s ∈ T, Ψ (t s + 1) := by
      calc ∑ s ∈ starts k R', Ψ (t s) ≤ ∑ s ∈ starts k R', Ψ (t (g s) + 1) :=
            Finset.sum_le_sum fun s hs => hmono _ _ (by have := (hg s hs).2.2; omega)
        _ = ∑ s ∈ (starts k R').image g, Ψ (t s + 1) :=
            (Finset.sum_image (f := fun s => Ψ (t s + 1)) hginj).symm
        _ ≤ ∑ s ∈ T, Ψ (t s + 1) :=
            Finset.sum_le_sum_of_subset_of_nonneg (fun s hs => by
              obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hs
              exact (hg v hv).1) (fun _ _ _ => Nat.zero_le _)
    calc ∑ w ∈ R, φ (t w) = ∑ w ∈ T, φ (t w) + ∑ w ∈ R', φ (t w) := hsplit
      _ ≤ ∑ w ∈ T, φ (t w) + ∑ s ∈ T, Ψ (t s + 1) := by linarith [hIH, h2]
      _ = ∑ s ∈ T, (φ (t s) + Ψ (t s + 1)) := by rw [Finset.sum_add_distrib]
      _ ≤ ∑ s ∈ T, Ψ (t s) := Finset.sum_le_sum fun s _ => hΨ _

/-! ### Ends and their successor blocks -/

/-- The `k` successors of `x`. -/
def block (k : ℕ) (x : Fin n) : Finset (Fin n) := Finset.univ.filter fun v => Band k x v

theorem block_card (k : ℕ) (x : Fin n) (hx : x.val + k < n) : k ≤ (block k x).card := by
  let f : ℕ → Fin n := fun j => if h : x.val + 1 + j < n then ⟨x.val + 1 + j, h⟩ else x
  have hmem : ∀ j ∈ Finset.range k, f j ∈ block k x := by
    intro j hj
    rw [Finset.mem_range] at hj
    have h : x.val + 1 + j < n := by omega
    have hfj : f j = ⟨x.val + 1 + j, h⟩ := by simp only [f, h, ↓reduceDIte]
    rw [hfj, block, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    unfold Band
    simp only
    omega
  have hinj : Set.InjOn f (Finset.range k) := by
    intro a ha b hb hab
    rw [Finset.coe_range, Set.mem_Iio] at ha hb
    have ha' : x.val + 1 + a < n := by omega
    have hb' : x.val + 1 + b < n := by omega
    have hfa : f a = ⟨x.val + 1 + a, ha'⟩ := by simp only [f, ha', ↓reduceDIte]
    have hfb : f b = ⟨x.val + 1 + b, hb'⟩ := by simp only [f, hb', ↓reduceDIte]
    rw [hfa, hfb, Fin.mk.injEq] at hab
    omega
  simpa using Finset.card_le_card_of_injOn f hmem hinj

theorem ends_block_disjoint (k : ℕ) (U : Finset (Fin n)) {a b : Fin n} (ha : a ∈ ends k U) (hb : b ∈ ends k U)
    (hab : a ≠ b) : Disjoint (block k a) (block k b) := by
  rw [Finset.disjoint_left]
  intro v hva hvb
  simp only [block, Finset.mem_filter, Finset.mem_univ, true_and] at hva hvb
  unfold Band at hva hvb
  have haU := (Finset.mem_filter.1 ha).1
  have hbU := (Finset.mem_filter.1 hb).1
  rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hab) with hlt | hlt
  · exact (Finset.mem_filter.1 ha).2 b hbU ⟨hlt, by omega⟩
  · exact (Finset.mem_filter.1 hb).2 a haU ⟨hlt, by omega⟩

theorem block_disjoint_of_end (k : ℕ) (U : Finset (Fin n)) {x : Fin n} (hx : x ∈ ends k U) :
    ∀ v ∈ block k x, v ∉ U := fun v hv hvU =>
  (Finset.mem_filter.1 hx).2 v hvU (Finset.mem_filter.1 hv).2

/-- At most one end has fewer than `k` successors. -/
theorem ends_final_card_le_one (k : ℕ) (U : Finset (Fin n)) :
    ((ends k U).filter fun x => ¬ x.val + k < n).card ≤ 1 := by
  rw [Finset.card_le_one]
  intro a ha b hb
  rw [Finset.mem_filter] at ha hb
  by_contra hne
  have haU := (Finset.mem_filter.1 ha.1).1
  have hbU := (Finset.mem_filter.1 hb.1).1
  have hbn := b.isLt
  have han := a.isLt
  rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne) with hlt | hlt
  · exact (Finset.mem_filter.1 ha.1).2 b hbU ⟨hlt, by omega⟩
  · exact (Finset.mem_filter.1 hb.1).2 a haU ⟨hlt, by omega⟩

/-- The non-final ends' blocks are disjoint `k`-sets inside any `Z` containing them. -/
theorem nonfinal_ends_mul_le (k : ℕ) (U Z : Finset (Fin n)) (E : Finset (Fin n))
    (hE : E ⊆ (ends k U).filter fun x => x.val + k < n) (hZ : ∀ x ∈ E, block k x ⊆ Z) :
    E.card * k ≤ Z.card := by
  classical
  have hdisj : (E : Set (Fin n)).PairwiseDisjoint (block k) := by
    intro a ha b hb hab
    exact ends_block_disjoint k U (Finset.mem_filter.1 (hE ha)).1 (Finset.mem_filter.1 (hE hb)).1 hab
  calc E.card * k = ∑ _x ∈ E, k := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ ∑ x ∈ E, (block k x).card :=
        Finset.sum_le_sum fun x hx => block_card k x (Finset.mem_filter.1 (hE hx)).2
    _ = (E.biUnion (block k)).card := (Finset.card_biUnion hdisj).symm
    _ ≤ Z.card := Finset.card_le_card (Finset.biUnion_subset.2 hZ)

/-! ### Weighted ends -/

/-- **Weighted ends.** Charge each node of `U` to the end of its segment. -/
theorem weighted_ends (k : ℕ) (r : Fin n → ℕ) (ψ Θ : ℕ → ℕ) (hΘ : ∀ x, ψ x + Θ (x - 1) ≤ Θ x)
    (hmono : ∀ x y, x ≤ y → Θ x ≤ Θ y) :
    ∀ U : Finset (Fin n), (∀ a b, Band k a b → a ∈ U → b ∈ U → r a < r b) →
      ∑ u ∈ U, ψ (r u) ≤ ∑ x ∈ ends k U, Θ (r x) := by
  classical
  intro U
  induction U using Finset.strongInduction with
  | H U ih =>
    intro hr
    rcases U.eq_empty_or_nonempty with h0 | hne
    · simp [h0]
    set E := ends k U with hEdef
    set U' := U \ E with hU'def
    have hEU : E ⊆ U := Finset.filter_subset _ _
    have hEne : E.Nonempty := by
      refine ⟨U.max' hne, Finset.mem_filter.2 ⟨U.max'_mem hne, fun b hb hab => ?_⟩⟩
      have := U.le_max' b hb
      exact absurd hab.1 (not_lt.2 (Fin.le_def.1 this))
    have hlt : U' ⊂ U := by
      refine Finset.ssubset_iff_subset_ne.2 ⟨Finset.sdiff_subset, fun h => ?_⟩
      obtain ⟨x, hx⟩ := hEne
      have : x ∈ U' := h ▸ hEU hx
      exact (Finset.mem_sdiff.1 this).2 hx
    have hIH := ih U' hlt fun a b h ha hb => hr a b h (Finset.mem_sdiff.1 ha).1 (Finset.mem_sdiff.1 hb).1
    let g : Fin n → Fin n := fun v => if hv : ∃ b ∈ U, Band k v b then Classical.choose hv else v
    have hg : ∀ v ∈ ends k U', g v ∈ E ∧ Band k v (g v) ∧ r v < r (g v) := by
      intro v hv
      have hv' := Finset.mem_sdiff.1 (Finset.filter_subset _ _ hv)
      have hex : ∃ b ∈ U, Band k v b := by
        by_contra hne'
        push Not at hne'
        exact hv'.2 (Finset.mem_filter.2 ⟨hv'.1, fun b hb h => hne' b hb h⟩)
      have hgv : g v = Classical.choose hex := by simp only [g, hex, ↓reduceDIte]
      obtain ⟨hS, hB⟩ := Classical.choose_spec hex
      rw [← hgv] at hS hB
      refine ⟨?_, hB, hr _ _ hB hv'.1 hS⟩
      by_contra hE
      exact (Finset.mem_filter.1 hv).2 (g v) (Finset.mem_sdiff.2 ⟨hS, hE⟩) hB
    have hginj : Set.InjOn g (ends k U') := by
      intro a ha b hb hab
      obtain ⟨-, hBa, -⟩ := hg a ha
      obtain ⟨-, hBb, -⟩ := hg b hb
      rw [hab] at hBa
      unfold Band at hBa hBb
      by_contra hne'
      rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne') with h | h
      · exact (Finset.mem_filter.1 ha).2 b (Finset.filter_subset _ _ hb) ⟨h, by omega⟩
      · exact (Finset.mem_filter.1 hb).2 a (Finset.filter_subset _ _ ha) ⟨h, by omega⟩
    have hsplit : ∑ u ∈ U, ψ (r u) = ∑ u ∈ E, ψ (r u) + ∑ u ∈ U', ψ (r u) := by
      rw [hU'def, ← Finset.sum_sdiff hEU, add_comm]
    have h2 : ∑ x ∈ ends k U', Θ (r x) ≤ ∑ x ∈ E, Θ (r x - 1) := by
      calc ∑ x ∈ ends k U', Θ (r x) ≤ ∑ x ∈ ends k U', Θ (r (g x) - 1) :=
            Finset.sum_le_sum fun x hx => hmono _ _ (by have := (hg x hx).2.2; omega)
        _ = ∑ x ∈ (ends k U').image g, Θ (r x - 1) :=
            (Finset.sum_image (f := fun x => Θ (r x - 1)) hginj).symm
        _ ≤ ∑ x ∈ E, Θ (r x - 1) :=
            Finset.sum_le_sum_of_subset_of_nonneg (fun x hx => by
              obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hx
              exact (hg v hv).1) (fun _ _ _ => Nat.zero_le _)
    calc ∑ u ∈ U, ψ (r u) = ∑ u ∈ E, ψ (r u) + ∑ u ∈ U', ψ (r u) := hsplit
      _ ≤ ∑ u ∈ E, ψ (r u) + ∑ x ∈ E, Θ (r x - 1) := by linarith [hIH, h2]
      _ = ∑ x ∈ E, (ψ (r x) + Θ (r x - 1)) := by rw [Finset.sum_add_distrib]
      _ ≤ ∑ x ∈ E, Θ (r x) := Finset.sum_le_sum fun x _ => hΘ _

/-! ### Triangular numbers -/

/-- `tri m = 0 + 1 + ⋯ + m`. -/
def tri (m : ℕ) : ℕ := ∑ i ∈ Finset.range (m + 1), i

theorem tri_succ (m : ℕ) : tri (m + 1) = tri m + (m + 1) := by
  unfold tri; rw [Finset.sum_range_succ]

theorem tri_mono {a b : ℕ} (h : a ≤ b) : tri a ≤ tri b :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 (by omega)) fun _ _ _ => Nat.zero_le _

/-- The step `weighted_ends` needs, for `ψ x = x − t`, `Θ x = tri (x − t)`. -/
theorem tri_step (t x : ℕ) : (x - t) + tri (x - 1 - t) ≤ tri (x - t) := by
  rcases Nat.lt_or_ge t x with h | h
  · obtain ⟨m, hm⟩ : ∃ m, x - t = m + 1 := ⟨x - t - 1, by omega⟩
    have : x - 1 - t = m := by omega
    rw [hm, this, tri_succ]; omega
  · have h1 : x - t = 0 := by omega
    have h2 : x - 1 - t = 0 := by omega
    rw [h1, h2]; simp

end PousP3Column
