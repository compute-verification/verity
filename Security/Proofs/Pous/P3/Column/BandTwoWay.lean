import Mathlib.Tactic.IntervalCases
import Proofs.Pous.P3.Column.BandWeights
import Proofs.Pous.P3.Column.TwoWay
import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# The band-graph certificate for the two-directional column game

`band_scheduleBound` proves the schedule bound of `columnHard_of_schedule` for `B(n, k)`, so
`band_columnHard_twoWay` gives `ColumnHard (bandGraph n k) d n X` from arithmetic side conditions.

**Proof.** Charge each rebuilt label to the start of its rebuild segment (`weighted_starts`); a start at round `t`
collects `Ψ t = Σ_{u ∈ U} (r u − t)`.
* **The truncated start.** At most one start is below `k`. It collects at most `Ψ 1 = Σ_u (r u − 1)`, bounded by the
  rank sum (`band_rank_sum`).
* **A full start** has a window of `k` seeds (held or promoted), all whole before it. Its benefit is charged to the
  dropped segments by their ends (`weighted_ends`).
  - **Rounds 1–2.** Every seed pays `n − |U|`.
  - **A late non-final dropped segment exists.** Every seed pays for that segment's `k` held successors, whose
    columns are not yet revealed, so the window pays `k²` per late segment.
  - **Free.** No non-final segment is late, so only the final segment benefits, by at most `tri (d − t)`.
* **Free starts from round `τ` on.** Every non-final dropped top finishes by `τ − 3`. So the rank sum is at most
  `⌊n/k⌋·tri (τ − 4) + tri (d − 1)`, and the free starts add at most `⌊n/(k+1)⌋·tri (d − τ)`.
-/

namespace PousP3Column

open Pous Pous.ColumnGame

variable {n : ℕ}

set_option maxHeartbeats 1600000 in
/-- **The band schedule bound.** -/
theorem band_scheduleBound (n k d X p q : ℕ) (hk : 1 ≤ k) (hq : 1 ≤ q)
    (hA : ∑ i ∈ Finset.range d, (q * i - p) ≤ p * k) (hX : p * (n + k) ≤ q * X * n)
    (hT : 2 * tri (d - 3) ≤ k * k)
    (hK : p * (n + k) ≤ q * (k * (n - d * (n + k) / (k + d))))
    (hB : ∀ τ, 3 ≤ τ → τ ≤ d → n / k * tri (τ - 4) + tri (d - 1) + n / (k + 1) * tri (d - τ) ≤ X * n) :
    ScheduleBound (bandGraph n k) d X := by
  classical
  intro U L0 P R r t hrU hrinc hP hR
  have hrB : ∀ a b, Band k a b → a ∈ U → b ∈ U → r a < r b := fun a b h =>
    hrinc a b ((mem_par_band hk a b).2 h)
  have htB : ∀ a b, Band k a b → a ∈ R → b ∈ R → t a < t b := fun a b h _ hb =>
    ((hR b hb).2.2 a ((mem_par_band hk a b).2 h)).2
  have hUn : U.card ≤ n := by have := Finset.card_le_univ U; rwa [Fintype.card_fin] at this
  -- the dropped tops fill at most `⌊d(n+k)/(k+d)⌋` nodes
  have hUmax : U.card ≤ d * (n + k) / (k + d) := by
    have hchains : ∀ p : List (Fin n), p.IsChain (Band k) → (∀ x ∈ p, x ∈ U) → p.length ≤ d :=
      chain_length_le_of_rank (fun x => x ∈ U) r d (fun a b h ha hb => hrB a b h ha hb) hrU
    have hUk := band_card_le k U d hchains
    have : U.card * (k + d) ≤ d * (n + k) := by
      have h1 : k * U.card ≤ d * (n - U.card) + d * k := by rw [← Nat.mul_add]; exact hUk
      have e : d * (n - U.card) + d * U.card = d * n := by rw [← Nat.mul_add, Nat.sub_add_cancel hUn]
      nlinarith
    exact (Nat.le_div_iff_mul_le (by omega)).2 this
  -- the rank sum
  set S1 := ∑ u ∈ U, (r u - 1) with hS1def
  have hS1q : q * S1 ≤ p * (n + k) := by
    have hrank := band_rank_sum k p q d U r hrB hrU
    have hst := starts_card_mul_le k U (Finset.univ \ U) (by
      intro v hv u hu
      rw [Finset.mem_sdiff]
      exact ⟨Finset.mem_univ u, fun huU => (Finset.mem_filter.1 hv).2 u huU (Finset.mem_filter.1 hu).2⟩)
    rw [Finset.card_sdiff_of_subset (Finset.subset_univ U), Finset.card_univ, Fintype.card_fin] at hst
    calc q * S1 ≤ p * U.card + (∑ i ∈ Finset.range d, (q * i - p)) * (starts k U).card := hrank
      _ ≤ p * U.card + p * k * (starts k U).card := by gcongr
      _ = p * U.card + p * ((starts k U).card * k) := by ring
      _ ≤ p * U.card + p * (n - U.card + k) := by gcongr
      _ = p * (n + k) := by rw [← Nat.mul_add]; congr 1; omega
  have hS1X : S1 ≤ X * n := Nat.le_of_mul_le_mul_left (hS1q.trans (by nlinarith [hX])) (by omega)
  have hS1k : S1 ≤ k * (n - U.card) := by
    have h1 : q * S1 ≤ q * (k * (n - U.card)) := by
      calc q * S1 ≤ p * (n + k) := hS1q
        _ ≤ q * (k * (n - d * (n + k) / (k + d))) := hK
        _ ≤ q * (k * (n - U.card)) := by gcongr
    exact Nat.le_of_mul_le_mul_left h1 (by omega)
  -- weights per rebuilt label and per start
  set φ : ℕ → ℕ := fun x => (U.filter fun u => x < r u).card with hφ
  set Ψ : ℕ → ℕ := fun x => ∑ u ∈ U, (r u - x) with hΨdef
  have hΨ : ∀ x, φ x + Ψ (x + 1) ≤ Ψ x := by
    intro x
    simp only [hφ, hΨdef]
    rw [Finset.card_filter, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun u _ => ?_
    split_ifs <;> omega
  have hmono : ∀ x y, x ≤ y → Ψ y ≤ Ψ x := fun x y h => Finset.sum_le_sum fun u _ => by omega
  have hWS : ∑ w ∈ R, φ (t w) ≤ ∑ s ∈ starts k R, Ψ (t s) := weighted_starts k t φ Ψ hΨ hmono R htB
  -- ends of the dropped set: non-final (with `k` successors) and final
  set NF := (ends k U).filter fun x => x.val + k < n with hNF
  -- seed costs
  set c : Fin n → ℕ := fun a => if a ∈ L0 then n - U.card else (ndHeld (bandGraph n k) U r (t a)).card with hc
  set F : Fin n → ℕ := fun s => if 3 ≤ t s ∧ ∀ x ∈ NF, r x + 3 ≤ t s then tri (d - t s) else 0 with hF
  set St := starts k R with hSt
  set Fu := St.filter fun v => k ≤ v.val with hFu
  set Sm := St.filter fun v => ¬ k ≤ v.val with hSm
  -- per full start
  have hper : ∀ s ∈ Fu, Ψ (t s) ≤ ∑ a ∈ window k s, c a + F s := by
    intro s hs
    have hsSt : s ∈ St := Finset.filter_subset _ _ hs
    have hsR : s ∈ R := starts_subset k R hsSt
    have hsk : k ≤ s.val := (Finset.mem_filter.1 hs).2
    have hwin : ∀ a ∈ window k s, (a ∈ L0 ∨ a ∈ P) ∧ t a < t s := by
      intro a ha
      have hBa : Band k a s := (Finset.mem_filter.1 ha).2
      obtain ⟨hmem, hlt⟩ := (hR s hsR).2.2 a ((mem_par_band hk a s).2 hBa)
      refine ⟨?_, hlt⟩
      rcases hmem with h | h | h
      · exact Or.inl h
      · exact Or.inr h
      · exact absurd hBa ((Finset.mem_filter.1 hsSt).2 a h)
    have hwk : k ≤ (window k s).card := window_card k s hsk
    have hts1 : 1 ≤ t s := (hR s hsR).2.1
    by_cases hts : t s ≤ 2
    · have hca : ∀ a ∈ window k s, n - U.card ≤ c a := by
        intro a ha
        obtain ⟨hm, hlt⟩ := hwin a ha
        by_cases haL : a ∈ L0
        · simp only [hc, haL, ↓reduceIte, le_refl]
        · simp only [hc, haL, ↓reduceIte]
          have hsub : Finset.univ \ U ⊆ ndHeld (bandGraph n k) U r (t a) := by
            intro v hv
            unfold ndHeld
            rw [Finset.mem_filter]
            exact ⟨Finset.mem_univ _, (Finset.mem_sdiff.1 hv).2, fun s' hs' => absurd hs' (by omega)⟩
          have := Finset.card_le_card hsub
          rwa [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin] at this
      calc Ψ (t s) ≤ Ψ 1 := hmono _ _ hts1
        _ = S1 := rfl
        _ ≤ k * (n - U.card) := hS1k
        _ ≤ (window k s).card * (n - U.card) := Nat.mul_le_mul_right _ hwk
        _ = ∑ _a ∈ window k s, (n - U.card) := by rw [Finset.sum_const, smul_eq_mul]
        _ ≤ ∑ a ∈ window k s, c a := Finset.sum_le_sum hca
        _ ≤ _ := Nat.le_add_right _ _
    · push Not at hts
      -- charge the benefit to the ends of the dropped segments
      have hWE := weighted_ends k r (fun x => x - t s) (fun x => tri (x - t s)) (fun x => tri_step (t s) x)
        (fun x y h => tri_mono (by omega)) U hrB
      have hsplitE := Finset.sum_filter_add_sum_filter_not (ends k U) (fun x => x.val + k < n)
        (fun x => tri (r x - t s))
      set FinE := (ends k U).filter fun x => ¬ x.val + k < n with hFinE
      have hEU : ∀ x ∈ ends k U, x ∈ U := fun x hx => Finset.filter_subset _ _ hx
      have hfinal : ∑ x ∈ FinE, tri (r x - t s) ≤ tri (d - t s) := by
        have hcard : FinE.card ≤ 1 := ends_final_card_le_one k U
        calc ∑ x ∈ FinE, tri (r x - t s) ≤ FinE.card • tri (d - t s) :=
              Finset.sum_le_card_nsmul _ _ _ fun x hx =>
                tri_mono (by have := (hrU x (hEU x (Finset.filter_subset _ _ hx))).2; omega)
          _ ≤ 1 * tri (d - t s) := by rw [smul_eq_mul]; exact Nat.mul_le_mul_right _ hcard
          _ = tri (d - t s) := one_mul _
      set LateNF := NF.filter fun x => t s ≤ r x + 2 with hLate
      have hNFsplit := Finset.sum_filter_add_sum_filter_not NF (fun x => t s ≤ r x + 2)
        (fun x => tri (r x - t s))
      have hearly : ∑ x ∈ NF.filter (fun x => ¬ t s ≤ r x + 2), tri (r x - t s) = 0 := by
        refine Finset.sum_eq_zero fun x hx => ?_
        have := (Finset.mem_filter.1 hx).2
        have h0 : r x - t s = 0 := by omega
        rw [h0]; rfl
      have hlate : ∑ x ∈ LateNF, tri (r x - t s) ≤ LateNF.card * tri (d - 3) := by
        calc ∑ x ∈ LateNF, tri (r x - t s) ≤ LateNF.card • tri (d - 3) :=
              Finset.sum_le_card_nsmul _ _ _ fun x hx => tri_mono (by
                have := (hrU x (hEU x (Finset.filter_subset _ _ (Finset.filter_subset _ _ hx)))).2
                omega)
          _ = LateNF.card * tri (d - 3) := smul_eq_mul _ _
      have hΨle : Ψ (t s) ≤ LateNF.card * tri (d - 3) + tri (d - t s) := by
        calc Ψ (t s) = ∑ u ∈ U, (r u - t s) := rfl
          _ ≤ ∑ x ∈ ends k U, tri (r x - t s) := hWE
          _ = ∑ x ∈ NF, tri (r x - t s) + ∑ x ∈ FinE, tri (r x - t s) := hsplitE.symm
          _ ≤ LateNF.card * tri (d - 3) + tri (d - t s) := by
              rw [← hNFsplit, hearly, add_zero]
              exact Nat.add_le_add hlate hfinal
      rcases Nat.eq_zero_or_pos LateNF.card with h0 | hpos
      · -- free: only the final segment benefits
        have hfree : 3 ≤ t s ∧ ∀ x ∈ NF, r x + 3 ≤ t s := by
          refine ⟨hts, fun x hx => ?_⟩
          by_contra hlt
          have : x ∈ LateNF := Finset.mem_filter.2 ⟨hx, by omega⟩
          rw [Finset.card_eq_zero] at h0
          rw [h0] at this
          exact absurd this (Finset.notMem_empty _)
        have hFs : F s = tri (d - t s) := ite_eq_left_of_eq_true _ _ (eq_true hfree)
        rw [hFs]
        rw [h0, zero_mul, zero_add] at hΨle
        exact hΨle.trans (Nat.le_add_left _ _)
      · -- a late non-final segment: the seeds pay `k` per late end
        set Kset := Finset.univ.filter fun v : Fin n =>
          v ∉ U ∧ ∃ x ∈ (bandGraph n k).par v, x ∈ U ∧ t s ≤ r x + 2 with hK'
        have hKcard : LateNF.card * k ≤ Kset.card := by
          refine nonfinal_ends_mul_le k U Kset LateNF (fun x hx => Finset.filter_subset _ _ hx) ?_
          intro x hx v hv
          have hxE : x ∈ ends k U := Finset.filter_subset _ _ (Finset.filter_subset _ _ hx)
          refine Finset.mem_filter.2 ⟨Finset.mem_univ _, block_disjoint_of_end k U hxE v hv,
            x, (mem_par_band hk x v).2 (Finset.mem_filter.1 hv).2, hEU x hxE, (Finset.mem_filter.1 hx).2⟩
        have hca : ∀ a ∈ window k s, Kset.card ≤ c a := by
          intro a ha
          obtain ⟨hm, hlt⟩ := hwin a ha
          by_cases haL : a ∈ L0
          · simp only [hc, haL, ↓reduceIte]
            have hsub : Kset ⊆ Finset.univ \ U := fun v hv =>
              Finset.mem_sdiff.2 ⟨Finset.mem_univ _, (Finset.mem_filter.1 hv).2.1⟩
            have := Finset.card_le_card hsub
            rwa [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin] at this
          · simp only [hc, haL, ↓reduceIte]
            refine Finset.card_le_card fun v hv => ?_
            obtain ⟨-, hvU, x, hx, hxU, hxr⟩ := Finset.mem_filter.1 hv
            unfold ndHeld
            rw [Finset.mem_filter]
            exact ⟨Finset.mem_univ _, hvU, fun s' hs' => ⟨x, hx, hxU, by omega⟩⟩
        have hsumK : k * Kset.card ≤ ∑ a ∈ window k s, c a :=
          calc k * Kset.card ≤ (window k s).card * Kset.card := Nat.mul_le_mul_right _ hwk
            _ = ∑ _a ∈ window k s, Kset.card := by rw [Finset.sum_const, smul_eq_mul]
            _ ≤ ∑ a ∈ window k s, c a := Finset.sum_le_sum hca
        have htri : tri (d - t s) ≤ tri (d - 3) := tri_mono (by omega)
        calc Ψ (t s) ≤ LateNF.card * tri (d - 3) + tri (d - t s) := hΨle
          _ ≤ LateNF.card * tri (d - 3) + LateNF.card * tri (d - 3) := by nlinarith
          _ = LateNF.card * (2 * tri (d - 3)) := by ring
          _ ≤ LateNF.card * (k * k) := Nat.mul_le_mul_left _ hT
          _ = k * (LateNF.card * k) := by ring
          _ ≤ k * Kset.card := Nat.mul_le_mul_left _ hKcard
          _ ≤ ∑ a ∈ window k s, c a := hsumK
          _ ≤ _ := Nat.le_add_right _ _
  -- the windows of full starts are disjoint and lie in the held and promoted labels
  have hcost : ∑ s ∈ Fu, ∑ a ∈ window k s, c a ≤
      L0.card * (n - U.card) + ∑ p ∈ P, (ndHeld (bandGraph n k) U r (t p)).card := by
    have hdisj : (Fu : Set (Fin n)).PairwiseDisjoint (window k) := by
      intro a ha b hb hab
      exact starts_window_disjoint k R (Finset.filter_subset _ _ ha) (Finset.filter_subset _ _ hb) hab
    have hsub : Fu.biUnion (window k) ⊆ L0 ∪ P := by
      intro a ha
      obtain ⟨s, hs, has⟩ := Finset.mem_biUnion.1 ha
      have hsSt : s ∈ St := Finset.filter_subset _ _ hs
      have hsR : s ∈ R := starts_subset k R hsSt
      have hBa : Band k a s := (Finset.mem_filter.1 has).2
      obtain ⟨hmem, -⟩ := (hR s hsR).2.2 a ((mem_par_band hk a s).2 hBa)
      rcases hmem with h | h | h
      · exact Finset.mem_union_left _ h
      · exact Finset.mem_union_right _ h
      · exact absurd hBa ((Finset.mem_filter.1 hsSt).2 a h)
    have hLP : Disjoint L0 P := Finset.disjoint_left.2 fun a ha hp => (hP a hp).1 ha
    calc ∑ s ∈ Fu, ∑ a ∈ window k s, c a = ∑ a ∈ Fu.biUnion (window k), c a := (Finset.sum_biUnion hdisj).symm
      _ ≤ ∑ a ∈ L0 ∪ P, c a := Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => Nat.zero_le _
      _ = ∑ a ∈ L0, c a + ∑ a ∈ P, c a := Finset.sum_union hLP
      _ = L0.card * (n - U.card) + ∑ p ∈ P, (ndHeld (bandGraph n k) U r (t p)).card := by
          congr 1
          · rw [Finset.sum_congr rfl fun a ha => (by simp only [hc, ha, ↓reduceIte] : c a = n - U.card),
              Finset.sum_const, smul_eq_mul]
          · refine Finset.sum_congr rfl fun a ha => ?_
            have : a ∉ L0 := (hP a ha).1
            simp only [hc, this, ↓reduceIte]
  -- the truncated start
  have hSmle : ∑ s ∈ Sm, Ψ (t s) ≤ S1 := by
    have hcard : Sm.card ≤ 1 := starts_small_card_le_one k R
    calc ∑ s ∈ Sm, Ψ (t s) ≤ Sm.card • Ψ 1 := Finset.sum_le_card_nsmul _ _ _ fun s hs =>
          hmono _ _ (hR s (starts_subset k R (Finset.filter_subset _ _ hs))).2.1
      _ ≤ 1 * Ψ 1 := by rw [smul_eq_mul]; exact Nat.mul_le_mul_right _ hcard
      _ = S1 := one_mul _
  -- the free starts
  have hfree : S1 + ∑ s ∈ Fu, F s ≤ X * n := by
    set Fr := Fu.filter fun s => 3 ≤ t s ∧ ∀ x ∈ NF, r x + 3 ≤ t s with hFr
    have hFsum : ∑ s ∈ Fu, F s = ∑ s ∈ Fr, tri (d - t s) :=
      (Finset.sum_filter (s := Fu) (fun s => 3 ≤ t s ∧ ∀ x ∈ NF, r x + 3 ≤ t s) (fun s => tri (d - t s))).symm
    rcases Fr.eq_empty_or_nonempty with he | hne
    · rw [hFsum, he, Finset.sum_empty, add_zero]; exact hS1X
    · set τ := (Fr.image t).min' (hne.image t) with hτ
      have hτle : ∀ s ∈ Fr, τ ≤ t s := fun s hs => Finset.min'_le _ _ (Finset.mem_image_of_mem t hs)
      obtain ⟨s0, hs0, hs0τ⟩ := Finset.mem_image.1 (Finset.min'_mem (Fr.image t) (hne.image t))
      have hs0' := (Finset.mem_filter.1 hs0).2
      rw [← hτ] at hs0τ
      have hτ3 : 3 ≤ τ := hs0τ ▸ hs0'.1
      have hNFτ : ∀ x ∈ NF, r x + 3 ≤ τ := fun x hx => hs0τ ▸ hs0'.2 x hx
      have hFrle : ∑ s ∈ Fr, tri (d - t s) ≤ Fu.card * tri (d - τ) := by
        calc ∑ s ∈ Fr, tri (d - t s) ≤ Fr.card • tri (d - τ) :=
              Finset.sum_le_card_nsmul _ _ _ fun s hs => tri_mono (by have := hτle s hs; omega)
          _ ≤ Fu.card * tri (d - τ) := by
              rw [smul_eq_mul]; exact Nat.mul_le_mul_right _ (Finset.card_le_card (Finset.filter_subset _ _))
      have hFu : Fu.card ≤ n / (k + 1) := (Nat.le_div_iff_mul_le (by omega)).2 (full_starts_mul_le k R)
      by_cases hτd : τ ≤ d
      · -- the rank sum, charged to ends: non-final ends finish by `τ − 3`
        have hWE := weighted_ends k r (fun x => x - 1) (fun x => tri (x - 1)) (fun x => tri_step 1 x)
          (fun x y h => tri_mono (by omega)) U hrB
        have hsplitE := Finset.sum_filter_add_sum_filter_not (ends k U) (fun x => x.val + k < n)
          (fun x => tri (r x - 1))
        have hEU : ∀ x ∈ ends k U, x ∈ U := fun x hx => Finset.filter_subset _ _ hx
        have hNFle : ∑ x ∈ NF, tri (r x - 1) ≤ NF.card * tri (τ - 4) := by
          calc ∑ x ∈ NF, tri (r x - 1) ≤ NF.card • tri (τ - 4) :=
                Finset.sum_le_card_nsmul _ _ _ fun x hx => tri_mono (by have := hNFτ x hx; omega)
            _ = NF.card * tri (τ - 4) := smul_eq_mul _ _
        have hFinle : ∑ x ∈ (ends k U).filter (fun x => ¬ x.val + k < n), tri (r x - 1) ≤ tri (d - 1) := by
          have hcard : ((ends k U).filter (fun x => ¬ x.val + k < n)).card ≤ 1 := ends_final_card_le_one k U
          calc ∑ x ∈ (ends k U).filter (fun x => ¬ x.val + k < n), tri (r x - 1)
              ≤ ((ends k U).filter (fun x => ¬ x.val + k < n)).card • tri (d - 1) :=
                Finset.sum_le_card_nsmul _ _ _ fun x hx =>
                  tri_mono (by have := (hrU x (hEU x (Finset.filter_subset _ _ hx))).2; omega)
            _ ≤ 1 * tri (d - 1) := by rw [smul_eq_mul]; exact Nat.mul_le_mul_right _ hcard
            _ = tri (d - 1) := one_mul _
        have hNFk : NF.card ≤ n / k := by
          refine (Nat.le_div_iff_mul_le (by omega)).2 ?_
          have := nonfinal_ends_mul_le k U Finset.univ NF (Finset.subset_of_eq hNF) fun x _ => Finset.subset_univ _
          rwa [Finset.card_univ, Fintype.card_fin] at this
        have hS1e : S1 ≤ n / k * tri (τ - 4) + tri (d - 1) := by
          calc S1 = ∑ u ∈ U, (r u - 1) := rfl
            _ ≤ ∑ x ∈ ends k U, tri (r x - 1) := hWE
            _ = ∑ x ∈ NF, tri (r x - 1) + ∑ x ∈ (ends k U).filter (fun x => ¬ x.val + k < n), tri (r x - 1) :=
                hsplitE.symm
            _ ≤ NF.card * tri (τ - 4) + tri (d - 1) := Nat.add_le_add hNFle hFinle
            _ ≤ n / k * tri (τ - 4) + tri (d - 1) := by gcongr
        calc S1 + ∑ s ∈ Fu, F s = S1 + ∑ s ∈ Fr, tri (d - t s) := by rw [hFsum]
          _ ≤ (n / k * tri (τ - 4) + tri (d - 1)) + n / (k + 1) * tri (d - τ) := by
              gcongr
              exact hFrle.trans (Nat.mul_le_mul_right _ hFu)
          _ ≤ X * n := hB τ hτ3 hτd
      · -- past the horizon the free starts collect nothing
        have h0 : ∑ s ∈ Fr, tri (d - t s) = 0 := by
          refine Finset.sum_eq_zero fun s hs => ?_
          have h1 : d - t s = 0 := by have := hτle s hs; omega
          rw [h1]; rfl
        rw [hFsum, h0, add_zero]; exact hS1X
  -- assemble
  have hsplitS : ∑ s ∈ St, Ψ (t s) = ∑ s ∈ Fu, Ψ (t s) + ∑ s ∈ Sm, Ψ (t s) :=
    (Finset.sum_filter_add_sum_filter_not St _ _).symm
  have hFuΨ : ∑ s ∈ Fu, Ψ (t s) ≤ ∑ s ∈ Fu, ∑ a ∈ window k s, c a + ∑ s ∈ Fu, F s := by
    rw [← Finset.sum_add_distrib]; exact Finset.sum_le_sum hper
  calc ∑ w ∈ R, (U.filter fun u => t w < r u).card = ∑ w ∈ R, φ (t w) := rfl
    _ ≤ ∑ s ∈ St, Ψ (t s) := hWS
    _ = ∑ s ∈ Fu, Ψ (t s) + ∑ s ∈ Sm, Ψ (t s) := hsplitS
    _ ≤ (∑ s ∈ Fu, ∑ a ∈ window k s, c a + ∑ s ∈ Fu, F s) + S1 := Nat.add_le_add hFuΨ hSmle
    _ ≤ (L0.card * (n - U.card) + ∑ p ∈ P, (ndHeld (bandGraph n k) U r (t p)).card + ∑ s ∈ Fu, F s) + S1 := by
        gcongr
    _ ≤ X * n + L0.card * (n - U.card) + ∑ p ∈ P, (ndHeld (bandGraph n k) U r (t p)).card := by omega

/-- **The band certificate, two-directional game.** -/
theorem band_columnHard_twoWay (n k d X p q : ℕ) (hk : 1 ≤ k) (hq : 1 ≤ q)
    (hA : ∑ i ∈ Finset.range d, (q * i - p) ≤ p * k) (hX : p * (n + k) ≤ q * X * n)
    (hT : 2 * tri (d - 3) ≤ k * k)
    (hK : p * (n + k) ≤ q * (k * (n - d * (n + k) / (k + d))))
    (hB : ∀ τ, 3 ≤ τ → τ ≤ d → n / k * tri (τ - 4) + tri (d - 1) + n / (k + 1) * tri (d - τ) ≤ X * n) :
    ColumnHard (bandGraph n k) d n X :=
  columnHard_of_schedule _ d X (band_scheduleBound n k d X p q hk hq hA hX hT hK hB)

/-! ### The operating points -/

/-- **`B(220, 12)` at `d = 10`** (`D = 20` oracle rounds): saving at most 3 labels (`p/q = 5/2`). -/
theorem band_twoWay_220_12 : ColumnHard (bandGraph 220 12) 10 220 3 :=
  band_columnHard_twoWay 220 12 10 3 5 2 (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)
    (by norm_num) (by intro τ h3 hd; interval_cases τ <;> decide)

/-- **`B(220, 12)` at `d = 11`** (`D ≤ 22`): saving at most 3 labels (`p/q = 13/5`). -/
theorem band_twoWay_220_12_d11 : ColumnHard (bandGraph 220 12) 11 220 3 :=
  band_columnHard_twoWay 220 12 11 3 13 5 (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)
    (by norm_num) (by intro τ h3 hd; interval_cases τ <;> decide)

/-- **`B(220, 16)` at `d = 12`** (`D ≤ 24`): saving at most 3 labels (`p/q = 13/5`). -/
theorem band_twoWay_220_16 : ColumnHard (bandGraph 220 16) 12 220 3 :=
  band_columnHard_twoWay 220 16 12 3 13 5 (by norm_num) (by norm_num) (by decide) (by norm_num) (by decide)
    (by norm_num) (by intro τ h3 hd; interval_cases τ <;> decide)

end PousP3Column
