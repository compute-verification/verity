import Definitions.Nci.Guarantees.Capability.Pointwise
import Proofs.Nci.Capability.Dependency
import Proofs.Core.Game.Prob
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-!
# The pointwise slowdown: proofs

* `Decomposition.represented_eq`: a region represents what it computes whole and what is split there;
* `Decomposition.Ω_eq`: every occurrence is represented in exactly one region or in the remainder, so `Ω` is the sum of
  the regions' represented counts and the remainder's;
* `DecompositionCount`: the regions are disjoint, of `c` gates each, so `N = c·B + U`;
* `PointwiseCount`: the regions carry at most `q` each, `q/c·(N − U)` together, and the remainder `ν_out·U`, so
  `Ω ≤ q/c·N + (ν_out − q/c)·U ≤ q/c·N + max(ν_out − q/c, 0)·f·N = Γ·N`;
* `PointwiseThreshold`: `Ω_* ≤ Ω ≤ Γ·N`;
* `PointwiseEvent`: on an accepted, successful execution below `min(L, N_max)`, either the decomposition hypotheses
  fail, or they hold and Theorem B leaves it below `Ω_*`; the union bound adds the two;
* `WholeCapLocal`: a region's whole occurrences are at most `r` per product it completes, so at most `r·h` times its
  served uses' weight (`WeightedProductCap`); those it serves are imported, at most its `k + k_F` unregistered input bits
  (`ImportOrRecompute`), or recomputed, at most `c/g` (`RecomputeByGates`).
-/

namespace Nci.Capability.Counted.Decomposition

open Verity Finset

variable {X : Counted} {c : ℕ} (D : X.Decomposition c)

theorem represented_eq (S : Finset (Fin X.C.N)) : D.represented S = X.whole S + D.split S := by
  have h := sum_filter_add_sum_filter_not (univ.filter fun o => D.rep o ∈ S) (fun o => X.G o ⊆ S) X.ω
  rw [filter_filter, filter_filter] at h
  rw [represented, ← h, Counted.whole, split]
  congr 1
  refine sum_congr (ext fun o => ?_) fun _ _ => rfl
  simp only [mem_filter, mem_univ, true_and]
  exact ⟨fun h => h.2, fun h => ⟨h (D.rep_mem o), h⟩⟩

theorem Ω_eq : X.Ω = ∑ b, D.represented (D.region b) + D.represented D.remainder := by
  classical
  have hdisj : ((univ : Finset (Fin D.B)) : Set (Fin D.B)).PairwiseDisjoint
      fun b => univ.filter fun o => D.rep o ∈ D.region b := fun b _ b' _ hbb' =>
    disjoint_filter.2 fun _ _ h1 h2 => disjoint_left.1 (D.region_disjoint b b' hbb') h1 h2
  have hbu : univ.biUnion (fun b => univ.filter fun o => D.rep o ∈ D.region b) =
      univ.filter fun o => D.rep o ∈ univ.biUnion D.region := by
    ext o
    simp
  have hrem : (univ.filter fun o => D.rep o ∈ D.remainder) =
      univ.filter fun o => ¬ D.rep o ∈ univ.biUnion D.region := by
    ext o
    simp only [remainder, mem_filter, mem_univ, true_and, mem_sdiff]
    exact ⟨fun h => h.2, fun h => ⟨D.rep_computed o, h⟩⟩
  rw [Counted.Ω, represented, hrem]
  simp only [represented]
  rw [← sum_biUnion hdisj, hbu]
  exact (sum_filter_add_sum_filter_not univ _ X.ω).symm

theorem card_eq : X.N = c * D.B + D.U := by
  classical
  have hdisj : ((univ : Finset (Fin D.B)) : Set (Fin D.B)).PairwiseDisjoint D.region := fun b _ b' _ h =>
    D.region_disjoint b b' h
  have hsub : univ.biUnion D.region ⊆ X.C.computed := biUnion_subset.2 fun b _ => D.region_computed b
  have hcard : (univ.biUnion D.region).card = c * D.B := by
    rw [card_biUnion hdisj]
    simp [D.region_card, mul_comm]
  have h := card_sdiff_add_card_eq_card hsub
  rw [hcard] at h
  rw [Counted.N, U, remainder, ← h, add_comm]

end Nci.Capability.Counted.Decomposition

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability
open scoped ENNReal

theorem DecompositionCount : Guarantees.DecompositionCount := fun _ _ D => D.card_eq

theorem PointwiseCount : Guarantees.PointwiseCount := by
  intro X c D qWhole qSplit ν f hW hS hR hU
  set q := qWhole + qSplit
  have hb : ∀ b, D.represented (D.region b) ≤ q := fun b => by
    rw [D.represented_eq]
    exact add_le_add (hW b) (hS b)
  have hreg : ∑ b, D.represented (D.region b) ≤ q / c * ((c * D.B : ℕ) : ℝ) := by
    rcases Nat.eq_zero_or_pos c with rfl | hc
    · have h0 : ∀ b, D.represented (D.region b) = 0 := fun b => by
        rw [card_eq_zero.1 (D.region_card b)]
        simp [Counted.Decomposition.represented]
      simp [h0]
    · have hc' : (c : ℝ) ≠ 0 := by positivity
      calc ∑ b, D.represented (D.region b) ≤ ∑ _b : Fin D.B, q := sum_le_sum fun b _ => hb b
        _ = q / c * ((c * D.B : ℕ) : ℝ) := by
          rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_mul]
          field_simp
  have hN : (X.N : ℝ) = ((c * D.B : ℕ) : ℝ) + D.U := by rw [D.card_eq, Nat.cast_add]
  have hU0 : (0 : ℝ) ≤ D.U := Nat.cast_nonneg _
  have h1 : (ν - q / c) * D.U ≤ max (ν - q / c) 0 * D.U :=
    mul_le_mul_of_nonneg_right (le_max_left _ _) hU0
  have h2 : max (ν - q / c) 0 * D.U ≤ max (ν - q / c) 0 * (f * X.N) :=
    mul_le_mul_of_nonneg_left hU (le_max_right _ _)
  have h3 : q / c * X.N = q / c * ((c * D.B : ℕ) : ℝ) + q / c * D.U := by rw [hN, mul_add]
  have hR' : D.represented D.remainder ≤ ν * D.U := hR
  rw [D.Ω_eq]
  unfold Gamma
  nlinarith

theorem GammaBelow : Guarantees.GammaBelow := by
  intro q c ν f h
  unfold Gamma
  rw [max_eq_left (sub_nonneg.2 h)]
  ring

theorem PointwiseThreshold : Guarantees.PointwiseThreshold := by
  intro X c D qWhole qSplit ν f Ωstar hW hS hR hU hΩ hΓ
  rw [div_le_iff₀ hΓ]
  have := PointwiseCount X c D qWhole qSplit ν f hW hS hR hU
  linarith [mul_comm (Gamma (qWhole + qSplit) c ν f) X.N]

theorem PointwiseEvent : Guarantees.PointwiseEvent := by
  intro α g s X accept success c qWhole qSplit ν f Ωstar Nmax δ η hΓ hδ hη
  refine le_trans (Game.prob_mono (fun a ha => ?_) g s)
    (le_trans (Game.prob_or_le (fun a => accept a ∧ ¬ (X a).Decomposes c qWhole qSplit ν f)
      (fun a => success a ∧ ((X a).N : ℝ) < Nmax ∧ ¬ Ωstar ≤ (X a).Ω) g s) (add_le_add hδ hη))
  obtain ⟨hacc, hsucc, hlt⟩ := ha
  by_cases hD : (X a).Decomposes c qWhole qSplit ν f
  · refine .inr ⟨hsucc, lt_of_lt_of_le hlt (min_le_right _ _), fun hΩ => ?_⟩
    obtain ⟨D, hW, hS, hR, hU⟩ := hD
    exact absurd (lt_of_lt_of_le hlt (min_le_left _ _))
      (not_lt.2 (PointwiseThreshold (X a) c D qWhole qSplit ν f Ωstar hW hS hR hU hΩ hΓ))
  · exact .inl ⟨hacc, hD⟩

/-- `WholeCapLocal` from the three bounds it composes. -/
theorem wholeCapLocal_of (hWP : Guarantees.WeightedProductCap) (hIR : Guarantees.ImportOrRecompute)
    (hRG : Guarantees.RecomputeByGates) : Guarantees.WholeCapLocal := by
  intro R c k kF h hP B W hSl hheld A G hG hA ω g hup hω hg hgG κ _ prod Go r hprod hm hr D hD b
  classical
  obtain ⟨u, hu⟩ := hD b
  set K := Consumers.ofUses B A G hG hA with hK
  set S := R.P.gates u with hSdef
  have hwt : ∀ q : Fin B.nw × Fin B.nt, 0 ≤ W.price q.1 * ω q.2 := fun q =>
    mul_nonneg (W.price_nonneg _) (hω.1 _)
  have hh : 0 ≤ h := by
    have := hheld u ∅ (fun _ => ∅) (fun i hi => absurd hi (notMem_empty i)) fun i hi => absurd hi (notMem_empty i)
    simpa using this
  have hocc : ∑ o ∈ univ.filter (fun o => Go o ⊆ S), W.price (prod o).1 * ω (prod o).2 ≤
      r * ∑ q ∈ completedIn B S, W.price q.1 * ω q.2 := by
    rw [← sum_fiberwise_of_maps_to (s := univ.filter fun o => Go o ⊆ S) (t := completedIn B S) (g := prod)
      (fun o ho => mem_filter.2 ⟨hprod o, (hm o).trans (mem_filter.1 ho).2⟩), mul_sum]
    refine sum_le_sum fun q _ => ?_
    calc ∑ o ∈ (univ.filter fun o => Go o ⊆ S).filter (fun o => prod o = q), W.price (prod o).1 * ω (prod o).2
        = ∑ _o ∈ (univ.filter fun o => Go o ⊆ S).filter (fun o => prod o = q), W.price q.1 * ω q.2 :=
          sum_congr rfl fun o ho => by rw [(mem_filter.1 ho).2]
      _ = ((univ.filter fun o => Go o ⊆ S).filter (fun o => prod o = q)).card * (W.price q.1 * ω q.2) := by
          rw [sum_const, nsmul_eq_mul]
      _ ≤ r * (W.price q.1 * ω q.2) := by
          refine mul_le_mul_of_nonneg_right ?_ (hwt q)
          exact_mod_cast (card_le_card (filter_subset_filter _ (subset_univ _))).trans (hr q)
  have hcap := hWP R B W ω h hSl hω.1 hheld u
  have hserved : served B S ⊆ K.servedIn S := by
    intro j hj
    obtain ⟨q, hq, rfl⟩ := mem_image.1 hj
    obtain ⟨hqP, hqS⟩ := mem_filter.1 hq
    exact mem_filter.2 ⟨mem_univ _, B.m q, mem_image.2 ⟨q, mem_filter.2 ⟨hqP, rfl⟩, rfl⟩, hqS⟩
  have hsrv : ∑ j ∈ served B S, ω j ≤ ∑ j ∈ K.servedIn S, ω j :=
    sum_le_sum_of_subset_of_nonneg hserved fun j _ _ => hω.1 j
  have himp := hIR R K ω hup hω u
  have hrec := hRG R.d K ω g hω hgG (K.servedIn S) S
  have hcard : (S.card : ℝ) = c := by exact_mod_cast hP.card u
  have hunreg : ((R.d.unregistered S).card : ℝ) ≤ (k + kF : ℕ) := by exact_mod_cast hP.reads u
  have hrec' : ∑ j ∈ (K.servedIn S).filter (fun j => K.G j ⊆ S), ω j ≤ c / g := by
    rw [le_div_iff₀ hg, ← hcard]
    linarith
  have hV : ∑ j ∈ served B S, ω j ≤ (k + kF : ℕ) + c / g := by linarith
  show ∑ o ∈ univ.filter (fun o => Go o ⊆ D.region b), W.price (prod o).1 * ω (prod o).2 ≤ _
  rw [hu]
  calc _ ≤ r * ∑ q ∈ completedIn B S, W.price q.1 * ω q.2 := hocc
    _ ≤ r * (h * ∑ j ∈ served B S, ω j) := mul_le_mul_of_nonneg_left hcap (Nat.cast_nonneg r)
    _ = r * h * ∑ j ∈ served B S, ω j := by ring
    _ ≤ r * h * ((k + kF : ℕ) + c / g) := mul_le_mul_of_nonneg_left hV (mul_nonneg (Nat.cast_nonneg r) hh)

theorem WholeCapLocal : Guarantees.WholeCapLocal :=
  wholeCapLocal_of WeightedProductCap ImportOrRecompute RecomputeByGates

end Nci.SecurityProofs
