import Models.Nci.Guarantees.Covering
import Mathlib.Combinatorics.Enumerative.DoubleCounting

/-!
# Proof of the covering bound

Charge each chained multiplication block `m` to a block `I` of the covering that computes its output gate `o`
(`Covers`). Tracing back from a gate `S` holds along a path that meets no input of `S` stays among the gates `S`
computes (`computes_of_reachesIn`), so:

* if `I` doesn't complete `m`, a gate of `m` is an input of `I` (`completes_or_mem_ins`), outside `θ` since `m` is
  computed; the `m`'s are disjoint, so `I` computes the output gates of at most `|in(I) \ θ| ≤ k` such blocks
  (`card_uncompleted_le`);
* if `I` completes a chained `m_{i,j}`, then `in(I) ∩ T_j` cuts `A_{i,j}` off from every `m_{i',j}` with `i' ∉ θ'_I`
  (`cuts_ins_inter`): a path inside `T_j` from `o_{i',j}` to `A_{i,j}` that misses it puts `o_{i',j}`, and with it all of
  `m_{i',j}`, among the gates `I` computes. Parameter-hardness gives `|θ'_I| ≤ k/c < D`, so it has at least `c` gates;
  the `T_j` are disjoint and miss `θ`, so `I` completes chained blocks of at most `k/c` tokens, and of at most
  `(k/c)²` pairs (`card_completed_chained_le`).
-/

namespace Nci.SecurityProofs.Covering

open Verity Finset Nci.Cover

variable {d : Declaration}

/-- Tracing back from a gate `t` of `S` along a path inside `U`, which holds no input of `S`, stays among the gates `S`
computes. -/
theorem computes_of_reachesIn {S : Subcircuit d.D} {U : Finset (Fin d.D.N)} (hU : ∀ x ∈ U, x ∉ S.ins)
    {g t : Fin d.D.N} (h : ReachesIn d.D U g t) (ht : t ∈ S.gates) : g ∈ S.computes := by
  induction h with
  | refl hu => exact mem_sdiff.2 ⟨ht, hU _ hu⟩
  | step hg hgh _ ih =>
    have hh := mem_sdiff.1 (ih ht)
    exact mem_sdiff.2 ⟨S.reads_subset _ hh.1 hh.2 hgh, hU _ hg⟩

variable {ι τ : Type} (M : Mults d ι τ)

/-- A subcircuit that computes the output gate of a multiplication block completes it, or has a gate of it among its
inputs. -/
theorem completes_or_mem_ins {p : ι × τ} (hp : p ∈ M.P) (S : Subcircuit d.D) (ho : M.o p.1 p.2 ∈ S.computes) :
    M.Completes S p.1 p.2 ∨ ∃ x ∈ M.m p.1 p.2, x ∈ S.ins := by
  by_cases h : ∃ x ∈ M.m p.1 p.2, x ∈ S.ins
  · exact Or.inr h
  · refine Or.inl fun g hg => computes_of_reachesIn (fun x hx hxS => h ⟨x, hx, hxS⟩)
      (M.reachesIn_o p hp g hg) (mem_sdiff.1 ho).1

open Classical in
/-- A subcircuit computes the output gates of at most `|in(S) \ θ|` multiplication blocks it doesn't complete. -/
theorem card_uncompleted_le (S : Subcircuit d.D) :
    (M.P.filter fun p => M.o p.1 p.2 ∈ S.computes ∧ ¬ M.Completes S p.1 p.2).card ≤ (S.ins \ d.θ).card := by
  refine card_le_card_of_forall_subsingleton (fun p x => x ∈ M.m p.1 p.2) ?_ ?_
  · intro p hp
    obtain ⟨hpP, ho, hnc⟩ := mem_filter.1 hp
    obtain ⟨x, hxm, hxS⟩ := (completes_or_mem_ins M hpP S ho).resolve_left hnc
    refine ⟨x, mem_sdiff.2 ⟨hxS, fun hθ => ?_⟩, hxm⟩
    have hc := (mem_filter.1 (M.m_computed p hpP hxm)).2
    rw [d.θ_input x hθ] at hc
    exact Bool.noConfusion hc
  · intro x _ p hp q hq
    by_contra hpq
    exact disjoint_left.1 (M.m_disjoint p (mem_filter.1 hp.1).1 q (mem_filter.1 hq.1).1 hpq) hp.2 hq.2

variable [DecidableEq ι] {D r b : ℝ}

/-- If `S` completes a chained `m_{i,j}`, then `in(S) ∩ T_j` cuts `A_{i,j}` off from every `m_{i',j}` with `i' ∉ θ'_S`. -/
theorem cuts_ins_inter {S : Subcircuit d.D} {A : Finset (Fin d.D.N)} {i : ι} {j : τ}
    (hA : ∀ a ∈ A, ∃ g ∈ M.m i j, a ∈ d.D.reads g) (hS : M.Completes S i j) {i' : ι} (hi'P : (i', j) ∈ M.P)
    (hi'T : M.m i' j ⊆ M.T j) (hi' : i' ∉ M.completed S) : M.Cuts (S.ins ∩ M.T j) A i' j := by
  classical
  by_contra hcut
  simp only [Mults.Cuts, not_or, not_not, not_nonempty_iff_eq_empty] at hcut
  obtain ⟨hmeet, a, haA, hpath⟩ := hcut
  obtain ⟨g, hg, hag⟩ := hA a haA
  have hgS := mem_sdiff.1 (hS hg)
  have haS : a ∈ S.gates := S.reads_subset g hgS.1 hgS.2 hag
  have hU : ∀ x ∈ M.T j \ (S.ins ∩ M.T j), x ∉ S.ins := fun x hx hxS =>
    (mem_sdiff.1 hx).2 (mem_inter.2 ⟨hxS, (mem_sdiff.1 hx).1⟩)
  have ho := computes_of_reachesIn hU hpath haS
  have hm : M.Completes S i' j := fun g' hg' => by
    refine computes_of_reachesIn (fun x hx hxS => ?_) (M.reachesIn_o (i', j) hi'P g' hg') (mem_sdiff.1 ho).1
    have : x ∈ (S.ins ∩ M.T j) ∩ M.m i' j := mem_inter.2 ⟨mem_inter.2 ⟨hxS, hi'T hx⟩, hx⟩
    rw [hmeet] at this
    exact absurd this (notMem_empty x)
  exact hi' (mem_image.2 ⟨(i', j), mem_filter.2 ⟨hi'P, hm⟩, rfl⟩)

open Classical in
/-- A block reading at most `k` gates outside `θ`, of at most `s` gates, completes chained multiplication blocks of at
most `(k/c)²` pairs, under parameter-hardness with `0 < c` and `k < cD`. -/
theorem card_completed_chained_le {s k : ℝ} (hc : 0 < r * b) (hkD : k < r * b * D)
    (hard : Assumptions.ParameterHard d M s r b) (S : Subcircuit d.D) (hS : Block d s k S) :
    (((M.P.filter fun p => Assumptions.ActivationChained d M D r b p.1 p.2).filter
        fun p => M.Completes S p.1 p.2).card : ℝ) ≤ (k / (r * b)) ^ 2 := by
  set c := r * b with hcdef
  set Mc := (M.P.filter fun p => Assumptions.ActivationChained d M D r b p.1 p.2).filter
    fun p => M.Completes S p.1 p.2
  set Θ := M.completed S
  set Js := Mc.image Prod.snd
  have hΘ : c * Θ.card ≤ k := (hard S hS.1).trans hS.2
  have hΘD : (Θ.card : ℝ) < D := by
    by_contra h
    exact absurd (hΘ.trans_lt hkD) (not_lt.2 (mul_le_mul_of_nonneg_left (not_lt.1 h) hc.le))
  have hΘsub : Θ ⊆ M.θ' := fun i hi => by
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
    exact (mem_product.1 (M.P_subset (mem_filter.1 hp).1)).1
  -- each token of a completed chained block contributes `c` gates of `in(S) ∩ T_j`
  have hJ : ∀ j ∈ Js, c ≤ (S.ins ∩ M.T j).card := by
    intro j hj
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hj
    obtain ⟨hp', hcomp⟩ := mem_filter.1 hp
    obtain ⟨_, hch⟩ := mem_filter.1 hp'
    obtain ⟨A, _, hA, W, _, hW, hX⟩ := hch
    refine hX Θ hΘsub hΘD _ inter_subset_right fun i' hi' => ?_
    obtain ⟨hi'W, hi'Θ⟩ := mem_sdiff.1 hi'
    obtain ⟨hi'P, hi'T, _⟩ := hW i' hi'W
    exact cuts_ins_inter M hA hcomp hi'P hi'T hi'Θ
  have hJsub : Js ⊆ M.J := fun j hj => by
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hj
    exact (mem_product.1 (M.P_subset (mem_filter.1 (mem_filter.1 hp).1).1)).2
  have hsum : ∑ j ∈ Js, ((S.ins ∩ M.T j).card : ℝ) ≤ (S.ins \ d.θ).card := by
    have hdisj : (Js : Set τ).PairwiseDisjoint fun j => S.ins ∩ M.T j := fun j hj j' hj' hjj' =>
      (M.T_disjoint j (hJsub hj) j' (hJsub hj') hjj').mono inter_subset_right inter_subset_right
    rw [← Nat.cast_sum, ← card_biUnion hdisj]
    refine Nat.cast_le.2 (card_le_card fun x hx => ?_)
    obtain ⟨j, hj, hx⟩ := mem_biUnion.1 hx
    exact mem_sdiff.2 ⟨(mem_inter.1 hx).1, disjoint_left.1 (M.T_disjoint_θ j (hJsub hj)) (mem_inter.1 hx).2⟩
  have hJs : c * Js.card ≤ k := by
    calc c * Js.card = ∑ _j ∈ Js, c := by rw [sum_const, nsmul_eq_mul, mul_comm]
      _ ≤ ∑ j ∈ Js, ((S.ins ∩ M.T j).card : ℝ) := sum_le_sum hJ
      _ ≤ (S.ins \ d.θ).card := hsum
      _ ≤ k := hS.2
  have hsub : Mc ⊆ Θ ×ˢ Js := fun p hp => by
    refine mem_product.2 ⟨mem_image.2 ⟨p, mem_filter.2 ⟨(mem_filter.1 (mem_filter.1 hp).1).1, ?_⟩, rfl⟩,
      mem_image_of_mem _ hp⟩
    exact (mem_filter.1 hp).2
  have hΘk : (Θ.card : ℝ) ≤ k / c := (le_div_iff₀ hc).2 (by linarith)
  have hJk : (Js.card : ℝ) ≤ k / c := (le_div_iff₀ hc).2 (by linarith)
  calc (Mc.card : ℝ) ≤ ((Θ ×ˢ Js).card : ℝ) := Nat.cast_le.2 (card_le_card hsub)
    _ = Θ.card * Js.card := by rw [card_product, Nat.cast_mul]
    _ ≤ (k / c) * (k / c) := mul_le_mul hΘk hJk (Nat.cast_nonneg _) ((Nat.cast_nonneg _).trans hΘk)
    _ = (k / c) ^ 2 := (sq _).symm

open Classical in
/-- The covering bound multiplied out: `|𝓜*| ≤ |𝓘| · ((k/c)² + k)`. -/
theorem card_chained_le {s k : ℝ} (hc : 0 < r * b) (hkD : k < r * b * D)
    (hard : Assumptions.ParameterHard d M s r b) {𝓘 : Finset (Subcircuit d.D)} (h𝓘 : Covers d s k 𝓘) :
    ((M.P.filter fun p => Assumptions.ActivationChained d M D r b p.1 p.2).card : ℝ) ≤
      𝓘.card * ((k / (r * b)) ^ 2 + k) := by
  set Ms := M.P.filter fun p => Assumptions.ActivationChained d M D r b p.1 p.2
  have hcover : Ms ⊆ 𝓘.biUnion fun I => Ms.filter fun p => M.o p.1 p.2 ∈ I.computes := fun p hp => by
    have hpP := (mem_filter.1 hp).1
    obtain ⟨I, hI, hoI⟩ := h𝓘.2 _ (M.m_computed p hpP (M.o_mem p hpP))
    exact mem_biUnion.2 ⟨I, hI, mem_filter.2 ⟨hp, hoI⟩⟩
  have hblock : ∀ I ∈ 𝓘, ((Ms.filter fun p => M.o p.1 p.2 ∈ I.computes).card : ℝ) ≤ (k / (r * b)) ^ 2 + k := by
    intro I hI
    have hsplit : (Ms.filter fun p => M.o p.1 p.2 ∈ I.computes) ⊆
        (Ms.filter fun p => M.Completes I p.1 p.2) ∪
          M.P.filter fun p => M.o p.1 p.2 ∈ I.computes ∧ ¬ M.Completes I p.1 p.2 := fun p hp => by
      obtain ⟨hpMs, ho⟩ := mem_filter.1 hp
      by_cases hcomp : M.Completes I p.1 p.2
      · exact mem_union_left _ (mem_filter.2 ⟨hpMs, hcomp⟩)
      · exact mem_union_right _ (mem_filter.2 ⟨(mem_filter.1 hpMs).1, ho, hcomp⟩)
    calc ((Ms.filter fun p => M.o p.1 p.2 ∈ I.computes).card : ℝ)
        ≤ (Ms.filter fun p => M.Completes I p.1 p.2).card +
            (M.P.filter fun p => M.o p.1 p.2 ∈ I.computes ∧ ¬ M.Completes I p.1 p.2).card := by
          rw [← Nat.cast_add]
          exact Nat.cast_le.2 ((card_le_card hsplit).trans (card_union_le _ _))
      _ ≤ (k / (r * b)) ^ 2 + k := add_le_add (card_completed_chained_le M hc hkD hard I (h𝓘.1 I hI))
          (le_trans (Nat.cast_le.2 (card_uncompleted_le M I)) (h𝓘.1 I hI).2)
  calc (Ms.card : ℝ) ≤ ∑ I ∈ 𝓘, ((Ms.filter fun p => M.o p.1 p.2 ∈ I.computes).card : ℝ) := by
        rw [← Nat.cast_sum]
        exact Nat.cast_le.2 ((card_le_card hcover).trans card_biUnion_le)
    _ ≤ ∑ _I ∈ 𝓘, ((k / (r * b)) ^ 2 + k) := sum_le_sum hblock
    _ = 𝓘.card * ((k / (r * b)) ^ 2 + k) := by rw [sum_const, nsmul_eq_mul]

end Nci.SecurityProofs.Covering

namespace Nci.SecurityProofs

open Verity Finset Nci.Cover Covering

theorem CoveringBound : Guarantees.CoveringBound := by
  intro ι τ _ d M b r s k D hb hr _ hkD hard 𝓘 h𝓘
  classical
  have hc : 0 < r * b := mul_pos hr hb
  have hcard := card_chained_le M hc hkD hard h𝓘
  refine ⟨hcard, ?_⟩
  set Ms := M.P.filter fun p => Assumptions.ActivationChained d M D r b p.1 p.2
  set den := (k / (r * b)) ^ 2 + k
  unfold Cover.slowdown
  by_cases hne : 𝓘.Nonempty
  · obtain ⟨I, hI⟩ := hne
    have hs : 0 ≤ s := (Nat.cast_nonneg _).trans (h𝓘.1 I hI).1
    have hk : 0 ≤ k := (Nat.cast_nonneg _).trans (h𝓘.1 I hI).2
    have hden : 0 ≤ den := add_nonneg (sq_nonneg _) hk
    have hN : (0 : ℝ) ≤ d.D.N := Nat.cast_nonneg _
    rcases hden.eq_or_lt with h0 | hpos
    · rw [← h0, div_zero]
      exact div_nonneg (mul_nonneg hs (Nat.cast_nonneg _)) hN
    · have hl : (Ms.card : ℝ) / d.D.N * s / den = Ms.card / den * (s / d.D.N) := by ring
      have hr' : s * 𝓘.card / d.D.N = 𝓘.card * (s / d.D.N) := by ring
      rw [hl, hr']
      exact mul_le_mul_of_nonneg_right ((div_le_iff₀ hpos).2 hcard) (div_nonneg hs hN)
  · rw [not_nonempty_iff_eq_empty] at hne
    have hMs : Ms = ∅ := eq_empty_of_forall_notMem fun p hp => by
      have hpP := (mem_filter.1 hp).1
      obtain ⟨I, hI, _⟩ := h𝓘.2 _ (M.m_computed p hpP (M.o_mem p hpP))
      rw [hne] at hI
      exact notMem_empty I hI
    simp [hMs, hne]

end Nci.SecurityProofs
