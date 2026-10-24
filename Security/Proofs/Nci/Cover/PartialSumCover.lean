import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Proofs.Nci.Cover.PartialSumCube
import Proofs.Nci.Cover.PartialSumRate

/-!
# The partial-sum bounds: the cover

`PartialSumCoverBound` assigns each counted product to a block that finishes it (`PartialSumsRead`'s `φ`). A product
assigned to `I` that `I` doesn't compute whole has a gate among `I`'s inputs inside its accumulation
(`card_le_inputs_accGates`), and those gates are not in `θ` (`inputs_accGates_card_le`): at most `k` of them per block.
The products `I` computes whole obey `block_counts`, and `rate_block` turns them into `I`'s share of the two read
budgets (`block_rate`).

The crossings' share is paid by the accumulations' reads (`crossings_sum_le`): an entry crossing at `I` has a product
`I` doesn't compute whole (`ActivationReads`), assigned to another block, so the entry's products span two blocks, or to
`I`, where it is one of `I`'s not-whole products. Either way each entry's crossings are at most `2 (m − 1) + n`, with
`m` its blocks and `n` its not-whole products, which sum to at most `3 Σ #(inputs ∩ accGates)` (`PartialSumsRead`).
`ps_cover_sum` adds the blocks up.
-/

namespace Nci.Cover.SecurityProofs

section

variable {C : Circuit} {J L Tok Val Mat Row Col : Type} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V w : ℕ}
  {K : Keyed M Tok Val d aOut c V w}

/-- The counted products `I` finishes but doesn't compute whole are at most `I`'s inputs among the accumulations. -/
theorem mem_finished_of [DecidableEq J] {I : Finset (Fin C.N)} {p : J × L} (hp : p ∈ M.P)
    (ho : M.o p ∈ C.computed I) : p ∈ M.finished I :=
  Finset.mem_filter.2 ⟨hp, ho⟩

theorem card_le_inputs_accGates [DecidableEq J] [DecidableEq L] [DecidableEq Tok] [DecidableEq Mat] [DecidableEq Row]
    (Q : Cube K Mat Row Col) {I : Finset (Fin C.N)} {B : Finset (J × L)} (hB : B ⊆ M.finished I \ M.whole I) :
    B.card ≤ (C.inputs I ∩ Q.accGates).card := by
  have hP : ∀ p ∈ B, p ∈ M.P := fun p hp => (Finset.mem_filter.1 (Finset.mem_sdiff.1 (hB hp)).1).1
  have hPall : ∀ p ∈ B, p ∈ M.Pall := fun p hp => M.P_subset (hP p hp)
  have hT : ∀ p ∈ B, (M.m p ∩ C.inputs I).Nonempty := by
    intro p hp
    obtain ⟨hpf, hpw⟩ := Finset.mem_sdiff.1 (hB hp)
    obtain ⟨hpP, hoI⟩ := Finset.mem_filter.1 hpf
    obtain ⟨x, hxm, hxc⟩ := Finset.not_subset.1 fun h => hpw (Finset.mem_filter.2 ⟨hpP, h⟩)
    obtain ⟨π, hπ, hπm⟩ := reachesIn_output C (M.m p) (M.o p) (M.m_outputs p (hPall p hp)) x hxm
    obtain ⟨g, hgπ, hgI⟩ := entry C I π x (M.o p) hπ (computed_subset C I hoI) hxc
    exact ⟨g, Finset.mem_inter.2 ⟨hπm g hgπ, hgI⟩⟩
  have hdisj : (B : Set (J × L)).PairwiseDisjoint fun p => M.m p ∩ C.inputs I :=
    (M.m_disjoint.subset fun p hp => hPall p hp).mono fun p => Finset.inter_subset_left
  refine (Finset.card_le_card_biUnion hdisj hT).trans (Finset.card_le_card ?_)
  intro x hx
  obtain ⟨p, hp, hx⟩ := Finset.mem_biUnion.1 hx
  obtain ⟨hxm, hxi⟩ := Finset.mem_inter.1 hx
  exact Finset.mem_inter.2
    ⟨hxi, Finset.mem_biUnion.2 ⟨Q.entry p, Finset.mem_image_of_mem _ (hP p hp), Q.acc_prod p (hP p hp) hxm⟩⟩

/-- The accumulations' gates hold no registered parameter. -/
theorem inputs_accGates_card_le [DecidableEq Tok] [DecidableEq Mat] [DecidableEq Row] (Q : Cube K Mat Row Col)
    (I : Finset (Fin C.N)) : (C.inputs I ∩ Q.accGates).card ≤ (C.inputs I \ C.θ).card := by
  refine Finset.card_le_card fun g hg => ?_
  obtain ⟨hgi, hga⟩ := Finset.mem_inter.1 hg
  refine Finset.mem_sdiff.2 ⟨hgi, fun hgθ => ?_⟩
  obtain ⟨E, hE, hgE⟩ := Finset.mem_biUnion.1 hga
  obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hE
  obtain ⟨y, hy⟩ := Q.acc_computed p hp g hgE
  rw [C.θ_input g hgθ] at hy
  exact Finset.notMem_empty y hy

/-- **The crossings are paid by the accumulations' reads**: for an assignment `φ` as in `PartialSumsRead` and sets
`Lo I` of products assigned to `I` at tokens low for `I`, the entries of the `Lo I` number at most three times the
accumulation gates among the blocks' inputs. -/
theorem crossings_sum_le [DecidableEq J] [DecidableEq L] [DecidableEq Tok] [DecidableEq Mat] [DecidableEq Row]
    (Q : Cube K Mat Row Col) {s k f a : ℕ} (hX : Assumptions.ActivationReads Q s k f a)
    {𝓘 : Finset (Finset (Fin C.N))} (h𝓘 : ∀ I ∈ 𝓘, C.InferenceBlock s k I) {φ : J × L → Finset (Fin C.N)}
    (hφ : ∀ p ∈ M.P, φ p ∈ 𝓘 ∧ M.o p ∈ C.computed (φ p))
    (hsum : ∑ E ∈ M.P.image Q.entry, (((M.P.filter fun p => Q.entry p = E).image φ).card - 1) ≤
      ∑ I ∈ 𝓘, (C.inputs I ∩ Q.accGates).card)
    (Lo : Finset (Fin C.N) → Finset (J × L))
    (hLo : ∀ I ∈ 𝓘, ∀ p ∈ Lo I, p ∈ M.P ∧ φ p = I ∧ K.tokReads I (K.tok (K.act p)) < f * a) :
    ∑ I ∈ 𝓘, ((Lo I).image Q.entry).card ≤ 3 * ∑ I ∈ 𝓘, (C.inputs I ∩ Q.accGates).card := by
  set Z := ∑ I ∈ 𝓘, (C.inputs I ∩ Q.accGates).card with hZ
  set N := M.P.filter fun p => p ∉ M.whole (φ p) with hN
  have hNZ : N.card ≤ Z := by
    rw [Finset.card_eq_sum_card_fiberwise (f := φ) (t := 𝓘) fun p hp => (hφ p (Finset.mem_filter.1 hp).1).1]
    refine Finset.sum_le_sum fun I _ => card_le_inputs_accGates Q fun p hp => ?_
    obtain ⟨hpN, hpI⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpP, hpw⟩ := Finset.mem_filter.1 hpN
    have ho := (hφ p hpP).2
    rw [hpI] at ho hpw
    exact Finset.mem_sdiff.2 ⟨mem_finished_of hpP ho, hpw⟩
  have hdc : ∑ I ∈ 𝓘, ((Lo I).image Q.entry).card =
      ∑ E ∈ M.P.image Q.entry, (𝓘.filter fun I => E ∈ (Lo I).image Q.entry).card := by
    have h : ∀ I ∈ 𝓘, ((Lo I).image Q.entry).card =
        ((M.P.image Q.entry).filter fun E => E ∈ (Lo I).image Q.entry).card := by
      intro I hI
      rw [Finset.filter_mem_eq_inter, Finset.inter_eq_right.2]
      intro E hE
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hE
      exact Finset.mem_image_of_mem _ (hLo I hI p hp).1
    rw [Finset.sum_congr rfl h]
    exact Finset.sum_card_bipartiteAbove_eq_sum_card_bipartiteBelow fun I E => E ∈ (Lo I).image Q.entry
  have hE : ∀ E ∈ M.P.image Q.entry, (𝓘.filter fun I => E ∈ (Lo I).image Q.entry).card ≤
      2 * (((M.P.filter fun p => Q.entry p = E).image φ).card - 1) + (N.filter fun p => Q.entry p = E).card := by
    intro E _
    have hSm : (𝓘.filter fun I => E ∈ (Lo I).image Q.entry).card ≤
        ((M.P.filter fun p => Q.entry p = E).image φ).card := by
      refine Finset.card_le_card fun I hI => ?_
      obtain ⟨hI𝓘, hIE⟩ := Finset.mem_filter.1 hI
      obtain ⟨p, hp, hpE⟩ := Finset.mem_image.1 hIE
      obtain ⟨hpP, hpI, -⟩ := hLo I hI𝓘 p hp
      exact Finset.mem_image.2 ⟨p, Finset.mem_filter.2 ⟨hpP, hpE⟩, hpI⟩
    rcases (𝓘.filter fun I => E ∈ (Lo I).image Q.entry).eq_empty_or_nonempty with hS0 | ⟨I, hI⟩
    · rw [hS0, Finset.card_empty]
      exact Nat.zero_le _
    obtain ⟨hI𝓘, hIE⟩ := Finset.mem_filter.1 hI
    obtain ⟨p, hp, hpE⟩ := Finset.mem_image.1 hIE
    obtain ⟨hpP, hpI, hlow⟩ := hLo I hI𝓘 p hp
    obtain ⟨p', hp'P, hp'E, hp'w⟩ := (hX I (h𝓘 I hI𝓘) _ hlow).1 p hpP rfl
    by_cases hm1 : ((M.P.filter fun p => Q.entry p = E).image φ).card ≤ 1
    · have hφ' : φ p' = I := by
        rw [← hpI]
        exact Finset.card_le_one.1 hm1 _ (Finset.mem_image_of_mem _ (Finset.mem_filter.2 ⟨hp'P, hp'E.trans hpE⟩))
          _ (Finset.mem_image_of_mem _ (Finset.mem_filter.2 ⟨hpP, hpE⟩))
      have hn : 1 ≤ (N.filter fun p => Q.entry p = E).card := by
        refine Finset.card_pos.2 ⟨p', Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hp'P, ?_⟩, hp'E.trans hpE⟩⟩
        rw [hφ']
        exact hp'w
      omega
    · omega
  have hNsum : ∑ E ∈ M.P.image Q.entry, (N.filter fun p => Q.entry p = E).card = N.card :=
    (Finset.card_eq_sum_card_fiberwise fun p hp => Finset.mem_image_of_mem _ (Finset.mem_filter.1 hp).1).symm
  calc ∑ I ∈ 𝓘, ((Lo I).image Q.entry).card
      = ∑ E ∈ M.P.image Q.entry, (𝓘.filter fun I => E ∈ (Lo I).image Q.entry).card := hdc
    _ ≤ ∑ E ∈ M.P.image Q.entry, (2 * (((M.P.filter fun p => Q.entry p = E).image φ).card - 1) +
          (N.filter fun p => Q.entry p = E).card) := Finset.sum_le_sum hE
    _ = 2 * ∑ E ∈ M.P.image Q.entry, (((M.P.filter fun p => Q.entry p = E).image φ).card - 1) + N.card := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, hNsum]
    _ ≤ 2 * Z + Z := Nat.add_le_add (Nat.mul_le_mul_left 2 hsum) hNZ
    _ = 3 * Z := by ring

/-- The rate is nonnegative: its fan-out face is. -/
theorem psRate_nonneg {b f a aOut K μ : ℝ} (hf : 0 ≤ f) (haOut : 0 ≤ aOut) (hμ0 : 0 ≤ μ) :
    0 ≤ Guarantees.psRate b f a aOut K μ :=
  (div_nonneg (mul_nonneg hμ0 hf) (haOut.trans (le_max_left _ _))).trans (le_max_right _ _)

/-- **One block's rate**: the products a block computes whole, at high and low tokens, cost at most its share
`k + μ k + (1 − μ) Y / 3` of the read budgets, `Y` the entries of its low ones. -/
theorem block_rate [DecidableEq J] [DecidableEq L] [DecidableEq Tok] [DecidableEq Mat] [DecidableEq Row]
    (Q : Cube K Mat Row Col) {s k f a : ℕ} {b μ : ℝ} (hb : 0 < b) (ha : 0 < a) (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 1)
    (hH : Assumptions.UpdateHardness M s b) (hX : Assumptions.ActivationReads Q s k f a) {I : Finset (Fin C.N)}
    (hI : C.InferenceBlock s k I) {B : Finset (J × L)} (hB : B ⊆ M.whole I) :
    Guarantees.psRate b f a aOut (2 * k) μ *
        (((B.filter fun p => ¬ K.tokReads I (K.tok (K.act p)) < f * a).card : ℝ) +
          (B.filter fun p => K.tokReads I (K.tok (K.act p)) < f * a).card) ≤
      k + μ * k + (1 - μ) * ((B.filter fun p => K.tokReads I (K.tok (K.act p)) < f * a).image Q.entry).card / 3 := by
  obtain ⟨X, Ah, Al, hX0, hAh, hAl, hXk, hAk, hHi, hLo, hF⟩ := block_counts Q hb hH hX hI hB
  have h := rate_block hb (by exact_mod_cast ha : (0 : ℝ) < a) (Nat.cast_nonneg f) (Nat.cast_nonneg aOut) hμ0 hμ1
    hX0 hAh hAl (Nat.cast_nonneg _) (Nat.cast_nonneg _) (Nat.cast_nonneg _) hXk hAk hHi hLo hF
  have hμA : μ * (Ah + Al) ≤ μ * k := mul_le_mul_of_nonneg_left hAk hμ0
  linarith

end

/-- **Adding up the blocks**: if each block's not-whole products and accumulation reads are at most `k`, the
crossings at most three times the accumulation reads, and each block's whole products pay their rate out of
`k + μ k + (1 − μ) Y / 3`, then `r (P − k #𝓘) ≤ 2 k #𝓘`. -/
theorem ps_cover_sum {ι : Type*} (𝓘 : Finset ι) (P k : ℕ) (N nH nL Y Z : ι → ℕ) {r μ : ℝ} (hr : 0 ≤ r)
    (hμ1 : μ ≤ 1) (hP : P ≤ ∑ I ∈ 𝓘, (N I + (nH I + nL I))) (hN : ∀ I ∈ 𝓘, N I ≤ k) (hZ : ∀ I ∈ 𝓘, Z I ≤ k)
    (hY : ∑ I ∈ 𝓘, Y I ≤ 3 * ∑ I ∈ 𝓘, Z I)
    (hblock : ∀ I ∈ 𝓘, r * ((nH I : ℝ) + nL I) ≤ k + μ * k + (1 - μ) * Y I / 3) :
    r * ((P : ℝ) - k * 𝓘.card) ≤ 2 * k * 𝓘.card := by
  have h1μ : 0 ≤ 1 - μ := sub_nonneg.2 hμ1
  have hP' : (P : ℝ) - k * 𝓘.card ≤ ∑ I ∈ 𝓘, ((nH I : ℝ) + nL I) := by
    have h1 : P ≤ 𝓘.card * k + ∑ I ∈ 𝓘, (nH I + nL I) := by
      refine hP.trans ?_
      rw [Finset.sum_add_distrib]
      refine Nat.add_le_add_right ?_ _
      calc ∑ I ∈ 𝓘, N I ≤ ∑ _I ∈ 𝓘, k := Finset.sum_le_sum hN
        _ = 𝓘.card * k := by rw [Finset.sum_const, smul_eq_mul]
    have h2 : (P : ℝ) ≤ 𝓘.card * k + ∑ I ∈ 𝓘, ((nH I : ℝ) + nL I) := by exact_mod_cast h1
    linarith
  have hY' : ∑ I ∈ 𝓘, (Y I : ℝ) ≤ 3 * (𝓘.card * k) := by
    have h1 : ∑ I ∈ 𝓘, Z I ≤ 𝓘.card * k := by
      calc ∑ I ∈ 𝓘, Z I ≤ ∑ _I ∈ 𝓘, k := Finset.sum_le_sum hZ
        _ = 𝓘.card * k := by rw [Finset.sum_const, smul_eq_mul]
    have h2 : ∑ I ∈ 𝓘, Y I ≤ 3 * (𝓘.card * k) := hY.trans (Nat.mul_le_mul_left 3 h1)
    exact_mod_cast h2
  calc r * ((P : ℝ) - k * 𝓘.card) ≤ r * ∑ I ∈ 𝓘, ((nH I : ℝ) + nL I) := mul_le_mul_of_nonneg_left hP' hr
    _ = ∑ I ∈ 𝓘, r * ((nH I : ℝ) + nL I) := Finset.mul_sum _ _ _
    _ ≤ ∑ I ∈ 𝓘, ((k : ℝ) + μ * k + (1 - μ) * Y I / 3) := Finset.sum_le_sum hblock
    _ = 𝓘.card * ((k : ℝ) + μ * k) + (1 - μ) / 3 * ∑ I ∈ 𝓘, (Y I : ℝ) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, Finset.mul_sum]
        congr 1
        exact Finset.sum_congr rfl fun I _ => by ring
    _ ≤ 𝓘.card * ((k : ℝ) + μ * k) + (1 - μ) / 3 * (3 * (𝓘.card * k)) :=
        add_le_add (le_refl _) (mul_le_mul_of_nonneg_left hY' (div_nonneg h1μ (by norm_num)))
    _ = 2 * k * 𝓘.card := by ring

theorem PartialSumCoverBound : Guarantees.PartialSumCoverBound := by
  intro C J L Tok Val Mat Row Col _ _ _ _ _ _ M d aOut c V w K Q s k f a b μ 𝓘 hb ha hμ0 hμ1 hH hX h𝓘 hcov
  obtain ⟨φ, hφ, hsum⟩ := PartialSumsRead C M d aOut c V w K Q 𝓘 hcov
  refine ps_cover_sum 𝓘 M.P.card k
    (fun I => ((M.P.filter fun p => φ p = I).filter fun p => p ∉ M.whole I).card)
    (fun I => (((M.P.filter fun p => φ p = I).filter fun p => p ∈ M.whole I).filter
      fun p => ¬ K.tokReads I (K.tok (K.act p)) < f * a).card)
    (fun I => (((M.P.filter fun p => φ p = I).filter fun p => p ∈ M.whole I).filter
      fun p => K.tokReads I (K.tok (K.act p)) < f * a).card)
    (fun I => ((((M.P.filter fun p => φ p = I).filter fun p => p ∈ M.whole I).filter
      fun p => K.tokReads I (K.tok (K.act p)) < f * a).image Q.entry).card)
    (fun I => (C.inputs I ∩ Q.accGates).card)
    (psRate_nonneg (Nat.cast_nonneg f) (Nat.cast_nonneg aOut) hμ0) hμ1 ?_ ?_ ?_ ?_ ?_
  · rw [Finset.card_eq_sum_card_fiberwise fun p hp => (hφ p hp).1]
    refine Finset.sum_le_sum fun I _ => ?_
    have h1 : ((M.P.filter fun p => φ p = I).filter fun p => p ∈ M.whole I).card +
        ((M.P.filter fun p => φ p = I).filter fun p => p ∉ M.whole I).card =
        (M.P.filter fun p => φ p = I).card := Finset.card_filter_add_card_filter_not _
    have h2 : (((M.P.filter fun p => φ p = I).filter fun p => p ∈ M.whole I).filter
          fun p => K.tokReads I (K.tok (K.act p)) < f * a).card +
        (((M.P.filter fun p => φ p = I).filter fun p => p ∈ M.whole I).filter
          fun p => ¬ K.tokReads I (K.tok (K.act p)) < f * a).card =
        ((M.P.filter fun p => φ p = I).filter fun p => p ∈ M.whole I).card :=
      Finset.card_filter_add_card_filter_not _
    omega
  · intro I hI
    refine (card_le_inputs_accGates Q fun p hp => ?_).trans ((inputs_accGates_card_le Q I).trans (h𝓘 I hI).2)
    obtain ⟨hpI, hpw⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpP, hpφ⟩ := Finset.mem_filter.1 hpI
    have ho := (hφ p hpP).2
    rw [hpφ] at ho
    exact Finset.mem_sdiff.2 ⟨mem_finished_of hpP ho, hpw⟩
  · intro I hI
    exact (inputs_accGates_card_le Q I).trans (h𝓘 I hI).2
  · refine crossings_sum_le Q hX h𝓘 hφ hsum _ fun I _ p hp => ?_
    obtain ⟨hpB, hlow⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpI, -⟩ := Finset.mem_filter.1 hpB
    obtain ⟨hpP, hpφ⟩ := Finset.mem_filter.1 hpI
    exact ⟨hpP, hpφ, hlow⟩
  · intro I hI
    exact block_rate Q hb ha hμ0 hμ1 hH hX (h𝓘 I hI) fun p hp => (Finset.mem_filter.1 hp).2

end Nci.Cover.SecurityProofs
