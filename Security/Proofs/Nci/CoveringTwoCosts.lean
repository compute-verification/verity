import Definitions.Nci.Guarantees.CoveringTwoCosts
import Proofs.Nci.Covering

/-!
# The covering bound at two costs

`CoveringBound`'s proof (`Proofs.Nci.Covering`) with each cost where it is used. Charge each chained multiplication block
to a block `S` of the covering that computes its output gate; at most `k` are not completed by `S`. The completed ones lie
in `θ'_S × J_S`: `|θ'_S| ≤ k / c_w` by parameter-hardness at `c_w`, so `|θ'_S| < D` since `c_w D > k`; with `Q = θ'_S`
the inputs of `S` in `T_j` cut the activation off for each token `j ∈ J_S`, so they are at least `c_a` and
`|J_S| ≤ k / c_a`. That count for one block is `CoveringTwoCosts.card_charged_le`, which the cover theorem's join to
the NCI policy uses at each block's own `k` (`Proofs.Nci.Cover.Chained`).

At `c_w = c_a = r · b` the assumptions are `ParameterHard` and `ActivationChained` by definition, and the bound is
`CoveringBound`'s (`coveringBound_of_twoCosts`). `CoveringBound` keeps its own proof.
-/

namespace Nci.SecurityProofs.CoveringTwoCosts

open Verity Finset Nci.Covering

variable {d : Registered} (B : MulBlocks d)

theorem parameterHard_iff (s : ℕ) (r b : ℝ) :
    Assumptions.ParameterHard d B s r b ↔ Assumptions.ParameterHardAt d B s (r * b) := Iff.rfl

theorem activationChained_iff (D : ℕ) (r b : ℝ) (p : Fin B.nw × Fin B.nt) :
    Assumptions.ActivationChained d B D r b p ↔ Assumptions.ActivationChainedAt d B D (r * b) p := Iff.rfl

theorem chainedBlocks_eq (D : ℕ) (r b : ℝ) :
    Assumptions.chainedBlocks d B D r b = Assumptions.chainedBlocksAt d B D (r * b) := rfl

/-- At one cost, `c_w = c_a = r · b`, the two-cost bound is `CoveringBound`'s statement. -/
theorem coveringBound_of_twoCosts (h : Guarantees.CoveringBoundTwoCosts) : Guarantees.CoveringBound := by
  intro d B s k D r b n I hkD hhard hcov
  have hc : 0 < r * b := by
    by_contra h'
    nlinarith [not_lt.1 h', (Nat.cast_nonneg D : (0 : ℝ) ≤ D), (Nat.cast_nonneg k : (0 : ℝ) ≤ k)]
  obtain ⟨h1, h2⟩ := h d B s k D (r * b) (r * b) n I hkD hc hhard hcov
  rw [chainedBlocks_eq, sq]
  exact ⟨h1, h2⟩

end Nci.SecurityProofs.CoveringTwoCosts

namespace Nci.SecurityProofs

open Verity Finset Nci.Covering Nci.SecurityProofs.Covering

/-- **One block's chained count at two costs**: an `(s, k)`-inference block `S` of an `(s, c_w)`-parameter-hard `d`,
with `c_w D > k` and `c_a > 0`, computes the output gates of at most `(k / c_w)(k / c_a) + k` multiplication blocks
chained at `(D, c_a)`: at most `k` that it doesn't complete, and the completed ones in `θ'_S × J_S`. -/
theorem CoveringTwoCosts.card_charged_le {d : Registered} (B : MulBlocks d) {s k D : ℕ} {cw ca : ℝ}
    (hkD : (k : ℝ) < cw * D) (hca : 0 < ca) (hhard : Assumptions.ParameterHardAt d B s cw) {S : Subcircuit d.D}
    (hS : InferenceBlock d s k S) :
    (((Assumptions.chainedBlocksAt d B D ca).filter fun p => B.o p ∈ S.inner).card : ℝ) ≤ k / cw * (k / ca) + k := by
  classical
  set M := Assumptions.chainedBlocksAt d B D ca with hM
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hcw : 0 < cw := by
    by_contra h
    nlinarith [not_lt.1 h, (Nat.cast_nonneg D : (0 : ℝ) ≤ D)]
  have hMP : ∀ p ∈ M, p ∈ B.P := fun p hp => (mem_filter.1 hp).1
  have hθ : ∀ g ∈ d.θ, g ∈ sources d.D := fun g hg => by
    simp [sources, d.D.input_reads g (d.θ_input g hg)]
  obtain ⟨hsz, hins⟩ := hS
  have hinsR : ((S.ins \ d.θ).card : ℝ) ≤ k := by exact_mod_cast hins
  have hQ : cw * (B.weightsCompleted S).card ≤ k := (hhard S hsz).trans hinsR
  have hQD : (B.weightsCompleted S).card < D := by
    have h : cw * (B.weightsCompleted S).card < cw * D := lt_of_le_of_lt hQ hkD
    exact_mod_cast lt_of_mul_lt_mul_left h hcw.le
  have hJ : ca * ((M.filter fun p => B.Completes S p).image Prod.snd).card ≤ k := by
    refine le_trans ?_ hinsR
    refine mul_card_le_of_disjoint _ B.T _ ca (fun i _ j _ hij => B.T_disjoint i j hij) ?_
    intro j hj
    obtain ⟨p, hpMC, rfl⟩ := mem_image.1 hj
    obtain ⟨hpM, hpC⟩ := mem_filter.1 hpMC
    obtain ⟨hpP, A, W, -, hAfeed, -, hW, hcut⟩ := (mem_filter.1 hpM).2
    have hX := hcut (B.weightsCompleted S) hQD (S.ins ∩ B.T p.2) inter_subset_right ?_
    · refine le_trans hX (Nat.cast_le.2 (card_le_card fun x hx => ?_))
      obtain ⟨hxi, hxT⟩ := mem_inter.1 hx
      exact mem_inter.2 ⟨hxT, mem_sdiff.2 ⟨hxi, disjoint_left.1 (B.T_θ p.2) hxT⟩⟩
    · intro i' hi'
      obtain ⟨hi'W, hi'Q⟩ := mem_sdiff.1 hi'
      obtain ⟨hi'P, hmT, -⟩ := hW i' hi'W
      by_cases hmeet : (S.ins ∩ B.T p.2 ∩ B.m (i', p.2)).Nonempty
      · exact Or.inl hmeet
      · refine Or.inr fun a ha hpath => ?_
        have hnot : ∀ x ∈ B.T p.2 \ (S.ins ∩ B.T p.2), x ∉ S.ins := fun x hx hxi =>
          (mem_sdiff.1 hx).2 (mem_inter.2 ⟨hxi, (mem_sdiff.1 hx).1⟩)
        obtain ⟨g, hg, hag⟩ := hAfeed a ha
        have hgin := mem_sdiff.1 (hpC hg)
        have hain : a ∈ S.inner :=
          mem_sdiff.2 ⟨S.closed g hgin.1 hgin.2 hag, hnot a (FeedsWithin.mem_right hpath)⟩
        have ho : B.o (i', p.2) ∈ S.inner := by
          rcases inner_or_meets hpath hain with h | ⟨x, hx, hxi⟩
          · exact h
          · exact absurd hxi (hnot x hx)
        rcases completes_or_meets B hi'P ho with hcomp | ⟨x, hxm, hxi⟩
        · exact hi'Q (mem_image.2 ⟨(i', p.2), mem_filter.2 ⟨hi'P, hcomp⟩, rfl⟩)
        · exact hmeet ⟨x, mem_inter.2 ⟨mem_inter.2 ⟨hxi, hmT hxm⟩, hxm⟩⟩
  have hNC : (((M.filter fun p => B.o p ∈ S.inner).filter fun p => ¬ B.Completes S p).card : ℝ) ≤ k := by
    refine le_trans ?_ hinsR
    have h := mul_card_le_of_disjoint ((M.filter fun p => B.o p ∈ S.inner).filter
        fun p => ¬ B.Completes S p) B.m (S.ins \ d.θ) 1
      (fun p hp q hq hpq => B.m_disjoint p (hMP p (mem_filter.1 (mem_filter.1 hp).1).1)
        q (hMP q (mem_filter.1 (mem_filter.1 hq).1).1) hpq) ?_
    · simpa using h
    · intro p hp
      obtain ⟨hpMo, hnc⟩ := mem_filter.1 hp
      obtain ⟨hpM, ho⟩ := mem_filter.1 hpMo
      have hpP := hMP p hpM
      rcases completes_or_meets B hpP ho with hcomp | ⟨x, hxm, hxi⟩
      · exact absurd hcomp hnc
      · have hx : x ∈ B.m p ∩ (S.ins \ d.θ) := mem_inter.2 ⟨hxm, mem_sdiff.2 ⟨hxi,
          fun hxθ => disjoint_left.1 (B.m_computed p hpP) hxm (hθ x hxθ)⟩⟩
        exact_mod_cast card_pos.2 ⟨x, hx⟩
  have hCP : (((M.filter fun p => B.o p ∈ S.inner).filter fun p => B.Completes S p).card : ℝ) ≤
      k / cw * (k / ca) := by
    have hsub : ((M.filter fun p => B.o p ∈ S.inner).filter fun p => B.Completes S p) ⊆
        B.weightsCompleted S ×ˢ ((M.filter fun p => B.Completes S p).image Prod.snd) := by
      intro p hp
      obtain ⟨hpMo, hcomp⟩ := mem_filter.1 hp
      have hpM := (mem_filter.1 hpMo).1
      exact mem_product.2 ⟨mem_image.2 ⟨p, mem_filter.2 ⟨hMP p hpM, hcomp⟩, rfl⟩,
        mem_image.2 ⟨p, mem_filter.2 ⟨hpM, hcomp⟩, rfl⟩⟩
    have h1 : ((B.weightsCompleted S).card : ℝ) ≤ k / cw := by
      rw [le_div_iff₀ hcw]; linarith
    have h2 : (((M.filter fun p => B.Completes S p).image Prod.snd).card : ℝ) ≤ k / ca := by
      rw [le_div_iff₀ hca]; linarith
    calc _ ≤ ((B.weightsCompleted S ×ˢ
            ((M.filter fun p => B.Completes S p).image Prod.snd)).card : ℝ) := by
          exact_mod_cast card_le_card hsub
      _ = (B.weightsCompleted S).card * ((M.filter fun p => B.Completes S p).image Prod.snd).card := by
          rw [card_product]; push_cast; rfl
      _ ≤ k / cw * (k / ca) := mul_le_mul h1 h2 (Nat.cast_nonneg _) (div_nonneg hk0 hcw.le)
  have hsplit := card_filter_add_card_filter_not (s := M.filter fun p => B.o p ∈ S.inner)
    (fun p => B.Completes S p)
  have hsplitR : ((M.filter fun p => B.o p ∈ S.inner).card : ℝ) =
      (((M.filter fun p => B.o p ∈ S.inner).filter fun p => B.Completes S p).card : ℝ) +
      (((M.filter fun p => B.o p ∈ S.inner).filter fun p => ¬ B.Completes S p).card : ℝ) := by
    exact_mod_cast hsplit.symm
  rw [hsplitR]
  linarith

theorem CoveringBoundTwoCosts : Guarantees.CoveringBoundTwoCosts := by
  classical
  intro d B s k D cw ca n I hkD hca hhard ⟨hblock, hcover⟩
  set M := Assumptions.chainedBlocksAt d B D ca with hM
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hcw : 0 < cw := by
    by_contra h
    nlinarith [not_lt.1 h, (Nat.cast_nonneg D : (0 : ℝ) ≤ D)]
  have hMP : ∀ p ∈ M, p ∈ B.P := fun p hp => (mem_filter.1 hp).1
  have hK : 0 ≤ k / cw * (k / ca) + k :=
    add_nonneg (mul_nonneg (div_nonneg hk0 hcw.le) (div_nonneg hk0 hca.le)) hk0
  have hblockbound : ∀ u, ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) ≤ k / cw * (k / ca) + k :=
    fun u => CoveringTwoCosts.card_charged_le B hkD hca hhard (hblock u)
  have hcount : (M.card : ℝ) ≤ n * (k / cw * (k / ca) + k) := by
    have hsub : M ⊆ univ.biUnion fun u => M.filter fun p => B.o p ∈ (I u).inner := by
      intro p hp
      have hpP := hMP p hp
      obtain ⟨u, hu⟩ := hcover _ fun h => disjoint_left.1 (B.m_computed p hpP) (B.o_mem p hpP) h
      exact mem_biUnion.2 ⟨u, mem_univ _, mem_filter.2 ⟨hp, hu⟩⟩
    calc (M.card : ℝ) ≤ ∑ u, ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) := by
          exact_mod_cast (card_le_card hsub).trans card_biUnion_le
      _ ≤ ∑ _u : Fin n, (k / cw * (k / ca) + k) := sum_le_sum fun u _ => hblockbound u
      _ = n * (k / cw * (k / ca) + k) := by simp
  refine ⟨hcount, ?_⟩
  have hdiv : (M.card : ℝ) / (k / cw * (k / ca) + k) ≤ n := by
    rcases hK.lt_or_eq with hK' | hK'
    · rw [div_le_iff₀ hK']; exact hcount
    · rw [← hK', div_zero]; exact Nat.cast_nonneg n
  calc (M.card : ℝ) / d.D.N * s / (k / cw * (k / ca) + k) =
        ((M.card : ℝ) / (k / cw * (k / ca) + k)) * ((s : ℝ) / d.D.N) := by ring
    _ ≤ (n : ℝ) * ((s : ℝ) / d.D.N) := mul_le_mul_of_nonneg_right hdiv (by positivity)
    _ = (s : ℝ) * n / d.D.N := by ring

end Nci.SecurityProofs
