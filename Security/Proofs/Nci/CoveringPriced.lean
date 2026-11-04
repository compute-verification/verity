import Definitions.Nci.Guarantees.CoveringPriced
import Proofs.Nci.CoveringTwoCosts
import Mathlib.Tactic.FieldSimp

/-!
# The covering bound at a price per weight

`CoveringBoundTwoCosts`'s proof (`Proofs.Nci.CoveringTwoCosts`) with the weights' prices added up where it counts weights.
Each chained multiplication block's output gate is computed by some block of the covering. `U` is the chained blocks that
no block computing their output completes: each is charged to a block computing its output, which has one of its gates as
an input outside `θ`, so at most `k` are charged to a block. The others are completed by a block `S` computing their
output: their weights are in `θ'_S`, whose prices add up to at most `k`, so `|θ'_S| < D` since any `D` weights cost more;
with `Q = θ'_S` the inputs of `S` in `T_j` cut the activation off for each token `j` it completes, so they are at least
`c_a` and those tokens number at most `k / c_a`. The completed blocks charged to `S` lie in `θ'_S × J_S` and cost at most
`(∑_{θ'_S} p) · |J_S| ≤ k · (k / c_a)`. A block of the covering may complete what another computes too, so the completed
blocks are bounded through a union over the covering (`sum_le_sum_sum_of_subset_biUnion`).

At `p = fun _ => c_w` the premise is `ParameterHardAt` (`parameterHardPricedAt_const_iff`) and the bound gives
`CoveringBoundTwoCosts` (`coveringBoundTwoCosts_of_priced`). `k < p i · D` for every `i` gives the two conditions on the
prices (`depth_of_lt_mul`).
-/

namespace Nci.SecurityProofs.CoveringPriced

open Verity Finset Nci.Covering

/-- A nonnegative function's sum over a subset of a union is at most the sum of its sums over the parts. -/
theorem sum_le_sum_sum_of_subset_biUnion {ι α : Type*} [DecidableEq α] (F : Finset ι) (t : ι → Finset α)
    (X : Finset α) (f : α → ℝ) (hf : ∀ a, 0 ≤ f a) (hX : X ⊆ F.biUnion t) :
    ∑ a ∈ X, f a ≤ ∑ i ∈ F, ∑ a ∈ t i, f a := by
  classical
  induction F using Finset.induction_on generalizing X with
  | empty =>
    rw [biUnion_empty, subset_empty] at hX
    simp [hX]
  | insert i F hi ih =>
    rw [sum_insert hi, ← sum_filter_add_sum_filter_not X (· ∈ t i)]
    refine add_le_add (sum_le_sum_of_subset_of_nonneg (fun a ha => (mem_filter.1 ha).2) fun a _ _ => hf a)
      (ih _ fun a ha => ?_)
    obtain ⟨haX, hat⟩ := mem_filter.1 ha
    obtain ⟨j, hj, haj⟩ := mem_biUnion.1 (hX haX)
    rcases mem_insert.1 hj with rfl | hj
    · exact absurd haj hat
    · exact mem_biUnion.2 ⟨j, hj, haj⟩

/-- `k < p i · D` for every weight gives the guarantee's two conditions on the prices: each is nonnegative, and any `D`
weights cost more than `k`. -/
theorem depth_of_lt_mul {ι : Type*} (p : ι → ℝ) (k : ℝ) (D : ℕ) (hk : 0 ≤ k) (hD : 0 < D)
    (h : ∀ i, k < p i * D) : (∀ i, 0 ≤ p i) ∧ ∀ W : Finset ι, D ≤ W.card → k < ∑ i ∈ W, p i := by
  have hD' : (0 : ℝ) < D := by exact_mod_cast hD
  refine ⟨fun i => ?_, fun W hW => ?_⟩
  · by_contra hp
    nlinarith [h i, not_le.1 hp]
  · have hW' : (D : ℝ) ≤ W.card := by exact_mod_cast hW
    have hne : W.Nonempty := card_pos.1 (lt_of_lt_of_le hD hW)
    have hlt : ∑ _i ∈ W, k < ∑ i ∈ W, p i * D := sum_lt_sum_of_nonempty hne fun i _ => h i
    rw [sum_const, nsmul_eq_mul, ← sum_mul] at hlt
    by_contra hle
    nlinarith [not_lt.1 hle]

variable {d : RegisteredCircuit} (B : MulBlocks d)

/-- At one price `c_w` for every weight, parameter-hardness at prices is `ParameterHardAt` at `c_w`. -/
theorem parameterHardPricedAt_const_iff (s : ℕ) (cw : ℝ) :
    Assumptions.ParameterHardPricedAt d B s (fun _ => cw) ↔ Assumptions.ParameterHardAt d B s cw := by
  refine forall_congr' fun S => imp_congr_right fun _ => ?_
  rw [sum_const, nsmul_eq_mul, mul_comm]

end Nci.SecurityProofs.CoveringPriced

namespace Nci.SecurityProofs

open Verity Finset Nci.Covering Nci.SecurityProofs.Covering Nci.SecurityProofs.CoveringPriced

theorem CoveringBoundPriced : Guarantees.CoveringBoundPriced := by
  classical
  intro d B s k D p ca n I hp hdepth hca hhard ⟨hblock, hcover⟩
  set M := Assumptions.chainedBlocksAt d B D ca with hM
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hMP : ∀ b ∈ M, b ∈ B.P := fun b hb => (mem_filter.1 hb).1
  have hθ : ∀ g ∈ d.θ, g ∈ sources d.D := fun g hg => by
    simp [sources, d.D.input_reads g (d.θ_input g hg)]
  have hNC : ∀ u, (((M.filter fun b => B.o b ∈ (I u).inner).filter fun b => ¬ B.Completes (I u) b).card : ℝ) ≤ k := by
    intro u
    have hinsR : (((I u).ins \ d.θ).card : ℝ) ≤ k := by exact_mod_cast (hblock u).2
    refine le_trans ?_ hinsR
    have h := mul_card_le_of_disjoint ((M.filter fun b => B.o b ∈ (I u).inner).filter
        fun b => ¬ B.Completes (I u) b) B.m ((I u).ins \ d.θ) 1
      (fun b hb c hc hbc => B.m_disjoint b (hMP b (mem_filter.1 (mem_filter.1 hb).1).1)
        c (hMP c (mem_filter.1 (mem_filter.1 hc).1).1) hbc) ?_
    · simpa using h
    · intro b hb
      obtain ⟨hbMo, hnc⟩ := mem_filter.1 hb
      obtain ⟨hbM, ho⟩ := mem_filter.1 hbMo
      have hbP := hMP b hbM
      rcases completes_or_meets B hbP ho with hcomp | ⟨x, hxm, hxi⟩
      · exact absurd hcomp hnc
      · have hx : x ∈ B.m b ∩ ((I u).ins \ d.θ) := mem_inter.2 ⟨hxm, mem_sdiff.2 ⟨hxi,
          fun hxθ => disjoint_left.1 (B.m_computed b hbP) hxm (hθ x hxθ)⟩⟩
        exact_mod_cast card_pos.2 ⟨x, hx⟩
  have hCP : ∀ u, ∑ b ∈ ((M.filter fun b => B.o b ∈ (I u).inner).filter fun b => B.Completes (I u) b), p b.1 ≤
      k * (k / ca) := by
    intro u
    obtain ⟨hsz, hins⟩ := hblock u
    have hinsR : (((I u).ins \ d.θ).card : ℝ) ≤ k := by exact_mod_cast hins
    have hQ : ∑ i ∈ B.weightsCompleted (I u), p i ≤ k := (hhard (I u) hsz).trans hinsR
    have hQD : (B.weightsCompleted (I u)).card < D := by
      by_contra h
      exact absurd (hdepth _ (not_lt.1 h)) (not_lt.2 hQ)
    have hJ : ca * ((M.filter fun b => B.Completes (I u) b).image Prod.snd).card ≤ k := by
      refine le_trans ?_ hinsR
      refine mul_card_le_of_disjoint _ B.T _ ca (fun i _ j _ hij => B.T_disjoint i j hij) ?_
      intro j hj
      obtain ⟨b, hbMC, rfl⟩ := mem_image.1 hj
      obtain ⟨hbM, hbC⟩ := mem_filter.1 hbMC
      obtain ⟨hbP, A, W, -, hAfeed, -, hW, hcut⟩ := (mem_filter.1 hbM).2
      have hX := hcut (B.weightsCompleted (I u)) hQD ((I u).ins ∩ B.T b.2) inter_subset_right ?_
      · refine le_trans hX (Nat.cast_le.2 (card_le_card fun x hx => ?_))
        obtain ⟨hxi, hxT⟩ := mem_inter.1 hx
        exact mem_inter.2 ⟨hxT, mem_sdiff.2 ⟨hxi, disjoint_left.1 (B.T_θ b.2) hxT⟩⟩
      · intro i' hi'
        obtain ⟨hi'W, hi'Q⟩ := mem_sdiff.1 hi'
        obtain ⟨hi'P, hmT, -⟩ := hW i' hi'W
        by_cases hmeet : ((I u).ins ∩ B.T b.2 ∩ B.m (i', b.2)).Nonempty
        · exact Or.inl hmeet
        · refine Or.inr fun a ha hpath => ?_
          have hnot : ∀ x ∈ B.T b.2 \ ((I u).ins ∩ B.T b.2), x ∉ (I u).ins := fun x hx hxi =>
            (mem_sdiff.1 hx).2 (mem_inter.2 ⟨hxi, (mem_sdiff.1 hx).1⟩)
          obtain ⟨g, hg, hag⟩ := hAfeed a ha
          have hgin := mem_sdiff.1 (hbC hg)
          have hain : a ∈ (I u).inner :=
            mem_sdiff.2 ⟨(I u).closed g hgin.1 hgin.2 hag, hnot a (FeedsWithin.mem_right hpath)⟩
          have ho : B.o (i', b.2) ∈ (I u).inner := by
            rcases inner_or_meets hpath hain with h | ⟨x, hx, hxi⟩
            · exact h
            · exact absurd hxi (hnot x hx)
          rcases completes_or_meets B hi'P ho with hcomp | ⟨x, hxm, hxi⟩
          · exact hi'Q (mem_image.2 ⟨(i', b.2), mem_filter.2 ⟨hi'P, hcomp⟩, rfl⟩)
          · exact hmeet ⟨x, mem_inter.2 ⟨mem_inter.2 ⟨hxi, hmT hxm⟩, hxm⟩⟩
    have hJ' : (((M.filter fun b => B.Completes (I u) b).image Prod.snd).card : ℝ) ≤ k / ca := by
      rw [le_div_iff₀ hca]; linarith
    have hsub : ((M.filter fun b => B.o b ∈ (I u).inner).filter fun b => B.Completes (I u) b) ⊆
        B.weightsCompleted (I u) ×ˢ ((M.filter fun b => B.Completes (I u) b).image Prod.snd) := by
      intro b hb
      obtain ⟨hbMo, hcomp⟩ := mem_filter.1 hb
      have hbM := (mem_filter.1 hbMo).1
      exact mem_product.2 ⟨mem_image.2 ⟨b, mem_filter.2 ⟨hMP b hbM, hcomp⟩, rfl⟩,
        mem_image.2 ⟨b, mem_filter.2 ⟨hbM, hcomp⟩, rfl⟩⟩
    calc ∑ b ∈ ((M.filter fun b => B.o b ∈ (I u).inner).filter fun b => B.Completes (I u) b), p b.1
        ≤ ∑ b ∈ B.weightsCompleted (I u) ×ˢ ((M.filter fun b => B.Completes (I u) b).image Prod.snd), p b.1 :=
          sum_le_sum_of_subset_of_nonneg hsub fun b _ _ => hp b.1
      _ = (∑ i ∈ B.weightsCompleted (I u), p i) *
            ((M.filter fun b => B.Completes (I u) b).image Prod.snd).card := by
          rw [sum_product, sum_mul]
          exact sum_congr rfl fun i _ => by simp only [sum_const, nsmul_eq_mul]; exact mul_comm _ _
      _ ≤ k * (k / ca) := mul_le_mul hQ hJ' (Nat.cast_nonneg _) hk0
  have hocov : ∀ b ∈ M, ∃ u, B.o b ∈ (I u).inner := fun b hb =>
    hcover _ fun h => disjoint_left.1 (B.m_computed b (hMP b hb)) (B.o_mem b (hMP b hb)) h
  set U := M.filter fun b => ∀ u, B.o b ∈ (I u).inner → ¬ B.Completes (I u) b with hU
  have hUcard : (U.card : ℝ) ≤ n * k := by
    have hsub : U ⊆ univ.biUnion fun u =>
        (M.filter fun b => B.o b ∈ (I u).inner).filter fun b => ¬ B.Completes (I u) b := by
      intro b hb
      obtain ⟨hbM, hbU⟩ := mem_filter.1 hb
      obtain ⟨u, hu⟩ := hocov b hbM
      exact mem_biUnion.2 ⟨u, mem_univ _, mem_filter.2 ⟨mem_filter.2 ⟨hbM, hu⟩, hbU u hu⟩⟩
    calc (U.card : ℝ)
        ≤ ∑ u, (((M.filter fun b => B.o b ∈ (I u).inner).filter fun b => ¬ B.Completes (I u) b).card : ℝ) := by
          exact_mod_cast (card_le_card hsub).trans card_biUnion_le
      _ ≤ ∑ _u : Fin n, (k : ℝ) := sum_le_sum fun u _ => hNC u
      _ = n * k := by simp
  have hCsum : ∑ b ∈ M \ U, p b.1 ≤ n * (k * (k / ca)) := by
    have hsub : M \ U ⊆ univ.biUnion fun u =>
        (M.filter fun b => B.o b ∈ (I u).inner).filter fun b => B.Completes (I u) b := by
      intro b hb
      obtain ⟨hbM, hbU⟩ := mem_sdiff.1 hb
      have hex : ∃ u, B.o b ∈ (I u).inner ∧ B.Completes (I u) b := by
        by_contra hno
        exact hbU (mem_filter.2 ⟨hbM, fun u hu hc => hno ⟨u, hu, hc⟩⟩)
      obtain ⟨u, hu, hc⟩ := hex
      exact mem_biUnion.2 ⟨u, mem_univ _, mem_filter.2 ⟨mem_filter.2 ⟨hbM, hu⟩, hc⟩⟩
    calc ∑ b ∈ M \ U, p b.1
        ≤ ∑ u, ∑ b ∈ ((M.filter fun b => B.o b ∈ (I u).inner).filter fun b => B.Completes (I u) b), p b.1 :=
          sum_le_sum_sum_of_subset_biUnion _ _ _ (fun b => p b.1) (fun b => hp b.1) hsub
      _ ≤ ∑ _u : Fin n, k * (k / ca) := sum_le_sum fun u _ => hCP u
      _ = n * (k * (k / ca)) := by simp
  refine ⟨⟨U, filter_subset _ _, hUcard, hCsum⟩, fun t ht => ?_⟩
  have hRHS : (0 : ℝ) ≤ s * n / d.D.N := by positivity
  rcases lt_or_ge t 0 with ht0 | ht0
  · have hM0 : M = ∅ :=
      eq_empty_of_forall_notMem fun b _ => absurd (ht b.1) (not_le.2 (ht0.trans_le (hp b.1)))
    rw [hM0, sum_empty, zero_div, zero_mul, zero_div]
    exact hRHS
  · have hK : 0 ≤ k * (k / ca) + t * k := add_nonneg (mul_nonneg hk0 (div_nonneg hk0 hca.le)) (mul_nonneg ht0 hk0)
    have hUsum : ∑ b ∈ U, p b.1 ≤ n * k * t :=
      calc ∑ b ∈ U, p b.1 ≤ ∑ _b ∈ U, t := sum_le_sum fun b _ => ht b.1
        _ = U.card * t := by rw [sum_const, nsmul_eq_mul]
        _ ≤ n * k * t := mul_le_mul_of_nonneg_right hUcard ht0
    have htot : ∑ b ∈ M, p b.1 ≤ n * (k * (k / ca) + t * k) := by
      rw [← sum_sdiff (filter_subset _ M : U ⊆ M)]
      have e : (n : ℝ) * (k * (k / ca) + t * k) = n * (k * (k / ca)) + n * k * t := by ring
      rw [e]
      exact add_le_add hCsum hUsum
    rcases hK.lt_or_eq with hK' | hK'
    · have hdiv : (∑ b ∈ M, p b.1) / (k * (k / ca) + t * k) ≤ n := by
        rw [div_le_iff₀ hK']; exact htot
      calc (∑ b ∈ M, p b.1) / d.D.N * s / (k * (k / ca) + t * k) =
            ((∑ b ∈ M, p b.1) / (k * (k / ca) + t * k)) * ((s : ℝ) / d.D.N) := by ring
        _ ≤ (n : ℝ) * ((s : ℝ) / d.D.N) := mul_le_mul_of_nonneg_right hdiv (by positivity)
        _ = (s : ℝ) * n / d.D.N := by ring
    · rw [← hK', div_zero]
      exact hRHS

end Nci.SecurityProofs

namespace Nci.SecurityProofs.CoveringPriced

open Verity Finset Nci.Covering

/-- **At uniform prices it is the two-cost bound**: at `p = fun _ => c_w` the priced bound's hypotheses follow from
`CoveringBoundTwoCosts`'s (`c_w D > k` makes `c_w` positive, and any `D` weights then cost `c_w |W| ≥ c_w D > k`), `U` and
the rest give its count `|𝓜*| ≤ n · ((k / c_w)(k / c_a) + k)`, and the slowdown form at `t = c_w` is its slowdown form. -/
theorem coveringBoundTwoCosts_of_priced (h : Guarantees.CoveringBoundPriced) : Guarantees.CoveringBoundTwoCosts := by
  classical
  intro d B s k D cw ca n I hkD hca hhard hcov
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hcw : 0 < cw := by
    by_contra h'
    nlinarith [not_lt.1 h', (Nat.cast_nonneg D : (0 : ℝ) ≤ D)]
  have hdepth : ∀ W : Finset (Fin B.nw), D ≤ W.card → (k : ℝ) < ∑ _i ∈ W, cw := by
    intro W hW
    rw [sum_const, nsmul_eq_mul]
    have hW' : (D : ℝ) ≤ W.card := by exact_mod_cast hW
    nlinarith
  obtain ⟨⟨U, hUM, hU, hC⟩, hslow⟩ := h d B s k D (fun _ => cw) ca n I (fun _ => hcw.le) hdepth hca
    ((parameterHardPricedAt_const_iff B s cw).2 hhard) hcov
  set M := Assumptions.chainedBlocksAt d B D ca with hM
  rw [sum_const, nsmul_eq_mul] at hC
  have hcard : (M.card : ℝ) = (M \ U).card + U.card := by
    exact_mod_cast (card_sdiff_add_card_eq_card hUM).symm
  have hX : k * (k / ca) + cw * k = cw * (k / cw * (k / ca) + k) := by
    field_simp
  refine ⟨?_, ?_⟩
  · have h1 : ((M \ U).card : ℝ) ≤ n * (k / cw * (k / ca)) := by
      rw [show (n : ℝ) * (k / cw * (k / ca)) = n * (k * (k / ca)) / cw by ring, le_div_iff₀ hcw]
      linarith
    rw [hcard]
    have e : (n : ℝ) * (k / cw * (k / ca) + k) = n * (k / cw * (k / ca)) + n * k := by ring
    rw [e]
    exact add_le_add h1 hU
  · have hs := hslow cw fun _ => le_rfl
    rw [sum_const, nsmul_eq_mul, hX] at hs
    calc (M.card : ℝ) / d.D.N * s / (k / cw * (k / ca) + k)
        = cw * ((M.card : ℝ) / d.D.N * s) / (cw * (k / cw * (k / ca) + k)) :=
          (mul_div_mul_left _ _ hcw.ne').symm
      _ = (M.card : ℝ) * cw / d.D.N * s / (cw * (k / cw * (k / ca) + k)) := by ring
      _ ≤ s * n / d.D.N := hs

end Nci.SecurityProofs.CoveringPriced
