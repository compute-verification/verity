import Proofs.Flock.Soundness.Discharge.ZkCoins.Weights

/-!
# Lazy sampling of committed rounds

A committed prover whose strategy does not depend on the rounds' leaves (`s : CStrat Salt g` fixed) and whose rounds
are drawn from a product law with uniform blocks (`μ j`'s block marginal is uniform, whatever the salts' law) does no
better than the optimal prover against live coins, provided no path opens a round twice (`lazy`): each round's block
is then drawn when it is opened.
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Finset Function
open scoped ENNReal

namespace TGame

variable {I Blk Salt α : Type}

theorem pr_le_one (p : α → Prop) : ∀ (g : TGame I Blk α) (s : CStrat Salt g) (ω : I → Blk × Salt), pr p g s ω ≤ 1
  | .ret a, _, _ => by
      show (@ite _ (p a) (Classical.propDecidable _) (1 : ℝ≥0∞) 0) ≤ 1
      split_ifs <;> simp
  | .send _ k, s, ω => pr_le_one p (k s.1) s.2 ω
  | .coin i k, s, ω => pr_le_one p (k (ω i).1) (s (ω i).1 (ω i).2) ω
  | @TGame.fresh _ _ _ C iF iN k, s, ω => by
      show (@avg C iF fun c => pr p (k c) (s c) ω) ≤ 1
      calc (@avg C iF fun c => pr p (k c) (s c) ω) ≤ @avg C iF fun _ => (1 : ℝ≥0∞) :=
            avg_mono fun c => pr_le_one p (k c) (s c) ω
        _ = 1 := avg_const 1

variable [DecidableEq I]

theorem pr_update (p : α → Prop) (i : I) (z : Blk × Salt) :
    ∀ (g : TGame I Blk α), g.NoRead i → ∀ (s : CStrat Salt g) (ω : I → Blk × Salt),
      pr p g s (update ω i z) = pr p g s ω
  | .ret _, _, _, _ => rfl
  | .send _ k, h, s, ω => pr_update p i z (k s.1) (h s.1) s.2 ω
  | .coin j k, h, s, ω => by
      show pr p (k (update ω i z j).1) (s (update ω i z j).1 (update ω i z j).2) (update ω i z) =
        pr p (k (ω j).1) (s (ω j).1 (ω j).2) ω
      rw [update_of_ne h.1]
      exact pr_update p i z (k (ω j).1) (h.2 _) _ ω
  | @TGame.fresh _ _ _ C iF iN k, h, s, ω => by
      show (@avg C iF fun c => pr p (k c) (s c) (update ω i z)) = @avg C iF fun c => pr p (k c) (s c) ω
      congr 1
      funext c
      exact pr_update p i z (k c) (h c) (s c) ω

end TGame

variable {I Blk Salt α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [Fintype Salt]

theorem sum_uniform_blocks {μ : I → Blk × Salt → ℝ≥0∞}
    (hμ : ∀ j b, ∑ y, μ j (b, y) = (Fintype.card Blk : ℝ≥0∞)⁻¹) (j : I) : ∑ z, μ j z = 1 := by
  rw [Fintype.sum_prod_type]
  simp only [hμ, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  exact ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- **Lazy sampling**: on rounds drawn from a product law whose block marginals are uniform, a strategy fixed in
advance succeeds with probability at most the optimal prover's against live coins, when no path opens a round
twice. -/
theorem lazy (p : α → Prop) (μ : I → Blk × Salt → ℝ≥0∞)
    (hμ : ∀ j b, ∑ y, μ j (b, y) = (Fintype.card Blk : ℝ≥0∞)⁻¹) :
    ∀ (g : TGame I Blk α), g.Distinct → ∀ s : TGame.CStrat Salt g,
      ∑ ω, W μ ω * TGame.pr p g s ω ≤ value p g.erase := by
  have h1 := sum_uniform_blocks hμ
  intro g
  induction g with
  | ret a =>
    intro _ s
    show ∑ ω, W μ ω * (@ite _ (p a) (Classical.propDecidable _) (1 : ℝ≥0∞) 0) ≤
      @ite _ (p a) (Classical.propDecidable _) (1 : ℝ≥0∞) 0
    refine le_of_eq ?_
    rw [← Finset.sum_mul, sum_W_one h1, one_mul]
  | send M k ih =>
    intro hg s
    show ∑ ω, W μ ω * TGame.pr p (k s.1) s.2 ω ≤ value p (.send M fun m => (k m).erase)
    rw [value_send]
    exact (ih s.1 (hg s.1) s.2).trans (le_iSup (fun m => value p (k m).erase) s.1)
  | coin i k ih =>
    intro hg s
    have hstep : ∀ ω z, TGame.pr p (TGame.coin i k) s (update ω i z) = TGame.pr p (k z.1) (s z.1 z.2) ω := by
      intro ω z
      show TGame.pr p (k (update ω i z i).1) (s (update ω i z i).1 (update ω i z i).2) (update ω i z) = _
      rw [update_self]
      exact TGame.pr_update p i z (k z.1) (hg z.1).1 _ ω
    rw [resample (h1 i) (TGame.pr p (TGame.coin i k) s)]
    simp only [hstep]
    calc ∑ ω, W μ ω * ∑ z, μ i z * TGame.pr p (k z.1) (s z.1 z.2) ω
        = ∑ z, μ i z * ∑ ω, W μ ω * TGame.pr p (k z.1) (s z.1 z.2) ω := by
          simp only [Finset.mul_sum]
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun z _ => Finset.sum_congr rfl fun ω _ => ?_
          ring
      _ ≤ ∑ z, μ i z * value p (k z.1).erase :=
          Finset.sum_le_sum fun z _ => mul_le_mul_right (ih z.1 (hg z.1).2 _) _
      _ = value p (TGame.coin i k).erase := by
          show _ = value p (.coin Blk fun b => (k b).erase)
          rw [value_coin, Fintype.sum_prod_type]
          refine Finset.sum_congr rfl fun b _ => ?_
          show ∑ y, μ i (b, y) * value p (k b).erase = _
          rw [← Finset.sum_mul, hμ]
  | @fresh C iF iN k ih =>
    intro hg s
    show ∑ ω, W μ ω * (@avg C iF fun c => TGame.pr p (k c) (s c) ω) ≤
      value p (@Game.coin _ C iF iN fun c => (k c).erase)
    rw [value_coin]
    unfold avg
    calc ∑ ω, W μ ω * ∑ c, (Fintype.card C : ℝ≥0∞)⁻¹ * TGame.pr p (k c) (s c) ω
        = ∑ c, (Fintype.card C : ℝ≥0∞)⁻¹ * ∑ ω, W μ ω * TGame.pr p (k c) (s c) ω := by
          simp only [Finset.mul_sum]
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun ω _ => ?_
          ring
      _ ≤ ∑ c, (Fintype.card C : ℝ≥0∞)⁻¹ * value p (k c).erase :=
          Finset.sum_le_sum fun c _ => mul_le_mul_right (ih c (hg c) (s c)) _

end FlockSoundness.Discharge.ZkCoins
