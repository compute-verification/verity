import FlockSoundness.Discharge.ZkTree.Defs

/-!
# Lazy sampling of committed rounds, strategy by strategy

`ZkCoins.lazy` bounds a committed prover that does not read the leaves by `value` against live coins. Here the bound is
an equality with a mixture of strategies of the model `G` (`ErD g G`): on rounds whose blocks are uniform and whose
salts are drawn, given the block `b`, from `κ i b`, a committed strategy `s` succeeds exactly as often as the mixture
over `r : I → Blk × (Blk → Salt)` of `toStrat h s r`, which at round `i`'s coin `c` takes the rest of the block from the
rest of `(r i).1`'s, and the salt `(r i).2 b` of the block `b` it makes. Each `(r i).1` is uniform and each `(r i).2 b`
drawn from `κ i b` (`lazy_mix`), provided no path opens a round twice.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins Finset Function
open scoped ENNReal

variable {I Blk Salt α : Type}

/-- **The model strategy of a committed strategy** at the randomness `r`: at round `i`'s coin `c`, the block
`d.e (c, rest)` with the rest of `(r i).1`'s block, and that block's salt `(r i).2 b`. -/
def toStrat : {g : TGame I Blk α} → {G : Game α} → ErD g G → TGame.CStrat Salt g → (I → Blk × (Blk → Salt)) →
    Strategy G
  | _, _, .ret _, _, _ => PUnit.unit
  | _, _, .send _ f, s, r => ⟨s.1, toStrat (f s.1) s.2 r⟩
  | _, _, @ErD.coin _ _ _ i C iF iN d k k' f, s, r => fun c =>
      toStrat (f c (d.e.symm (r i).1).2) (s (d.e (c, (d.e.symm (r i).1).2)) ((r i).2 (d.e (c, (d.e.symm (r i).1).2)))) r
  | _, _, @ErD.fresh _ _ _ C iF iN k k' f, s, r => fun c => toStrat (f c) (s c) r

variable [DecidableEq I]

theorem toStrat_update {i : I} (z : Blk × (Blk → Salt)) :
    ∀ {g : TGame I Blk α} {G : Game α} (h : ErD g G), g.NoRead i → ∀ (s : TGame.CStrat Salt g) (r : I → Blk × (Blk → Salt)),
      toStrat h s (update r i z) = toStrat h s r
  | _, _, .ret _, _, _, _ => rfl
  | _, _, .send _ f, hn, s, r => by
      show (⟨s.1, toStrat (f s.1) s.2 (update r i z)⟩ : Strategy (Game.send _ _)) = ⟨s.1, toStrat (f s.1) s.2 r⟩
      rw [toStrat_update z (f s.1) (hn s.1) s.2 r]
  | _, _, @ErD.coin _ _ _ j C iF iN d k k' f, hn, s, r => by
      funext c
      show toStrat (f c (d.e.symm (update r i z j).1).2) _ (update r i z) = toStrat (f c (d.e.symm (r j).1).2) _ r
      rw [update_of_ne hn.1]
      exact toStrat_update z (f c _) (hn.2 _) _ r
  | _, _, @ErD.fresh _ _ _ C iF iN k k' f, hn, s, r => by
      funext c
      exact toStrat_update z (f c) (hn c) (s c) r

section Weights

variable {J Zt : Type} [Fintype J] [DecidableEq J] [Fintype Zt]

/-- **The marginal of a product law** at coordinate `k`. -/
theorem sum_W_marg {μ : J → Zt → ℝ≥0∞} (hμ : ∀ j, ∑ z, μ j z = 1) (k : J) (F : Zt → ℝ≥0∞) :
    ∑ ω, W μ ω * F (ω k) = ∑ z, μ k z * F z := by
  rw [resample (hμ k) (fun ω => F (ω k))]
  simp only [update_self]
  rw [← Finset.sum_mul, sum_W_one hμ, one_mul]

/-- **Pulling out coordinate `k`** of a product law, under a function that reads `ω k` only through its first
argument. -/
theorem sum_W_pull {μ : J → Zt → ℝ≥0∞} (hμ : ∀ j, ∑ z, μ j z = 1) (k : J) (F : Zt → (J → Zt) → ℝ≥0∞)
    (hF : ∀ z ω z', F z (update ω k z') = F z ω) :
    ∑ ω, W μ ω * F (ω k) ω = ∑ z, μ k z * ∑ ω, W μ ω * F z ω := by
  rw [resample (hμ k) (fun ω => F (ω k) ω)]
  simp only [update_self, hF]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun z _ => Finset.sum_congr rfl fun ω _ => ?_
  ring

end Weights

variable [Fintype I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk] [Fintype Salt]

/-- Round `j`'s law when its salt is drawn from `κ j b` given its uniform block `b`. -/
noncomputable def wtK (κ : I → Blk → Salt → ℝ≥0∞) (j : I) (z : Blk × Salt) : ℝ≥0∞ :=
  (Fintype.card Blk : ℝ≥0∞)⁻¹ * κ j z.1 z.2

/-- The law of `r j`: a uniform block, and for every block `b` a salt from `κ j b`, independently. -/
noncomputable def wtR (κ : I → Blk → Salt → ℝ≥0∞) (j : I) (z : Blk × (Blk → Salt)) : ℝ≥0∞ :=
  (Fintype.card Blk : ℝ≥0∞)⁻¹ * W (κ j) z.2

omit [DecidableEq I] [Fintype I] [DecidableEq Blk] in
theorem sum_wtK {κ : I → Blk → Salt → ℝ≥0∞} (hκ : ∀ j b, ∑ y, κ j b y = 1) (j : I) : ∑ z, wtK κ j z = 1 := by
  simp only [wtK, Fintype.sum_prod_type, ← Finset.mul_sum, hκ, mul_one, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul]
  exact ENNReal.inv_mul_cancel (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := Blk))) (ENNReal.natCast_ne_top _)

omit [DecidableEq I] [Fintype I] in
theorem sum_wtR {κ : I → Blk → Salt → ℝ≥0∞} (hκ : ∀ j b, ∑ y, κ j b y = 1) (j : I) : ∑ z, wtR κ j z = 1 := by
  simp only [wtR, Fintype.sum_prod_type, ← Finset.mul_sum, sum_W_one (hκ j), mul_one, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul]
  exact ENNReal.inv_mul_cancel (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := Blk))) (ENNReal.natCast_ne_top _)

/-- **Lazy sampling, strategy by strategy.** On rounds with uniform blocks and salts drawn from `κ` given the block, a
committed strategy `s` of `g` succeeds exactly as often as the mixture of the model strategies `toStrat h s r`, `r`
drawn from `wtR κ`, when no path of `g` opens a round twice. -/
theorem lazy_mix (p : α → Prop) {κ : I → Blk → Salt → ℝ≥0∞} (hκ : ∀ j b, ∑ y, κ j b y = 1) :
    ∀ {g : TGame I Blk α} {G : Game α} (h : ErD g G), g.Distinct → ∀ s : TGame.CStrat Salt g,
      ∑ ω, W (wtK κ) ω * TGame.pr p g s ω = ∑ r, W (wtR κ) r * prob p G (toStrat h s r)
  | _, _, .ret a, _, _ => by
      show ∑ ω, W (wtK κ) ω * (@ite _ (p a) (Classical.propDecidable _) (1 : ℝ≥0∞) 0) =
        ∑ r, W (wtR κ) r * (@ite _ (p a) (Classical.propDecidable _) (1 : ℝ≥0∞) 0)
      rw [← Finset.sum_mul, ← Finset.sum_mul, sum_W_one (sum_wtK hκ), sum_W_one (sum_wtR hκ)]
  | _, _, .send _ f, hg, s => lazy_mix p hκ (f s.1) (hg s.1) s.2
  | _, _, @ErD.fresh _ _ _ C iF iN k k' f, hg, s => by
      show ∑ ω, W (wtK κ) ω * avg (fun c => TGame.pr p _ (s c) ω) =
        ∑ r, W (wtR κ) r * avg fun c => prob p _ (toStrat (f c) (s c) r)
      rw [sum_mul_avg, sum_mul_avg]
      exact congrArg avg (funext fun c => lazy_mix p hκ (f c) (hg c) (s c))
  | _, _, @ErD.coin _ _ _ i C iF iN d k k' f, hg, s => by
      obtain ⟨Yv, hY⟩ : ∃ Yv : C → d.Rest → Salt → ℝ≥0∞, ∀ c rest y,
          Yv c rest y = ∑ r, W (wtR κ) r * prob p (k' c) (toStrat (f c rest) (s (d.e (c, rest)) y) r) :=
        ⟨_, fun _ _ _ => rfl⟩
      have hL : ∑ ω, W (wtK κ) ω * TGame.pr p (TGame.coin i k) s ω =
          ∑ c, ∑ rest, ∑ y, (Fintype.card Blk : ℝ≥0∞)⁻¹ * κ i (d.e (c, rest)) y * Yv c rest y := by
        calc ∑ ω, W (wtK κ) ω * TGame.pr p (TGame.coin i k) s ω
            = ∑ ω, W (wtK κ) ω * TGame.pr p (k (ω i).1) (s (ω i).1 (ω i).2) ω := rfl
          _ = ∑ z, wtK κ i z * ∑ ω, W (wtK κ) ω * TGame.pr p (k z.1) (s z.1 z.2) ω :=
              sum_W_pull (sum_wtK hκ) i (fun z ω => TGame.pr p (k z.1) (s z.1 z.2) ω)
                fun z ω z' => TGame.pr_update p i z' (k z.1) (hg z.1).1 _ ω
          _ = ∑ b, ∑ y, (Fintype.card Blk : ℝ≥0∞)⁻¹ * κ i b y *
                ∑ ω, W (wtK κ) ω * TGame.pr p (k b) (s b y) ω := by
              rw [Fintype.sum_prod_type]
              rfl
          _ = ∑ cr : C × d.Rest, ∑ y, (Fintype.card Blk : ℝ≥0∞)⁻¹ * κ i (d.e cr) y *
                ∑ ω, W (wtK κ) ω * TGame.pr p (k (d.e cr)) (s (d.e cr) y) ω :=
              (Equiv.sum_comp d.e (fun b => ∑ y, (Fintype.card Blk : ℝ≥0∞)⁻¹ * κ i b y *
                ∑ ω, W (wtK κ) ω * TGame.pr p (k b) (s b y) ω)).symm
          _ = _ := by
              rw [Fintype.sum_prod_type]
              refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun rest _ =>
                Finset.sum_congr rfl fun y _ => ?_
              rw [hY, lazy_mix p hκ (f c rest) (hg _).2 (s (d.e (c, rest)) y)]
      obtain ⟨Ψ, hΨ⟩ : ∃ Ψ : Blk × (Blk → Salt) → (I → Blk × (Blk → Salt)) → ℝ≥0∞, ∀ z r,
          Ψ z r = avg fun c => prob p (k' c) (toStrat (f c (d.e.symm z.1).2)
            (s (d.e (c, (d.e.symm z.1).2)) (z.2 (d.e (c, (d.e.symm z.1).2)))) r) :=
        ⟨_, fun _ _ => rfl⟩
      have hΨu : ∀ z r z', Ψ z (update r i z') = Ψ z r := fun z r z' => by
        rw [hΨ, hΨ]
        congr 1
        funext c
        rw [toStrat_update z' (f c _) (hg _).1]
      have hC : (Fintype.card C : ℝ≥0∞) * (Fintype.card C : ℝ≥0∞)⁻¹ = 1 :=
        ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := C))) (ENNReal.natCast_ne_top _)
      have hR : ∑ r, W (wtR κ) r * prob p (Game.coin C k') (toStrat (ErD.coin i d f) s r) =
          ∑ c, ∑ rest, ∑ y, (Fintype.card Blk : ℝ≥0∞)⁻¹ * κ i (d.e (c, rest)) y * Yv c rest y := by
        calc ∑ r, W (wtR κ) r * prob p (Game.coin C k') (toStrat (ErD.coin i d f) s r)
            = ∑ r, W (wtR κ) r * Ψ (r i) r := by
              refine Finset.sum_congr rfl fun r _ => ?_
              rw [hΨ]
              rfl
          _ = ∑ z, wtR κ i z * ∑ r, W (wtR κ) r * Ψ z r := sum_W_pull (sum_wtR hκ) i Ψ hΨu
          _ = ∑ z, wtR κ i z * avg fun c => Yv c (d.e.symm z.1).2 (z.2 (d.e (c, (d.e.symm z.1).2))) := by
              refine Finset.sum_congr rfl fun z _ => ?_
              simp only [hΨ]
              rw [sum_mul_avg]
              congr 2
              funext c
              rw [hY]
          _ = ∑ β, (Fintype.card Blk : ℝ≥0∞)⁻¹ * avg fun c =>
                ∑ y, κ i (d.e (c, (d.e.symm β).2)) y * Yv c (d.e.symm β).2 y := by
              rw [Fintype.sum_prod_type]
              refine Finset.sum_congr rfl fun β _ => ?_
              simp only [wtR, mul_assoc, ← Finset.mul_sum]
              congr 1
              rw [sum_mul_avg]
              congr 1
              funext c
              exact sum_W_marg (hκ i) _ (fun y => Yv c (d.e.symm β).2 y)
          _ = ∑ cr : C × d.Rest, (Fintype.card Blk : ℝ≥0∞)⁻¹ * avg fun c =>
                ∑ y, κ i (d.e (c, cr.2)) y * Yv c cr.2 y := by
              rw [← Equiv.sum_comp d.e]
              simp only [Equiv.symm_apply_apply]
          _ = _ := by
              rw [Fintype.sum_prod_type]
              dsimp only
              rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Finset.sum_comm, Finset.mul_sum]
              refine Finset.sum_congr rfl fun rest _ => ?_
              unfold avg
              rw [Finset.mul_sum, Finset.mul_sum]
              refine Finset.sum_congr rfl fun c _ => ?_
              rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]
              refine Finset.sum_congr rfl fun y _ => ?_
              calc (Fintype.card C : ℝ≥0∞) * ((Fintype.card Blk : ℝ≥0∞)⁻¹ * ((Fintype.card C : ℝ≥0∞)⁻¹ *
                    (κ i (d.e (c, rest)) y * Yv c rest y)))
                  = ((Fintype.card C : ℝ≥0∞) * (Fintype.card C : ℝ≥0∞)⁻¹) *
                    ((Fintype.card Blk : ℝ≥0∞)⁻¹ * κ i (d.e (c, rest)) y * Yv c rest y) := by ring
                _ = _ := by rw [hC, one_mul]
      exact hL.trans hR.symm

end FlockSoundness.Discharge.ZkTree
