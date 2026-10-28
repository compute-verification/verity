import Proofs.Flock.Soundness.ZK.Hiding

/-!
# One committed round's leaf

* `hiding_fun`: T1 (`Claims.Hm96Hiding`) for `[0,1]`-valued tests: from an event bound to a bound on every
  expectation, by the statistical-distance argument (the event is where the real leaf's mass exceeds the ideal's).
* `kern g v`: the salt's posterior given leaf `v`, for the leaf map `g` of a fixed block: uniform on the fiber of `v`
  (uniform on all salts when the fiber is empty, a value the real round never takes).
* `fiber_avg`: redrawing the salt from its posterior given its own leaf leaves a uniform salt's average unchanged.
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Finset
open scoped ENNReal

section Mass

variable {Ω V : Type} [Fintype Ω] [DecidableEq V]

/-- The mass of `v` under the pushforward of a uniform `Ω` by `g`. -/
noncomputable def mass (g : Ω → V) (v : V) : ℝ≥0∞ :=
  ∑ ω, (Fintype.card Ω : ℝ≥0∞)⁻¹ * if g ω = v then 1 else 0

theorem avg_comp_eq (g : Ω → V) (S : Finset V) (hS : ∀ ω, g ω ∈ S) (F : V → ℝ≥0∞) :
    avg (fun ω => F (g ω)) = ∑ v ∈ S, mass g v * F v := by
  unfold avg mass
  simp only [Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun ω _ => ?_
  simp only [mul_assoc, ite_mul, one_mul, zero_mul]
  rw [← Finset.mul_sum, Finset.sum_ite_eq S (g ω) F, if_pos (hS ω)]

theorem mass_ne_top [Nonempty Ω] (g : Ω → V) (v : V) : mass g v ≠ ∞ := by
  unfold mass
  refine ENNReal.sum_ne_top.2 fun ω _ => ENNReal.mul_ne_top
    (ENNReal.inv_ne_top.2 (Nat.cast_ne_zero.2 Fintype.card_ne_zero)) ?_
  split_ifs <;> simp

/-- **From events to `[0,1]`-valued tests**: if no event separates the pushforwards of `g` and `g'` by more than `δ`,
no test with values in `[0,1]` does. -/
theorem avg_le_of_prCoin {Ω' : Type} [Fintype Ω'] [Nonempty Ω'] (g : Ω → V) (g' : Ω' → V) (S : Finset V)
    (hS : ∀ ω, g ω ∈ S) (hS' : ∀ ω, g' ω ∈ S) {δ : ℝ≥0∞}
    (h : ∀ P : V → Prop, prCoin (fun ω => P (g ω)) ≤ prCoin (fun ω => P (g' ω)) + δ)
    (F : V → ℝ≥0∞) (hF : ∀ v, F v ≤ 1) :
    avg (fun ω => F (g ω)) ≤ avg (fun ω => F (g' ω)) + δ := by
  rw [avg_comp_eq g S hS, avg_comp_eq g' S hS']
  have hE := h fun v => mass g' v < mass g v
  beta_reduce at hE
  have e1 : prCoin (fun ω => mass g' (g ω) < mass g (g ω)) =
      ∑ v ∈ S, mass g v * @ite _ (mass g' v < mass g v) (Classical.propDecidable _) 1 0 :=
    avg_comp_eq g S hS fun v => @ite _ (mass g' v < mass g v) (Classical.propDecidable _) 1 0
  have e2 : prCoin (fun ω => mass g' (g' ω) < mass g (g' ω)) =
      ∑ v ∈ S, mass g' v * @ite _ (mass g' v < mass g v) (Classical.propDecidable _) 1 0 :=
    avg_comp_eq g' S hS' fun v => @ite _ (mass g' v < mass g v) (Classical.propDecidable _) 1 0
  rw [e1, e2] at hE
  have split : ∑ v ∈ S, mass g v * @ite _ (mass g' v < mass g v) (Classical.propDecidable _) 1 0 =
      ∑ v ∈ S, @ite _ (mass g' v < mass g v) (Classical.propDecidable _) (mass g v - mass g' v) 0 +
        ∑ v ∈ S, mass g' v * @ite _ (mass g' v < mass g v) (Classical.propDecidable _) 1 0 := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun v _ => ?_
    by_cases hp : mass g' v < mass g v
    · simp only [if_pos hp, mul_one]
      exact (tsub_add_cancel_of_le hp.le).symm
    · simp only [if_neg hp, mul_zero, add_zero]
  have hgap : ∑ v ∈ S, @ite _ (mass g' v < mass g v) (Classical.propDecidable _) (mass g v - mass g' v) 0 ≤ δ := by
    rw [split, add_comm] at hE
    refine (ENNReal.add_le_add_iff_left ?_).1 hE
    exact ENNReal.sum_ne_top.2 fun v _ => ENNReal.mul_ne_top (mass_ne_top g' v) (by split_ifs <;> simp)
  have hpt : ∀ v ∈ S, mass g v * F v ≤ mass g' v * F v +
      @ite _ (mass g' v < mass g v) (Classical.propDecidable _) (mass g v - mass g' v) 0 := by
    intro v _
    by_cases hp : mass g' v < mass g v
    · rw [if_pos hp]
      calc mass g v * F v = (mass g v - mass g' v) * F v + mass g' v * F v := by
            rw [← add_mul, tsub_add_cancel_of_le hp.le]
        _ ≤ (mass g v - mass g' v) * 1 + mass g' v * F v := by gcongr; exact hF v
        _ = mass g' v * F v + (mass g v - mass g' v) := by rw [mul_one, add_comm]
    · rw [if_neg hp, add_zero]
      exact mul_le_mul_left (not_lt.1 hp) _
  calc ∑ v ∈ S, mass g v * F v
      ≤ ∑ v ∈ S, (mass g' v * F v +
          @ite _ (mass g' v < mass g v) (Classical.propDecidable _) (mass g v - mass g' v) 0) :=
        Finset.sum_le_sum hpt
    _ = ∑ v ∈ S, mass g' v * F v +
          ∑ v ∈ S, @ite _ (mass g' v < mass g v) (Classical.propDecidable _) (mass g v - mass g' v) 0 :=
        Finset.sum_add_distrib
    _ ≤ ∑ v ∈ S, mass g' v * F v + δ := by gcongr

end Mass

section Hiding

variable {Salt B Cs D : Type} [Fintype Salt] [Nonempty Salt] [AddCommGroup B] [Fintype B]

/-- **T1 for `[0,1]`-valued tests**, at a leaf shifted by any digest `x`: replacing the leaf `H(x + M·y, c(y))` by the
ideal `H(U, c(y))` changes the expectation of any test with values in `[0,1]` by at most `δ`. -/
theorem hiding_fun (My : Salt → B) (cs : Salt → Cs) {δ : ℝ≥0∞} (hT : Claims.Hm96Hiding My cs δ)
    (H : B × Cs → D) (x : B) (Γ : D → ℝ≥0∞) (hΓ : ∀ d, Γ d ≤ 1) :
    avg (fun y => Γ (H (x + My y, cs y))) ≤ avg (fun q : B × Salt => Γ (H (q.1, cs q.2))) + δ := by
  classical
  have h := avg_le_of_prCoin (fun y => (My y, cs y)) (fun q : B × Salt => (q.1, cs q.2))
    ((univ : Finset B) ×ˢ univ.image cs)
    (fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem cs (mem_univ y)⟩)
    (fun q => mem_product.2 ⟨mem_univ _, mem_image_of_mem cs (mem_univ q.2)⟩) (Assumptions.hm96Hiding_apply hT)
    (fun v => Γ (H (x + v.1, v.2))) (fun v => hΓ _)
  refine h.trans (le_of_eq ?_)
  congr 1
  unfold avg
  exact Fintype.sum_equiv ((Equiv.addLeft x).prodCongr (Equiv.refl Salt)) _ _ (fun _ => rfl)

end Hiding

section Kernel

variable {Salt D : Type} [Fintype Salt]

open Classical in
/-- **The salt's posterior given leaf `v`** under the leaf map `g`: uniform on the fiber of `v`, or on all salts when
the fiber is empty. -/
noncomputable def kern (g : Salt → D) (v : D) (y : Salt) : ℝ≥0∞ :=
  if (univ.filter fun y' => g y' = v).Nonempty then
    (if g y = v then ((univ.filter fun y' => g y' = v).card : ℝ≥0∞)⁻¹ else 0)
  else (Fintype.card Salt : ℝ≥0∞)⁻¹

theorem sum_kern [Nonempty Salt] (g : Salt → D) (v : D) : ∑ y, kern g v y = 1 := by
  classical
  unfold kern
  split_ifs with h
  · rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 h.card_pos.ne') (ENNReal.natCast_ne_top _)
  · rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

theorem sum_kern_le_one [Nonempty Salt] (g : Salt → D) (v : D) (G : Salt → ℝ≥0∞) (hG : ∀ y, G y ≤ 1) :
    ∑ y, kern g v y * G y ≤ 1 :=
  calc ∑ y, kern g v y * G y ≤ ∑ y, kern g v y * 1 := Finset.sum_le_sum fun y _ => mul_le_mul_right (hG y) _
    _ = 1 := by simp only [mul_one, sum_kern]

/-- **The fiber identity**: a uniform salt `y`, with `Φ` read at `y` and its leaf `g y`, has the same total as a uniform
salt whose leaf is kept and which is then redrawn from its posterior given that leaf. -/
theorem fiber_avg (g : Salt → D) (Φ : Salt → D → ℝ≥0∞) :
    ∑ y, Φ y (g y) = ∑ y, ∑ y', kern g (g y) y' * Φ y' (g y) := by
  classical
  have hk : ∀ y y', kern g (g y) y' * Φ y' (g y) =
      if g y = g y' then ((univ.filter fun y'' => g y'' = g y').card : ℝ≥0∞)⁻¹ * Φ y' (g y') else 0 := by
    intro y y'
    unfold kern
    rw [if_pos (show (univ.filter fun y'' => g y'' = g y).Nonempty from ⟨y, mem_filter.2 ⟨mem_univ _, rfl⟩⟩)]
    by_cases h : g y' = g y
    · rw [if_pos h, if_pos h.symm, h]
    · rw [if_neg h, if_neg (Ne.symm h), zero_mul]
  simp only [hk]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y' _ => ?_
  have hc : ((univ.filter fun y'' => g y'' = g y').card : ℝ≥0∞) ≠ 0 := by
    refine Nat.cast_ne_zero.2 (Finset.card_ne_zero.2 ⟨y', ?_⟩)
    simp
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul, ← mul_assoc,
    ENNReal.mul_inv_cancel hc (ENNReal.natCast_ne_top _), one_mul]

end Kernel

end FlockSoundness.Discharge.ZkCoins
