import Proofs.Flock.EndToEnd.Period.Statement
import Proofs.Flock.EndToEnd.Period.Union
import Proofs.Flock.EndToEndHidden
import Proofs.Flock.Soundness.Audit.Horizon
import Proofs.Flock.Soundness.Game.Union

/-!
# Proof of the integrity audit over a period, and over a horizon

* `Flock.SecurityProofs.IntegrityPeriod`: `period_union` over the sampling blocks, each sampling block's bound
  `Flock.SecurityProofs.EndToEndHidden` at the strategy that plays it (`SamplingBlock.accepted_wrong_le`), with its
  knowledge term at the worst registration.
* `Flock.SecurityProofs.IntegrityHorizon`: a run in which every sampling block is accepted and the wrong units total at
  least `M` either has a sampling block accepted with a drawn unit wrong, at most `Σ_w (ε_ks,w + δ_link,w)` by
  `period_union` over `Flock.SecurityProofs.EndToEndHiddenDrawn` (`SamplingBlock.accepted_drawnWrong_le`), or every
  sampling block's draw misses its wrong units with at least `M` of them in all, at most `(1 − q)^M` by
  `period_horizon`: sampling block `w`'s draw misses its `|W|` wrong units with probability at most
  `C(n − |W|, kd)/C(n, kd) ≤ (1 − kd/n)^{|W|}` under A3 (`SamplingBlock.drawMiss_le`; `Law.execOS_subset_escape_le`,
  `Law.choose_div_le_one_sub_pow`). At `M = ⌈ln(1/δ)/q⌉`, `(1 − q)^M ≤ e^{−qM} ≤ δ`.
-/

namespace Flock.SecurityProofs

open scoped ENNReal

/-- `(1 − q)^M ≤ e^{−qM} ≤ δ` once `M ≥ ln(1/δ)/q`. -/
theorem one_sub_pow_le_of_log {q δ : ℝ} (hq : 0 < q) (hδ : 0 < δ) {M : ℕ} (hM : Real.log (1 / δ) / q ≤ M) :
    ENNReal.ofReal (1 - q) ^ M ≤ ENNReal.ofReal δ := by
  calc ENNReal.ofReal (1 - q) ^ M ≤ ENNReal.ofReal (Real.exp (-q)) ^ M :=
        pow_le_pow_left₀ zero_le (ENNReal.ofReal_le_ofReal (by linarith [Real.add_one_le_exp (-q)])) M
    _ = ENNReal.ofReal (Real.exp (-q * M)) := by
        rw [← ENNReal.ofReal_pow (Real.exp_pos _).le, ← Real.exp_nat_mul, mul_comm]
    _ ≤ ENNReal.ofReal (Real.exp (-q * (Real.log (1 / δ) / q))) :=
        ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2 (mul_le_mul_of_nonpos_left hM (by linarith)))
    _ = ENNReal.ofReal δ := by
        rw [neg_mul, mul_div_cancel₀ _ hq.ne', one_div, Real.log_inv, neg_neg, Real.exp_log hδ]

end Flock.SecurityProofs

namespace Flock.Guarantees.SamplingBlock

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

variable (w : SamplingBlock)

open Classical in
/-- **One sampling block's `EndToEndHidden`**, at the strategy `τ` that plays it. -/
theorem accepted_wrong_le (hA3 : FlockSoundness.Assumptions.UniformRandomBytes w.Lb w.src) (τ : Strategy w.game)
    (hcr : w.CR τ) :
    prob (fun o => LiveAcceptsZK w.I o ∧ w.K₀ ≤ (w.wrongSet τ).card) w.game τ ≤ w.miss + w.ks + w.link := by
  have h := Flock.SecurityProofs.EndToEndHidden w.n w.Reg w.I w.ht w.D w.hD w.hU w.pub w.hpub w.hpt w.hscope w.kd
    w.hkd w.hdiv w.dj w.S₀ w.d₀ w.y₀ w.wr w.dg w.N w.Ω w.Lb w.src hA3 w.hTags w.hCust w.k w.Rw w.M w.hk w.hRw w.hM
    w.ρ w.hρ w.hr w.t' w.ht' w.K₀ w.hK₀ τ w.qF w.qS hcr.1 hcr.2
  calc prob (fun o => LiveAcceptsZK w.I o ∧ w.K₀ ≤ (w.wrongSet τ).card) w.game τ
      ≤ w.miss + ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS (reg (w.sim τ)) + w.link := h
    _ ≤ w.miss + w.ks + w.link := by
        gcongr
        exact le_iSup (fun R => ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS R) _

open Classical in
/-- **One sampling block's `EndToEndHiddenDrawn`**, at the strategy `τ` that plays it: accepted with a drawn unit
wrong. -/
theorem accepted_drawnWrong_le (τ : Strategy w.game) (hcr : w.CR τ) :
    prob (fun o => LiveAcceptsZK w.I o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ ≤ w.ks + w.link := by
  have h := Flock.SecurityProofs.EndToEndHiddenDrawn w.n w.Reg w.I w.ht w.D w.hD w.hU w.pub w.hpub w.hpt w.hscope
    w.kd w.hdiv w.dj w.S₀ w.d₀ w.y₀ w.wr w.dg w.N w.Ω w.Lb w.src w.hTags w.hCust w.k w.Rw w.M w.hk w.hRw w.hM
    w.ρ w.hρ w.hr w.t' w.ht' τ w.qF w.qS hcr.1 hcr.2
  calc prob (fun o => LiveAcceptsZK w.I o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ
      ≤ ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS (reg (w.sim τ)) + w.link := h
    _ ≤ w.ks + w.link := by
        gcongr
        exact le_iSup (fun R => ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS R) _

private theorem prob_const {α : Type} (Q : Prop) [Decidable Q] (g : Game α) (s : Strategy g) :
    prob (fun _ => Q) g s = if Q then 1 else 0 := by
  by_cases hQ : Q
  · simp only [hQ, if_true]
    exact prob_true g s
  · simp only [hQ, if_false]
    exact prob_false g s

open Classical in
/-- **A sampling block's draw misses its wrong units** with the escape probability of its live draw. -/
theorem prob_drawMiss (B : Finset (Fin w.n)) (τ : Strategy w.game) :
    prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) w.game τ = w.L.escape B := by
  show avg (fun ω => prob _ ((liveOfZJ w.dg (planZHJ w.dj w.y₀) w.N w.wr (w.L.draw ω) τ.1).map
    fun t => (τ.1, w.L.draw ω, t)) (τ.2 ω)) = prCoin fun ω => Disjoint (w.L.draw ω) B
  unfold prCoin
  refine congrArg avg (funext fun ω => ?_)
  rw [prob_map, prob_const (¬ ∃ u ∈ w.L.draw ω, u ∈ B)]
  beta_reduce
  have hiff : (¬ ∃ u ∈ w.L.draw ω, u ∈ B) ↔ Disjoint (w.L.draw ω) B := by simp [Finset.disjoint_left]
  by_cases hd : Disjoint (w.L.draw ω) B
  · rw [if_pos (hiff.2 hd), if_pos hd]
  · rw [if_neg (mt hiff.1 hd), if_neg hd]

open Classical in
/-- **A sampling block's draw misses its wrong units** with probability at most `(1 − kd/n)^{|W|}`, under A3. -/
theorem drawMiss_le (hA3 : FlockSoundness.Assumptions.UniformRandomBytes w.Lb w.src) (B : Finset (Fin w.n))
    (τ : Strategy w.game) :
    prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) w.game τ ≤ (1 - (w.kd : ℝ≥0∞) / w.n) ^ B.card := by
  rw [prob_drawMiss]
  refine (Law.execOS_subset_escape_le w.n w.kd w.hkd w.Lb w.src hA3 B).trans ?_
  rw [Law.subset_escape]
  exact Law.choose_div_le_one_sub_pow w.hkd (by simpa using B.card_le_univ)


/-- A draw covering a fraction `q` of the sampling block's units: `1 − kd/n ≤ 1 − q`. -/
theorem one_sub_le {q : ℝ} (hq : 0 < q) (hqk : q ≤ (w.kd : ℝ) / w.n) :
    1 - (w.kd : ℝ≥0∞) / w.n ≤ ENNReal.ofReal (1 - q) := by
  have hn : (0 : ℝ) < w.n := by
    rcases Nat.eq_zero_or_pos w.n with h0 | h0
    · rw [h0, Nat.cast_zero, div_zero] at hqk
      linarith
    · exact_mod_cast h0
  rw [show (w.kd : ℝ≥0∞) / w.n = ENNReal.ofReal ((w.kd : ℝ) / w.n) by
    rw [ENNReal.ofReal_div_of_pos hn, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]]
  rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_sub _ (div_nonneg (Nat.cast_nonneg _) hn.le)]
  exact ENNReal.ofReal_le_ofReal (by linarith)

/-- **The sampling term at `δ`**: `C(n − K₀, kd)/C(n, kd) ≤ (1 − q)^{K₀} ≤ δ` once `K₀ ≥ ln(1/δ)/q`. -/
theorem miss_le {q δ : ℝ} (hq : 0 < q) (hδ : 0 < δ) (hqk : q ≤ (w.kd : ℝ) / w.n)
    (hK : Real.log (1 / δ) / q ≤ w.K₀) : w.miss ≤ ENNReal.ofReal δ :=
  (Law.choose_div_le_one_sub_pow w.hkd w.hK₀).trans
    ((pow_le_pow_left₀ zero_le (w.one_sub_le hq hqk) _).trans (Flock.SecurityProofs.one_sub_pow_le_of_log hq hδ hK))

end Flock.Guarantees.SamplingBlock

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- **The integrity audit over a period.** -/
theorem IntegrityPeriod : Flock.Guarantees.IntegrityPeriod := by
  intro W T hA3 s hT hL
  have hs : PeriodGood (fun t => (W t).game) (fun t => (W t).CR) 0 T s :=
    periodGood_of_plays _ _ T 0 s fun t τ hp => And.intro (hT t τ hp) (hL t τ hp)
  have h := period_union (fun t => (W t).game) (fun t => (W t).record t)
    (fun r => r.accepted ∧ (W r.idx).K₀ ≤ r.wrong) (fun t τ => t < T ∧ (W t).CR τ)
    (fun t => (W t).miss + (W t).ks + (W t).link)
    (fun t τ hτ => (W t).accepted_wrong_le (hA3 t hτ.1) τ hτ.2) T 0 s
    (periodGood_mono _ T 0 s (fun j τ _ hj hcr => ⟨by omega, hcr⟩) hs)
  simpa [Period.run] using h

/-- **The integrity audit over a period, at `δ`.** -/
theorem IntegrityPeriodLog : Flock.Guarantees.IntegrityPeriodLog := by
  intro W T q δ hq hδ hqt hK hA3 s hT hL
  refine (IntegrityPeriod W T hA3 s hT hL).trans (Finset.sum_le_sum fun t ht => ?_)
  have ht' := Finset.mem_range.1 ht
  gcongr
  exact (W t).miss_le hq hδ (hqt t ht') (hK t ht')

/-- **The integrity audit over a horizon.** -/
theorem IntegrityHorizon : Flock.Guarantees.IntegrityHorizon := by
  intro W T q δ hq hδ hqt hA3 s hT hL
  classical
  have hs : PeriodGood (fun t => (W t).game) (fun t => (W t).CR) 0 T s :=
    periodGood_of_plays _ _ T 0 s fun t τ hp => And.intro (hT t τ hp) (hL t τ hp)
  set M : ℕ := ⌈Real.log (1 / δ) / q⌉₊
  set x : ℝ≥0∞ := ENNReal.ofReal (1 - q)
  have hx : x ≤ 1 := ENNReal.ofReal_le_one.2 (by linarith)
  have hfac : ∀ t < T, ∀ m : ℕ, (1 - ((W t).kd : ℝ≥0∞) / (W t).n) ^ m ≤ x ^ m :=
    fun t ht m => pow_le_pow_left₀ zero_le ((W t).one_sub_le hq (hqt t ht)) m
  have hgood := periodGood_mono _ T 0 s (fun j τ _ hj hcr => (⟨by omega, hcr⟩ : j < T ∧ (W j).CR τ)) hs
  -- every sampling block accepted with a drawn unit wrong
  have hA := period_union (fun t => (W t).game) (fun t => (W t).record t)
    (fun r => r.accepted ∧ r.drawnWrong) (fun t τ => t < T ∧ (W t).CR τ)
    (fun t => (W t).ks + (W t).link)
    (fun t τ hτ => (W t).accepted_drawnWrong_le τ hτ.2) T 0 s hgood
  -- every draw misses, with at least `M` wrong units in all
  have hB := period_horizon (fun t => (W t).game) (fun t => (W t).record t) (fun r => ¬ r.drawnWrong)
    WindowRec.wrong (fun t τ => t < T ∧ (W t).CR τ) x hx
    (fun t τ hτ => ⟨((W t).wrongSet τ).card, fun _ => rfl,
      ((W t).drawMiss_le (hA3 t hτ.1) _ τ).trans (hfac t hτ.1 _)⟩) T 0 s M hgood
  have hxM : x ^ M ≤ ENNReal.ofReal δ := one_sub_pow_le_of_log hq hδ (Nat.le_ceil _)
  calc value (fun l => (∀ r ∈ l, r.accepted) ∧ Real.log (1 / δ) / q ≤ ((l.map WindowRec.wrong).sum : ℝ))
        (Period.run W T s)
      ≤ value (fun l => (∃ r ∈ l, r.accepted ∧ r.drawnWrong) ∨
          ((∀ r ∈ l, ¬ r.drawnWrong) ∧ M ≤ (l.map WindowRec.wrong).sum)) (Period.run W T s) := by
        refine value_mono (fun l hl => ?_) _
        by_cases hd : ∃ r ∈ l, r.drawnWrong
        · obtain ⟨r, hr, hdr⟩ := hd
          exact Or.inl ⟨r, hr, hl.1 r hr, hdr⟩
        · push_neg at hd
          exact Or.inr ⟨hd, Nat.ceil_le.2 hl.2⟩
    _ ≤ value (fun l => ∃ r ∈ l, r.accepted ∧ r.drawnWrong) (Period.run W T s) +
          value (fun l => (∀ r ∈ l, ¬ r.drawnWrong) ∧ M ≤ (l.map WindowRec.wrong).sum) (Period.run W T s) :=
        value_or_le _ _ _
    _ ≤ (∑ t ∈ Finset.range T, ((W t).ks + (W t).link)) + x ^ M := by
        gcongr
        · simpa [Period.run] using hA
        · exact hB
    _ ≤ ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).ks + (W t).link) := by
        rw [add_comm]
        gcongr

end Flock.SecurityProofs
