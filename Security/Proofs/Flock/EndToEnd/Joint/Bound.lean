import Proofs.Flock.EndToEnd.Joint.Drawn
import Proofs.Flock.EndToEnd.Joint.Prod
import Proofs.Flock.Soundness.Game.Union

/-!
# Proof of the joint stratified bound, per sampling block and pooled over a period

* `Flock.SecurityProofs.JointStratified`: if the sampling block's wrong units `B` are in the confidence set
  (`δ ≤` their escape), they are within the one budget (`StratifiedBlock.within_of_le`), so the event is empty.
  Otherwise an accepted sampling block outside the budget has a drawn unit wrong, at most `ε_ks + ε_bind`
  (`StratifiedBlock.accepted_drawnWrong_le`), or its draw misses `B`, at most `B`'s escape `< δ` under A3
  (`StratifiedBlock.drawMiss_le`).
* `Flock.SecurityProofs.JointStratifiedHorizon`: a run in which every sampling block is accepted and the wrong units are
  not within the period's budget either has a sampling block accepted with a drawn unit wrong, at most
  `Σ_t (ε_ks,t + ε_bind,t)` by `period_union`, or every draw misses with the product of the escapes below `δ` (else the
  wrong units would be within the budget, `Stratified.within_of_le`), at most `δ` by `period_prod`.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

namespace BlockRec

variable {W : ℕ → StratifiedBlock}

/-- The run's escapes, multiplied, are at most `exp(−Σ spend)`. -/
theorem escape_prod_le (l : List (BlockRec W)) :
    (l.map BlockRec.escape).prod ≤ ENNReal.ofReal (Real.exp (-(l.map BlockRec.spend).sum)) := by
  induction l with
  | nil => simp
  | cons r l ih =>
      rw [List.map_cons, List.prod_cons, List.map_cons, List.sum_cons, neg_add, Real.exp_add,
        ENNReal.ofReal_mul (Real.exp_pos _).le]
      exact mul_le_mul' (Law.stratified_escape_le_exp (W r.idx).sub (W r.idx).ks (W r.idx).hks r.wrong) ih

end BlockRec

namespace Stratified

variable {W : ℕ → StratifiedBlock}

/-- A run whose escapes multiply to at least `δ` holds at most the period's `maxHarm` of any weight. -/
theorem harm_le_maxHarm (h : (t : ℕ) → Fin (W t).n → ℕ) {δ : ℝ} {l : List (BlockRec W)}
    (hl : ENNReal.ofReal δ ≤ (l.map BlockRec.escape).prod) :
    (((l.map (BlockRec.harm h)).sum : ℕ) : ℝ) ≤ maxHarm W h δ l := by
  have hB' : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, BlockRec.escape l[i.1] := by
    rw [Fin.prod_univ_fun_getElem]; exact hl
  have hB : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, (W l[i].idx).ideal.escape l[i].wrong :=
    hB'.trans_eq (Finset.prod_congr rfl fun _ _ => rfl)
  have e : (l.map (BlockRec.harm h)).sum = ∑ i : Fin l.length, ∑ u ∈ l[i].wrong, h l[i].idx u := by
    rw [← Fin.sum_univ_fun_getElem]; exact Finset.sum_congr rfl fun _ _ => rfl
  have hb : BddAbove (Set.range fun B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).n) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)} => ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ)) :=
    (Set.finite_range _).bddAbove
  rw [e]
  exact le_ciSup hb ⟨fun i => l[i].wrong, hB⟩

/-- **A run whose escapes multiply to at least `δ` is within the period's one budget.** -/
theorem within_of_le {δ : ℝ} (hδ : 0 < δ) (ands bits : (t : ℕ) → Fin (W t).n → ℕ) {l : List (BlockRec W)}
    (hl : ENNReal.ofReal δ ≤ (l.map BlockRec.escape).prod) : Within W δ ands bits l := by
  refine ⟨hl, ?_, harm_le_maxHarm ands hl, harm_le_maxHarm bits hl⟩
  have h1 := hl.trans (BlockRec.escape_prod_le l)
  rw [ENNReal.ofReal_le_ofReal_iff (Real.exp_pos _).le] at h1
  have h2 := Real.log_le_log hδ h1
  rw [Real.log_exp] at h2
  rw [one_div, Real.log_inv]
  linarith

end Stratified

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.HiddenDec
  Flock.Guarantees
open scoped ENNReal

/-- **The joint stratified bound, per sampling block.** -/
theorem JointStratified : Flock.Guarantees.JointStratified := by
  intro w δ ands bits hδ hA3 τ hCR hA2
  classical
  by_cases hB : ENNReal.ofReal δ ≤ w.ideal.escape (w.wrongSet τ)
  · have hW := w.within_of_le hδ ands bits hB
    calc prob (fun o => LiveAcceptsZK w.I o ∧ ¬ w.Within δ ands bits (w.wrongSet τ)) w.game τ
        ≤ prob (fun _ => False) w.game τ := prob_mono (fun _ ho => ho.2 hW) _ _
      _ = 0 := prob_false _ _
      _ ≤ _ := zero_le
  · rw [not_le] at hB
    calc prob (fun o => LiveAcceptsZK w.I o ∧ ¬ w.Within δ ands bits (w.wrongSet τ)) w.game τ
        ≤ prob (fun o => (LiveAcceptsZK w.I o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) ∨ ¬ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ)
            w.game τ := by
          refine prob_mono (fun o ho => ?_) _ _
          by_cases hd : ∃ u ∈ o.2.1, u ∈ w.wrongSet τ
          · exact Or.inl ⟨ho.1, hd⟩
          · exact Or.inr hd
      _ ≤ prob (fun o => LiveAcceptsZK w.I o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ +
            prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ := prob_or_le _ _ _ _
      _ ≤ (w.epsKs + w.link) + ENNReal.ofReal δ :=
          add_le_add (w.accepted_drawnWrong_le τ hCR hA2) ((w.drawMiss_le hA3 (w.wrongSet τ) τ).trans hB.le)
      _ = ENNReal.ofReal δ + w.epsKs + w.link := (add_comm _ _).trans (add_assoc _ _ _).symm

/-- **The joint stratified bound over a period, pooled.** -/
theorem JointStratifiedHorizon : Flock.Guarantees.JointStratifiedHorizon := by
  intro W T δ ands bits hδ hA3 s hT hL
  classical
  let cr : (t : ℕ) → Strategy (W t).game → Prop := fun t τ => (∀ ω j,
      TableCRZ execArith hs512 (planAtZKH (W t).dj (W t).y₀ ((W t).L.draw ω) (reg ((W t).sim τ))) j
        (zcStrat execArith hs512 (planAtZKH (W t).dj (W t).y₀) (cont ((W t).sim τ) ω)) (W t).qF (W t).qS) ∧
      LinkCRZC execArith hs512 (W t).P (planAtZKH (W t).dj (W t).y₀) (tabAtZKH (W t).dj (W t).y₀) (W t).vb
        (reg ((W t).sim τ)) (cont ((W t).sim τ)) (W t).k (W t).Rw (W t).M (W t).t'
  have hs : PeriodGood (fun t => (W t).game) (fun t τ => t < T ∧ cr t τ) 0 T s :=
    periodGood_mono _ T 0 s (fun j τ _ hj hcr => ⟨by omega, hcr⟩)
      (periodGood_of_plays (fun t => (W t).game) cr T 0 s fun t τ hp => And.intro (hT t τ hp) (hL t τ hp))
  -- some sampling block accepted with a drawn unit wrong
  have hA := period_union (fun t => (W t).game) (StratifiedBlock.record W)
    (fun r => r.accepted ∧ r.drawnWrong) (fun t τ => t < T ∧ cr t τ) (fun t => (W t).epsKs + (W t).link)
    (fun t τ hτ => (W t).accepted_drawnWrong_le τ hτ.2.1 hτ.2.2) T 0 s hs
  -- every draw misses, with the product of the escapes below `δ`
  have hB := period_prod (fun t => (W t).game) (StratifiedBlock.record W) (fun r => ¬ r.drawnWrong)
    BlockRec.escape (fun t τ => t < T ∧ cr t τ)
    (fun t τ hτ => ⟨(W t).ideal.escape ((W t).wrongSet τ),
      ne_top_of_le_ne_top ENNReal.one_ne_top (Law.escape_le_one _ _), fun _ => rfl,
      (W t).drawMiss_le (hA3 t hτ.1) ((W t).wrongSet τ) τ⟩) T 0 s (ENNReal.ofReal δ) hs
  calc value (fun l => (∀ r ∈ l, r.accepted) ∧ ¬ Stratified.Within W δ ands bits l) (Stratified.run W T s)
      ≤ value (fun l => (∃ r ∈ l, r.accepted ∧ r.drawnWrong) ∨
          ((∀ r ∈ l, ¬ r.drawnWrong) ∧ (l.map BlockRec.escape).prod < ENNReal.ofReal δ)) (Stratified.run W T s) := by
        refine value_mono (fun l hl => ?_) _
        by_cases hd : ∃ r ∈ l, r.drawnWrong
        · obtain ⟨r, hr, hdr⟩ := hd
          exact Or.inl ⟨r, hr, hl.1 r hr, hdr⟩
        · refine Or.inr ⟨fun r hr hdr => hd ⟨r, hr, hdr⟩, ?_⟩
          by_contra hlt
          exact hl.2 (Stratified.within_of_le hδ ands bits (not_lt.1 hlt))
    _ ≤ value (fun l => ∃ r ∈ l, r.accepted ∧ r.drawnWrong) (Stratified.run W T s) +
          value (fun l => (∀ r ∈ l, ¬ r.drawnWrong) ∧ (l.map BlockRec.escape).prod < ENNReal.ofReal δ)
            (Stratified.run W T s) :=
        value_or_le _ _ _
    _ ≤ (∑ t ∈ Finset.range T, ((W t).epsKs + (W t).link)) + ENNReal.ofReal δ := by
        gcongr
        · simpa [Stratified.run] using hA
        · exact hB
    _ = ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).epsKs + (W t).link) := add_comm _ _

end Flock.SecurityProofs
