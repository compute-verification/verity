import Proofs.Flock.EndToEnd.Joint.MT.Drawn
import Proofs.Flock.EndToEnd.Joint.Prod
import Proofs.Flock.Soundness.Game.Union

/-!
# Proof of the joint stratified bound over sampling blocks of `J` tables, per sampling block and pooled

The proofs of `JointStratified` and `JointStratifiedHorizon` (`Joint/Bound.lean`), at `StratifiedBlockMT`:

* `Flock.SecurityProofs.JointStratifiedMT`: if the sampling block's wrong units `B` are in the confidence set, they are
  within the one budget (`StratifiedBlockMT.within_of_le`), so the event is empty. Otherwise an accepted sampling block
  outside the budget has a drawn unit wrong, at most `ε_ks + ε_bind` (`StratifiedBlockMT.accepted_drawnWrong_le`), or
  its draw misses `B`, at most `B`'s escape `< δ` under A3 (`StratifiedBlockMT.drawMiss_le`).
* `Flock.SecurityProofs.JointStratifiedHorizonMT`: a run in which every sampling block is accepted and the wrong units
  are not within the period's budget either has a sampling block accepted with a drawn unit wrong, at most
  `Σ_t (ε_ks,t + ε_bind,t)` by `period_union`, or every draw misses with the product of the escapes below `δ`, at most
  `δ` by `period_prod`.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

namespace BlockRecMT

variable {W : ℕ → StratifiedBlockMT}

/-- The run's escapes, multiplied, are at most `exp(−Σ spend)`. -/
theorem escape_prod_le (l : List (BlockRecMT W)) :
    (l.map BlockRecMT.escape).prod ≤ ENNReal.ofReal (Real.exp (-(l.map BlockRecMT.spend).sum)) := by
  induction l with
  | nil => simp
  | cons r l ih =>
      rw [List.map_cons, List.prod_cons, List.map_cons, List.sum_cons, neg_add, Real.exp_add,
        ENNReal.ofReal_mul (Real.exp_pos _).le]
      exact mul_le_mul' (Law.stratified_escape_le_exp (W r.idx).sub (W r.idx).ks (W r.idx).hks r.wrong) ih

end BlockRecMT

namespace StratifiedMT

variable {W : ℕ → StratifiedBlockMT}

/-- A run whose escapes multiply to at least `δ` holds at most the period's `maxHarm` of any weight. -/
theorem harm_le_maxHarm (h : (t : ℕ) → Fin (W t).n → ℕ) {δ : ℝ} {l : List (BlockRecMT W)}
    (hl : ENNReal.ofReal δ ≤ (l.map BlockRecMT.escape).prod) :
    (((l.map (BlockRecMT.harm h)).sum : ℕ) : ℝ) ≤ maxHarm W h δ l := by
  have hB' : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, BlockRecMT.escape l[i.1] := by
    rw [Fin.prod_univ_fun_getElem]; exact hl
  have hB : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, (W l[i].idx).ideal.escape l[i].wrong :=
    hB'.trans_eq (Finset.prod_congr rfl fun _ _ => rfl)
  have e : (l.map (BlockRecMT.harm h)).sum = ∑ i : Fin l.length, ∑ u ∈ l[i].wrong, h l[i].idx u := by
    rw [← Fin.sum_univ_fun_getElem]; exact Finset.sum_congr rfl fun _ _ => rfl
  have hb : BddAbove (Set.range fun B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).n) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)} => ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ)) :=
    (Set.finite_range _).bddAbove
  rw [e]
  exact le_ciSup hb ⟨fun i => l[i].wrong, hB⟩

/-- **A run whose escapes multiply to at least `δ` is within the period's one budget.** -/
theorem within_of_le {δ : ℝ} (hδ : 0 < δ) (ands bits : (t : ℕ) → Fin (W t).n → ℕ) {l : List (BlockRecMT W)}
    (hl : ENNReal.ofReal δ ≤ (l.map BlockRecMT.escape).prod) : Within W δ ands bits l := by
  refine ⟨hl, ?_, harm_le_maxHarm ands hl, harm_le_maxHarm bits hl⟩
  have h1 := hl.trans (BlockRecMT.escape_prod_le l)
  rw [ENNReal.ofReal_le_ofReal_iff (Real.exp_pos _).le] at h1
  have h2 := Real.log_le_log hδ h1
  rw [Real.log_exp] at h2
  rw [one_div, Real.log_inv]
  linarith

end StratifiedMT

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden Flock.Guarantees
open scoped ENNReal

/-- **The joint stratified bound, per sampling block of `J` tables.** -/
theorem JointStratifiedMT : Flock.Guarantees.JointStratifiedMT := by
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

/-- **The joint stratified bound over a period of sampling blocks of `J` tables, pooled.** -/
theorem JointStratifiedHorizonMT : Flock.Guarantees.JointStratifiedHorizonMT := by
  intro W T δ ands bits hδ hA3 s hT hL
  classical
  let cr : (t : ℕ) → Strategy (W t).game → Prop := fun t τ => (∀ ω j,
      TableCRZ execArith hs512 (planZHJ (W t).dj (W t).y₀ ((W t).L.draw ω) (reg ((W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (W t).dj (W t).y₀) (cont ((W t).sim τ) ω)) (W t).qF (W t).qS) ∧
      LinkCRZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
        (reg ((W t).sim τ)) (cont ((W t).sim τ)) (W t).k (W t).Rw (W t).M (W t).t'
  have hs : PeriodGood (fun t => (W t).game) (fun t τ => t < T ∧ cr t τ) 0 T s :=
    periodGood_mono _ T 0 s (fun j τ _ hj hcr => ⟨by omega, hcr⟩)
      (periodGood_of_plays (fun t => (W t).game) cr T 0 s fun t τ hp => And.intro (hT t τ hp) (hL t τ hp))
  have hA := period_union (fun t => (W t).game) (StratifiedBlockMT.record W)
    (fun r => r.accepted ∧ r.drawnWrong) (fun t τ => t < T ∧ cr t τ) (fun t => (W t).epsKs + (W t).link)
    (fun t τ hτ => (W t).accepted_drawnWrong_le τ hτ.2.1 hτ.2.2) T 0 s hs
  have hB := period_prod (fun t => (W t).game) (StratifiedBlockMT.record W) (fun r => ¬ r.drawnWrong)
    BlockRecMT.escape (fun t τ => t < T ∧ cr t τ)
    (fun t τ hτ => ⟨(W t).ideal.escape ((W t).wrongSet τ),
      ne_top_of_le_ne_top ENNReal.one_ne_top (Law.escape_le_one _ _), fun _ => rfl,
      (W t).drawMiss_le (hA3 t hτ.1) ((W t).wrongSet τ) τ⟩) T 0 s (ENNReal.ofReal δ) hs
  calc value (fun l => (∀ r ∈ l, r.accepted) ∧ ¬ StratifiedMT.Within W δ ands bits l) (StratifiedMT.run W T s)
      ≤ value (fun l => (∃ r ∈ l, r.accepted ∧ r.drawnWrong) ∨
          ((∀ r ∈ l, ¬ r.drawnWrong) ∧ (l.map BlockRecMT.escape).prod < ENNReal.ofReal δ))
          (StratifiedMT.run W T s) := by
        refine value_mono (fun l hl => ?_) _
        by_cases hd : ∃ r ∈ l, r.drawnWrong
        · obtain ⟨r, hr, hdr⟩ := hd
          exact Or.inl ⟨r, hr, hl.1 r hr, hdr⟩
        · refine Or.inr ⟨fun r hr hdr => hd ⟨r, hr, hdr⟩, ?_⟩
          by_contra hlt
          exact hl.2 (StratifiedMT.within_of_le hδ ands bits (not_lt.1 hlt))
    _ ≤ value (fun l => ∃ r ∈ l, r.accepted ∧ r.drawnWrong) (StratifiedMT.run W T s) +
          value (fun l => (∀ r ∈ l, ¬ r.drawnWrong) ∧ (l.map BlockRecMT.escape).prod < ENNReal.ofReal δ)
            (StratifiedMT.run W T s) :=
        value_or_le _ _ _
    _ ≤ (∑ t ∈ Finset.range T, ((W t).epsKs + (W t).link)) + ENNReal.ofReal δ := by
        gcongr
        · simpa [StratifiedMT.run] using hA
        · exact hB
    _ = ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).epsKs + (W t).link) := add_comm _ _

end Flock.SecurityProofs
