import Proofs.Flock.EndToEnd.Joint.MT.Statement
import Proofs.Flock.Soundness.Discharge.Composed.DrawnJ
import Proofs.Flock.Soundness.Audit.HarmBound

/-!
# A sampling block of `J` tables stratified by subcircuit: its drawn units, its draw's miss, and its one budget

* `StratifiedBlockMT.accepted_drawnWrong_le`: flock-verify accepts the sampling block with a drawn unit wrong with
  probability at most `ε_ks + ε_bind`: `zk_flock_drawnC_reg` at `planZHJ`, carried to flock-verify's acceptance at `J`
  tables by `execRefinesZK_tables_drawn`, as `ZkHidden.zk_session_soundHJ_custody_drawn_reg` is at a subset draw.
* `StratifiedBlockMT.drawMiss_le`: the live draw misses a set `B` with probability at most `Law.stratified`'s escape of
  `B`, under A3 (`Law.execOS_escape_le`).
* `StratifiedBlockMT.within_of_le`: a set in the confidence set, `δ ≤` its escape, is within the one budget.
-/

namespace Flock.Guarantees.StratifiedBlockMT

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.HiddenDec
  FlockSoundness.Discharge.ZkSession
open scoped ENNReal

variable (w : StratifiedBlockMT)

open Classical in
/-- **A sampling block of `J` tables accepted with a drawn unit wrong**, at the strategy `τ` that plays it: at most
`ε_ks + ε_bind`. -/
theorem accepted_drawnWrong_le (τ : Strategy w.game)
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZHJ w.dj w.y₀ (w.L.draw ω) (reg (w.sim τ))) j
      (zcStrat execArith hs512 (planZHJ w.dj w.y₀) (cont (w.sim τ) ω)) w.qF w.qS)
    (hA2 : LinkCRZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ))
      (cont (w.sim τ)) w.k w.Rw w.M w.t') :
    prob (fun o => LiveAcceptsZK w.I o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ ≤ w.epsKs + w.link := by
  refine (execRefinesZK_tables_drawn execArith hs512 (planZHJ w.dj w.y₀) w.I w.wr
    (opnAtZHJ (planZHJ w.dj w.y₀) w.I w.dj) w.dg (decAtZJ (planZHJ w.dj w.y₀)) (fun _ _ _ _ => true)
    (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) w.N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ w.I w.wr _ _ _ _ _
      (runsZCJH_liveA w.I w.ht (Flock.Desc.parse_flat w.hD) (Discharge.Layout.scopeOkH_spec w.hU w.hscope).2.1 w.dj
        w.y₀ w.wr w.L w.hTags w.hCust.1
        fun ω R o hacc => drawOkZKHJ_of_accepts_stratified w.I w.ht (Flock.Desc.parse_flat w.hD) w.sub w.ks w.hks
          w.St w.hS w.Kd w.pop w.Lb w.src w.dj w.hCust.2 w.hCust.1 ω R o hacc)) τ
    (fun S => ∃ u ∈ S, u ∈ w.wrongSet τ)).trans ?_
  refine (prob_mono (q := fun o : Bool × Finset (Fin w.n) => o.1 = true ∧ ¬ Disjoint (w.wrongSet τ) o.2)
    (fun o h => ⟨h.1, Finset.not_disjoint_iff.2 (let ⟨u, hu, hw⟩ := h.2; ⟨u, hw, hu⟩)⟩) _ _).trans ?_
  refine (zk_flock_drawnC_reg execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb
    (loweringSoundZK_of_place execArith _ (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀)
      (placeZHJ w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU))
    (Discharge.Layout.keyProg_readsOwn _ w.hU w.n _)
    w.hRw w.hM w.hρ w.hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ w.dj w.y₀) w.hk) w.ht'
    (w.sim τ)).trans ?_
  refine add_le_add ((ksAvgZC_le_strict execArith hs512 (planZHJ w.dj w.y₀) w.k (reg (w.sim τ)) (cont (w.sim τ))
    hCR).trans (le_iSup (fun R => ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS R) _)) ?_
  exact le_of_eq (if_pos hA2)

private theorem prob_const {α : Type} (Q : Prop) [Decidable Q] (g : Game α) (s : Strategy g) :
    prob (fun _ => Q) g s = if Q then 1 else 0 := by
  by_cases hQ : Q
  · simp only [hQ, if_true]
    exact prob_true g s
  · simp only [hQ, if_false]
    exact prob_false g s

open Classical in
/-- **A sampling block's draw misses a set `B`** with the escape probability of its live draw. -/
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
/-- **A sampling block's draw misses a set `B`** with probability at most `Law.stratified`'s escape of `B`, under A3. -/
theorem drawMiss_le (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (B : Finset (Fin w.n)) (τ : Strategy w.game) :
    prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) w.game τ ≤ w.ideal.escape B := by
  rw [prob_drawMiss]
  exact Law.execOS_escape_le w.sub w.ks w.hks w.St w.hS w.Kd w.pop w.Lb w.src hA3 B

/-- A set in the confidence set holds at most `maxHarm` of any weight. -/
theorem harm_le_maxHarm (h : Fin w.n → ℕ) {δ : ℝ} {B : Finset (Fin w.n)}
    (hB : ENNReal.ofReal δ ≤ w.ideal.escape B) : ((∑ u ∈ B, h u : ℕ) : ℝ) ≤ w.maxHarm h δ :=
  le_ciSup (f := fun B : {B : Finset (Fin w.n) // ENNReal.ofReal δ ≤ w.ideal.escape B} =>
    ((∑ u ∈ B.1, h u : ℕ) : ℝ)) (Set.finite_range _).bddAbove ⟨B, hB⟩

/-- **A set in the confidence set is within the one budget.** -/
theorem within_of_le {δ : ℝ} (hδ : 0 < δ) (ands bits : Fin w.n → ℕ) {B : Finset (Fin w.n)}
    (hB : ENNReal.ofReal δ ≤ w.ideal.escape B) : w.Within δ ands bits B := by
  refine ⟨hB, ?_, w.harm_le_maxHarm ands hB, w.harm_le_maxHarm bits hB⟩
  have h := (Law.stratified_risk_le w.sub w.ks w.hks (ENNReal.ofReal_pos.2 hδ) hB).2
  rw [ENNReal.toReal_ofReal hδ.le, ← one_div] at h
  exact h

end Flock.Guarantees.StratifiedBlockMT
