import Proofs.Flock.Soundness.Discharge.DrawWords.Prob

/-!
# R11 at the `good` draws, and the headline with the decoder needed only at the draws of `∑ s, k s` units

* `execRefines_where`: at a draw `good` admits the simulated session is `simAuditOn`'s, and `live_le_sessionBL` bounds
  the live acceptance as in `execRefines_on`. At a draw it refuses the live event `¬ good` has probability one.
* `decodesWhere_of_on`: `DecodesOn` at the one-outcome law of a `good` draw.
* `mPts_at_card`: `setupH_mPts_of_proved` at a draw file that proves exactly its draw's units.
* `headline_card`: `Prog.flock_headline_execL` with `hRef := execRefines_where` at `good S := S.card = ∑ s, k s`. Its
  `Pr[bad]` is the law's `Pr[(L.draw ω).card ≠ ∑ s, k s]` (`prob_auditLive_draw`), at most `(∑ s, k s) · 2^-64` under
  A3 (`draw_card_rare`).
-/

namespace FlockSoundness.Discharge.DrawWords

open FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.Law FlockSoundness.Discharge.Exec Finset
open Model Accounting Refine Audit.Partition
open scoped ENNReal

/-- An event of a mapped game that holds on every image or on none has probability one or zero. -/
theorem prob_map_const {α β : Type} (p : β → Prop) (f : α → β) (Q : Prop) [Decidable Q] (hf : ∀ a, p (f a) ↔ Q)
    (g : Game α) (s : Strategy (g.map f)) : prob p (g.map f) s = if Q then 1 else 0 := by
  rw [prob_map]
  by_cases hQ : Q
  · have e : (fun a => p (f a)) = fun _ => True := funext fun a => propext ⟨fun _ => trivial, fun _ => (hf a).2 hQ⟩
    rw [e, ite_eq_left hQ, prob_true]
  · have e : (fun a => p (f a)) = fun _ => False := funext fun a => propext ⟨fun h => hQ ((hf a).1 h), False.elim⟩
    rw [e, ite_eq_right hQ, prob_false]

/-- **An event of the live audit's draw alone has the law's probability.** -/
theorem prob_auditLive_draw {n : ℕ} {L : Law n} {Reg : Type} (live : Finset (Fin n) → Reg → Game Transcript)
    (τ : Strategy (auditLive L Reg live)) (Q : Finset (Fin n) → Prop) :
    prob (fun o => Q o.2.1) (auditLive L Reg live) τ = prCoin (fun ω => Q (L.draw ω)) := by
  classical
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => Q o.2.1) ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) = _
  unfold prCoin avg
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [prob_map_const (fun o : Reg × Finset (Fin n) × Transcript => Q o.2.1) (fun t => (τ.1, L.draw ω, t))
    (Q (L.draw ω)) (fun _ => Iff.rfl)]

theorem execRefines_where : ExecRefinesWhere := by
  intro D Sl _ _ _ mPts n Reg H E Lf T I wr opn dg dec one fin capOf N L good hdec τ
  unfold ExecRefines
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAccepts I) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionL H E Lf (fun S R => [T S R]) (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simAuditWhere H E Lf T I wr opn dg dec one fin capOf N good hdec τ).2 ω) +
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => ¬ good o.2.1) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun ω _ => ?_
  rw [← mul_add]
  refine mul_le_mul_right ?_ _
  by_cases hg : good (L.draw ω)
  · refine le_trans ?_ le_self_add
    unfold simAuditWhere batchedSessionL
    rw [prob_strategy_map, prob_strategy_map, prob_map]
    unfold liveOf
    rw [prob_map]
    refine le_trans (prob_mono (q := fun o => good (L.draw ω) ∧ Accepts I (wr (L.draw ω) τ.1 o))
      (fun a ha => ⟨hg, ha⟩) _ _) ?_
    refine (live_le_sessionBL execArith H E Lf (T (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
      (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (fun o ho => hdec ω ho.1 τ.1 o ho.2)
      _).trans_eq ?_
    congr 1
    funext outs
    classical
    exact propext decide_eq_true_iff.symm
  · classical
    refine le_trans ?_ le_add_self
    rw [prob_map_const (fun o : Reg × Finset (Fin n) × Transcript => ¬ good o.2.1) (fun t => (τ.1, L.draw ω, t))
      (¬ good (L.draw ω)) (fun _ => Iff.rfl), ite_eq_left hg]
    exact prob_le_one _ _ _

theorem decodesWhere_of_on : DecodesWhereOfOn := by
  intro D Sl _ _ _ mPts n Reg H E Lf T I wr opn dec one fin capOf L good h ω hω R
  exact h ({ Ω := Unit, draw := fun _ => L.draw ω } : Law n) (fun _ => hω) () R

theorem mPts_at_card : MPtsAtCard := by
  intro c n I ht hc dj hdj S st h
  exact setupH_mPts_of_proved (by simp only [ht, Bool.false_eq_true, ↓reduceIte]; exact hc) (hdj S) h

theorem headline_card : HeadlineCard := by
  intro N' n p outs Reg mPts m σs ks hks St hS Kd pop Ω _ _ Lb src hA3 hLb T tab pp ones hConst zeros hZero lay rs
    hHm k hk Rw M hRw hM ρ hρ hr t' ht K I wr opn dg dec one fin capOf N hdec τ hCR qF qS hKS
  have h := p.flock_headline_execL outs (fun S R => [T S R]) tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst
    zeros hZero lay rs hHm hk hRw hM hρ hr ht K I (liveOf dg mPts N wr) τ
    (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ)
    (execRefines_where H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ) hCR hKS
  refine h.trans (add_le_add le_rfl ?_)
  rw [prob_auditLive_draw (liveOf dg mPts N wr) τ fun S => ¬ S.card = ∑ s, ks s]
  exact draw_card_rare σs ks hks St hS Kd pop Lb src hA3 hLb

end FlockSoundness.Discharge.DrawWords
