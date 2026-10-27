import Proofs.Flock.EndToEnd.Joint.Statement
import Proofs.Flock.Soundness.Discharge.Composed.DrawnJ
import Proofs.Flock.Soundness.Audit.HarmBound

/-!
# A sampling block stratified by subcircuit: its drawn units, its draw's miss, and its one budget

* `Composed.execRefinesZK_on_drawn`: `execRefinesZK_on` with an event `E` of the draw on both sides, at one table, as
  `execRefinesZK_tables_drawn` is at `J` tables: both audits' outcomes carry the draw, and the refinement is one
  inequality at each of the law's draws, where `E` is fixed.
* `StratifiedBlock.accepted_drawnWrong_le`: flock-verify accepts the sampling block with a drawn unit wrong with
  probability at most `ε_ks + ε_bind`: `zk_flock_drawnC_reg` at `planAtZKH`, carried to flock-verify's acceptance.
* `StratifiedBlock.drawMiss_le`: the live draw misses a set `B` with probability at most `Law.stratified`'s escape of
  `B`, under A3 (`Law.execOS_escape_le`).
* `StratifiedBlock.within_of_le`: a set in the confidence set, `δ ≤` its escape, is within the one budget:
  `Law.stratified_risk_le` for its spend, and `maxHarm`'s supremum for its ANDs and output bits.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) {L : Law n}

/-- **R11 at `--zk` at one table, with an event of the draw**: flock-verify `--zk` accepts the live audit at a draw in
`E` no likelier than the compiled batched audit, played by `simZ … τ`, accepts at a draw in `E`. -/
theorem execRefinesZK_on_drawn (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (E : Finset (Fin n) → Prop) :
    prob (fun o => LiveAcceptsZK I o ∧ E o.2.1) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤
      prob (fun o => o.1 = true ∧ E o.2) (audit L Reg (batchedSessionZC A Hs (planOf T)))
        (simZ A Hs T I wr opn dg dec one fin capOf N hdec τ) := by
  have hc : ∀ (g : Finset (Fin n) → Reg → Game Bool)
      (h : ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = g),
      prob (fun o => o.1 = true ∧ E o.2) (audit L Reg g)
          (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
            (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)) =
        prob (fun o => o.1 = true ∧ E o.2) (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
          (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ) := by
    intro g h
    subst h
    rfl
  refine le_of_le_of_eq ?_ (hc _ (sessionZC_planOf A Hs T)).symm
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => LiveAcceptsZK I o ∧ E o.2.1)
        ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true ∧ E o.2)
        ((ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditZKOn ZkExec.sessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  classical
  dsimp only
  simp only [prob_and_const]
  split_ifs
  · exact live_le (outz Hs T (L.draw ω) τ.1) _ (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg
      (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N
      (Gz A Hs T (L.draw ω) τ.1) (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1) _
  · exact le_rfl

end FlockSoundness.Discharge.Composed

namespace Flock.Guarantees.StratifiedBlock

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.HiddenDec
  FlockSoundness.Discharge.ZkSession
open scoped ENNReal

variable (w : StratifiedBlock)

open Classical in
/-- **A sampling block accepted with a drawn unit wrong**, at the strategy `τ` that plays it: at most
`ε_ks + ε_bind`. -/
theorem accepted_drawnWrong_le (τ : Strategy w.game)
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH w.dj w.y₀ (w.L.draw ω) (reg (w.sim τ))) j
      (zcStrat execArith hs512 (planAtZKH w.dj w.y₀) (cont (w.sim τ) ω)) w.qF w.qS)
    (hA2 : LinkCRZC execArith hs512 w.P (planAtZKH w.dj w.y₀) (tabAtZKH w.dj w.y₀) w.vb (reg (w.sim τ))
      (cont (w.sim τ)) w.k w.Rw w.M w.t') :
    prob (fun o => LiveAcceptsZK w.I o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ ≤ w.epsKs + w.link := by
  refine (execRefinesZK_on_drawn execArith hs512 _ w.I w.wr _ w.dg _ _ _ _ w.N
    (decodesOnZK_of_runsOnZC execArith hs512 _ w.I w.wr _ _ _ _
      (runsZC_liveH w.I w.ht (Flock.Desc.parse_flat w.hD) (Discharge.Layout.scopeOkH_spec w.hU w.hscope).2.1 w.dj
        w.y₀ w.wr _ tablesAtZKH_live w.hTags w.hCust.1
        (fun ω => drawOkZKH_of_exec w.I (Flock.Desc.parse_flat w.hD) w.sub w.ks w.hks w.St w.hS w.Kd w.pop w.Lb
          w.src w.dj w.hCust.2 ω))) τ
    (fun S => ∃ u ∈ S, u ∈ w.wrongSet τ)).trans ?_
  refine (prob_mono (q := fun o : Bool × Finset (Fin w.n) => o.1 = true ∧ ¬ Disjoint (w.wrongSet τ) o.2)
    (fun o h => ⟨h.1, Finset.not_disjoint_iff.2 (let ⟨u, hu, hw⟩ := h.2; ⟨u, hw, hu⟩)⟩) _ _).trans ?_
  refine (zk_flock_drawnC_reg execArith hs512 w.P (planAtZKH w.dj w.y₀) (tabAtZKH w.dj w.y₀) w.vb
    (loweringSoundZK_of_place execArith _ (planAtZKH w.dj w.y₀) (tabAtZKH w.dj w.y₀)
      (placeZH w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU))
    (Discharge.Layout.keyProg_readsOwn _ w.hU w.n _)
    w.hRw w.hM w.hρ w.hr (jointZC_ksBound execArith execArith_correct hs512 (planAtZKH w.dj w.y₀) w.hk) w.ht'
    (w.sim τ)).trans ?_
  refine add_le_add ((ksAvgZC_le_strict execArith hs512 (planAtZKH w.dj w.y₀) w.k (reg (w.sim τ)) (cont (w.sim τ))
    hCR).trans (le_iSup (fun R => ksAvgStrictZ (planAtZKH w.dj w.y₀) w.L w.k w.qF w.qS R) _)) ?_
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
  show avg (fun ω => prob _ ((liveOf w.dg (mPtsOf w.D.unroll.g (∑ s, w.ks s)) w.N w.wr (w.L.draw ω) τ.1).map
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

end Flock.Guarantees.StratifiedBlock
