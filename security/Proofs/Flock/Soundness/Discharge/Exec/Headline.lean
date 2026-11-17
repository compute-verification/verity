module

public import Proofs.Flock.Soundness.Headline
public import Proofs.Flock.Soundness.Discharge.Exec.Event
public import Proofs.Flock.Soundness.Discharge.Exec.SessionL

/-! A draft of the headline at the executable, on the live audit with the gap a named refinement `ExecRefines`. It does not
close the refinement for `hm96Rows` statements, whose honest sessions `batchedSession` rejects: those need the salted
session (`flock_headline_exec_L`), and `hRef` itself is not proved. -/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Audit Model Accounting Refine Partition
open scoped ENNReal

variable {n : ℕ}

open Classical in
theorem exec_le_of_refines {β : Type} {G : Game β} {τ : Strategy G} {acc bad : β → Prop}
    {Gm : Game (Bool × Finset (Fin n))} {σ : Strategy Gm} (hRef : ExecRefines G τ acc bad Gm σ) (Q : Prop)
    {B : ℝ≥0∞} (hB : prob (fun o => o.1 = true ∧ Q) Gm σ ≤ B) :
    prob (fun b => acc b ∧ Q) G τ ≤ B + prob bad G τ := by
  rw [prob_and_const]
  rw [prob_and_const] at hB
  by_cases hQ : Q
  · simp only [hQ, ite_true] at hB ⊢
    exact hRef.trans (add_le_add hB le_rfl)
  · simp only [hQ, ite_false, zero_le]

theorem ofNever_finderCR {D β : Type} (H : List UInt8 → D) {G : Game β} {S : Strategy G} {bad : β → Prop}
    (h : ∀ b, ¬ bad b) (q : ℝ) : (StrictCR.TreeRewind.ofNever (H := H) (G := G) (S := S) h).finder.CR H q := by
  intro _
  have h0 : prob (fun b => ∃ x y, id b = some (x, y) ∧ x ≠ y ∧ H x = H y)
      (StrictCR.TreeRewind.ofNever (H := H) (G := G) (S := S) h).finder.g
      (StrictCR.TreeRewind.ofNever (H := H) (G := G) (S := S) h).finder.s = 0 := by
    simp [StrictCR.TreeRewind.finder, StrictCR.TreeRewind.ofNever, StrictCR.twoRuns, StrictCR.conflictPick,
      StrictCR.collOf, Game.bind, Game.map, prob]
  rw [h0, ENNReal.toReal_zero]
  positivity

variable {N : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- Draft. `pp`, `hConst`, `hZero`, `lay`, `rs` and `hHm` are open as in `Prog.flock_headline`, and `hRef` is the
refinement; for `hm96Rows` statements `hRef` needs the salted session in place of `batchedSession`. -/
theorem _root_.FlockSoundness.Prog.flock_headline_exec
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ)
    (I : Inputs) (live : Finset (Fin n) → Reg → Game Transcript)
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live))
    (σ : Strategy (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSession H512 enc512 plan)))
    {bad : Reg × Finset (Fin n) × Transcript → Prop}
    (hRef : ExecRefines (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ
      (LiveAccepts I) bad
      (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ)
    (hCR : (p.partition outs).LinkCR H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H512 enc512
      (plan ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStrat H512 enc512 plan (cont σ ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros ((p.partition outs).Xplur H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        prob bad (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := p.flock_headline outs plan tab (execAccept := fun o => o.1 = true) σs ks hks St hS Kd pop Lb src hA3 pp
    ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K σ hCR hKS (StrictCR.TreeRewind.ofNever hnever)
    (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  exact exec_le_of_refines hRef _ hH

/-- Draft. Its bound `hB`, the headline restated over `Merkle.Leaf`, is open. -/
theorem flock_headline_exec_L {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    {L : SamplingDesign n} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (I : Inputs) (live : Finset (Fin n) → Reg → Game Transcript) (τ : Strategy (auditLive L Reg live))
    (σ : Strategy (audit L Reg (batchedSessionL H512 enc512 Lp plan)))
    {bad : Reg × Finset (Fin n) × Transcript → Prop}
    (hRef : ExecRefines (auditLive L Reg live) τ (LiveAccepts I) bad
      (audit L Reg (batchedSessionL H512 enc512 Lp plan)) σ)
    (Q : Prop) {B : ℝ≥0∞}
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionL H512 enc512 Lp plan)) σ ≤ B) :
    prob (fun o => LiveAccepts I o ∧ Q) (auditLive L Reg live) τ ≤ B + prob bad (auditLive L Reg live) τ :=
  exec_le_of_refines hRef Q hB

end FlockSoundness.Discharge.Exec
