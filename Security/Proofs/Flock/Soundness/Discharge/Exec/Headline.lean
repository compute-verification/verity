import Proofs.Flock.Soundness.Headline
import Proofs.Flock.Soundness.Discharge.Exec.Event
import Proofs.Flock.Soundness.Discharge.Exec.SessionL

/-!
# A draft of the headline at the executable (R11)

`Prog.flock_headline` bounds an event `execAccept` of the audit as Lean defines it, and carries the gap between that
audit and flock-verify as `tr`, a rewinding that covers `execAccept ∧ o.1 = false`. This file states the headline the
other way round: on the live audit (`auditLive`, whose outcome carries the session directory), for flock-verify's
verdict (`LiveAccepts I`), with the gap a named refinement `ExecRefines` (R11 at one strategy pair, `Event.lean`).

* `exec_le_of_refines`: a bound on the audit as Lean defines it, for its acceptance conjoined with a fact `Q` fixed by
  its strategy, carries to the live event conjoined with `Q`, plus the refinement's live event `bad`.
* `ofNever_finderCR`: `TreeRewind.ofNever`'s finder finds nothing, so `SHA512CRStrict` holds for it at every budget.
* `Prog.flock_headline_exec`: `Prog.flock_headline` at `execAccept := (·.1 = true)`, `tr := TreeRewind.ofNever`
  (`δ_tree = 0`), carried to the live audit by `exec_le_of_refines`. Its hypotheses are `Prog.flock_headline`'s, with
  `pp`, `hConst`, `hZero`, `lay`, `rs` and `hHm` still open, and `hRef` in place of `tr` and `hT`.
* `flock_headline_exec_L`: the same carry for the salted session `batchedSessionL`, from a bound on it.

**What this does not close.** `Prog.flock_headline` is about `batchedSession`, which checks openings from the unsalted
leaf `H(encode row)`. Every `hm96Rows` statement commits its proof rows with `hm96-sha512/v1` leaves, so flock-verify
accepts an honest session that `batchedSession` rejects, and `hRef` against `batchedSession` holds only with a `bad`
as likely as acceptance itself. For those statements the refinement targets `batchedSessionL H512 enc512 leaf512 plan`
(`SessionL.lean`; under a tag without hidden outputs, each accepted table is one accepting run of `tableAfterL`,
`accepts_tables_hm96`), and what is open
is the headline restated there: knowledge soundness and the link theorem over `Merkle.Leaf`, the bound
`flock_headline_exec_L` takes as `hB`. The other open half of R11 is `hRef` itself: a strategy of the audit built from
the live prover, the challenger's per-round coins (A6) as the audit's coins, and the live event `bad` bounded.
-/

namespace FlockSoundness.Discharge.Exec

open Game Audit Model Accounting Refine Partition
open scoped ENNReal

variable {n : ℕ}

open Classical in
/-- **A bound on the audit as Lean defines it carries to the live audit.** If `σ` accepts `Gm` with the fact `Q`
(fixed by the strategies, such as "at least `K` units of `σ`'s committed values are wrong") with probability at most
`B`, and `τ`'s event `acc` refines `σ`'s acceptance up to `bad`, then `acc ∧ Q` has probability at most `B` plus
`bad`'s. -/
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

/-- **`TreeRewind.ofNever`'s finder finds nothing**: it opens no position, so it never outputs a collision, and
`SHA512CRStrict` holds for it at every budget, with no assumption. -/
theorem ofNever_finderCR {D β : Type} (H : List UInt8 → D) {G : Game β} {S : Strategy G} {bad : β → Prop}
    (h : ∀ b, ¬ bad b) (q : ℝ) : (StrictCR.TreeRewind.ofNever (H := H) (G := G) (S := S) h).finder.CR H q := by
  intro _
  have h0 : prob (fun b => ∃ x y, id b = some (x, y) ∧ x ≠ y ∧ H x = H y)
      (StrictCR.TreeRewind.ofNever (H := H) (G := G) (S := S) h).finder.g
      (StrictCR.TreeRewind.ofNever (H := H) (G := G) (S := S) h).finder.s = 0 := by
    simp [StrictCR.TreeRewind.finder, StrictCR.TreeRewind.ofNever, StrictCR.twoRuns, StrictCR.conflictPick,
      StrictCR.onePick, StrictCR.collOf, Game.bind, Game.map, prob]
  rw [h0, ENNReal.toReal_zero]
  positivity

variable {N : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Draft: the headline at the executable.** Let the verifier draw the units as in `Prog.flock_headline` (`Lx`, the
live stratified draw under A3), and let the live audit `auditLive Lx Reg live`, played by the prover `τ`, refine the
audit with `batchedSession` played by `σ`, up to the live event `bad` (`hRef`). If `σ`'s committed values have at least
`K` wrong units, then flock-verify accepts with probability at most
`miss(K) + ε_ks + δ_link + Pr[bad]`, the headline's bound at `δ_tree = 0` plus the refinement's event. `pp`, `hConst`,
`hZero`, `lay`, `rs` and `hHm` are open as in `Prog.flock_headline`; `hRef` is R11. For `hm96Rows` statements `hRef`
needs the salted session in place of `batchedSession` (the module docstring, `flock_headline_exec_L`). -/
theorem _root_.FlockSoundness.Prog.flock_headline_exec
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
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
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live))
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSession H512 enc512 plan)))
    {bad : Reg × Finset (Fin n) × Transcript → Prop}
    (hRef : ExecRefines (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ
      (LiveAccepts I) bad
      (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ)
    (hCR : (p.partition outs).LinkCR H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H512 enc512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStrat H512 enc512 plan (cont σ ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros ((p.partition outs).Xplur H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        prob bad (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := p.flock_headline outs plan tab (execAccept := fun o => o.1 = true) σs ks hks St hS Kd pop Lb src hA3 pp
    ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K σ hCR hKS (StrictCR.TreeRewind.ofNever hnever)
    (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  exact exec_le_of_refines hRef _ hH

/-- **Draft: the headline at the executable, for the salted session.** For `hm96Rows` statements the audit flock-verify
refines is the one with `batchedSessionL H512 enc512 leaf512 plan`. A bound `B` on its acceptance with at least `K`
wrong units of `σ`'s committed values (`hB`, the headline restated over `Merkle.Leaf`, which is open) carries to
flock-verify's acceptance in the live audit, plus the refinement's event. -/
theorem flock_headline_exec_L {L : Law n} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (I : Inputs) (live : Finset (Fin n) → Reg → Game Transcript) (τ : Strategy (auditLive L Reg live))
    (σ : Strategy (audit L Reg (batchedSessionL H512 enc512 leaf512 plan)))
    {bad : Reg × Finset (Fin n) × Transcript → Prop}
    (hRef : ExecRefines (auditLive L Reg live) τ (LiveAccepts I) bad
      (audit L Reg (batchedSessionL H512 enc512 leaf512 plan)) σ)
    (Q : Prop) {B : ℝ≥0∞}
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionL H512 enc512 leaf512 plan)) σ ≤ B) :
    prob (fun o => LiveAccepts I o ∧ Q) (auditLive L Reg live) τ ≤ B + prob bad (auditLive L Reg live) τ :=
  exec_le_of_refines hRef Q hB

end FlockSoundness.Discharge.Exec
