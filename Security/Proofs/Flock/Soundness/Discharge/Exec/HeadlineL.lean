import Proofs.Flock.Soundness.Discharge.Exec.BindingE2EL
import Proofs.Flock.Soundness.Discharge.Exec.Headline

/-!
# The headline over the salted session, and at the executable

`Prog.flock_headline` (`Headline.lean`) restated for the session whose leaves are `hm96-sha512/v1`'s
(`batchedSessionL H512 enc512 leaf512 plan`), the session flock-verify checks for `hm96Rows` statements:

* `Prog.flock_headline_L`: the count curve on the statement's rows, at the live draw, with the same bound as
  `Prog.flock_headline`: `miss(K) + ksAvgStrict + linkBoundE + δ_tree`, under the same assumptions
  (`SHA512CRStrict` through `hKS` and `hT`, `SHA512CRExpected` through `hCR`, A3 through `hA3`) and the same named
  obligations (`pp`, `hConst`, `hZero`, `lay`, `rs`, `hHm`, `tr`). Its knowledge and link terms are
  `ksAvgBE_le_strictL` and `flock_batched_linkSoundEL`, through `flock_e2e_count_hm96L`.
* `Prog.flock_headline_execL`: the headline at the executable for the salted session. Flock-verify's verdict on the
  live audit (`LiveAccepts I`) with at least `K` wrong units of `σ`'s committed values has probability at most
  `miss(K) + ksAvgStrict + linkBoundE + Pr[bad]`, given the refinement `hRef` (R11 at one strategy pair, `Event.lean`)
  of the live audit by the audit with `batchedSessionL H512 enc512 leaf512 plan`. Its hypotheses are
  `Prog.flock_headline_L`'s at `tr := TreeRewind.ofNever` (`δ_tree = 0`), with `hRef` in place of `tr` and `hT`.

What stays open of R11 is `hRef` itself: a strategy `σ` of this audit built from the live prover `τ`, the challenger's
per-round coins (A6) as the audit's coins, and the live event `bad` bounded.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Refine Audit Audit.Partition
open scoped ENNReal

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **The headline over the salted session**: `Prog.flock_headline` for the batched session with `hm96-sha512/v1`
leaves (`batchedSessionL H512 enc512 leaf512 plan`). If `execAccept` holds, the probability that at least `K` units of
the program's rows circuit are wrong at the committed values (1 on `ones`, 0 on `zeros`, the plurality of the prover's
openings elsewhere) is at most `miss(K) + ε_ks + δ_link + δ_tree`:
* `miss(K)`, `Law.stratified`'s count curve, under A3 (`hA3`);
* `ε_ks = ksAvgStrict`, under `SHA512CRStrict` (`cr/sha-512`) for each table's knowledge finders at budgets `qF`, `qS`
  (`hKS`, `TableCRL`);
* `δ_link = linkBoundE`, under `SHA512CRExpected` (`ecr/sha-512`) for the prover's link finders (`hCR`, `LinkCRL`);
* `δ_tree = TreeRewind.bound qT`, under `SHA512CRStrict` (`cr/sha-512`) for the serving roots' rewinding finder (`hT`). -/
theorem _root_.FlockSoundness.Prog.flock_headline_L
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionL H512 enc512 leaf512 plan)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H512
      (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (batchedSessionL H512 enc512 leaf512 plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H512 qT) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionL H512 enc512 leaf512 plan)) σ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (flock_e2e_count_hm96L (p.partition outs) H512 enc512 leaf512 plan tab pp.derived ones hConst zeros hZero lay rs hHm
    hk hRw hM hρ hr ht K σ hCR hKS tr hT).trans
    (add_le_add (add_le_add (add_le_add (Law.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K) le_rfl) le_rfl) le_rfl)

/-- `Prog.flock_headline_L` at rows each under its own encoding (`Binding.HmRowsX`). -/
theorem _root_.FlockSoundness.Prog.flock_headline_LX
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionL H512 enc512 leaf512 plan)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H512
      (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (batchedSessionL H512 enc512 leaf512 plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H512 qT) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionL H512 enc512 leaf512 plan)) σ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (flock_e2e_count_hm96LX (p.partition outs) H512 enc512 leaf512 plan tab pp.derived ones hConst zeros hZero lay rs hHm
    hk hRw hM hρ hr ht K σ hCR hKS tr hT).trans
    (add_le_add (add_le_add (add_le_add (Law.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K) le_rfl) le_rfl) le_rfl)

/-- **The headline at the executable, for the salted session.** Let the verifier draw the units as in
`Prog.flock_headline_L` (the live stratified draw under A3), and let the live audit `auditLive Lx Reg live`, played by
the prover `τ`, refine the audit with `batchedSessionL H512 enc512 leaf512 plan` played by `σ`, up to the live event
`bad` (`hRef`). If `σ`'s committed values have at least `K` wrong units, flock-verify accepts with probability at most
`miss(K) + ε_ks + δ_link + Pr[bad]`. `pp`, `hConst`, `hZero`, `lay`, `rs` and `hHm` are open as in
`Prog.flock_headline`; `hRef` is R11. -/
theorem _root_.FlockSoundness.Prog.flock_headline_execL
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
      (batchedSessionL H512 enc512 leaf512 plan)))
    {bad : Reg × Finset (Fin n) × Transcript → Prop}
    (hRef : ExecRefines (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ
      (LiveAccepts I) bad
      (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (batchedSessionL H512 enc512 leaf512 plan)) σ)
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 plan (cont σ ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        prob bad (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := p.flock_headline_L outs plan tab (execAccept := fun o => o.1 = true) σs ks hks St hS Kd pop Lb src hA3 pp
    ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K σ hCR hKS (StrictCR.TreeRewind.ofNever hnever)
    (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  exact flock_headline_exec_L plan I live τ σ hRef _ hH

/-- `Prog.flock_headline_execL` at rows each under its own encoding (`Binding.HmRowsX`). -/
theorem _root_.FlockSoundness.Prog.flock_headline_execLX
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ)
    (I : Inputs) (live : Finset (Fin n) → Reg → Game Transcript)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live))
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionL H512 enc512 leaf512 plan)))
    {bad : Reg × Finset (Fin n) × Transcript → Prop}
    (hRef : ExecRefines (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ
      (LiveAccepts I) bad
      (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (batchedSessionL H512 enc512 leaf512 plan)) σ)
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 plan (cont σ ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        prob bad (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg live) τ := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := p.flock_headline_LX outs plan tab (execAccept := fun o => o.1 = true) σs ks hks St hS Kd pop Lb src hA3 pp
    ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K σ hCR hKS (StrictCR.TreeRewind.ofNever hnever)
    (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  exact flock_headline_exec_L plan I live τ σ hRef _ hH

end FlockSoundness.Discharge.Exec
