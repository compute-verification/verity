import Proofs.Flock.Soundness.Binding.E2E
import Proofs.Flock.Soundness.ExecDrawOS

/-!
# The headline: the program's rows, SHA-512 throughout, the live draw

`Prog.flock_headline` is the one end-to-end statement the others specialize into. It combines:
* the circuit of the statement's rows (`p.circuit outs`, `Prog.circuit`), each drawn unit's place from its table's
  statement (`ProgPlaces`);
* SHA-512 (`Refine.H512`, `Refine.enc512`) as the session's hash and the rows' commit hash, with the rows' value binding
  (`flock_e2e_count_hm96`);
* the live draw (`Law.execOS`, `Flock.Draw.drawOS`'s stratified draw on a byte source), its count curve at most
  `Law.stratified`'s under A3 (`Law.execOS_miss_le`).

Its cryptographic hypotheses are instances of three assumptions, each for the finders built from the prover bounded:
* `SHA512CRStrict` (`cr/sha-512`) through `hKS` (`StrictCR.TableCR`) and `hT` (`StrictCR.Finder.CR`);
* `SHA512CRExpected` (`ecr/sha-512`) through `hCR` (`LinkCR`);
* A3 (`uniform/io-getrandombytes`) through `hA3` (`UniformRandomBytes`).

The others are the named obligations `assumption-notes/e2e-checklist.md` lists: W6 and `Layout.Aliased` (`pp.placed`,
`pp.aliased`), `HmRowComputes` (`hHm`), the layout's facts (`lay`, `rs`, in place of `vb`), the serving roots' rewinding
(`tr`, in place of `hExec`; `TreeRewind.ofNever` builds it from `hExec` at `δ_tree = 0`), `hConst` and `hZero`. The rest
(`hks`, `hS`, `hk`, `hRw`, `hM`, `hρ`, `hr`, `ht`) are numeric conditions and the strata's match. The budgets must bound
their finders' real SHA-512 evaluations, `qF, qT ≥ 2t′ + 2v` and `qS ≥ t′ + v` for `t′` one session run's and `v` one
opening's verification, which Lean doesn't check (`ASSUMPTIONS.md`).

It bounds wrong units of the rows' circuit. That a correct unit computes a Definition additionally needs the unit's
certificate (`RowsCert`, `IsRowsUnit.computes_of_cert`) and the Definition's Boolean semantics in Lean.

Its proof coins are the model's: uniform, drawn each round after the prover's message. That the challenger's round
coins are those is A6 (`uniform/flock-coin-server`, `ASSUMPTIONS.md`), which the model embodies rather than a hypothesis
states, and which covers only per-round OS coins: the circuit prover's seed-derived coins are outside the headline.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder Refine Partition
open scoped ENNReal

/-- SHA-512's digests are inhabited, as the session's tables ask of their digests. -/
instance digInhabited : Inhabited Dig := ⟨H512 []⟩

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **The headline: the count curve on the statement's rows, under SHA-512, at the live draw.** Let the verifier draw
the units with `Flock.Draw.drawOS`'s stratified law (strata `St`, matching `Law.stratified σs ks` by `hS`) on a byte
source `src` that A3 makes uniform (`hA3`), and run the batched session with SHA-512 leaves, nodes and rows. If
`execAccept` holds, the probability that at least `K` units of the program's rows circuit are wrong at the committed
values (1 on `ones`, 0 on `zeros`, the plurality of the prover's openings elsewhere) is at most
`miss(K) + ε_ks + δ_link + δ_tree`:
* `miss(K)`, `Law.stratified`'s count curve, which `stratified_miss_eq_greedy` computes, under A3 (`hA3`);
* `ε_ks = ksAvgStrict`, under `SHA512CRStrict` (`cr/sha-512`) for each table's knowledge finders at budgets `qF`, `qS`
  (`hKS`);
* `δ_link = linkBoundE`, under `SHA512CRExpected` (`ecr/sha-512`) for the prover's link finders (`hCR`);
* `δ_tree = TreeRewind.bound qT`, under `SHA512CRStrict` (`cr/sha-512`) for the serving roots' rewinding finder (`hT`). -/
theorem _root_.FlockSoundness.Prog.flock_headline
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
      (batchedSession H512 enc512 plan)))
    (hCR : (p.partition outs).LinkCR H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H512 enc512
      (plan ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStrat H512 enc512 plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H512
      (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H512 qT) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros ((p.partition outs).Xplur H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (Binding.flock_e2e_count_hm96 (p.partition outs) H512 enc512 plan tab pp.derived ones hConst zeros hZero lay rs hHm
    hk hRw hM hρ hr ht K σ hCR hKS tr hT).trans
    (add_le_add (add_le_add (add_le_add (Law.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K) le_rfl) le_rfl) le_rfl)

end FlockSoundness.Audit
