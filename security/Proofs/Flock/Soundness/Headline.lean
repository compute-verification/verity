module

public import Proofs.Flock.Soundness.Binding.E2E
public import Proofs.Flock.Soundness.ExecSampleOS

/-! `Prog.flock_headline`, the end-to-end statement the others specialize into: the program's rows, SHA-512 throughout,
at the live sample. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder Refine Partition
open scoped ENNReal

instance digInhabited : Inhabited Dig := ⟨H512 []⟩

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- Lean doesn't check that the budgets bound their finders' real SHA-512 evaluations (`qF, qT ≥ 2t′ + 2v`,
`qS ≥ t′ + v`). No theorem ties a unit's rows to its Definition's gate circuit, and the round coins' uniformity (A6) is
the model's, not a hypothesis. -/
theorem _root_.FlockSoundness.Prog.flock_headline
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
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
    (σ : Strategy (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSession H512 enc512 plan)))
    (hCR : (p.partition outs).LinkCR H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H512 enc512
      (plan ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStrat H512 enc512 plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H512
      (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H512 qT) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros ((p.partition outs).Xplur H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (Binding.flock_e2e_count_hm96 (p.partition outs) H512 enc512 plan tab pp.derived ones hConst zeros hZero lay rs hHm
    hk hRw hM hρ hr ht K σ hCR hKS tr hT).trans
    (add_le_add (add_le_add (add_le_add (SamplingDesign.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K) le_rfl) le_rfl) le_rfl)

end FlockSoundness.Audit
