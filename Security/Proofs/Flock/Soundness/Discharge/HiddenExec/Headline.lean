import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted

/-!
# The headline over hidden-output sessions, with every open guarantee named

`Prog.flock_headline_live_on` at the one-table plan of a hidden-output session's inputs (`planAtH`, `tablesAtH`): at a
draw flock-verify sets up under hidden outputs, the table is `stmtOf` of the `Stmt.setupHidden` statement it accepted
(`tableAtH_of_accepts`), and elsewhere no witness satisfies it (`tableAtH_unsat`).

`flock_headline_hidden` is that instance for any program `p`, outputs `outs`, places `pp`, constant and zero gates and
binding (`lay`, `rs`, `hHm`) at those tables. What it leaves open is named in its hypotheses:

* `hdec` (`DecodesOn`, R11b, at the law's draws only), as in #828's `flock_headline_exec`, at the hidden tables;
* `pp`, `hConst`, `hZero`, `lay`, `rs`, `hHm` at the hidden tables: what [wave2-hidden] establishes about
  `setupHidden`'s statements (the program's places in `stmtOf` of the hidden statement, its constant and zero
  columns, and the binding of each drawn unit's rows, outputs included, through the rows' `b ‖ c`), through its
  interface;
* the assumptions of record: A3 (`hA3`), `cr/sha-512` (`hKS`), `ecr/sha-512` (`hCR`); A6 is the live session.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

/-- **The headline over hidden-output sessions.** Let `I` be a verify command's inputs (one table) under a tag with
hidden outputs, and let the challenger write draw `S` as the JSON `dj S` (`x₀` one draw it sets up with
`Stmt.setupHidden`). Let the verifier draw the units with `Flock.Draw.drawOS`'s stratified law on a byte source A3
makes uniform, and let each draw's session be the challenger's one table (`liveTable`, A6), written out by `wr`.
Against any live prover `τ`, flock-verify accepts with at least `K` wrong units of the program `p`, at the simulated
prover's committed values, with probability at most `miss(K) + ε_ks + δ_link`, given the program's places, constant
and zero columns and binding at the hidden tables (`pp`, `hConst`, `hZero`, `lay`, `rs`, `hHm`) and `hdec`. -/
theorem flock_headline_hidden {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (pp : ProgPlaces p outs (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀))
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) (planAtH dj x₀) (tabAtH dj x₀)
      ((p.partition outs).placeDecoder (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place))
    (rs : Binding.HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm)
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm)
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm) k Rw M ρ t' :=
  p.flock_headline_live_onX outs σs ks hks St hS Kd pop Lb src hA3 (tablesAtH dj x₀) (tabAtH dj x₀) pp ones hConst
    zeros hZero lay rs hHm hk hRw hM hρ hr ht' K I wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.HiddenExec
