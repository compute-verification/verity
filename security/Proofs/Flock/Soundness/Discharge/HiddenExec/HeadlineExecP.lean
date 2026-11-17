module

public import Proofs.Flock.Soundness.Discharge.HiddenExec.RowsAtHP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.LiveOnP

@[expose] public section

set_option Elab.async false

/-! The headline at the executable over hidden-output sessions at the verifier's own public input words (`flock_headline_exec_hiddenP`). -/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {own : Option ByteArray} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hcS : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupHP I own mPts S₀ (dj S₀))
  (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
  {pub : Public} (hpub : HmIn.loadPublic I.tags c pi I.publicFile = .ok pub) {Reg : Type}

theorem flock_headline_exec_hiddenP {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtHP (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dec one fin capOf
      (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) H512 enc512 Lp
      (planAtHP dj x₀) (tabAtHP dj x₀) ((rsAtHP ht hcS hscope dj x₀ hU hpi hpub (Reg := Reg)).binding
        (rsAtHP_computes ht hcS hscope dj x₀ hU hpi hpub))
      (reg (simAuditOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 Lp
      (planAtHP dj x₀ ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 Lp (planAtHP dj x₀)
        (cont (simAuditOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dg dec one fin capOf N hdec τ) ω))
      qF qS) :
    prob (fun o => LiveAcceptsP I own o ∧
        K ≤ (((Placed.keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)).incorrect
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) H512 enc512 Lp
              (planAtHP dj x₀) (tabAtHP dj x₀) ((rsAtHP ht hcS hscope dj x₀ hU hpi hpub (Reg := Reg)).binding
                (rsAtHP_computes ht hcS hscope dj x₀ hU hpi hpub))
              (reg (simAuditOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtHP dj x₀) (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOnP H512 enc512 Lp (tablesAtHP dj x₀) I own wr opn dg dec one fin capOf N hdec τ)) +
        ((Placed.keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)).linkBoundE
          (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtHP dj x₀) (tabAtHP dj x₀)
          ((rsAtHP ht hcS hscope dj x₀ hU hpi hpub (Reg := Reg)).binding
            (rsAtHP_computes ht hcS hscope dj x₀ hU hpi hpub)) k Rw M ρ t' :=
  (Placed.keyProgP c pi hU n).flock_headline_live_onXP (Layout.outsOfP c hU n pi) σs ks hks St hS Kd pop Lb src hA3
    (tablesAtHP dj x₀) (tabAtHP dj x₀)
    (ppHP ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀ hU hpi (Layout.outsOfP c hU n pi))
    {Layout.oneG} (hConstHP ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀ hU hpi (Layout.outsOfP c hU n pi))
    {Layout.zeroG} (hZeroHP ht hcS (rowsFit_of_stmt ht hcS hscope) dj x₀ hU hpi (Layout.outsOfP c hU n pi))
    (layAtHP ht hcS hscope dj x₀ hU hpi (pop := pub)) (rsAtHP ht hcS hscope dj x₀ hU hpi hpub)
    (rsAtHP_computes ht hcS hscope dj x₀ hU hpi hpub) hk hRw hM hρ hr ht' K I own wr opn dg dec one fin capOf N hdec τ
    hCR hKS

end FlockSoundness.Discharge.HiddenExec
