import FlockSoundness.Discharge.HdecCor.Headlines
import FlockSoundness.Discharge.ScopeW.Headline
import FlockSoundness.Discharge.ScopeW.HeadlineExec

set_option Elab.async false

/-!
# The widened headlines with R11b's message half discharged

`HdecCor`'s corollaries at the widened scope: `flock_headline_exec_decW` is `flock_headline_execW` with `hdec`
replaced by `decodesOn_exec`'s hypotheses, `flock_headline_exec_hidden_decW` is `flock_headline_exec_hiddenW` with
`hdec` replaced by `decodesOn_execH`'s. The decoders read no row's shape, so the hypotheses that replace `hdec` are `HdecCor`'s.
-/

namespace FlockSoundness.Discharge.ScopeW

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

section Public

open FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL

variable {c : Flock.Circuit} (hU : Built c.unit) {n : ℕ} {I : Inputs} (ht : I.tags.typed = false)
  (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {pub : Public}
  (hpub : HmRow.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none)
  (hscope : scopeOkW c = true) {Reg : Type}

/-- **`flock_headline_execW` with `hdec` discharged**, as `HdecCor.flock_headline_exec_dec`. -/
theorem flock_headline_exec_decW (ho : c.outNet = c.unitNet) (hT : TagsOk I)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (Integrate.planAt (Reg := Reg) ht dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOk I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512
      (Integrate.planAt ht dj x₀) (Integrate.tabAt ht dj x₀) ((rsAtW hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAtW_computes hU ht hc dj x₀ hpub hpt hscope))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj) dg
        (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj) dg
        (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (Integrate.planAt ht dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj) dg
          (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ))) j
      (batchStratL H512 enc512 leaf512 (Integrate.planAt ht dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj) dg
          (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (Integrate.planAt ht dj x₀)
              (Integrate.tabAt ht dj x₀) ((rsAtW hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
                (rsAtW_computes hU ht hc dj x₀ hpub hpt hscope))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj) dg
                (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N
                (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj) dg
                (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N
                (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (Integrate.planAt ht dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj) dg
            (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (Integrate.planAt ht dj x₀) (Integrate.tabAt ht dj x₀)
          ((rsAtW hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
            (rsAtW_computes hU ht hc dj x₀ hpub hpt hscope)) k Rw M ρ t' :=
  flock_headline_execW hU ht hc dj x₀ hpub hpt hscope σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr _ dg
    _ _ _ _ N (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ hCR hKS

end Public

section Hidden

open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : scopeOkW c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) {Reg : Type}

/-- **`flock_headline_exec_hiddenW` with `hdec` discharged**, as `HdecCor.flock_headline_exec_hidden_dec`. -/
theorem flock_headline_exec_hidden_decW (ho : c.outNet = c.unitNet) (hT : TagsOkH I)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtHW ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
        (rsAtHW_computes ht hc hscope dj x₀ hU hpub hpt))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtHW ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
                (rsAtHW_computes ht hc hscope dj x₀ hU hpub hpt))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtHW ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
            (rsAtHW_computes ht hc hscope dj x₀ hU hpub hpt)) k Rw M ρ t' :=
  flock_headline_exec_hiddenW ht hc hscope dj x₀ hU hpub hpt σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr
    _ dg _ _ _ _ N (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ hCR hKS

end Hidden

end FlockSoundness.Discharge.ScopeW
