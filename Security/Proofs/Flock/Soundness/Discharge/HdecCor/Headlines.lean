import Proofs.Flock.Soundness.Discharge.Integrate.Headline
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
import Proofs.Flock.Soundness.Discharge.DecodeZL
import Proofs.Flock.Soundness.Discharge.HiddenDec
import Proofs.Flock.Soundness.Discharge.DecodeZLJ
import Proofs.Flock.Soundness.Discharge.Tables

/-!
# The headlines with R11b's message half discharged

Each headline takes `hdec` (`DecodesOn`, or `DecodesOnJ` at `J` tables). These corollaries fix the decoder, the
openings, single coins, no final messages and `capOfRoot` at the values the decoding theorems are about, and replace
`hdec` by those theorems' hypotheses:
* `flock_headline_exec_dec`: #828's `flock_headline_exec` with #862's `decodesOn_exec` at `wOP`, `wLIG`: `hdec` becomes
  `c.outNet = c.unitNet`, `TagsOk`, `RecordsLive` and `DrawOk` at the law's draws;
* `flock_headline_exec_hidden_dec`: #869's `flock_headline_exec_hidden` with #875's `decodesOn_execH`: `hdec` becomes
  `c.outNet = c.unitNet`, `TagsOkH`, `RecordsLive` and `DrawOkH` at the law's draws;
* `headline_tables_subset_dec`: `headline_tables_subset` at the tables flock-verify sets up (`planJ`) with #867's
  `decodesOnJ_exec`: `hdec` becomes `TagsOkJ`, `RecordsLiveJ` and `DrawOkJ` at the law's draws. The decoder at `J`
  tables reads no circuit, so there is no `c.outNet = c.unitNet` here.

`FailClosed.flock_headline_exec_dec` and `_hidden_dec` take none of the hypotheses on the statement's form
(`c.outNet = c.unitNet`, `scopeOk`, the parse and load, retained rounds and leaves), which flock-verify refuses a statement
failing (PROTOCOL.md §16.15).
-/

namespace FlockSoundness.Discharge.HdecCor

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
  (hscope : Layout.scopeOk c = true) {Reg : Type}

/-- **`flock_headline_exec` with `hdec` discharged** by `decodesOn_exec` at DecodeZL's wires `wOP`, `wLIG`. -/
theorem flock_headline_exec_dec (ho : c.outNet = c.unitNet) (hT : TagsOk I)
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
      (Integrate.planAt ht dj x₀) (Integrate.tabAt ht dj x₀) ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAt_computes hU ht hc dj x₀ hpub hpt hscope))
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
              (Integrate.tabAt ht dj x₀) ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
                (rsAt_computes hU ht hc dj x₀ hpub hpt hscope))
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
          ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
            (rsAt_computes hU ht hc dj x₀ hpub hpt hscope)) k Rw M ρ t' :=
  flock_headline_exec hU ht hc dj x₀ hpub hpt hscope σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr _ dg
    _ _ _ _ N (decodesOn_exec ht hc ho dj x₀ _ wr wOP wLIG op_carries lig_carries hT hwr hdraw) τ hCR hKS

end Public

section Hidden

open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) {Reg : Type}

/-- **`flock_headline_exec_hidden` with `hdec` discharged** by `decodesOn_execH`. -/
theorem flock_headline_exec_hidden_dec (ho : c.outNet = c.unitNet) (hT : TagsOkH I)
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
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
        (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt))
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
              (tabAtH dj x₀) ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
                (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt))
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
          ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
            (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt)) k Rw M ρ t' :=
  flock_headline_exec_hidden ht hc hscope dj x₀ hU hpub hpt σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr
    _ dg _ _ _ _ N (decodesOn_execH ht hc ho dj x₀ _ wr hT hwr hdraw) τ hCR hKS

end Hidden

section Tables

open FlockSoundness.Discharge.Tables FlockSoundness.Discharge.DecodeZLJ FlockSoundness.Discharge.DecodeZL

/-- **`headline_tables_subset` with `hdec` discharged**, at the tables flock-verify sets up (`planJ ht T₀ dj`, any
fallback `T₀`), by `decodesOnJ_exec`. -/
theorem headline_tables_subset_dec
    {N' n : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N')) {Reg : Type} {mPts : ℕ}
    {I : Inputs} (ht : I.tags.typed = false) (T₀ : TabSpec mPts) (dj : Finset (Fin n) → Lean.Json) (hT : TagsOkJ I)
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planJ ht T₀ dj S R).length)
    (pp : ProgPlaces p outs (planJ ht T₀ dj) tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols (planJ ht T₀ dj) tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols (planJ ht T₀ dj) tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) (planJ ht T₀ dj) tab
      ((p.partition outs).placeDecoder (planJ ht T₀ dj) tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S (R : Reg) j, rateB (planJ ht T₀ dj S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (hK : K ≤ n)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planJ ht T₀ dj S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLiveJ I dj (fun S R => (planJ ht T₀ dj S R).length) wr)
    (hdraw : ∀ ω, DrawOkJ I mPts dj ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planJ ht T₀ dj) N wr))) :
    let σ := simAuditJ H512 enc512 leaf512 (planJ ht T₀ dj) I wr (opnJ I (planJ ht T₀ dj) dj) dg
      (decJ (planJ ht T₀ dj)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnJ_exec ht T₀ dj _ wr hT hwr hdraw) τ
    LinkCRL (p.partition outs) H512 enc512 leaf512 (planJ ht T₀ dj) tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' →
    ∀ {qF qS : ℝ}, (∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planJ ht T₀ dj ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 (planJ ht T₀ dj) (cont σ ω)) qF qS) →
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 (planJ ht T₀ dj) tab (rs.binding hHm) (reg σ) (cont σ)
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planJ ht T₀ dj) N wr)) τ ≤
      ((n - K).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        StrictCR.ksAvgStrict (planJ ht T₀ dj) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.subset kd) n)
          (planJ ht T₀ dj) tab (rs.binding hHm) k Rw M ρ t' :=
  headline_tables_subset p outs kd hkd Lb src hA3 (planJ ht T₀ dj) tab pp ones hConst zeros hZero lay rs hHm hk hRw hM
    hρ hr ht' K hK I wr _ dg _ _ _ _ N (decodesOnJ_exec ht T₀ dj _ wr hT hwr hdraw) τ

end Tables

end FlockSoundness.Discharge.HdecCor
