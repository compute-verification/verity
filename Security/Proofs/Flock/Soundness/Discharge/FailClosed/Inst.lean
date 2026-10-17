import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHidden
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegistered
import Proofs.Flock.Soundness.Discharge.FailClosed.Exec
import Proofs.Flock.Soundness.Discharge.ZkReg.PublicInst
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundRuns
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineInst

set_option Elab.async false

/-!
# The hidden-output headlines at the template instance's program, over what flock-verify accepts

Under hidden outputs the check passes the general instance form: one unit, repeated, reading any number of committed
input groups, and any number of stages the verifier evaluates, each reading wires from earlier outputs. Where the
circuit is not of the unit form (`¬ UnitForm`: the unit reads past its leaf group, or a tail stage returns the outputs)
the check gives `wiresAcyclic` (`Checks.wires`), and under hidden outputs `Layout.scopeOkI` (`Checks.scopeI`), so the
circuit conditions of the headlines at `instProg` hold: the rows' fit (`Checks.rowsFit`), the unit's reads
(`Checks.portReads`) and the wires' order. Each theorem here is such a headline over what `flock-verify verify` accepts,
with no hypothesis on the statement's form but which headline applies, as `FailClosed.ZkHidden` and `FailClosed.Exec`
are at the unit's program:

* `flock_headline_exec_hidden_decI`: `HiddenExec.flock_headline_exec_hidden_inst` over `Verifies`, at the verifier's
  decoder (`HiddenDec.decodesOn_execHI`);
* `zk_session_soundHI`: one table under `--zk`, the law stratified, over `VerifiesZK` (`ZkHidden.zk_session_soundHI`,
  `zk_session_soundH` at `instProg`: its runs `runsZC_liveHI`, its count curve `zk_flock_countC_execOS_atHI`);
* `zk_session_soundHJI`: `J` tables under `--zk`, the live subset draw, over `VerifiesZK`
  (`ZkHidden.zk_session_soundHJI`);
* `zk_session_soundHJRI`: the same with `--registered F` over `VerifiesZKR`, at a public file with or without shared
  rows, and its registered reads checked (`ZkReg.checked_of_acceptsHJI`).
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec

theorem Checks.rowsFit {t : Tags} {zk : Bool} {I : Inputs} {reg : Bool} (h : Checks t zk I reg)
    (hh : t.hiddenOutputs = true) : Hidden.Any.RowsFitS (circ t I) :=
  HiddenExec.rowsFit_of_scopeIH h.typed (h.parseH hh) (h.scopeI hh)

theorem Checks.portReads {t : Tags} {zk : Bool} {I : Inputs} {reg : Bool} (h : Checks t zk I reg)
    (hh : t.hiddenOutputs = true) : Layout.PortReads (circ t I) h.built :=
  Layout.scopeOkI_portReads h.built (h.scopeI hh)

theorem Scope.rowsFit {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true) :
    Hidden.Any.RowsFitS (circS I) := Checks.rowsFit h hh

theorem Scope.portReads {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true) :
    Layout.PortReads (circS I) h.built := Checks.portReads h hh

theorem ScopeZ.rowsFit {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) :
    Hidden.Any.RowsFitS (circZ I) := Checks.rowsFit h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.portReads {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) :
    Layout.PortReads (circZ I) h.built := Checks.portReads h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZR.rowsFit {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = true) :
    Hidden.Any.RowsFitS (circZ I) := Checks.rowsFit h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZR.portReads {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = true) :
    Layout.PortReads (circZ I) h.built := Checks.portReads h (by rw [zkTags_hiddenOutputs]; exact hh)

end FlockSoundness.Discharge.FailClosed

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Inst
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session is sound, at `instProg`.** `zk_session_soundH` with the template
instance's program, its circuit conditions the rows' fit (`hrf`), the unit's reads (`hpr`) and the wires' order (`hw`):
the runs at the law's draws are `runsZC_liveHI`, which read nothing of where the outputs are returned, and the count
curve `zk_flock_countC_execOS_atHI`. -/
theorem zk_session_soundHI
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the challenger's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the challenger's record of the live session
    (hTags : TagsOkH I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((instProg c n).partition (outsOfI c n)).wrong
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZHI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ)) := by
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hc' := zk_flock_countC_execOS_atHI dj y₀ ht hc hrf hpub hU hpr hw hs512 σs ks hks St hS Kd pop Lb src hA3 hk
    hRw hM hρ hr ht' K₀ (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR (execAccept := fun o => o.1 = true)
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR hs512.H hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hc'
  exact zk_headline_exec_on execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ _ _ hc'

end FlockSoundness.Discharge.ZkHidden

namespace FlockSoundness.Discharge.FailClosed

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.Inst
open scoped ENNReal

section Exec

/-- **One hidden-output non-ZK sampled session that flock-verify accepts is sound, at `instProg`, with the decoder
discharged.** `HiddenExec.flock_headline_exec_hidden_inst` over `Verifies` at a circuit not of the unit form, at the
verifier's decoder (`decodesOn_execHI`, which reads nothing of where the outputs are returned). -/
theorem flock_headline_exec_hidden_decI
    {n : ℕ} {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hu : ¬ UnitForm (circS I)) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {Reg : Type}
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
    (hCR : ∀ hs : Scope I, LinkCRL ((instProg (circS I) n).partition (outsOfI (circS I) n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀)
      ((rsAtHI hs.typed (hs.parseH hh) (hs.rowsFit hh) dj x₀ (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu) (Reg := Reg)).binding
        (rsAtHI_computes hs.typed (hs.parseH hh) (hs.rowsFit hh) dj x₀ (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveVerifies I o ∧ ∀ hs : Scope I,
        K ≤ (((instProg (circS I) n).partition (outsOfI (circS I) n)).wrong
          (Xpub {instOneG (c := circS I) (n := n)} {instZeroG (c := circS I) (n := n)}
            (XplurL ((instProg (circS I) n).partition (outsOfI (circS I) n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀)
              ((rsAtHI hs.typed (hs.parseH hh) (hs.rowsFit hh) dj x₀ (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu) (Reg := Reg)).binding
                (rsAtHI_computes hs.typed (hs.parseH hh) (hs.rowsFit hh) dj x₀ (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((instProg (circS I) n).partition (outsOfI (circS I) n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtHI hs.typed (hs.parseH hh) (hs.rowsFit hh) dj x₀ (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu) (Reg := Reg)).binding
            (rsAtHI_computes hs.typed (hs.parseH hh) (hs.rowsFit hh) dj x₀ (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu))) k Rw M ρ t') := by
  refine prob_scope (A := LiveAccepts I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact HiddenExec.flock_headline_exec_hidden_inst hs.typed (hs.parseH hh) (hs.rowsFit hh) dj x₀ (hs.loadH hh)
    hs.built (hs.portReads hh) (hs.wires hu) σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr _ dg _ _ _ _ N
    (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ (hCR hs) (hKS hs)

end Exec

section Zk

open ZkSession ZkLink ZkExec ZkBind Composed Placed StrictCR ZkHidden FlockLevel3
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.ZkReg (LiveAcceptsZKR OwnReadsHJ regPathsHJ acceptsZK_of_R checked_of_acceptsHJI)

variable {mPts n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session that flock-verify accepts is sound, at `instProg`.**
`ZkHidden.zk_session_soundHI` over `VerifiesZK`, at a circuit not of the unit form. -/
theorem zk_session_soundHI
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hu : ¬ UnitForm (circZ I)) (hn : I.count ≤ 1)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I, ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I hs.typed (hs.parseH hh) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I hs.typed (hs.parseH hh) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        K₀ ≤ (((instProg (circZ I) n).partition (outsOfI (circZ I) n)).wrong
          (XplurZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZHI dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)).bindingZ
              (rsZHI_computes dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I hs.typed (hs.parseH hh) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I hs.typed (hs.parseH hh) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : ScopeZ I, ((Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I hs.typed (hs.parseH hh) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHI dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)).bindingZ
            (rsZHI_computes dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I hs.typed (hs.parseH hh) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHI I hs.typed (hs.parseH hh) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ))) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.ZkHidden.zk_session_soundHI I hs.typed dj y₀ wr dg N (hs.parseH hh) hs.built
    (hs.loadH hh) (hs.rowsFit hh) (hs.portReads hh) (hs.wires hu) σs ks hks St hS Kd pop Lb src hA3 hDraw
    (hs.tagsOkH hh hn) hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk` sampled session of `J` tables that flock-verify accepts is sound, at `instProg`.**
`ZkHidden.zk_session_soundHJI` over `VerifiesZK`, at a circuit not of the unit form, at every number of tables. -/
theorem zk_session_soundHJI
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hu : ¬ UnitForm (circZ I))
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseH hh) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseH hh) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((instProg (circZ I) n).partition (outsOfI (circZ I) n)).wrong
          (XplurZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)).bindingZ
              (rsZHJI_computes dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseH hh) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)).bindingZ
            (rsZHJI_computes dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
          k Rw M ρ t' (reg σ) (cont σ)) :=
  prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs =>
    FlockSoundness.Discharge.ZkHidden.zk_session_soundHJI I hs.typed dj y₀ wr dg N (hs.parseH hh) hs.built
      (hs.loadH hh) (hs.rowsFit hh) (hs.portReads hh) (hs.wires hu) kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ hh) hRec
      hk hRw hM hρ hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk --registered` sampled session of `J` tables that flock-verify accepts is sound, and its
registered reads are checked, at `instProg`.** `zk_session_soundHJR` at a circuit not of the unit form, over
`VerifiesZKR`, at a public file with or without shared rows: the bound is `ZkHidden.zk_session_soundHJI`'s, since what
`--registered F` accepts `verify --zk` accepts at the same statement (`acceptsZK_of_R`), and the registered check's
consequence is `checked_of_acceptsHJI`. -/
theorem zk_session_soundHJRI
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hu : ¬ UnitForm (circZ I)) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZR I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseH hh) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZKR I own o ∧ ∀ hs : ScopeZR I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseH hh) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((instProg (circZ I) n).partition (outsOfI (circZ I) n)).wrong
          (XplurZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)).bindingZ
              (rsZHJI_computes dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZR I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseH hh) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)).bindingZ
            (rsZHJI_computes dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)))
          k Rw M ρ t' (reg σ) (cont σ)) ∧
    ∀ hs : ScopeZR I, ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveVerifiesZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX
          (layShHJI (Reg := Reg) dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (pub := pubZ I))
          (rsShHJI dj y₀ hs.typed (hs.parseH hh) (hs.rowsFit hh) (hs.loadH hh) hs.built (hs.portReads hh) (hs.wires hu)),
        OwnReadsHJ I own (circZ I) (pubZ I) o.2.2 rd.isRead rd.port rd.readAt →
          rd.Checked o.1 (regPathsHJ I (circZ I) (pubZ I) o.2.2 rd.port) :=
  ⟨prob_scope (A := LiveAcceptsZKR I own) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs =>
      le_trans (prob_mono (q := fun o => LiveAcceptsZK I o ∧ _) (fun _ h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
        (FlockSoundness.Discharge.ZkHidden.zk_session_soundHJI I hs.typed dj y₀ wr dg N (hs.parseH hh) hs.built
          (hs.loadH hh) (hs.rowsFit hh) (hs.portReads hh) (hs.wires hu) kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ hh) hRec
          hk hRw hM hρ hr ht' K₀ τ (hCR hs)),
    fun hs o hv rd hown =>
      checked_of_acceptsHJI I hs.typed own dj y₀ (hs.parseH hh) (hs.rowsFit hh) hs.built (hs.loadH hh)
        (hs.portReads hh) (hs.wires hu) (hs.tagsOkZHJ hh) o hv.1 rd hown⟩

end Zk

end FlockSoundness.Discharge.FailClosed
