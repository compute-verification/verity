import Proofs.Flock.Soundness.Discharge.General.Hidden
import Proofs.Flock.Soundness.Discharge.ZkReg
import Proofs.Flock.Soundness.Discharge.FailClosed.Exec
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHidden
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegistered

/-!
# The hidden-output headlines at the general form, over what flock-verify accepts

Where the wires feed the unit or the outputs (`ProvedScope.wiresFeed`), a hidden-output statement the check passes is
in the general form (`Checks.scopeGen`: `Layout.scopeGenH`), and the headline there is the general form's
(`Discharge.General`) at the template instance's program `instProg c n`, over what the verifier accepts, as
`FailClosed.Exec`, `FailClosed.ZkHidden` and `FailClosed.ZkRegistered` are over it where the wires feed neither:

* `flock_headline_exec_hiddenG`: `General.flock_headline_exec_hidden_gen` over `Verifies`;
* `zk_session_soundHG`: `General.zk_session_soundH_gen` over `VerifiesZK`, at one table;
* `zk_session_soundHJG`: `ZkHidden.zk_session_soundHJI` over `VerifiesZK`, at every number of tables;
* `zk_session_soundHJRG`: `ZkHidden.zk_session_soundHJI` over `VerifiesZKR`, and the registered reads checked
  (`ZkReg.checked_of_acceptsHJI`), at a public file with or without shared rows.

Each takes the circuit, the public file and the conditions `scopeGenH` gives from the check at each statement it passes
(`hs`), and at a statement the check refuses the event is empty.
-/

namespace FlockSoundness.Discharge.FailClosed

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Inst
open scoped ENNReal

section Hidden

open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.General (rsAtG rsAtG_computes)

/-- **One hidden-output non-ZK sampled session that flock-verify accepts, where the wires feed, is sound.**
`General.flock_headline_exec_hidden_gen` over `Verifies`. -/
theorem flock_headline_exec_hiddenG
    {n : ℕ} {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = true) {mPts : ℕ}
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
    (hCR : ∀ hs : Scope I, LinkCRL ((instProg (circS I) n).partition (outsOfI (circS I) n)) H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀) (tabAtH dj x₀)
      ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
        (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built))
      (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveVerifies I o ∧ ∀ hs : Scope I,
        K ≤ (((instProg (circS I) n).partition (outsOfI (circS I) n)).wrong
          (Xpub {instOneG (c := circS I) (n := n)} {instZeroG (c := circS I) (n := n)}
            (XplurL ((instProg (circS I) n).partition (outsOfI (circS I) n)) H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
              (tabAtH dj x₀)
              ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
                (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built))
              (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((instProg (circS I) n).partition (outsOfI (circS I) n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
            (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)) k Rw M ρ t') := by
  refine prob_scope (A := LiveAccepts I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.General.flock_headline_exec_hidden_gen hs.typed (hs.parseH hh) (hs.scopeGen hh hf)
    dj x₀ (hs.loadH hh) hs.built (hs.tagsOkH hh) σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr dg N hwr
    hdraw τ (hCR hs) (hKS hs)

end Hidden

section Zk

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.General (rsZG rsZG_computes rsZHJG rsZHJG_computes layShHJG rsShHJG)
open FlockSoundness.Discharge.ZkReg (AcceptsZKR LiveAcceptsZKR OwnReadsHJ regPathsHJ acceptsZK_of_R
  checked_of_acceptsHJI)
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {mPts n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session that flock-verify accepts, where the wires feed, is sound.**
`General.zk_session_soundH_gen` over `VerifiesZK`. -/
theorem zk_session_soundHG
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circZ I) = true) (hn : I.count ≤ 1)
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
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_liveHI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planAtZKH dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_liveHI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ
        K₀ ≤ (((instProg (circZ I) n).partition (outsOfI (circZ I) n)).wrong
          (XplurZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planAtZKH dj y₀)
            (tabAtZKH dj y₀)
            ((rsZG dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built).bindingZ
              (rsZG_computes dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_liveHI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ tablesAtZKH_liveI (hs.tagsOkH hh hn) hRec hDraw)) τ
        (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planAtZKH dj y₀)
          (tabAtZKH dj y₀)
          ((rsZG dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built).bindingZ
            (rsZG_computes dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.General.zk_session_soundH_gen I hs.typed dj y₀ wr dg N (hs.parseHG hh hf) hs.built
    (hs.loadH hh) (hs.scopeGen hh hf) σs ks hks St hS Kd pop Lb src hA3 hDraw (hs.tagsOkH hh hn) hRec hk hRw hM hρ
    hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk` sampled session of `J` tables that flock-verify accepts, where the wires feed, is
sound.** `ZkHidden.zk_session_soundHJI` over `VerifiesZK`, at every number of tables. -/
theorem zk_session_soundHJG
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circZ I) = true)
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
          (runsZCJH_liveI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((instProg (circZ I) n).partition (outsOfI (circZ I) n)).wrong
          (XplurZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀)
            (tabZHJ dj y₀)
            ((rsZHJG dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built).bindingZ
              (rsZHJG_computes dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀)
          (tabZHJ dj y₀)
          ((rsZHJG dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built).bindingZ
            (rsZHJG_computes dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.ZkHidden.zk_session_soundHJI I hs.typed dj y₀ wr dg N (hs.parseHG hh hf) hs.built
    (hs.loadH hh) (General.rowsFit_of_scopeGenH ((zkTags_typed I).trans hs.typed) (hs.parseHG hh hf) (hs.scopeGen hh hf))
    (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).1 (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).2
    kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ hh) hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk --registered` sampled session of `J` tables that flock-verify accepts, where the wires
feed, is sound, and its registered reads are checked.** `ZkHidden.zk_session_soundHJI` over `VerifiesZKR`, at a public
file with or without shared rows (`rsZHJI` reads none), and `ZkReg.checked_of_acceptsHJI` at every accepted outcome. -/
theorem zk_session_soundHJRG
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circZ I) = true)
    (own : Array Flock.Registered.Port)
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
          (runsZCJH_liveI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZKR I own o ∧ ∀ hs : ScopeZR I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((instProg (circZ I) n).partition (outsOfI (circZ I) n)).wrong
          (XplurZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀)
            (tabZHJ dj y₀)
            ((rsZHJG dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built).bindingZ
              (rsZHJG_computes dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZR I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveI I hs.typed (hs.parseHG hh hf) dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg (circZ I) n).partition (outsOfI (circZ I) n)) (planZHJ dj y₀)
          (tabZHJ dj y₀)
          ((rsZHJG dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built).bindingZ
            (rsZHJG_computes dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built))
          k Rw M ρ t' (reg σ) (cont σ)) ∧
    ∀ hs : ScopeZR I, ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveVerifiesZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX
          (layShHJG (Reg := Reg) dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (pub := pubZ I))
          (rsShHJG dj y₀ hs.typed (hs.parseHG hh hf) (hs.scopeGen hh hf) (hs.loadH hh) hs.built),
        OwnReadsHJ I own (circZ I) (pubZ I) o.2.2 rd.isRead rd.port rd.readAt →
          rd.Checked o.1 (regPathsHJ I (circZ I) (pubZ I) o.2.2 rd.port) :=
  ⟨prob_scope (A := LiveAcceptsZK I) (fun _ h => acceptsZK_of_R h.1) (fun _ h => h.2) _ _ _ _ fun hs =>
      FlockSoundness.Discharge.ZkHidden.zk_session_soundHJI I hs.typed dj y₀ wr dg N (hs.parseHG hh hf) hs.built
        (hs.loadH hh) (General.rowsFit_of_scopeGenH ((zkTags_typed I).trans hs.typed) (hs.parseHG hh hf) (hs.scopeGen hh hf))
        (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).1 (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).2
        kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ hh) hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs),
    fun hs o hv rd hown =>
      checked_of_acceptsHJI I hs.typed own dj y₀ (hs.parseHG hh hf)
        (General.rowsFit_of_scopeGenH ((zkTags_typed I).trans hs.typed) (hs.parseHG hh hf) (hs.scopeGen hh hf)) hs.built
        (hs.loadH hh) (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).1
        (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).2 (hs.tagsOkZHJ hh) o hv.1 rd hown⟩

end Zk

end FlockSoundness.Discharge.FailClosed
