import Proofs.Flock.Soundness.Discharge.ZkHidden
import Proofs.Flock.Soundness.Discharge.HiddenDec.Assemble

set_option Elab.async false

/-!
# The headlines at the general form, under hidden outputs

The general form (`Layout.scopeGenH`): a template instance is one repeated unit stage and any number of stages the
verifier evaluates, the units reading any number of committed input groups and every other input port filled by a wire
from an earlier stage's output, the outputs returned by any of them; on the wires, only an order that does not compute
is out (`ProvedScope.wiresAcyclic`). Each headline here is the headline at the template instance's program
`instProg c n` (`Inst/Prog.lean`), the units those of the whole template instance partitioned at the output words' bits
(`outsOfI`), with its circuit conditions from `scopeGenH` (`rowsFit_of_scopeGenH`, `Layout.scopeGenH_spec`) and its open
guarantees discharged by the decoders that read no shape of the circuit (`HiddenDec.decodesOn_execHI`,
`ZkHidden.runsZC_liveHI`):

* `flock_headline_exec_hidden_gen`: `HdecCor.flock_headline_exec_hidden_dec` at `instProg c n`
  (`HiddenExec.flock_headline_exec_hidden_inst`);
* `zk_session_soundH_gen`: `ZkHidden.zk_session_soundH` at `instProg c n` (`ZkHidden.zk_flock_countC_execOS_atHI`);
* `zk_session_soundH_custody_gen`: `ZkHidden.zk_session_soundH_custody` at `instProg c n`.

None takes `c.unit.inGroups.size = 1`, `c.outNet = c.unitNet` or `pub.tables = none`. At `J` tables and against the
registration, `ZkHidden.zk_session_soundHJI` and `ZkReg.zk_session_soundHJR_custody_regI` already take only the
conditions `scopeGenH` gives (`hrf`, `hpr`, `hw`).
-/

namespace FlockSoundness.Discharge.General

open Flock

/-- **`scopeGenH` gives the rows' fit** on a circuit an untyped statement's `HmOut.parse` accepts
(`HiddenExec.rowsFit_of_rows`). -/
theorem rowsFit_of_scopeGenH {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (ht : tags.typed = false) (hc : HmOut.parse tags cf tables = .ok c) (h : Layout.scopeGenH c = true) :
    Hidden.Any.RowsFitS c :=
  HiddenExec.rowsFit_of_rows (HiddenExec.parse_untyped ht hc).1 (HiddenExec.parse_untyped ht hc).2
    (Layout.scopeGenH_rowsB h)

/-! ## Without `--zk` -/

section Hidden

open Game Model Accounting Refine Audit Audit.Partition
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec
  FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.Inst
open scoped ENNReal

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hgen : Layout.scopeGenH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hU : Built c.unit) {Reg : Type}

/-- **`rs` at the hidden tables, at the general form**: `rsAtHI` with its circuit conditions from `scopeGenH`. -/
noncomputable abbrev rsAtG :=
  rsAtHI ht hc (rowsFit_of_scopeGenH ht hc hgen) dj x₀ hpub hU (Layout.scopeGenH_spec hU hgen).1
    (Layout.scopeGenH_spec hU hgen).2 (Reg := Reg)

theorem rsAtG_computes : (rsAtG ht hc hgen dj x₀ hpub hU (Reg := Reg)).Computes :=
  rsAtHI_computes ht hc (rowsFit_of_scopeGenH ht hc hgen) dj x₀ hpub hU (Layout.scopeGenH_spec hU hgen).1
    (Layout.scopeGenH_spec hU hgen).2

/-- **The headline at the executable over hidden-output sessions, at the general form.** Against any live prover `τ`,
flock-verify accepts with at least `K` wrong units of the template instance's program `instProg c n`, at the simulated
prover's committed values, with probability at most `miss(K) + ε_ks + δ_link`. Its circuit condition is `scopeGenH`;
`hdec` is `decodesOn_execHI`'s, from the tags, the challenger's record and its draws, at the model's leaf of the tags'
leaf scheme (`leafOfId`: `hm96-sha512/v1` or plain SHA-512 leaves). -/
theorem flock_headline_exec_hidden_gen (hT : TagsOkHL I)
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
    {qF qS : ℝ} :
    let σ := simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
      (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
      (decodesOn_execHI ht hc dj x₀ _ wr hT hwr hdraw) τ
    LinkCRL ((instProg c n).partition (outsOfI c n)) H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀) (tabAtH dj x₀)
      ((rsAtG ht hc hgen dj x₀ hpub hU (Reg := Reg)).binding (rsAtG_computes ht hc hgen dj x₀ hpub hU))
      (reg σ) (cont σ) k Rw M t' →
    (∀ ω j, TableCRL execArith H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStratL H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((instProg c n).partition (outsOfI c n)).wrong
          (Xpub {instOneG (c := c) (n := n)} {instZeroG (c := c) (n := n)}
            (XplurL ((instProg c n).partition (outsOfI c n)) H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtG ht hc hgen dj x₀ hpub hU (Reg := Reg)).binding
                (rsAtG_computes ht hc hgen dj x₀ hpub hU))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        ((instProg c n).partition (outsOfI c n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtG ht hc hgen dj x₀ hpub hU (Reg := Reg)).binding
            (rsAtG_computes ht hc hgen dj x₀ hpub hU)) k Rw M ρ t' := by
  intro σ hCR hKS
  exact flock_headline_exec_hidden_inst ht hc (rowsFit_of_scopeGenH ht hc hgen) dj x₀ hpub hU
    (Layout.scopeGenH_spec hU hgen).1 (Layout.scopeGenH_spec hU hgen).2 σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ
    hr ht' K wr _ dg _ _ _ _ N (decodesOn_execHI ht hc dj x₀ _ wr hT hwr hdraw) τ hCR hKS

end Hidden

/-! ## Under `--zk` -/

section Zk

open Game Audit Exec Refine FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.Inst
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **`rs` at the hidden-output `--zk` tables, at the general form**: `rsZHI` with its circuit conditions from
`scopeGenH`. -/
noncomputable abbrev rsZG {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hgen : Layout.scopeGenH c = true) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hU : Built c.unit) :=
  rsZHI dj y₀ ht hc (rowsFit_of_scopeGenH ((zkTags_typed I).trans ht) hc hgen) hpub hU
    (Layout.scopeGenH_spec hU hgen).1 (Layout.scopeGenH_spec hU hgen).2 (Reg := Reg)

theorem rsZG_computes {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hgen : Layout.scopeGenH c = true) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hU : Built c.unit) :
    (rsZG (Reg := Reg) dj y₀ ht hc hgen hpub hU).Computes :=
  rsZHI_computes dj y₀ ht hc (rowsFit_of_scopeGenH ((zkTags_typed I).trans ht) hc hgen) hpub hU
    (Layout.scopeGenH_spec hU hgen).1 (Layout.scopeGenH_spec hU hgen).2

/-- **One hidden-output `--zk` sampled session is sound, at the general form.** `ZkHidden.zk_session_soundH` at the
template instance's program `instProg c n`: for any live prover `τ`,
`Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`. Its circuit condition is
`scopeGenH`; `RunsZC` at the law's draws is `runsZC_liveHI`'s. -/
theorem zk_session_soundH_gen
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hgen : Layout.scopeGenH c = true)
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
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ} :
    let σ := simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
      (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planAtZKH dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((instProg c n).partition (outsOfI c n)).wrong
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZG dj y₀ ht hc hgen hpub hU).bindingZ (rsZG_computes dj y₀ ht hc hgen hpub hU))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZG dj y₀ ht hc hgen hpub hU).bindingZ (rsZG_computes dj y₀ ht hc hgen hpub hU))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hRec hDraw)
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hc' := zk_flock_countC_execOS_atHI dj y₀ ht hc (rowsFit_of_scopeGenH ((zkTags_typed I).trans ht) hc hgen)
    hpub hU (Layout.scopeGenH_spec hU hgen).1 (Layout.scopeGenH_spec hU hgen).2 hs512 σs ks hks St hS Kd pop Lb src
    hA3 hk hRw hM hρ hr ht' K₀ (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR
    (execAccept := fun o => o.1 = true) (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR hs512.H hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hc'
  exact zk_headline_exec_on execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ _ _ hc'

/-- **One hidden-output `--zk` sampled session is sound under `live-verifier`, at the general form.**
`zk_session_soundH_gen` at the link points of the law's own draws, `mPtsOf c.g (∑ s, ks s)`, with the record's custody
(`RecordCustodyZK`) in place of `hRec` and `hDraw`, as `ZkHidden.zk_session_soundH_custody` is
`zk_session_soundH`'s. -/
theorem zk_session_soundH_custody_gen
    (I : Inputs) (ht : I.tags.typed = false)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hgen : Layout.scopeGenH c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the record, at the link points of the law's draws
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I (mPtsOf c.g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf c.g (∑ s, ks s)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verify command's tags, and live-verifier: the record and its draw file are the live session's
    (hTags : TagsOkH I) (hCust : Assumptions.Zk.RecordCustodyZK I dj wr (.stratified Kd (Array.ofFn St)) pop)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)))
    {qF qS : ℝ} :
    let σ := simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
      (runsZC_liveHI I ht hc dj y₀ wr _ tablesAtZKH_liveI hTags hCust.1
        (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planAtZKH dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((instProg c n).partition (outsOfI c n)).wrong
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZG dj y₀ ht hc hgen hpub hU).bindingZ (rsZG_computes dj y₀ ht hc hgen hpub hU))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZG dj y₀ ht hc hgen hpub hU).bindingZ (rsZG_computes dj y₀ ht hc hgen hpub hU))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundH_gen I ht dj y₀ wr dg N hc hU hpub hgen σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

/-! At `J` tables, `ZkHidden.zk_session_soundHJI` and `ZkReg.checked_of_acceptsHJI` take the conditions `scopeGenH`
gives; these are their rows with those conditions read off it. -/

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hgen : Layout.scopeGenH c = true) {pub : Public}
  (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hU : Built c.unit)

/-- **The public plan's layout at `J` tables, at the general form** (`layShHJI`). -/
noncomputable abbrev layShHJG :=
  layShHJI dj y₀ ht hc (rowsFit_of_scopeGenH ((zkTags_typed I).trans ht) hc hgen) (pub := pub) (Reg := Reg)

/-- **`rs` at the public plan at `J` tables, at the general form** (`rsShHJI`). -/
noncomputable abbrev rsShHJG :=
  rsShHJI dj y₀ ht hc (rowsFit_of_scopeGenH ((zkTags_typed I).trans ht) hc hgen) hpub hU
    (Layout.scopeGenH_spec hU hgen).1 (Layout.scopeGenH_spec hU hgen).2 (Reg := Reg)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables, at the general form** (`rsZHJI`). -/
noncomputable abbrev rsZHJG :=
  rsZHJI dj y₀ ht hc (rowsFit_of_scopeGenH ((zkTags_typed I).trans ht) hc hgen) hpub hU
    (Layout.scopeGenH_spec hU hgen).1 (Layout.scopeGenH_spec hU hgen).2 (Reg := Reg)

theorem rsZHJG_computes : (rsZHJG dj y₀ ht hc hgen hpub hU (Reg := Reg)).Computes :=
  rsZHJI_computes dj y₀ ht hc (rowsFit_of_scopeGenH ((zkTags_typed I).trans ht) hc hgen) hpub hU
    (Layout.scopeGenH_spec hU hgen).1 (Layout.scopeGenH_spec hU hgen).2

end Zk

end FlockSoundness.Discharge.General
