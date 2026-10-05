import FlockSoundness.Protocol.Discharge.ZkHidden.View
import FlockSoundness.Protocol.Discharge.Canonical.Session
import FlockSoundness.Protocol.Discharge.Canonical.Verifier

/-!
# Soundness: one theorem over everything flock-verify accepts

`flock_verify_sound` is the soundness theorem over what `flock-verify verify` accepts (`FlockVerifyAccepts`), with or
without `--zk`, public or hidden outputs, at any number of tables, with no hypothesis on the circuit file or the
statement's form. It splits on the statement inside its conclusion (its hidden-output tag, `--zk`, and under `--zk`
whether the session has one table) and states in each case the strongest claim the `FailClosed` headlines make there:

- without `--zk`, public outputs: `SoundExec` (`flock_headline_exec_dec`); hidden outputs: `SoundExecH`
  (`flock_headline_exec_hidden_dec`). The check refuses more than one table without `--zk`.
- `--zk`, public outputs: at one table `SoundZ` (`zk_session_sound`, the stratified law) and zero knowledge `ViewZ`
  (`zk_session_composed`); at every number of tables `SoundZJ` (`zk_session_soundJ`, the live subset draw).
- `--zk`, hidden outputs, the circuit in `Layout.scopeOk`: at one table `SoundZH` (`zk_session_soundH`); at every number
  of tables `SoundZHJ` (`zk_session_soundHJ`). The circuit outside it, which the check accepts only in the canonical
  format (`PROTOCOL.md` §16.16): at every number of tables `SoundZHJC` (`Canonical.zk_session_soundC_verifies`, the
  live subset draw; `ProvedScope.canonicalLaw` refuses a stratified draw there). Zero knowledge `ViewZHJ`
  (`zk_session_composedHJ`) at both.

Each claim's event is `FlockVerifyAccepts zk I` at the live session's transcript, and each is the headline's statement
with `hh`, `hv` (`Layout.scopeOk` of the circuit) and `hn` read off the case.

The sixteen `FailClosed` headlines make eight distinct claims: the six soundness bounds (each its own game: the law,
stratified or the live subset draw, and the outputs, public or hidden) and the two zero-knowledge views. The other eight
are corollaries of these: `flock_headline_exec` and `flock_headline_exec_hidden` state the `_dec` bounds at any decoder
given its `DecodesOn` (`hdec`), which the `_dec` forms prove for the verifier's decoder; the soundness halves of
`zk_session_composed` and `zk_session_composedHJ` are `zk_session_sound` and `zk_session_soundHJ`; the four `_custody`
forms take the record (`hRec`) and draw (`hDraw`) facts from one named assumption (`Assumptions.Zk.RecordCustodyZK`,
`RecordCustodyZKJ`), and the two `_custody_json` forms the draw fact from the draw file being `flock-verify draw`'s
(`hdj`). The headlines remain as lemmas.
-/

namespace FlockSoundness.Discharge.FailClosed

section Accepts

open Flock Exec ZkExec

/-- **`flock-verify verify` accepted**, with `--zk` (`zk = true`) or without: `Main.verifyCmd` prints `ACCEPT` exactly when
the setup and verdict `Accepts` (`AcceptsZK`) models hold and the statement check passes (`Verifies`, `VerifiesZK`). -/
def FlockVerifyAccepts (zk : Bool) (I : Inputs) (t : Exec.Transcript) : Prop :=
  if zk then VerifiesZK I t else Verifies I t

end Accepts

/-! ## The rerun count

Every soundness claim takes the extractor's number `k` of accepted reruns and a rate `ρ < 1` with `hr`: each table of
each plan has `2^{logLen ℓ₀}/(e·k) ≤ ρ`. `k` is the analysis's parameter, not the verifier's, and every plan reads the
draw alone (`plan S R` does not depend on the registration `R`), so the plans have finitely many tables between them and
a large enough `k` meets `hr` at every statement, draw file and setup, for every `ρ > 0` (`hr_exec`, `hr_execH`, `hr_z`,
`hr_zj`, `hr_zh`, `hr_zhj`). The bound states its dependence on `k`.

Every claim also takes a reference setup (`x₀`, `y₀`) at some draw: at one table a setup of the statement, and in the
claims at every number of tables (`SoundZJ`, `SoundZHJ`, `ViewZHJ`) a setup of table 0 under its tags (`inputsJ I 0`).
Every session flock-verify accepts at a draw the coin server's file proves, with the session's link points, sets one up
(`setup_of_verifies`, `setupH_of_verifies`, `setupZ_of_verifiesZK`, `setupZH_of_verifiesZK`, and at any number of
tables `setupZJ_of_verifiesZK`, `setupZHJ_of_verifiesZK`), so where there is none it accepts no such session. -/

section

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL

set_option linter.unusedVariables false in
/-- The claim at a public-output statement without `--zk`: `flock_headline_exec_dec`'s. -/
def SoundExec (I : Inputs) (hh : I.tags.hiddenOutputs = false) : Prop :=
  ∀ {n : ℕ}
    {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀))
    {Reg : Type}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ hs : Scope I, ∀ S R j, rateB (Integrate.planAt (Reg := Reg) hs.typed dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOk I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : ∀ hs : Scope I, LinkCRL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512
      (Integrate.planAt hs.typed dj x₀) (Integrate.tabAt hs.typed dj x₀) ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables hs.scope (Reg := Reg)).binding
        (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables hs.scope))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
        (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
        (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (Integrate.planAt hs.typed dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
          (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 leaf512 (Integrate.planAt hs.typed dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
          (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ) ω)) qF qS),
    prob (fun o => FlockVerifyAccepts false I o.2.2 ∧ ∀ hs : Scope I,
        K ≤ (((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512 (Integrate.planAt hs.typed dj x₀)
              (Integrate.tabAt hs.typed dj x₀) ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables hs.scope (Reg := Reg)).binding
                (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables hs.scope))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
                (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N
                (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
                (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N
                (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (Integrate.planAt hs.typed dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAt hs.typed dj x₀) I wr (opnAt (tablesAt hs.typed dj x₀) I dj) dg
            (decAt (tablesAt hs.typed dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_exec hs.typed (hs.parse hh) hs.outNet dj x₀ _ wr wOP wLIG op_carries lig_carries (hs.tagsOk hh) hwr hdraw) τ)) +
        ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (Integrate.planAt hs.typed dj x₀) (Integrate.tabAt hs.typed dj x₀)
          ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables hs.scope (Reg := Reg)).binding
            (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables hs.scope)) k Rw M ρ t')

end

section

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL

set_option linter.unusedVariables false in
/-- The claim at a hidden-output statement without `--zk`: `flock_headline_exec_hidden_dec`'s. -/
def SoundExecH (I : Inputs) (hh : I.tags.hiddenOutputs = true) : Prop :=
  ∀ {n : ℕ}
    {mPts : ℕ}
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
    (hCR : ∀ hs : Scope I, LinkCRL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) hs.scope dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
        (rsAtH_computes hs.typed (hs.parseH hh) hs.scope dj x₀ hs.built (hs.loadH hh) hs.tables))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS),
    prob (fun o => FlockVerifyAccepts false I o.2.2 ∧ ∀ hs : Scope I,
        K ≤ (((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) hs.scope dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
                (rsAtH_computes hs.typed (hs.parseH hh) hs.scope dj x₀ hs.built (hs.loadH hh) hs.tables))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execH hs.typed (hs.parseH hh) hs.outNet dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtH hs.typed (hs.parseH hh) hs.scope dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
            (rsAtH_computes hs.typed (hs.parseH hh) hs.scope dj x₀ hs.built (hs.loadH hh) hs.tables)) k Rw M ρ t')

end

section

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

set_option linter.unusedVariables false in
/-- The claim at a public-output statement of one table under `--zk`, the law stratified: `zk_session_sound`'s. -/
def SoundZ (I : Inputs) (hh : I.tags.hiddenOutputs = false) (hn : I.count ≤ 1) : Prop :=
  ∀ {n : ℕ} {mPts : ℕ} {Reg : Type}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I, ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I hs.typed dj y₀ wr _ tablesAtZK_live (hs.tagsOk hh hn) hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I hs.typed dj y₀ wr _ tablesAtZK_live (hs.tagsOk hh hn) hRec hDraw)) τ) ω)) qF qS),
    prob (fun o => FlockVerifyAccepts true I o.2.2 ∧ ∀ hs : ScopeZ I,
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAt hs.typed dj y₀ (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)).bindingZ (rsZAt_computes hs.typed dj y₀ (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I hs.typed dj y₀ wr _ tablesAtZK_live (hs.tagsOk hh hn) hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I hs.typed dj y₀ wr _ tablesAtZK_live (hs.tagsOk hh hn) hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : ScopeZ I, ((Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I hs.typed dj y₀ wr _ tablesAtZK_live (hs.tagsOk hh hn) hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAt hs.typed dj y₀ (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)).bindingZ (rsZAt_computes hs.typed dj y₀ (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I hs.typed dj y₀ wr _ tablesAtZK_live (hs.tagsOk hh hn) hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I hs.typed dj y₀ wr _ tablesAtZK_live (hs.tagsOk hh hn) hRec hDraw)) τ)))

set_option linter.unusedVariables false in
/-- The claim at a public-output statement of any number of tables under `--zk`, the law the live subset draw: `zk_session_soundJ`'s. -/
def SoundZJ (I : Inputs) (hh : I.tags.hiddenOutputs = false) : Prop :=
  ∀ {n : ℕ} {mPts : ℕ} {Reg : Type}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _ (runsZCJ_live I hs.typed dj y₀ wr _ (hs.tagsOkZJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZJ dj y₀) (cont σ ω)) qF qS),
    prob (fun o => FlockVerifyAccepts true I o.2.2 ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _ (runsZCJ_live I hs.typed dj y₀ wr _ (hs.tagsOkZJ hh) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZJ dj y₀) (tabZJ dj y₀)
            ((rsZJ dj y₀ hs.typed (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)).bindingZ (rsZJ_computes dj y₀ hs.typed (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _ (runsZCJ_live I hs.typed dj y₀ wr _ (hs.tagsOkZJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ hs.typed (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)).bindingZ (rsZJ_computes dj y₀ hs.typed (hs.parse hh) hs.built (hs.load hh) hs.tables (hs.scope hh)))
          k Rw M ρ t' (reg σ) (cont σ))

set_option linter.unusedVariables false in
/-- Zero knowledge at a public-output statement of one table under `--zk`, at every statement the check passes:
`zk_session_composed`'s second half. -/
def ViewZ (I : Inputs) : Prop :=
  ∀ (P : LigProver) {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    -- hash-derived-key
    (hKey : Assumptions.Zk.HashDerivedKeyHm96),
    ScopeZ I → ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
      ZkViewAt (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ))

end

section

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

set_option linter.unusedVariables false in
/-- The claim at a hidden-output statement of one table under `--zk`, the law stratified: `zk_session_soundH`'s. -/
def SoundZH (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hv : Layout.scopeOk (circZ I) = true) (hn : I.count ≤ 1) :
    Prop :=
  ∀ {n : ℕ} {mPts : ℕ} {Reg : Type}
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
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ) ω)) qF qS),
    prob (fun o => FlockVerifyAccepts true I o.2.2 ∧ ∀ hs : ScopeZ I,
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZH dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : ScopeZ I, ((Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)))

set_option linter.unusedVariables false in
/-- The claim at a hidden-output statement of any number of tables under `--zk`, the law the live subset draw: `zk_session_soundHJ`'s. -/
def SoundZHJ (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hv : Layout.scopeOk (circZ I) = true) : Prop :=
  ∀ {n : ℕ} {mPts : ℕ} {Reg : Type}
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
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS),
    prob (fun o => FlockVerifyAccepts true I o.2.2 ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOk_spec hs.built hv).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hv hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg σ) (cont σ))

set_option linter.unusedVariables false in
/-- The claim at a `--zk` statement of any number of tables whose circuit is outside `Layout.scopeOk`, the law the live
subset draw: `Canonical.zk_session_soundC_verifies`'s. The check accepts such a circuit only in the canonical format, at
hidden outputs. -/
def SoundZHJC (I : Inputs) (hv : Layout.scopeOk (circZ I) = false) : Prop :=
  ∀ {n : ℕ} {mPts : ℕ} {Reg : Type}
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
          (Canonical.runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _ (hs.tagsOkZHJ (hs.hidden hv)) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS),
    prob (fun o => FlockVerifyAccepts true I o.2.2 ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (Canonical.runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _ (hs.tagsOkZHJ (hs.hidden hv)) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((Canonical.rsZC dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))).bindingZ (Canonical.rsZC_computes dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (Canonical.runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _ (hs.tagsOkZHJ (hs.hidden hv)) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((Canonical.rsZC dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))).bindingZ (Canonical.rsZC_computes dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))))
          k Rw M ρ t' (reg σ) (cont σ))

set_option linter.unusedVariables false in
/-- Zero knowledge at a hidden-output statement of any number of tables under `--zk`, at every statement the check
passes: `zk_session_composedHJ`'s second half. -/
def ViewZHJ (I : Inputs) : Prop :=
  ∀ (P : LigProver) {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    -- hash-derived-key
    (hKey : Assumptions.Zk.HashDerivedKeyHm96),
    ScopeZ I → ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ))

end

section

open Flock Exec ZkExec

end

end FlockSoundness.Discharge.FailClosed
