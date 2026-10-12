import Proofs.Flock.Soundness.Discharge.FailClosed.Composed
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHidden
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegistered
import Proofs.Flock.Soundness.Discharge.FailClosed.Exec

/-!
# Soundness: one theorem over everything flock-verify accepts

`flock_verify_sound` is the soundness theorem over what `flock-verify verify` accepts (`FlockVerifyAccepts`, and with
`--registered F` `FlockVerifyAcceptsR`), with or without `--zk`, public or hidden outputs, at any number of tables, with
no hypothesis on the circuit file or the statement's form. It splits on the statement inside its conclusion (its
hidden-output tag, `--zk`, and under `--zk` whether the session has one table) and states in each case the strongest
claim the `FailClosed` headlines make there:

- without `--zk`, public outputs: `SoundExec` (`flock_headline_exec_dec`); hidden outputs: `SoundExecH`
  (`flock_headline_exec_hidden_dec`). The check refuses more than one table without `--zk`.
- `--zk`, public outputs: at one table `SoundZ` (`zk_session_sound`, the stratified law) and zero knowledge `ViewZ`
  (`zk_session_composed`); at every number of tables `SoundZJ` (`zk_session_soundJ`, the live subset draw).
- `--zk`, hidden outputs: at one table `SoundZH` (`zk_session_soundH`); at every number of tables `SoundZHJ`
  (`zk_session_soundHJ`) and zero knowledge `ViewZHJ` (`zk_session_composedHJ`); and with `--registered F`, at a
  public file with or without shared rows, `SoundZHJR` (`zk_session_soundHJR`, the live subset draw, and the registered
  reads checked) and zero knowledge `ViewZHJR` (`ViewZHJ` at the check with the verifier's registered reads).

Each claim's event is `FlockVerifyAccepts zk I` (the registered ones' `FlockVerifyAcceptsR zk I own`) at the live
session's transcript, and each is the headline's statement with `hh` and `hn` read off the case. Its second conjunct
covers `--registered F` everywhere else: without `--zk` or at public outputs it accepts no more than the verifier does
without it (`verifiesZK_of_R`), so the claims above hold of it.

The sixteen `FailClosed` headlines make eight distinct claims: the six soundness bounds (each its own game: the law,
stratified or the live subset draw, and the outputs, public or hidden) and the two zero-knowledge views. The registered
headline (`FailClosed.ZkRegistered`) makes a ninth, `SoundZHJR`, and `ViewZHJR` is `ViewZHJ`'s view at the statements
only its check passes. The other eight
are corollaries of these: `flock_headline_exec` and `flock_headline_exec_hidden` state the `_dec` bounds at any decoder
given its `DecodesOn` (`hdec`), which the `_dec` forms prove for the verifier's decoder; the soundness halves of
`zk_session_composed` and `zk_session_composedHJ` are `zk_session_sound` and `zk_session_soundHJ`; the four `_custody`
forms take the record (`hRec`) and draw (`hDraw`) facts from one named assumption (`Assumptions.Zk.RecordCustodyZK`,
`RecordCustodyZKJ`), and the two `_custody_json` forms the draw fact from the draw file being `flock-verify draw`'s
(`hdj`). The headlines remain, pinned, as lemmas.
-/

namespace FlockSoundness.Discharge.FailClosed

section Accepts

open Flock Exec ZkExec

/-- **`flock-verify verify` accepted**, with `--zk` (`zk = true`) or without: `Main.verifyCmd` prints `ACCEPT` exactly when
the setup and verdict `Accepts` (`AcceptsZK`) models hold and the statement check passes (`Verifies`, `VerifiesZK`). -/
def FlockVerifyAccepts (zk : Bool) (I : Inputs) (t : Exec.Transcript) : Prop :=
  if zk then VerifiesZK I t else Verifies I t

/-- **`flock-verify verify --registered F` accepted** (`own` F's ports): under `--zk`, `VerifiesZKR` (the registered check
and the statement check with the verifier's registered reads); without, `Verifies`, as without `--registered`: the
statement check is the same without `--zk`, and the registered check only refuses more. -/
def FlockVerifyAcceptsR (zk : Bool) (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  if zk then VerifiesZKR I own t else Verifies I t

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

/-- **A rerun count meeting a rate**: when a plan reads the draw alone, some `k ≥ 1` puts every table's
`2^{w T}/(e·k)` at most `ρ`. -/
theorem exists_k {α : Type} {n : ℕ} {Reg : Type} (w : α → ℕ) (plan : Finset (Fin n) → Reg → List α)
    (hp : ∀ S R R', plan S R = plan S R') {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R (j : Fin (plan S R).length),
      (2 : ℝ) ^ w ((plan S R).get j) / (Real.exp 1 * k) ≤ ρ := by
  rcases isEmpty_or_nonempty Reg with hR | ⟨⟨R₀⟩⟩
  · exact ⟨1, le_rfl, fun _ R => isEmptyElim R⟩
  set B : ℕ := ∑ S : Finset (Fin n), ((plan S R₀).map fun x => 2 ^ w x).sum
  refine ⟨⌈(B : ℝ) / (Real.exp 1 * ρ)⌉₊ + 1, Nat.le_add_left 1 _, fun S R j => ?_⟩
  have hmem : (plan S R).get j ∈ plan S R₀ := by rw [← hp S R R₀]; exact List.get_mem _ _
  have hle : 2 ^ w ((plan S R).get j) ≤ B :=
    (List.le_sum_of_mem (List.mem_map.2 ⟨_, hmem, rfl⟩)).trans
      (Finset.single_le_sum (f := fun S => ((plan S R₀).map fun x => 2 ^ w x).sum)
        (fun _ _ => Nat.zero_le _) (Finset.mem_univ S))
  have hB : (2 : ℝ) ^ w ((plan S R).get j) ≤ B := by exact_mod_cast hle
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  have hc : (B : ℝ) / (Real.exp 1 * ρ) ≤ ⌈(B : ℝ) / (Real.exp 1 * ρ)⌉₊ := Nat.le_ceil _
  rw [div_le_iff₀ (by positivity)] at hc
  rw [div_le_iff₀ (by positivity)]
  push_cast
  nlinarith [mul_pos he hρ]

/-- `exists_k` at the rate of a plan without `--zk` (`rateB`). -/
theorem exists_k_rateB {n mPts : ℕ} {Reg : Type} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (hp : ∀ S R R', plan S R = plan S R') {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R j, rateB (plan S R) j k ≤ ρ :=
  exists_k (fun T => T.l₀.logLen) plan hp hρ

/-- `exists_k` at the rate of a `--zk` plan (`rateZ`). -/
theorem exists_k_rateZ {F : Type} [Field F] [Fintype F] [DecidableEq F] {n mPts : ℕ} {Reg : Type}
    (plan : Finset (Fin n) → Reg → List (ZkSession.TabZK F mPts)) (hp : ∀ S R R', plan S R = plan S R')
    {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R j, ZkLink.rateZ (plan S R) j k ≤ ρ :=
  exists_k (fun T : ZkSession.TabZK F mPts => (lvOf T.sch 0).logLen) plan hp hρ

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
      (Integrate.planAt hs.typed dj x₀) (Integrate.tabAt hs.typed dj x₀) ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh) (Reg := Reg)).binding
        (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh)))
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
              (Integrate.tabAt hs.typed dj x₀) ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh) (Reg := Reg)).binding
                (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh)))
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
          ((rsAt hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh) (Reg := Reg)).binding
            (rsAt_computes hs.built hs.typed (hs.parse hh) dj x₀ (hs.load hh) hs.tables (hs.scope hh))) k Rw M ρ t')

theorem soundExec (I : Inputs) (hh : I.tags.hiddenOutputs = false) : SoundExec I hh :=
  @fun _ _ dj _ x₀ _ _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 _ hk _ _ hRw hM _ hρ hr _ ht' K wr dg N hwr hdraw τ hCR _ _ hKS =>
    flock_headline_exec_dec hh dj x₀ σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr dg N hwr hdraw τ hCR hKS

/-- `SoundExec`'s `hr` holds for some rerun count at every statement, draw file and setup. -/
theorem hr_exec (I : Inputs) {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (x₀ : DrawSetup I mPts S₀ (dj S₀)) (Reg : Type) {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ hs : Scope I, ∀ S R j, rateB (Integrate.planAt (Reg := Reg) hs.typed dj x₀ S R) j k ≤ ρ := by
  by_cases hs : Scope I
  · obtain ⟨k, hk, h⟩ := exists_k_rateB (Integrate.planAt (Reg := Reg) hs.typed dj x₀) (fun _ _ _ => rfl) hρ
    exact ⟨k, hk, fun _ => h⟩
  · exact ⟨1, le_rfl, fun h => absurd h hs⟩

/-- **`SoundExec`'s setup exists at every accepted draw.** Where flock-verify accepts a session whose record's draw is
`d`, setupH set up at `d` (`st`), `d` decodes (`D`), and when `D` proves `S` and `st`'s link points are `mPts`, an
`x₀ : DrawSetup I mPts S d` has statement `st`. -/
theorem setup_of_verifies (I : Inputs) (hh : I.tags.hiddenOutputs = false) {n mPts : ℕ} {S : Finset (Fin n)}
    {d : Lean.Json} {t : Exec.Transcript} (h : Verifies I t) (hd : recordDraw t.record = some d) :
    ∃ st D, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st ∧
      Draw.UnitDraw.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : DrawSetup I mPts S d, x.st = st) :=
  drawSetup_of_accepts h.2.typed (h.2.parse hh) h.2.outNet hh h.2.count h.1 hd

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
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
        (rsAtH_computes hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables))
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
              (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
                (rsAtH_computes hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables))
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
          ((rsAtH hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
            (rsAtH_computes hs.typed (hs.parseH hh) hs.scopeH dj x₀ hs.built (hs.loadH hh) hs.tables)) k Rw M ρ t')

theorem soundExecH (I : Inputs) (hh : I.tags.hiddenOutputs = true) : SoundExecH I hh :=
  @fun _ _ dj _ x₀ _ _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 _ hk _ _ hRw hM _ hρ hr _ ht' K wr dg N hwr hdraw τ hCR _ _ hKS =>
    flock_headline_exec_hidden_dec hh dj x₀ σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K wr dg N hwr hdraw τ hCR hKS

/-- `SoundExecH`'s `hr` holds for some rerun count at every statement, draw file and setup. -/
theorem hr_execH (I : Inputs) {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (Reg : Type) {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ :=
  exists_k_rateB (planAtH dj x₀) (fun _ _ _ => rfl) hρ

/-- **`SoundExecH`'s setup exists at every accepted draw** (`setup_of_verifies` under hidden outputs). -/
theorem setupH_of_verifies (I : Inputs) (hh : I.tags.hiddenOutputs = true) {n mPts : ℕ} {S : Finset (Fin n)}
    {d : Lean.Json} {t : Exec.Transcript} (h : Verifies I t) (hd : recordDraw t.record = some d) :
    ∃ st D, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program =
        .ok st ∧ Draw.UnitDraw.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : DrawSetupH I mPts S d, x.st = st) :=
  drawSetupH_of_accepts h.2.typed (h.2.parseH hh) h.2.outNet hh h.2.count h.1 hd

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

theorem soundZ (I : Inputs) (hh : I.tags.hiddenOutputs = false) (hn : I.count ≤ 1) : SoundZ I hh hn :=
  @fun _ _ _ dj _ y₀ wr dg N _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_sound I hh hn dj y₀ wr dg N σs ks hks St hS Kd pop Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

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

theorem soundZJ (I : Inputs) (hh : I.tags.hiddenOutputs = false) : SoundZJ I hh :=
  @fun _ _ _ dj _ _ y₀ wr dg N kd hkd _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_soundJ I hh dj y₀ wr dg N kd hkd Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

/-- `SoundZ`'s `hr` holds for some rerun count at every statement, draw file and setup. -/
theorem hr_z (I : Inputs) {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (Reg : Type) {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ :=
  exists_k_rateZ (planZAt dj y₀) (fun _ _ _ => rfl) hρ

/-- `SoundZJ`'s `hr` holds for some rerun count at every statement, draw file and setup. -/
theorem hr_zj (I : Inputs) {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀) (Reg : Type) {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ :=
  exists_k_rateZ (planZJ dj y₀) (fun _ _ _ => rfl) hρ

/-- **`SoundZ`'s setup exists at every accepted draw**: where flock-verify `--zk` accepts a one-table session whose
record's draw is `dj S`, at a draw the coin server's file proves with the session's link points (`DrawOkZK`), there is a
`y₀ : DrawSetupZK I mPts S (dj S)`. -/
theorem setupZ_of_verifiesZK (I : Inputs) (hh : I.tags.hiddenOutputs = false) (hn : I.count ≤ 1) {n mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : VerifiesZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZK I dj mPts S) :
    Nonempty (DrawSetupZK I mPts S (dj S)) := by
  obtain ⟨st, hst, hsh, hv⟩ := acceptsZK_one I _ hh hn h.1
  rw [hd] at hst
  obtain ⟨-, sess, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs, c, hz, -⟩ := hall 0 (by simp)
  have hz' : Zk.schedule st.m = .ok (schs[0].1, schs[0].2) := hz
  generalize schs[0].1 = sch at hz'
  generalize schs[0].2 = tPad at hz'
  obtain ⟨-, -, sch₀, hf⟩ := schedule_spec hz'
  have ht' : (zkInputs I).tags.typed = false := (zkTags_typed I).trans h.2.typed
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, rfl⟩ := drawSetup_of_setupH_g (I := zkInputs I) (S := S) ht' hst hf hD (hok.1 D hD) (hok.2 _ hst)
  obtain ⟨y, -, -, -⟩ := drawSetupZK_of h.2.typed x hsh hz'
  exact ⟨y⟩

/-- **`SoundZJ`'s setup exists at every accepted draw, at every number of tables**: where flock-verify `--zk` accepts a
session whose record's draw is `dj S`, at a draw whose parts the coin server's file proves with the session's link
points (`DrawOkZKJ`), table 0 sets up, so there is a `y₀ : DrawSetupZJ (inputsJ I 0) mPts _ _`. -/
theorem setupZJ_of_verifiesZK (I : Inputs) (hh : I.tags.hiddenOutputs = false) {n mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : VerifiesZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKJ I dj mPts S) :
    Nonempty (DrawSetupZJ (inputsJ I 0) mPts (partS I (dj S) S ⟨0, nTab_pos I⟩) (drawJ I (dj S) 0)) := by
  obtain ⟨spec, sts, hs, -, hv⟩ := h.1
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJ hh hd hs
  have hj : 0 < sts.size := hsz ▸ nTab_pos I
  have hst := hstJ 0 hj
  obtain ⟨-, sess, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, c0, hz0, -⟩ := hall 0 (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Zk.schedule sts[0].m = .ok (schs[0].1, schs[0].2) := hz0
  obtain ⟨-, -, sch₀, hf⟩ := schedule_spec hz'
  have ht' : (inputsJ I 0).tags.typed = false := typedJ h.2.typed 0
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I 0) (S := partS I (dj S) S ⟨0, nTab_pos I⟩) (mPts := mPts)
    ht' hst hf hD ((hok ⟨0, nTab_pos I⟩).1 D hD) ((hok ⟨0, nTab_pos I⟩).2 _ hst)
  obtain ⟨y, -, -, -⟩ := drawSetupZJ_of ht' x (sch := schs[0].1) (tPad := schs[0].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

/-- Zero knowledge at a public-output statement of one table under `--zk`, at every statement the check passes:
`zk_session_composed`'s second half, at the pinned key's hiding gap (`Composed.zk_session_view`).

The bound is per hidden leaf the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`, `ZkView.saltHash`),
`hidingGap My saltHash`, where hm96 hiding holds with no premise (`CROnly.hm96Hiding_gap`). Nothing here proves it small
for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and that `K*` is not one of the
others is unproved. -/
def ViewZ (I : Inputs) : Prop :=
  ∀ (P : LigProver) {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)),
    ScopeZ I → ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
      ZkViewAt (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st) (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))

theorem viewZ (I : Inputs) : ViewZ I :=
  @fun P _ _ _ dj _ y₀ hs => Composed.zk_session_view P I dj y₀ hs.typed (fun _ y => DrawSetupZK.avoidsMask y)
    (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)

end

section

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

set_option linter.unusedVariables false in
/-- The claim at a hidden-output statement of one table under `--zk`, the law stratified: `zk_session_soundH`'s. -/
def SoundZH (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1) : Prop :=
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
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ) ω)) qF qS),
    prob (fun o => FlockVerifyAccepts true I o.2.2 ∧ ∀ hs : ScopeZ I,
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZH dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : ScopeZ I, ((Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)))

theorem soundZH (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1) : SoundZH I hh hn :=
  @fun _ _ _ dj _ y₀ wr dg N _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_soundH I hh hn dj y₀ wr dg N σs ks hks St hS Kd pop Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

set_option linter.unusedVariables false in
/-- The claim at a hidden-output statement of any number of tables under `--zk`, the law the live subset draw: `zk_session_soundHJ`'s. -/
def SoundZHJ (I : Inputs) (hh : I.tags.hiddenOutputs = true) : Prop :=
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
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS),
    prob (fun o => FlockVerifyAccepts true I o.2.2 ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg σ) (cont σ))

theorem soundZHJ (I : Inputs) (hh : I.tags.hiddenOutputs = true) : SoundZHJ I hh :=
  @fun _ _ _ dj _ _ y₀ wr dg N kd hkd _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_soundHJ I hh dj y₀ wr dg N kd hkd Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

/-- `SoundZH`'s `hr` holds for some rerun count at every statement, draw file and setup. -/
theorem hr_zh (I : Inputs) {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (Reg : Type) {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ :=
  exists_k_rateZ (planAtZKH dj y₀) (fun _ _ _ => rfl) hρ

/-- `SoundZHJ`'s `hr` holds for some rerun count at every statement, draw file and setup. -/
theorem hr_zhj (I : Inputs) {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (Reg : Type) {ρ : ℝ} (hρ : 0 < ρ) :
    ∃ k : ℕ, 1 ≤ k ∧ ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ :=
  exists_k_rateZ (planZHJ dj y₀) (fun _ _ _ => rfl) hρ

/-- **`SoundZH`'s setup exists at every accepted draw** (`setupZ_of_verifiesZK` under hidden outputs). -/
theorem setupZH_of_verifiesZK (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1) {n mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : VerifiesZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKH I dj mPts S) :
    Nonempty (DrawSetupZKH I mPts S (dj S)) :=
  ZkHidden.drawSetupZKH_of_accepts ZkHidden.zk_schedOK h.2.typed (h.2.parseH hh) (Checks.outNet h.2) hh hn h.1 hd hok

/-- **`SoundZHJ`'s setup exists at every accepted draw, at every number of tables** (`setupZJ_of_verifiesZK` under
hidden outputs). -/
theorem setupZHJ_of_verifiesZK (I : Inputs) (hh : I.tags.hiddenOutputs = true) {n mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : VerifiesZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKHJ I dj mPts S) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) mPts (partS I (dj S) S ⟨0, nTab_pos I⟩) (drawJ I (dj S) 0)) := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h.1
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hh hd hs
  have hj : 0 < sts.size := hsz ▸ nTab_pos I
  have hst := hstJ 0 hj
  obtain ⟨-, sess, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, c0, hz0, -⟩ := hall 0 (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Zk.schedule sts[0].m = .ok (schs[0].1, schs[0].2) := hz0
  obtain ⟨sch₀, hf⟩ := fast100_of_schedule hz'
  have ht' : (inputsJ I 0).tags.typed = false := typedJ h.2.typed 0
  have hcJ : HmOut.parse (inputsJ I 0).tags (inputsJ I 0).circuitFile (inputsJ I 0).tables = .ok (circZ I) :=
    (parseH_tagsAt ((zkTags_typed I).trans h.2.typed) _ _ _ _).trans (h.2.parseH hh)
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[0].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := drawSetupH_of_setupHidden (I := inputsJ I 0) (S := partS I (dj S) S ⟨0, nTab_pos I⟩)
    (mPts := mPts) ht' hcJ (Checks.outNet h.2) hst hf hD ((hok ⟨0, nTab_pos I⟩).1 D hD) ((hok ⟨0, nTab_pos I⟩).2 _ hst)
  obtain ⟨y, -, -, -⟩ := drawSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape 0 hj)
    (sch := schs[0].1) (tPad := schs[0].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

/-- Zero knowledge at a hidden-output statement of any number of tables under `--zk`, at every statement the check
passes: `zk_session_composedHJ`'s second half, at the pinned key's hiding gap (`ZkHidden.zk_session_viewHJ`).

The bound is per hidden leaf the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`, `ZkView.saltHash`),
`hidingGap My saltHash`, where hm96 hiding holds with no premise (`CROnly.hm96Hiding_gap`). Nothing here proves it small
for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and that `K*` is not one of the
others is unproved. -/
def ViewZHJ (I : Inputs) : Prop :=
  ∀ (P : LigProver) {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀),
    ScopeZ I → ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))

theorem viewZHJ (I : Inputs) : ViewZHJ I :=
  @fun P _ _ _ dj _ _ y₀ hs =>
    ZkHidden.zk_session_viewHJ P I dj y₀ hs.typed (fun _ _ y => DrawSetupZHJ.avoidsMask y)
      (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)

open FlockSoundness.Discharge.ZkReg (OwnReadsHJ regPathsHJ rsShHJS rsZHJS rsZHJS_computes)

set_option linter.unusedVariables false in
/-- The claim at a hidden-output statement of any number of tables under `--zk --registered F` (`own` F's ports), the law
the live subset draw, at a public file with or without shared rows: `zk_session_soundHJR`'s, its bound and the
registered check's consequence at every accepted outcome. -/
def SoundZHJR (I : Inputs) (hh : I.tags.hiddenOutputs = true) : Prop :=
  ∀ {n : ℕ} {mPts : ℕ} {Reg : Type} (own : Array Flock.Registered.Port)
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
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS),
    prob (fun o => FlockVerifyAcceptsR true I own o.2.2 ∧ ∀ hs : ScopeZR I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJS dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)).bindingZ (rsZHJS_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZR I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJS dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)).bindingZ (rsZHJS_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)))
          k Rw M ρ t' (reg σ) (cont σ)) ∧
    ∀ hs : ScopeZR I, ∀ o : Reg × Finset (Fin n) × Exec.Transcript, FlockVerifyAcceptsR true I own o.2.2 →
      ∀ rd : FlockSoundness.Registered.ReadsX
          (layShHJ (Reg := Reg) dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (pub := pubZ I))
          (rsShHJS dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh)),
        OwnReadsHJ I own (circZ I) (pubZ I) o.2.2 rd.isRead rd.port rd.readAt →
          rd.Checked o.1 (regPathsHJ I (circZ I) (pubZ I) o.2.2 rd.port)

theorem soundZHJR (I : Inputs) (hh : I.tags.hiddenOutputs = true) : SoundZHJR I hh :=
  @fun _ _ _ own dj _ _ y₀ wr dg N kd hkd _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_soundHJR I hh own dj y₀ wr dg N kd hkd Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

/-- Zero knowledge at a hidden-output statement of any number of tables under `--zk --registered F`, at every statement
the check passes with the verifier's registered reads: `ViewZHJ` at `ScopeZR`, at the same bound. The view reads the
statement's tags, not its public file, so a shared-row one changes nothing in it. -/
def ViewZHJR (I : Inputs) : Prop :=
  ∀ (P : LigProver) {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀),
    ScopeZR I → ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))

theorem viewZHJR (I : Inputs) : ViewZHJR I :=
  @fun P _ _ _ dj _ _ y₀ hs =>
    ZkHidden.zk_session_viewHJ P I dj y₀ hs.typed (fun _ _ y => DrawSetupZHJ.avoidsMask y)
      (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)

end

section

open Flock Exec ZkExec

/-- **Soundness, over everything flock-verify accepts.** At every statement `I`, with or without `--zk` and with or
without `--registered F`: the live session's probability that `flock-verify verify` accepts (`FlockVerifyAccepts zk I`,
with `--registered F` `FlockVerifyAcceptsR zk I own`) while at least `K` of the statement's proof units are wrong is at
most the case's bound. Without `--zk`, `SoundExec` (public outputs) or `SoundExecH` (hidden); under `--zk`, at one table
`SoundZ` with zero knowledge `ViewZ` (public) or `SoundZH` (hidden), and at every number of tables `SoundZJ` (public) or
`SoundZHJ` with zero knowledge `ViewZHJ` (hidden), and with `--registered F` at hidden outputs `SoundZHJR` with zero
knowledge `ViewZHJR`, at a public file with or without shared rows. Everywhere else `--registered F` accepts no more than
the verifier does without it (the second conjunct), so the claims without it hold. No hypothesis on the circuit file or
the statement's form: the verifier refuses every statement the claims do not cover (`Flock.ProvedScope.check`), and
there the event is empty.

The zero-knowledge views are within `2·|Hid|` times the commitments' hiding gap at hm96-sha512's pinned key `K*`,
`hidingGap My saltHash`, and take no premise on the key or on SHA-512. Nothing here proves the gap small for `K*`: it is
below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and that `K*` is not one of the others is
unproved. -/
theorem flock_verify_sound (zk : Bool) (I : Inputs) :
    (if hh : I.tags.hiddenOutputs = true then
      if zk then (∀ hn : I.count ≤ 1, SoundZH I hh hn) ∧ SoundZHJ I hh ∧ ViewZHJ I ∧ SoundZHJR I hh ∧ ViewZHJR I
      else SoundExecH I hh
    else
      if zk then
        (∀ hn : I.count ≤ 1, SoundZ I (Bool.eq_false_of_ne_true hh) hn ∧ ViewZ I) ∧ SoundZJ I (Bool.eq_false_of_ne_true hh)
      else SoundExec I (Bool.eq_false_of_ne_true hh)) ∧
    ∀ (own : Array Flock.Registered.Port) (t : Exec.Transcript), FlockVerifyAcceptsR zk I own t →
      (zk = true ∧ I.tags.hiddenOutputs = true) ∨ FlockVerifyAccepts zk I t := by
  refine ⟨?_, fun own t h => ?_⟩
  · split
    · next hh =>
      cases zk
      · show SoundExecH I hh
        exact @soundExecH I hh
      · show (∀ hn : I.count ≤ 1, SoundZH I hh hn) ∧ SoundZHJ I hh ∧ ViewZHJ I ∧ SoundZHJR I hh ∧ ViewZHJR I
        exact ⟨fun hn => @soundZH I hh hn, @soundZHJ I hh, @viewZHJ I, @soundZHJR I hh, @viewZHJR I⟩
    · next hh =>
      cases zk
      · show SoundExec I (Bool.eq_false_of_ne_true hh)
        exact @soundExec I _
      · show (∀ hn : I.count ≤ 1, SoundZ I (Bool.eq_false_of_ne_true hh) hn ∧ ViewZ I) ∧
          SoundZJ I (Bool.eq_false_of_ne_true hh)
        exact ⟨fun hn => ⟨@soundZ I _ hn, @viewZ I⟩, @soundZJ I _⟩
  · cases zk
    · change Verifies I t at h
      exact Or.inr h
    · change VerifiesZKR I own t at h
      by_cases hh : I.tags.hiddenOutputs = true
      · exact Or.inl ⟨rfl, hh⟩
      · exact Or.inr (verifiesZK_of_R h (Bool.eq_false_of_ne_true hh))

end

end FlockSoundness.Discharge.FailClosed
