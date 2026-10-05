import FlockSoundness.Discharge.FailClosed.Composed
import FlockSoundness.Discharge.FailClosed.ZkHidden
import FlockSoundness.Discharge.FailClosed.Exec
import FlockSoundness.Discharge.Canonical.Headline
import FlockSoundness.Protocol.Discharge.FailClosed.One

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

theorem soundZ (I : Inputs) (hh : I.tags.hiddenOutputs = false) (hn : I.count ≤ 1) : SoundZ I hh hn :=
  @fun _ _ _ dj _ y₀ wr dg N _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_sound I hh hn dj y₀ wr dg N σs ks hks St hS Kd pop Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

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

theorem viewZ (I : Inputs) : ViewZ I :=
  @fun P _ _ _ dj _ y₀ hKey hs => Composed.zk_session_view P I dj y₀ hs.typed (fun _ y => DrawSetupZK.avoidsMask y) hKey

end

section

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

theorem soundZH (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hv : Layout.scopeOk (circZ I) = true)
    (hn : I.count ≤ 1) : SoundZH I hh hv hn :=
  @fun _ _ _ dj _ y₀ wr dg N _ σs ks hks St hS Kd pop _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_soundH I hh hv hn dj y₀ wr dg N σs ks hks St hS Kd pop Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

theorem soundZHJ (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hv : Layout.scopeOk (circZ I) = true) :
    SoundZHJ I hh hv :=
  @fun _ _ _ dj _ _ y₀ wr dg N kd hkd _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    zk_session_soundHJ I hh hv dj y₀ wr dg N kd hkd Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

theorem soundZHJC (I : Inputs) (hv : Layout.scopeOk (circZ I) = false) : SoundZHJC I hv :=
  @fun _ _ _ dj _ _ y₀ wr dg N kd hkd _ _ _ Lb src hA3 hDraw hRec _ _ _ hk hRw hM _ hρ hr _ ht' K₀ τ _ _ hCR =>
    Canonical.zk_session_soundC_verifies I hv dj y₀ wr dg N kd hkd Lb src hA3 hDraw hRec hk hRw hM hρ hr ht' K₀ τ hCR

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

theorem viewZHJ (I : Inputs) : ViewZHJ I :=
  @fun P _ _ _ dj _ _ y₀ hKey hs =>
    ZkHidden.zk_session_viewHJ P I dj y₀ hs.typed (fun _ _ y => DrawSetupZHJ.avoidsMask y) hKey

end

section

open Flock Exec ZkExec

/-- **Soundness, over everything flock-verify accepts.** At every statement `I`, with or without `--zk`: the live session's
probability that `flock-verify verify` accepts (`FlockVerifyAccepts zk I`) while at least `K` of the statement's proof
units are wrong is at most the case's bound. Without `--zk`, `SoundExec` (public outputs) or `SoundExecH` (hidden);
under `--zk`, at one table `SoundZ` with zero knowledge `ViewZ` (public) or `SoundZH` (hidden), and at every number of
tables `SoundZJ` (public) or `SoundZHJ` with zero knowledge `ViewZHJ` (hidden); at hidden outputs, a circuit outside
`Layout.scopeOk` is in the canonical format, and the claim there is `SoundZHJC`. No hypothesis on the circuit file or
the statement's form: the verifier refuses every statement the claims do not cover (`Flock.ProvedScope.check`), and
there the event is empty. -/
theorem flock_verify_sound (zk : Bool) (I : Inputs) :
    if hh : I.tags.hiddenOutputs = true then
      if zk then
        (if hv : Layout.scopeOk (circZ I) = true then (∀ hn : I.count ≤ 1, SoundZH I hh hv hn) ∧ SoundZHJ I hh hv
          else SoundZHJC I (Bool.eq_false_of_ne_true hv)) ∧ ViewZHJ I
      else SoundExecH I hh
    else
      if zk then
        (∀ hn : I.count ≤ 1, SoundZ I (Bool.eq_false_of_ne_true hh) hn ∧ ViewZ I) ∧ SoundZJ I (Bool.eq_false_of_ne_true hh)
      else SoundExec I (Bool.eq_false_of_ne_true hh) := by
  split
  · next hh =>
    cases zk
    · show SoundExecH I hh
      exact @soundExecH I hh
    · show (if hv : Layout.scopeOk (circZ I) = true then (∀ hn : I.count ≤ 1, SoundZH I hh hv hn) ∧ SoundZHJ I hh hv
          else SoundZHJC I (Bool.eq_false_of_ne_true hv)) ∧ ViewZHJ I
      refine ⟨?_, @viewZHJ I⟩
      split
      · next hv => exact ⟨fun hn => @soundZH I hh hv hn, @soundZHJ I hh hv⟩
      · next hv => exact @soundZHJC I _
  · next hh =>
    cases zk
    · show SoundExec I (Bool.eq_false_of_ne_true hh)
      exact @soundExec I _
    · show (∀ hn : I.count ≤ 1, SoundZ I (Bool.eq_false_of_ne_true hh) hn ∧ ViewZ I) ∧
        SoundZJ I (Bool.eq_false_of_ne_true hh)
      exact ⟨fun hn => ⟨@soundZ I _ hn, @viewZ I⟩, @soundZJ I _⟩

end

end FlockSoundness.Discharge.FailClosed
