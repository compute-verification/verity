module

public import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Statements
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
import Proofs.Flock.Soundness.Discharge.Layout.Drawn

/-! A sample verity-verify-proof accepts under hidden outputs sets up (`sampleSetupH_of_accepts`, `tableAtH_of_accepts`). -/

@[expose] public section

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs Accepts recordDraw)
open FlockSoundness.Discharge.Integrate (model_sched fits_blocks)

variable {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
  (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
include ht hc

theorem sampleSetupH_of_setupHiddenI {st : Flock.Stmt}
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitSample}
    (hD : Draw.UnitSample.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : SampleSetupH I mPts S d, x.st = st := by
  have hcst : st.c = c := setupHidden_c_of_parse hc h
  have hg : 0 < st.c.g := by
    have P := parse_facts (parse_untyped ht hc).1
    obtain ⟨ls, hls⟩ := P.check
    have hp := (check_facts P.typed hls).g_pow2
    simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
    exact hcst ▸ hp.1
  obtain ⟨pop, -, hdr⟩ := setupHidden_sampled h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.sampled_spec (c := { st.c with outPorts := #[] }) hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := setupHidden_n h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts, setupHidden_wf h⟩, rfl⟩

set_option linter.unusedVariables false in
theorem sampleSetupH_of_setupHidden (ho : c.outNet = c.unitNet) {st : Flock.Stmt}
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitSample}
    (hD : Draw.UnitSample.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : SampleSetupH I mPts S d, x.st = st :=
  sampleSetupH_of_setupHiddenI ht hc h hf hD hS hmPts

theorem sampleSetupH_of_accepts (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1) {t : Exec.Transcript}
    (h : Accepts I t) (hd : recordDraw t.record = some d) :
    ∃ st D, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program =
        .ok st ∧ Draw.UnitSample.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : SampleSetupH I mPts S d, x.st = st) := by
  obtain ⟨st, hst, hv⟩ := accepts_oneAny hc1 h
  rw [setupOne_hidden hh, hd] at hst
  obtain ⟨-, sch, -, -, hf, -, -⟩ := verify_one_ok hv
  obtain ⟨pop, -, hdr⟩ := setupHidden_sampled hst
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec (c := { st.c with outPorts := #[] }) hdr
  exact ⟨st, D, hst, hD, fun hS hm => sampleSetupH_of_setupHiddenI ht hc hst hf hD hS hm⟩

theorem tableAtH_of_acceptsI (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀)) {t : Exec.Transcript} (h : Accepts I t)
    (hd : recordDraw t.record = some (dj S))
    (hS : ∀ D, Draw.UnitSample.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition
      I.program = .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
        .ok st ∧ ∃ hwf : StmtWF st, (tableAtH dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  obtain ⟨st, D, hst, hD, hx⟩ := sampleSetupH_of_accepts (S := S) (mPts := mPts) ht hc hh hc1 h hd
  obtain ⟨x, rfl⟩ := hx (hS D hD) (hmPts _ hst)
  obtain ⟨y, hy, e⟩ := tableAtH_stmt dj x₀ x
  exact ⟨x.st, hst, hy ▸ ⟨y.wf.1, congrArg (fun T : TableH I mPts S => T.T.S) e⟩⟩

omit ht hc in
theorem tableAtH_of_accepts : TableAtHOfAccepts := by
  intro I mPts n S c ht hc _ hh hc1 dj S₀ x₀ t h hd hS hmPts
  exact tableAtH_of_acceptsI ht hc hh hc1 dj x₀ h hd hS hmPts

end FlockSoundness.Discharge.HiddenExec
