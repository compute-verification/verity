import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts

/-!
# A draw flock-verify accepts under hidden outputs sets up

Integrate's `Accepted.lean` for `Stmt.setupHidden`. A `DrawSetupH` is more than setupHidden's acceptance: the draw proves
the drawn units, `0 < c.g`, the proved units fit the blocks, `m` has a `fast100` schedule of at least two levels, and
the statement's link points are `mPts`. All but the first and the last follow from a one-table acceptance, as on the
public path:

* `0 < c.g`: an untyped `HmOut.parse` is `HmRow.parse` then `HmOut.checkRows` (`parse_untyped`), and `HmRow.check`
  holds a power of two of VUs per block (`CheckFacts.g_pow2`), wherever the outputs are returned;
* the blocks: setupHidden's `nbl` is setupH's function of `n` (`setupHidden_n`), and a drawn statement's `n` is the
  draw's proved units (`HmOut.drawn` is `HmRow.drawn` with no output word, `Layout.drawn_spec`);
* the schedule: `verify_one_ok` and `model_sched`, as on the public path.

`drawSetupH_of_accepts` and `tableAtH_of_accepts` are `drawSetup_of_accepts` and `tableAt_of_accepts` under hidden
outputs (`hh : I.tags.hiddenOutputs = true`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs Accepts recordDraw)
open FlockSoundness.Discharge.Integrate (model_sched fits_blocks)

variable {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
  (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
include ht hc

/-- **A `DrawSetupH` from setupHidden's acceptance at a draw**, a `fast100` schedule of its `m`, the draw proving `S`,
and its link points `mPts`, wherever the outputs are returned. -/
theorem drawSetupH_of_setupHiddenI {st : Flock.Stmt}
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : DrawSetupH I mPts S d, x.st = st := by
  have hcst : st.c = c := setupHidden_c_of_parse hc h
  have hg : 0 < st.c.g := by
    have P := parse_facts (parse_untyped ht hc).1
    obtain ⟨ls, hls⟩ := P.check
    have hp := (check_facts P.typed hls).g_pow2
    simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
    exact hcst ▸ hp.1
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
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
/-- `drawSetupH_of_setupHiddenI` where the unit returns the outputs. -/
theorem drawSetupH_of_setupHidden (ho : c.outNet = c.unitNet) {st : Flock.Stmt}
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : DrawSetupH I mPts S d, x.st = st :=
  drawSetupH_of_setupHiddenI ht hc h hf hD hS hmPts

/-- **A draw flock-verify accepts under hidden outputs sets up.** A one-table acceptance whose record's draw is `d`:
setupHidden accepted its inputs at `d` (`st`), the draw decodes (`D`), and whenever `D` proves `S` and `st`'s link points
are `mPts`, a `DrawSetupH` at `d` has statement `st`. -/
theorem drawSetupH_of_accepts (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1) {t : Exec.Transcript}
    (h : Accepts I t) (hd : recordDraw t.record = some d) :
    ∃ st D, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program =
        .ok st ∧ Draw.UnitDraw.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : DrawSetupH I mPts S d, x.st = st) := by
  obtain ⟨st, hst, hv⟩ := accepts_oneAny hc1 h
  rw [setupOne_hidden hh, hd] at hst
  obtain ⟨-, sch, -, -, hf, -, -⟩ := verify_one_ok hv
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  exact ⟨st, D, hst, hD, fun hS hm => drawSetupH_of_setupHiddenI ht hc hst hf hD hS hm⟩

/-- **At a draw flock-verify accepts under hidden outputs, the headline's table is `stmtOf` of the statement it
accepted**, wherever the outputs are returned (`TableAtHOfAccepts` without `c.outNet = c.unitNet`). -/
theorem tableAtH_of_acceptsI (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {t : Exec.Transcript} (h : Accepts I t)
    (hd : recordDraw t.record = some (dj S))
    (hS : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition
      I.program = .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
        .ok st ∧ ∃ hwf : StmtWF st, (tableAtH dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  obtain ⟨st, D, hst, hD, hx⟩ := drawSetupH_of_accepts (S := S) (mPts := mPts) ht hc hh hc1 h hd
  obtain ⟨x, rfl⟩ := hx (hS D hD) (hmPts _ hst)
  obtain ⟨y, hy, e⟩ := tableAtH_stmt dj x₀ x
  exact ⟨x.st, hst, hy ▸ ⟨y.wf.1, congrArg (fun T : TableH I mPts S => T.T.S) e⟩⟩

omit ht hc in
theorem tableAtH_of_accepts : TableAtHOfAccepts := by
  intro I mPts n S c ht hc _ hh hc1 dj S₀ x₀ t h hd hS hmPts
  exact tableAtH_of_acceptsI ht hc hh hc1 dj x₀ h hd hS hmPts

end FlockSoundness.Discharge.HiddenExec
