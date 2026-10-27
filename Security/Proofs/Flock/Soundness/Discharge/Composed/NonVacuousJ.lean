import Proofs.Flock.Soundness.Discharge.Composed.CustodyJJson
import Proofs.Flock.Soundness.Assumptions.CustodyZk

/-!
# The `J`-table statements' draw hypotheses hold together

`zk_session_soundJ`, `zk_session_soundJ_custody` and `zk_session_soundJ_custody_json` take a reference setup `y₀` (the
refused table's stand-in, `tableZJ`) and, directly or through the draw file's custody, `DrawOkZKJ` at every draw of the
law, all at one link-point count. `y₀` is a setup at table 0's inputs, not a `DrawSetupZK` of the whole statement at the
whole draw: that one has the whole draw's link points, `mPtsOf c.g kd`, while the tables have their part's,
`mPtsOf c.g (kd / nTab I)`; at 16 drawn units, two tables and one unit per block these are 28 and 27, and no such `y₀`
exists. This file proves the hypotheses satisfiable together and the whole-draw binder empty:

* `setupZJ_of_acceptsZK`: every table of a session `verify --zk` accepts at a draw `DrawOkZKJ` admits is set up
  (`DrawSetupZJ` at table `j`'s inputs and draw).
* `drawHypsZJ`: at the coin server's draw file (`DrawFileZK`, `subset kd`) with the tables dividing `kd`, `DrawOkZKJ`
  holds at every draw and every table of an accepted session is set up, all at `mPtsOf c.g (kd / nTab I)` link points.
* `zk_session_soundJ_nonvacuous`, `zk_session_soundJ_custody_nonvacuous`, `zk_session_soundJ_custody_json_nonvacuous`:
  each statement's `y₀` and draw hypotheses at one accepted session, each proof applying its statement to the `y₀` it
  builds, so a binder that no longer admits it fails the build.
* `drawSetupZK_whole_mPts`, `e2e_forces_mPts`, `drawSetupZK_whole_empty_16_2_1`: a whole-draw `DrawSetupZK` at the draw
  file has `mPtsOf c.g kd` link points, so at 16 drawn units, two tables and `g = 1` that binder is empty.
* `y0_binder_16_2_1`: at the same draw the statements' `y₀` exists at 27 link points and the whole-draw one doesn't.

What none of this covers: that a session `verify --zk` accepts exists at a draw of the file (the live run's fact, which
the kernel can't evaluate: setupH hashes and parses the inputs), `RecordsLiveZKJ` (custody of `wr`), the input facts
`hU`, `hpub`, `hpt`, `hscope`, `hkd`, A3, and the count curve's parameters and `TableCRZ`.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {n mPts : ℕ}

/-- **Every table of a session `verify --zk` accepts is set up.** On untyped tags with public outputs, at a session
`verify --zk` accepts whose record carries the draw `dj S`, at which `DrawOkZKJ` holds at `mPts` link points: table `j`
has a `DrawSetupZJ` at its inputs, its draw's units of `S` and its draw. -/
theorem setupZJ_of_acceptsZK (I : Inputs) (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : Assumptions.Zk.DrawAtZKJ I dj mPts S)
    (j : Fin (nTab I)) :
    Nonempty (DrawSetupZJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) := by
  obtain ⟨spec, sts, hs, -, hv⟩ := h
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJ hh hd hs
  have hj : j.val < sts.size := by have := j.isLt; omega
  have hst := hstJ j.val hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, _, hz0, -⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨-, -, _, hf⟩ := schedule_spec hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ ht j.val
  obtain ⟨_, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I j.val) (S := partS I (dj S) S j) (mPts := mPts)
    ht' hst hf hD ((hok j).1 D hD) ((hok j).2 _ hst)
  obtain ⟨y, -, -, -⟩ := drawSetupZJ_of ht' x (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

/-- **The `J`-table draw hypotheses hold together at an accepted session.** On untyped inputs under `TagsOkZJ`'s tags,
whose circuit file parses to `c`: when the record's draw file is the verifier's `subset kd` draw (`DrawFileZK`) and the
tables divide `kd`, `DrawOkZKJ` holds at every draw, and at a session `verify --zk` accepts at the draw `dj S` every
table is set up, all at `mPtsOf c.g (kd / nTab I)` link points. -/
theorem drawHypsZJ (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    (∀ S', DrawOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S') ∧
      ∀ j : Fin (nTab I),
        Nonempty (DrawSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j)) :=
  ⟨drawOkZKJ_of_file I ht hc kd n hdiv dj hdj,
    setupZJ_of_acceptsZK I hTags.1 ht dj h hd (drawOkZKJ_of_file I ht hc kd n hdiv dj hdj S)⟩

/-- **`zk_session_soundJ`'s `y₀` and `hDraw` hold together.** At the coin server's draw file (`DrawFileZK`, `subset kd`,
the tables dividing `kd`) and one session `verify --zk` accepts at the draw `dj S`, on the inputs `ht`, `hTags` and `hc`
describe: at `mPts = mPtsOf c.g (kd / nTab I)` there is a `y₀` (table 0's setup), every table is set up, and `DrawOkZKJ`
holds at every draw, so at every draw of the law (`hDraw`). The proof applies `zk_session_soundJ` to this `y₀` and
`hDraw`. Not covered: the session's existence, `hRec`, `hU`, `hpub`, `hpt`, `hscope`, `hkd`, `hA3`, the count curve's
parameters and `TableCRZ`. -/
theorem zk_session_soundJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZJ I ht hTags hc kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (Lb : Fin 1 → ℕ) (src : Unit → (i : Fin 1) → Fin (Lb i) → Fin 256) =>
    zk_session_soundJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀) (c := c)
      (pub := pub) (kd := kd) (Ω := Unit) (Lb := Lb) (src := src) (hDraw := fun _ => hok _)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **`zk_session_soundJ_custody`'s `y₀` and the custody's draw file hold together.** At the draw file `hCust.2` is
(`DrawFileZK`, `subset kd`), the tables dividing `kd` (`hdiv`), and a session `verify --zk` accepts at the draw `dj S`,
on the inputs `ht`, `hTags` and `hc` describe: there is a `y₀` at `mPtsOf c.g (kd / nTab I)` link points (table 0's
setup), every table is set up there, and `DrawOkZKJ`, which the statement derives from the draw file, holds at every
draw at the same count. The proof applies `zk_session_soundJ_custody` to this `y₀`. Not covered: the session's
existence, `hCust.1` (`RecordsLiveZKJ`), `hU`, `hpub`, `hpt`, `hscope`, `hkd`, `hA3`, the count curve's parameters and
`TableCRZ`. -/
theorem zk_session_soundJ_custody_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZJ I ht hTags hc kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) =>
    zk_session_soundJ_custody (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **`zk_session_soundJ_custody_json`'s `y₀` and `hdj` hold together.** As `zk_session_soundJ_custody_nonvacuous`, with
each draw file the JSON value `flock-verify draw` printed for the verifier's `subset kd` draw (`hdj`). The proof applies
`zk_session_soundJ_custody_json` to the `y₀` it builds and to `hdj`. Not covered: the session's existence, `hRec`,
`hU`, `hpub`, `hpt`, `hscope`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundJ_custody_json_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZJ I ht hTags hc kd hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (k Rw M : ℕ) =>
    zk_session_soundJ_custody_json (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
      (Ω := Unit) (hdj := hdj) (k := k) (Rw := Rw) (M := M)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-! ## The whole-draw binder -/

/-- A draw file of the verifier's `subset kd` draw proves `kd` units. -/
theorem drawFileZK_proved_size {kd pop : ℕ} {dj : Finset (Fin n) → Lean.Json} (hdj : DrawFileZK dj (.subset kd) pop)
    (S : Finset (Fin n)) : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → D.proved.size = kd := by
  intro D hD
  have hDe := hdj S D hD
  have hsz : D.units.size = kd := (ofJson_shape hD).2.2 kd (by rw [hDe]; rfl)
  rw [hDe, drawOf_proved, ← hDe, hsz]

/-- **A whole-draw setup has the whole draw's link points.** On untyped tags whose circuit file parses to `c` under the
`--zk` tags, a `DrawSetupZK` of the whole statement at a draw file of the verifier's `subset kd` draw has
`mPtsOf c.g kd` link points. -/
theorem drawSetupZK_whole_mPts (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) {kd pop : ℕ}
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) : mPts = mPtsOf c.g kd := by
  obtain ⟨ud, hud, hm⟩ := setupH_mPts_drawn y₀.x.h
  have hcst := (setupH_facts y₀.x.h).1
  rw [zkTyped ht] at hcst
  simp only [Bool.false_eq_true, ↓reduceIte] at hcst
  have hst : y₀.x.st.c = c := Except.ok.inj (hcst.symm.trans hc)
  rw [← y₀.x.hmPts, hm, hst, drawFileZK_proved_size hdj S₀ ud hud]

/-- **A whole-draw `y₀` forces the whole draw's link points onto a table's**: a whole-draw `DrawSetupZK` at
`mPtsOf c.g (kd / nTab I)` link points, at a draw file of the verifier's `subset kd` draw, makes both counts equal. -/
theorem e2e_forces_mPts (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I (mPtsOf c.g (kd / nTab I)) S₀ (dj S₀))
    (hdj : DrawFileZK dj (.subset kd) n) :
    mPtsOf c.g kd = mPtsOf c.g (kd / nTab I) :=
  (drawSetupZK_whole_mPts I ht hc dj hdj y₀).symm

/-- 16 drawn units at one unit per block take 28 link points, 8 take 27. -/
theorem mPtsOf_16_8 : mPtsOf 1 16 = 28 ∧ mPtsOf 1 8 = 27 := by decide

theorem mPtsOf_16_ne_8 : mPtsOf 1 16 ≠ mPtsOf 1 8 := by decide

/-- **At 16 drawn units, two tables and `g = 1`, the whole-draw binder is empty**: no whole-draw `DrawSetupZK` at
`mPtsOf c.g (16 / nTab I)` link points exists at a draw file of the verifier's `subset 16` draw. -/
theorem drawSetupZK_whole_empty_16_2_1 (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hJ : nTab I = 2) (hg : c.g = 1)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset 16) n) (S₀ : Finset (Fin n)) :
    IsEmpty (DrawSetupZK I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) :=
  ⟨fun y₀ => by
    have h := e2e_forces_mPts I ht hc 16 dj y₀ hdj
    rw [hJ, hg] at h
    exact mPtsOf_16_ne_8 h⟩

/-- **At 16 drawn units, two tables and `g = 1`, the statements' `y₀` exists and a whole-draw one doesn't.** At the
verifier's `subset 16` draw file and one session `verify --zk` accepts at the draw `dj S`: the tables' count is 27, a
`y₀` (`zk_session_soundJ`'s `DrawSetupZJ` at table 0's inputs) exists at it, and no whole-draw `DrawSetupZK` does. -/
theorem y0_binder_16_2_1 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hJ : nTab I = 2) (hg : c.g = 1)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset 16) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    mPtsOf c.g (16 / nTab I) = 27 ∧
      Nonempty (DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (16 / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      ∀ S₀ : Finset (Fin n), IsEmpty (DrawSetupZK I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) := by
  refine ⟨?_, (drawHypsZJ I ht hTags hc 16 (by simp [hJ]) dj hdj h hd).2 ⟨0, nTab_pos I⟩,
    drawSetupZK_whole_empty_16_2_1 I ht hc hJ hg dj hdj⟩
  rw [hJ, hg]
  exact mPtsOf_16_8.2

end FlockSoundness.Discharge.Composed
