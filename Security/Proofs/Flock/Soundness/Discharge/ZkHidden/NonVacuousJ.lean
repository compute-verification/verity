import Proofs.Flock.Soundness.Discharge.Composed.NonVacuousJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJJson
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk
import Proofs.Flock.Soundness.Discharge.ZkHidden.View

/-!
# The hidden-output `J`-table statements' draw hypotheses hold together

`Composed/NonVacuousJ.lean` under hidden outputs (`Stmt.setupHidden`). `zk_session_soundHJ`, its custody forms,
`zk_session_composedHJ` and `zk_session_viewHJ` take a reference setup `y₀ : DrawSetupZHJ` at table 0's inputs and, the
first four, `DrawOkZKHJ` at every draw of the law, all at one link-point count; a `DrawSetupZKH` of the whole statement
at the whole draw has the whole draw's count, and none exists at the tables' at 16 drawn units, two tables and `g = 1`.

* `setupZHJ_of_acceptsZK`: every table of a session `verify --zk` accepts at a draw `DrawOkZKHJ` admits is set up.
* `drawHypsZHJ`: at the coin server's draw file with the tables dividing `kd`, `DrawOkZKHJ` at every draw and every
  table of an accepted session set up, at `mPtsOf c.g (kd / nTab I)` link points.
* `zk_session_soundHJ_nonvacuous`, `zk_session_soundHJ_custody_nonvacuous`,
  `zk_session_soundHJ_custody_json_nonvacuous`, `zk_session_composedHJ_nonvacuous`, `zk_session_viewHJ_nonvacuous`: each
  statement's `y₀` and draw hypotheses at one accepted session, each proof applying its statement to the `y₀` it builds.
* `drawSetupZKH_whole_mPts`, `drawSetupZKH_whole_empty_16_2_1`, `y0_binderH_16_2_1`: the whole-draw binder's count, its
  emptiness at 16 drawn units, two tables and `g = 1`, and the statements' `y₀` at the same draw.

What none of this covers: that a session `verify --zk` accepts exists at a draw of the file, `RecordsLiveZKJ`, `hpub`,
`hpt`, `hkd`, A3, the count curve's parameters, `TableCRZ`, `zk_session_composedHJ`'s and `zk_session_viewHJ`'s `hdm`
(`avoidsMask`, which no check implies), and `zk_session_composedHJ`'s `hKey` (a named assumption); `zk_session_viewHJ`'s
`hT1` holds at the pinned key's hiding gap (`CROnly.hm96Hiding_gap`).
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {n mPts : ℕ}

/-- **Every hidden-output table of a session `verify --zk` accepts is set up.** On untyped tags with hidden outputs,
whose circuit file `HmOut.parse` accepts under the `--zk` tags as `c`, a circuit whose unit returns the outputs, at a
session `verify --zk` accepts whose record carries the draw `dj S`, at which `DrawOkZKHJ` holds at `mPts` link points:
table `j` has a `DrawSetupZHJ` at its inputs, its draw's units of `S` and its draw. -/
theorem setupZHJ_of_acceptsZK (I : Inputs) (hh : I.tags.hiddenOutputs = true) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKHJ I dj mPts S) (j : Fin (nTab I)) :
    Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hh hd hs
  have hj : j.val < sts.size := by have := j.isLt; omega
  have hst := hstJ j.val hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, _, hz0, -⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨_, hf⟩ := fast100_of_schedule hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ ht j.val
  have hcJ : HmOut.parse (inputsJ I j.val).tags (inputsJ I j.val).circuitFile (inputsJ I j.val).tables = .ok c :=
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  obtain ⟨_, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := drawSetupH_of_setupHidden (I := inputsJ I j.val) (S := partS I (dj S) S j) (mPts := mPts)
    ht' hcJ ho hst hf hD ((hok j).1 D hD) ((hok j).2 _ hst)
  obtain ⟨y, -, -, -⟩ := drawSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape j.val hj)
    (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

/-- **The hidden-output `J`-table draw hypotheses hold together at an accepted session.** On untyped inputs under
`TagsOkZHJ`'s tags, whose circuit file `HmOut.parse` accepts as `c` (`hU`, `hscope`: its unit returns the outputs):
when the record's draw file is the verifier's `subset kd` draw (`DrawFileZK`) and the tables divide `kd`, `DrawOkZKHJ`
holds at every draw, and at a session `verify --zk` accepts at the draw `dj S` every table is set up, all at
`mPtsOf c.g (kd / nTab I)` link points. -/
theorem drawHypsZHJ (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) (hscope : Layout.scopeOkH c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0) (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    (∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S') ∧
      ∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j)) :=
  ⟨drawOkZKHJ_of_file I ht hc kd n hdiv dj hdj,
    setupZHJ_of_acceptsZK I hTags.1 ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj h hd
      (drawOkZKHJ_of_file I ht hc kd n hdiv dj hdj S)⟩

/-- **`zk_session_soundHJ`'s `y₀` and `hDraw` hold together.** At the coin server's draw file (`DrawFileZK`,
`subset kd`, the tables dividing `kd`) and one session `verify --zk` accepts at the draw `dj S`, on the inputs `ht`,
`hTags`, `hc`, `hU` and `hscope` describe: at `mPts = mPtsOf c.g (kd / nTab I)` there is a `y₀` (table 0's setup),
every table is set up, and `DrawOkZKHJ` holds at every draw, so at every draw of the law (`hDraw`). The proof applies
`zk_session_soundHJ` to this `y₀` and `hDraw`. Not covered: the session's existence, `hRec`, `hpub`, `hpt`, `hkd`,
`hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundHJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (Lb : Fin 1 → ℕ) (src : Unit → (i : Fin 1) → Fin (Lb i) → Fin 256) =>
    zk_session_soundHJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀) (c := c)
      (pub := pub) (kd := kd) (Ω := Unit) (Lb := Lb) (src := src) (hDraw := fun _ => hok _)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **`zk_session_soundHJ_custody`'s `y₀` and the custody's draw file hold together.** At the draw file `hCust.2` is
(`DrawFileZK`, `subset kd`), the tables dividing `kd` (`hdiv`), and a session `verify --zk` accepts at the draw `dj S`,
on the inputs `ht`, `hTags`, `hc`, `hU` and `hscope` describe: there is a `y₀` at `mPtsOf c.g (kd / nTab I)` link points
(table 0's setup), every table is set up there, and `DrawOkZKHJ`, which the statement derives from the draw file, holds
at every draw at the same count. The proof applies `zk_session_soundHJ_custody` to this `y₀`. Not covered: the
session's existence, `hCust.1` (`RecordsLiveZKJ`), `hpub`, `hpt`, `hkd`, `hA3`, the count curve's parameters and
`TableCRZ`. -/
theorem zk_session_soundHJ_custody_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) =>
    zk_session_soundHJ_custody (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **`zk_session_soundHJ_custody_json`'s `y₀` and `hdj` hold together.** As `zk_session_soundHJ_custody_nonvacuous`,
with each draw file the JSON value `flock-verify draw` printed for the verifier's `subset kd` draw (`hdj`). The proof
applies `zk_session_soundHJ_custody_json` to the `y₀` it builds and to `hdj`. Not covered: the session's existence,
`hRec`, `hpub`, `hpt`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundHJ_custody_json_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ :=
    drawHypsZHJ I ht hTags hc hU hscope kd hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (k Rw M : ℕ) =>
    zk_session_soundHJ_custody_json (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
      (Ω := Unit) (hdj := hdj) (k := k) (Rw := Rw) (M := M)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **`zk_session_composedHJ`'s `y₀` and `hDraw` hold together**, as `zk_session_soundHJ_nonvacuous`; the proof applies
`zk_session_composedHJ` to this `y₀` and `hDraw`. Not covered: the session's existence, `hRec`, `hpub`, `hpt`, `hkd`,
`hA3`, the count curve's parameters, `TableCRZ`, `hdm` (`avoidsMask`, which no check implies) and `hKey` (a named
assumption). -/
theorem zk_session_composedHJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (Lb : Fin 1 → ℕ) (src : Unit → (i : Fin 1) → Fin (Lb i) → Fin 256) =>
    zk_session_composedHJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀) (c := c)
      (pub := pub) (kd := kd) (Ω := Unit) (Lb := Lb) (src := src) (hDraw := fun _ => hok _)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **`zk_session_viewHJ`'s `y₀` exists, at the draw `dj S` its conclusion's setups `ys` exist too.** At the coin
server's draw file (`DrawFileZK`, `subset kd`, the tables dividing `kd`) and one session `verify --zk` accepts at the
draw `dj S`, on the inputs `ht`, `hTags`, `hc`, `hU` and `hscope` describe: at `mPts = mPtsOf c.g (kd / nTab I)` there
is a `y₀` (table 0's setup) and every table of `dj S` is set up, so the conclusion's `ys` exist at `S`. The proof
applies `zk_session_viewHJ` to this `y₀`. Not covered: the session's existence and `hdm` (`avoidsMask`, which no check
implies); `hT1` holds at the pinned key's hiding gap (`CROnly.hm96Hiding_gap`). -/
theorem zk_session_viewHJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      ∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j)) := by
  obtain ⟨-, hy⟩ := drawHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := zk_session_viewHJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀)
  exact ⟨⟨y₀⟩, hy⟩

/-! ## The whole-draw binder -/

/-- **A whole-draw hidden-output setup has the whole draw's link points**: a `DrawSetupZKH` of the whole statement at
a draw file of the verifier's `subset kd` draw has `mPtsOf c.g kd` link points, `c` the circuit file's `HmOut.parse`
under the `--zk` tags. -/
theorem drawSetupZKH_whole_mPts (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) {kd pop : ℕ}
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) : mPts = mPtsOf c.g kd :=
  y₀.x.hmPts.symm.trans (setupHidden_mPts_of_proved hc (drawFileZK_proved_size hdj S₀) y₀.x.h)

/-- **At 16 drawn units, two tables and `g = 1`, the whole-draw hidden-output binder is empty**: no whole-draw
`DrawSetupZKH` at `mPtsOf c.g (16 / nTab I)` link points exists at a draw file of the verifier's `subset 16` draw. -/
theorem drawSetupZKH_whole_empty_16_2_1 (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hJ : nTab I = 2) (hg : c.g = 1)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset 16) n) (S₀ : Finset (Fin n)) :
    IsEmpty (DrawSetupZKH I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) :=
  ⟨fun y₀ => by
    have h := drawSetupZKH_whole_mPts I hc dj hdj y₀
    rw [hJ, hg] at h
    exact mPtsOf_16_ne_8 h.symm⟩

/-- **At 16 drawn units, two tables and `g = 1`, the hidden-output statements' `y₀` exists and a whole-draw one
doesn't.** At the verifier's `subset 16` draw file and one session `verify --zk` accepts at the draw `dj S`: the tables'
count is 27, a `y₀` (`zk_session_soundHJ`'s `DrawSetupZHJ` at table 0's inputs) exists at it, and no whole-draw
`DrawSetupZKH` does. -/
theorem y0_binderH_16_2_1 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) (hscope : Layout.scopeOkH c = true)
    (hJ : nTab I = 2) (hg : c.g = 1) (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset 16) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    mPtsOf c.g (16 / nTab I) = 27 ∧
      Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (16 / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      ∀ S₀ : Finset (Fin n), IsEmpty (DrawSetupZKH I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) := by
  refine ⟨?_, (drawHypsZHJ I ht hTags hc hU hscope 16 (by simp [hJ]) dj hdj h hd).2 ⟨0, nTab_pos I⟩,
    drawSetupZKH_whole_empty_16_2_1 I hc hJ hg dj hdj⟩
  rw [hJ, hg]
  exact mPtsOf_16_8.2

end FlockSoundness.Discharge.ZkHidden
