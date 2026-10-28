import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-!
# The theorems of `Statements.lean`

The tables' theorems hold from the session alone (`drawSetupZKH_of_acceptsU`, `tablesAtZKH_liveU`), so each circuit
form, `HmOut.parse`'s and the statement circuit's (`…St`, `HmOut.parseStmt`), reads them.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec ZkExec HiddenExec

theorem acceptsZK_setupOne : AcceptsZKSetupOne := by
  intro I t h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  obtain ⟨hpos, hall⟩ := setupTables_setupOne hs
  exact ⟨spec, sts, hs, hv, hpos, fun j hj => ⟨array_forM_ok hshape j hj, hall j hj⟩⟩

theorem acceptsZK_oneAny : AcceptsZKOneAny := by
  intro I t hc h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  rcases setupTables_okAny hs with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · exact ⟨st, hst, by simpa using array_forM_ok hshape 0 (by simp), by simpa using hv⟩
  · change 1 < I.count at hc'
    omega

theorem acceptsZK_retainedAny : AcceptsZKRetainedAny := by
  intro I t hr h
  obtain ⟨spec, sts, hs, -, hv⟩ := h
  obtain ⟨-, sess, -, hd, -⟩ := zk_verify_ok _ _ _ _ hv
  have hsr : spec.retainRounds = true := ((setupTables_retainAny hs).trans (zkTags_retainRounds I)).trans hr
  exact ⟨spec, sts, sess, hs, hd, decode_allRetained hsr hd⟩

theorem acceptsZK_acceptedH : AcceptsZKAcceptedH := by
  intro I t hh ht c hc hv1 hc1 h
  obtain ⟨st, hst, hshape, -⟩ := acceptsZK_oneAny I t hc1 h
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  rw [setupOne_hidden hh'] at hst
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have hcst : st.c = c := setupHidden_c_of_parse hc hst
  refine ⟨st, hst, hshape, fun regs => ⟨_, ⟨zkTags I, I.circuitFile, I.publicFile, zkCoins I, I.tables,
    recordDraw t.record, I.partition, I.program, st, ht', Hidden.setupHidden_specCC hst,
    Hidden.unpin_layout ht' hst, unitRowsFlat_unpin ht' hst (by rw [hcst]; exact hv1.1.toP), regs, rfl⟩, rfl, HEq.rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩⟩

theorem zk_schedOK : ZkSchedOK := zk_schedOK'

/-- **A one-table hidden-output `--zk` acceptance at the challenger's draw sets the draw up**, from the session alone:
with `ZkSchedOK`, a `DrawSetupZKH` at `dj S`. -/
theorem drawSetupZKH_of_acceptsU (hsched : ZkSchedOK) {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)} (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1)
    {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKH I dj mPts S) :
    Nonempty (DrawSetupZKH I mPts S (dj S)) := by
  obtain ⟨st, hst, hshape, hv⟩ := acceptsZK_oneAny I t hc1 h
  rw [setupOne_hidden ((zkTags_hiddenOutputs I).trans hh), hd] at hst
  obtain ⟨-, sess, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, cap, hz, -⟩ := hall 0 (by simp)
  have hz' : Flock.Zk.schedule st.m = .ok (schs[0].1, schs[0].2) := by simp only [List.getElem_toArray] at hz; exact hz
  obtain ⟨sch, hf⟩ := fast100_of_schedule hz'
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  obtain ⟨x, rfl⟩ := drawSetupH_of_setupHiddenU (I := zkInputs I) ((zkTags_typed I).trans ht) hst hf hD
    (hok.1 D hD) (hok.2 _ hst)
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hsched x.st _ _ x.wf.1 x.wf.2 x.hm hshape hz'
  exact ⟨⟨x, _, _, hz', h1, h2, h3, h4, h5, h6 _⟩⟩

set_option linter.unusedVariables false in
/-- **A one-table hidden-output `--zk` acceptance at the challenger's draw sets the draw up**: with `ZkSchedOK`, a
`DrawSetupZKH` at `dj S`, wherever the outputs are returned. -/
theorem drawSetupZKH_of_acceptsI (hsched : ZkSchedOK) {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)} (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1)
    {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKH I dj mPts S) :
    Nonempty (DrawSetupZKH I mPts S (dj S)) :=
  drawSetupZKH_of_acceptsU hsched ht hh hc1 h hd hok

set_option linter.unusedVariables false in
/-- `drawSetupZKH_of_acceptsI` at the statement circuit (`HmOut.parseStmt`). -/
theorem drawSetupZKH_of_acceptsISt (hsched : ZkSchedOK) {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)} (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1)
    {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKH I dj mPts S) :
    Nonempty (DrawSetupZKH I mPts S (dj S)) :=
  drawSetupZKH_of_acceptsU hsched ht hh hc1 h hd hok

set_option linter.unusedVariables false in
/-- `drawSetupZKH_of_acceptsI` where the unit returns the outputs. -/
theorem drawSetupZKH_of_accepts (hsched : ZkSchedOK) {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)} (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1)
    {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) (hok : DrawOkZKH I dj mPts S) :
    Nonempty (DrawSetupZKH I mPts S (dj S)) :=
  drawSetupZKH_of_acceptsI hsched ht hc hh hc1 h hd hok

theorem tablesAtZKH_liveU_of (hsched : ZkSchedOK) {n mPts : ℕ} {Reg : Type} :
    TablesAtZKHLiveU (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I ht dj S₀ y₀ wr hh hc1 hrec S R o hok h
  obtain ⟨hd, -, -⟩ := hrec S R o h
  have hne := drawSetupZKH_of_acceptsU hsched ht hh hc1 h hd hok
  refine ⟨(Classical.choice hne).x.st, (Classical.choice hne).sch, (Classical.choice hne).tPad,
    (Classical.choice hne).x.wf.1, (Classical.choice hne).x.wf.2, (Classical.choice hne).x.h,
    (Classical.choice hne).hz, Classical.choice hne, rfl, rfl, rfl, ?_, fun points hpts hsz => ?_⟩
  · show tableAtZKH dj y₀ S = _
    unfold tableAtZKH
    rw [dite_eq_left_of_eq_true (eq_true hne)]
  · have e := ofCircuit_extra (Classical.choice hne).x.st (Classical.choice hne).x.wf.1 (Classical.choice hne).x.wf.2
      points hpts (by rw [(Classical.choice hne).x.hmPts]; exact hsz)
    rw [(Classical.choice hne).x.hmPts] at e
    exact e

theorem tablesAtZKH_liveI_of (hsched : ZkSchedOK) {n mPts : ℕ} {Reg : Type} :
    TablesAtZKHLiveI (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I ht c _
  exact tablesAtZKH_liveU_of hsched I ht

theorem tablesAtZKH_liveISt_of (hsched : ZkSchedOK) {n mPts : ℕ} {Reg : Type} :
    TablesAtZKHLiveISt (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I ht c _
  exact tablesAtZKH_liveU_of hsched I ht

theorem tablesAtZKH_live_of (hsched : ZkSchedOK) {n mPts : ℕ} {Reg : Type} :
    TablesAtZKHLive (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I ht c hc _
  exact tablesAtZKH_liveI_of hsched I ht hc

theorem tablesAtZKH_liveSt_of (hsched : ZkSchedOK) {n mPts : ℕ} {Reg : Type} :
    TablesAtZKHLiveSt (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I ht c hc _
  exact tablesAtZKH_liveISt_of hsched I ht hc

theorem tablesAtZKH_liveU {n mPts : ℕ} {Reg : Type} : TablesAtZKHLiveU (n := n) (mPts := mPts) (Reg := Reg) :=
  tablesAtZKH_liveU_of zk_schedOK

theorem tablesAtZKH_liveI {n mPts : ℕ} {Reg : Type} : TablesAtZKHLiveI (n := n) (mPts := mPts) (Reg := Reg) :=
  tablesAtZKH_liveI_of zk_schedOK

theorem tablesAtZKH_live {n mPts : ℕ} {Reg : Type} : TablesAtZKHLive (n := n) (mPts := mPts) (Reg := Reg) :=
  tablesAtZKH_live_of zk_schedOK

theorem tablesAtZKH_liveISt {n mPts : ℕ} {Reg : Type} : TablesAtZKHLiveISt (n := n) (mPts := mPts) (Reg := Reg) :=
  tablesAtZKH_liveISt_of zk_schedOK

theorem tablesAtZKH_liveSt {n mPts : ℕ} {Reg : Type} : TablesAtZKHLiveSt (n := n) (mPts := mPts) (Reg := Reg) :=
  tablesAtZKH_liveSt_of zk_schedOK

end FlockSoundness.Discharge.ZkHidden
