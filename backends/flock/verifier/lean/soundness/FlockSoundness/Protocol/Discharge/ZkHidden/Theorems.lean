import FlockSoundness.Discharge.HiddenExec.Flat
import FlockSoundness.Discharge.HiddenExec.Accepted
import FlockSoundness.Protocol.Discharge.ZkHidden.Sched

/-!
# The theorems of `Statements.lean`
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec ZkExec HiddenExec

theorem acceptsZK_oneAny : AcceptsZKOneAny := by
  intro I t hc h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  rcases setupTables_okAny hs with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · exact ⟨st, hst, by simpa using array_forM_ok hshape 0 (by simp), by simpa using hv⟩
  · change 1 < I.count at hc'
    omega

theorem zk_schedOK : ZkSchedOK := zk_schedOK'

/-- **A one-table hidden-output `--zk` acceptance at the coin server's draw sets the draw up**: with `ZkSchedOK`, a
`DrawSetupZKH` at `dj S`. -/
theorem drawSetupZKH_of_accepts (hsched : ZkSchedOK) {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
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
  obtain ⟨x, rfl⟩ := drawSetupH_of_setupHidden (I := zkInputs I) ((zkTags_typed I).trans ht) hc ho hst hf hD
    (hok.1 D hD) (hok.2 _ hst)
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hsched x.st _ _ x.wf.1 x.wf.2 x.hm hshape hz'
  exact ⟨⟨x, _, _, hz', h1, h2, h3, h4, h5, h6 _⟩⟩

theorem tablesAtZKH_live_of (hsched : ZkSchedOK) {n mPts : ℕ} {Reg : Type} :
    TablesAtZKHLive (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I ht c hc ho dj S₀ y₀ wr hh hc1 hrec S R o hok h
  obtain ⟨hd, -, -⟩ := hrec S R o h
  have hne := drawSetupZKH_of_accepts hsched ht hc ho hh hc1 h hd hok
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

theorem tablesAtZKH_live {n mPts : ℕ} {Reg : Type} : TablesAtZKHLive (n := n) (mPts := mPts) (Reg := Reg) :=
  tablesAtZKH_live_of zk_schedOK

end FlockSoundness.Discharge.ZkHidden
