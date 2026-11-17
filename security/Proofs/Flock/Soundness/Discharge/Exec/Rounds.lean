module

public import Proofs.Flock.Soundness.Discharge.Exec.FramesRep
public import Proofs.Flock.Soundness.Discharge.Exec.Event
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Discharge.Exec.Retained
import Proofs.Flock.Soundness.Discharge.Exec.Tables
import Proofs.Flock.Soundness.Refine.Seam

/-! verity-verify-proof's acceptance round by round (`RepRounds`); which of `repC`'s messages each chunk carries is the open part of `DecodesAll`. -/

@[expose] public section

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine

def RepRounds (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (sess : RunRecord) (i r : ℕ)
    (bytes : ByteArray) : Prop :=
  ∃ (stream : StreamRec) (p : Proof) (extras : Array Flock.Claim) (T : F128),
    sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
    st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
    (chunks (repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T)).1.length =
      stream.rounds.size ∧
    ∀ k < stream.rounds.size, Retains (stream.rounds[k]!)
      (encs ((chunks (repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T)).1.getD k []))

theorem rep_rounds_retained {spec : SessionSpec} {st : Setup} {sch : Flock.Schedule} {sess : RunRecord} {i r : ℕ}
    {bytes cap : ByteArray} (hm : 14 ≤ st.m) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hret : ∀ s ∈ sess.streams.toList, AllRetained s.rounds) (h : verifyRep spec st sch sess i r bytes = .ok cap) :
    RepRounds spec st sch sess i r bytes := by
  obtain ⟨stream, p, extras, T, hs, hex, hp, hR⟩ := rep_rounds spec st sch sess i r bytes cap hm hdim hlv h
  obtain ⟨hlen, hk⟩ := hR (hret stream (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs)))
  exact ⟨stream, p, extras, T, hs, hex, hp, hlen, hk⟩

theorem accepts_rep_rounds {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false)
    (hr : I.tags.retainRounds = true) (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 0 t.proofs[2 * j]! ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 1 t.proofs[2 * j + 1]! := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verify_ok_tables hv
  have hret := decode_allRetained ((setupTables_retain hh hs).trans hr) hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨hdim, -⟩ := fast100_dims _ _ hf
  have hlv : ∀ i < schs[j].levels.size, LevelSeam (schs[j].levels.getD i default) := fun i hi =>
    fast100_level _ _ hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hm : 14 ≤ (Setup.ofCircuit sts[j]).m := by have := exec_fast100_ge hf; omega
  exact ⟨schs[j], hf, rep_rounds_retained hm hdim hlv hret h0, rep_rounds_retained hm hdim hlv hret h1⟩

end FlockSoundness.Discharge.Exec
