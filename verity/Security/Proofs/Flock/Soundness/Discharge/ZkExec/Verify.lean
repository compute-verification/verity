import Proofs.Flock.Soundness.Discharge.ZkExec.Statements
import Proofs.Flock.Soundness.Discharge.Exec.Tables

/-!
# flock-verify `--zk` accepts, table by table and rep by rep

* `zk_verify_ok : ZkVerifyOk`: `Zk.verify`'s session checks (the coin tree, the record, the tables, the proof files),
  then every table's two reps with one cap.
* `zk_verifyRep_ok : ZkRepOk`: one rep's checks: its stream and proof file, the PCS parameters, the pads' root, the
  transcript program `repProg` on the stream's rounds consuming all of them, and the cap `root_B`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec

theorem zk_verify_ok : ZkVerifyOk := by
  intro spec tabs record proofs h
  unfold Flock.Zk.verify at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  by_cases c1 : spec.coinTree.isNone = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rename_i hp
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hloop, _⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one] at hloop
  have c0' : tabs.size = spec.tables.size ∧ 0 < tabs.size := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, not_not] at c0
    exact ⟨c0.1, Nat.pos_of_ne_zero c0.2⟩
  have c1' : spec.coinTree.isSome = true := by
    cases hc : spec.coinTree with
    | none => rw [hc] at c1; exact absurd rfl c1
    | some _ => rfl
  have hp' : proofs.size = spec.repStreams.size := by
    simpa only [bne_iff_ne, ne_eq, not_not] using hp
  obtain ⟨hsz, hsch⟩ := array_mapM_ok_getElem hschs
  have hP := range'_forIn_each tabs.size _
    (fun i => ∃ c, Flock.Zk.verifyRep spec tabs[i]! (schs[i]!).1 (schs[i]!).2 sess i 0 proofs[2 * i]! = .ok c ∧
      Flock.Zk.verifyRep spec tabs[i]! (schs[i]!).1 (schs[i]!).2 sess i 1 proofs[2 * i + 1]! = .ok c)
    (fun a s y hy => by
      obtain ⟨c0, h0, hy⟩ := ebind_ok hy
      obtain ⟨c1, h1, hy⟩ := ebind_ok hy
      by_cases c3 : (c0 != c1) = true
      · rw [ite_eq_left c3] at hy; obtain ⟨_, h0, _⟩ := ebind_ok hy; cases h0
      rw [ite_eq_right c3] at hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      have hb : (c0 == c1) = true := by
        cases hb : (c0 == c1)
        · exact absurd (by simp [bne, hb]) c3
        · rfl
      rw [bytes_eq_of_beq hb] at h0
      exact ⟨⟨c1, h0, h1⟩, _, hy.symm⟩) _ _ hloop
  refine ⟨c1', sess, schs, hsess, c0'.1, c0'.2, hp', hsz, fun i hi => ?_⟩
  have hs : i < schs.size := hsz ▸ hi
  obtain ⟨c, h0, h1⟩ := hP i hi
  rw [getElem!_pos tabs i hi, getElem!_pos schs i hs] at h0 h1
  exact ⟨hs, c, hsch i hi hs, h0, h1⟩

theorem zk_verifyRep_ok : ZkRepOk := by
  intro spec st sch tPad sess i r bytes cap h
  unfold Flock.Zk.verifyRep at h
  dsimp only at h
  split at h
  swap
  · cases h
  rename_i stream hstream
  obtain ⟨extras, hext, h⟩ := ebind_ok h
  obtain ⟨zp, hzp, h⟩ := ebind_ok h
  by_cases c0 : (zp.proof.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  by_cases c1 : (zp.inner.root.size != 64) = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  obtain ⟨⟨⟨⟩, tr⟩, hrun, h⟩ := ebind_ok h
  dsimp only at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c2] at h
  by_cases c3 : (zp.proof.cap.foldl (· ++ ·) ByteArray.empty != sess.rootB[i]!) = true
  · rw [ite_eq_left c3] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c3] at h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  have hb : (zp.proof.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!) = true := by
    cases hb : (zp.proof.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!)
    · exact absurd (by simp [bne, hb]) c3
    · rfl
  refine ⟨stream, extras, zp, tr, hstream, hext, hzp, Bool.eq_false_iff.2 c0,
    by simpa only [bne_iff_ne, ne_eq, not_not] using c1, hrun,
    by simpa only [bne_iff_ne, ne_eq, not_not] using c2, rfl, bytes_eq_of_beq hb⟩

end FlockSoundness.Discharge.ZkExec
