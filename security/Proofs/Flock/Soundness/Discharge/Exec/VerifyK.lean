module

public import Flock.With.Deployed
public import Flock.With.Eq
public import Flock.With.VerifyE
public import Proofs.Flock.Soundness.Discharge.Exec.Event
public import Proofs.Flock.Soundness.Refine.Frames
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Discharge.Exec.Tables

/-! What the acceptance walkers read of `verify` with its digest comparisons read through `keep`; at `keep := id` each statement is the shipped one. -/

@[expose] public section

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

theorem decodeWithK_id_std (spec : SessionSpec) (text : String) :
    Record.decodeWithK id With.std spec text = Record.decode spec text :=
  (Record.decodeWith_eq_decodeWithK With.std spec text).symm.trans (With.decodeWith_std spec text)

theorem verifyWithMK_tables_cap (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {record : String} {proofs : Array ByteArray} (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) :
    ∃ (sess : RunRecord) (schs : Array Flock.Schedule), Record.decodeWithK keep hs spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧ 0 < tabs.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs' : i < schs.size) (c : ByteArray),
        Flock.fast100 tabs[i].m = some schs[i] ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c := by
  unfold verifyWithMK at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hloop, _⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one] at hloop
  have c0' : tabs.size = spec.tables.size ∧ 0 < tabs.size := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, not_not] at c0
    exact ⟨c0.1, Nat.pos_of_ne_zero c0.2⟩
  obtain ⟨hsz, hsch⟩ := array_mapM_ok_getElem hschs
  have hP := range'_forIn_each tabs.size _
    (fun i => ∃ c, verifyRepWith hs spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! = .ok c ∧
      verifyRepWith hs spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! = .ok c)
    (fun a s y hy => by
      obtain ⟨c₀, h₀, hy⟩ := ebind_ok hy
      obtain ⟨c₁, h₁, hy⟩ := ebind_ok hy
      by_cases c3 : ((c₀ : ByteArray) != c₁) = true
      · rw [ite_eq_left c3] at hy; obtain ⟨_, h0, _⟩ := ebind_ok hy; cases h0
      rw [ite_eq_right c3] at hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      have hb : (c₀ == c₁) = true := by
        cases hb : (c₀ == c₁)
        · exact absurd (by simp [bne, hb]) c3
        · rfl
      rw [bytes_eq_of_beq hb] at h₀
      exact ⟨⟨c₁, h₀, h₁⟩, _, hy.symm⟩) _ _ hloop
  refine ⟨sess, schs, hsess, c0'.1, c0'.2, hsz, fun i hi => ?_⟩
  have hs' : i < schs.size := hsz ▸ hi
  obtain ⟨c, h₀, h₁⟩ := hP i hi
  rw [getElem!_pos tabs i hi, getElem!_pos schs i hs'] at h₀ h₁
  refine ⟨hs', c, ?_, h₀, h₁⟩
  have hf := hsch i hi hs'
  cases hm : Flock.fast100 tabs[i].m with
  | none => rw [hm] at hf; cases hf
  | some s => rw [hm] at hf; cases hf; rfl

theorem verifyK_ok_tables (keep : Bool → Bool) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} (hN : ∀ st ∈ tabs, st.merkle.Named)
    (h : verifyWithMK keep With.std spec tabs record proofs = .ok ()) :
    ∃ (sess : RunRecord) (schs : Array Flock.Schedule), Record.decodeWithK keep With.std spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧ 0 < tabs.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs : i < schs.size) (c : ByteArray),
        Flock.fast100 tabs[i].m = some schs[i] ∧
        verifyRep spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c ∧
        verifyRep spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c := by
  obtain ⟨sess, schs, hd, hsz, hpos, hsch, hall⟩ := verifyWithMK_tables_cap keep With.std h
  refine ⟨sess, schs, hd, hsz, hpos, hsch, fun i hi => ?_⟩
  obtain ⟨hs, c, hf, h₀, h₁⟩ := hall i hi
  rw [With.verifyRepWith_std _ _ (hN _ (Array.getElem_mem hi))] at h₀ h₁
  exact ⟨hs, c, hf, h₀, h₁⟩

theorem verifyK_one_ok (keep : Bool → Bool) {spec : SessionSpec} {st : Setup} {record : String}
    {proofs : Array ByteArray} (hN : st.merkle.Named)
    (h : verifyWithMK keep With.std spec #[st] record proofs = .ok ()) :
    ∃ sess sch c, Record.decodeWithK keep With.std spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      verifyRep spec st sch sess 0 0 proofs[0]! = .ok c ∧ verifyRep spec st sch sess 0 1 proofs[1]! = .ok c := by
  obtain ⟨sess, schs, hd, -, -, -, hall⟩ := verifyK_ok_tables keep (fun s hs => by simp at hs; exact hs ▸ hN) h
  obtain ⟨hs, c, hf, h₀, h₁⟩ := hall 0 (by simp)
  exact ⟨sess, schs[0], c, hd, hf, h₀, h₁⟩

set_option maxHeartbeats 1000000 in
theorem decodeWithK_points (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    sess.points.size = spec.points ∧ ∀ p ∈ sess.points.toList, p.size = spec.mPts := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact check_pts (by assumption)

set_option maxHeartbeats 1000000 in
theorem decodeWithK_roundsWF (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∀ s ∈ sess.streams.toList, RoundsWF s.rounds := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro s hs
    try dsimp only at hs
    obtain ⟨x, hx⟩ := mapM_ok_mem (by assumption) s hs
    try dsimp only at hx
    repeat' (first | (obtain ⟨_, _, hx⟩ := ebind_ok hx) | (split at hx))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hx
      subst hx
      apply roundsWF_of
      intro r hr
      obtain ⟨k, y, hg⟩ := mapIdxM_ok_mem (by assumption) r hr
      try dsimp only at hg
      repeat' (first | (obtain ⟨_, _, hg⟩ := ebind_ok hg) | (split at hg))
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at hg
        subst hg
        exact check_coins (by assumption)

set_option maxHeartbeats 1000000 in
theorem decodeWithK_retained (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} (hr : spec.retainRounds = true) (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∀ s ∈ sess.streams.toList, ∀ r ∈ s.rounds.toList, ∃ b, r.bytes = some b := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro s hs
    try dsimp only at hs
    obtain ⟨x, hx⟩ := mapM_ok_mem (by assumption) s hs
    try dsimp only at hx
    repeat' (first | (obtain ⟨_, _, hx⟩ := ebind_ok hx) | (split at hx))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hx
      subst hx
      intro r hr'
      obtain ⟨k, y, hg⟩ := mapIdxM_ok_mem (by assumption) r hr'
      try dsimp only at hg
      iterate 9 obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨_, hret, hg⟩ := ebind_ok hg
      have hn : Record.isNull y "msg" = false := by
        have := check_ok_iff.1 hret
        rw [hr] at this
        simpa using this
      rw [hn] at hg
      simp only [Bool.false_eq_true, ↓reduceIte] at hg
      obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨b, _, hg⟩ := ebind_ok hg
      iterate 3 obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨x, hx, hg⟩ := ebind_ok hg
      simp only [pure, Except.pure, Except.ok.injEq] at hx hg
      subst hx hg
      exact ⟨b, rfl⟩

theorem decodeWithK_allRetained (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} (hr : spec.retainRounds = true) (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∀ s ∈ sess.streams.toList, AllRetained s.rounds := fun s hs' i hi => by
  rw [getElem!_pos s.rounds i hi]
  exact decodeWithK_retained keep hs hr h s hs' _ (Array.getElem_mem_toList hi)

def AcceptsK (keep : Bool → Bool) (I : Inputs) (t : Transcript) : Prop :=
  ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
    verifyWithMK keep With.std spec (sts.map Setup.ofCircuit) t.record t.proofs = .ok ()

theorem named_of_setupOf {I : Inputs} {record : String} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (h : setupOf I record = .ok (spec, sts)) : ∀ s ∈ sts.map Setup.ofCircuit, s.merkle.Named :=
  With.named_ofCircuit (With.setupTables_named _ _ _ _ _ _ _ _ _ _ _ _ _ _ h)

theorem accepts_iff_acceptsK_id (I : Inputs) (t : Transcript) : Accepts I t ↔ AcceptsK id I t := by
  unfold Accepts AcceptsK
  refine exists_congr fun spec => exists_congr fun sts => and_congr_right fun hs => ?_
  rw [verify_eq_verifyWith _ _ _ _ (named_of_setupOf hs)]
  exact Iff.rfl

theorem acceptsK_true_of_accepts {I : Inputs} {t : Transcript} (h : Accepts I t) : AcceptsK (fun _ => true) I t := by
  obtain ⟨spec, sts, hs, hv⟩ := (accepts_iff_acceptsK_id I t).1 h
  exact ⟨spec, sts, hs, verifyWithE_of_verifyWith _ _ _ _ _ hv⟩

end FlockSoundness.Discharge.Exec
