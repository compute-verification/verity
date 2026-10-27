import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Exec.Tables
import Proofs.Flock.Soundness.Refine.Frames

/-!
# The production statements retain every round's bytes

`liveTable` (`Refine/Live.lean`) is the challenger that keeps each round's framed bytes (`roundOf`'s
`bytes := some b`), and the alignment `DecodesAll` asks for rests on it: with retained bytes, round `k`'s coins answer
only the framed bytes absorbed since round `k - 1`, exactly (`Transcript.squeeze`, R2). This file shows that every
`hm96Rows` tag pinned today is of that kind, and that a record flock-verify accepts under it has every round's bytes.

* `retainRounds_circuitEb90718f`, `retainRounds_circuit`, `retainRounds_circuit967b8d06`, `retainRounds_circuitTypes`:
  each `hm96Rows` tag has `retainRounds = true`, inherited from `circuit631567f7` (rfl).
* `setupH_tags`, `setupTables_retain`: the session's spec retains rounds when its tags do.
* `decode_retained`: when the spec retains rounds, `Record.decode` gives every round of every stream its bytes (S12).
* `accepts_retained`: so every round of a record flock-verify accepts under such tags carries its bytes.
* `accepts_allRetained`: the same as `Refine.AllRetained` for every stream, the hypothesis of `Refine/Frames.lean`'s
  squeeze and sampling lemmas: these tags retain too, so those lemmas apply to the production statements.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

theorem retainRounds_circuitEb90718f : Tags.circuitEb90718f.retainRounds = true := rfl
theorem retainRounds_circuit : Tags.circuit.retainRounds = true := rfl
theorem retainRounds_circuit967b8d06 : Tags.circuit967b8d06.retainRounds = true := rfl
theorem retainRounds_circuitTypes : Tags.circuitTypes.retainRounds = true := rfl

set_option maxHeartbeats 4000000 in
/-- `setupH`'s statement carries the tags it was set up under. -/
theorem setupH_tags {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) : st.tags = tags := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}

set_option maxHeartbeats 4000000 in
/-- The session's spec retains rounds exactly when its tags do, for one table or several. -/
theorem setupTables_retain (hh : tags.hiddenOutputs = false)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    spec.retainRounds = tags.retainRounds := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  have hn : ¬ tags.hiddenOutputs = true := by simp [hh]
  simp only [throw_bind', pure_bind, ite_eq_right hn] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  · exact congrArg Tags.retainRounds (setupH_tags (ok_of_refuse (by assumption)))
  all_goals rfl

set_option maxHeartbeats 1000000 in
/-- **An accepted record has every round's bytes** when its spec retains rounds: `Record.decode` refuses a round
without them (S12). -/
theorem decode_retained {text : String} {sess : Session} (hr : spec.retainRounds = true)
    (h : Record.decode spec text = .ok sess) :
    ∀ s ∈ sess.streams.toList, ∀ r ∈ s.rounds.toList, ∃ b, r.bytes = some b := by
  unfold Record.decode at h
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
      iterate 8 obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨_, hret, hg⟩ := ebind_ok hg
      have hn : Record.isNull y "msg" = false := by
        have := check_ok_iff.1 hret
        rw [hr] at this
        simpa using this
      rw [hn] at hg
      simp only [Bool.false_eq_true, ↓reduceIte] at hg
      obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨b, _, hg⟩ := ebind_ok hg
      obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨x, hx, hg⟩ := ebind_ok hg
      simp only [pure, Except.pure, Except.ok.injEq] at hx hg
      subst hx hg
      exact ⟨b, rfl⟩

/-- **flock-verify's acceptance under retaining tags has every round's bytes**: the session's spec retains rounds, and
every round of every stream of the record it decodes carries its framed bytes. -/
theorem accepts_retained {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false)
    (hr : I.tags.retainRounds = true) (h : Accepts I t) :
    ∃ spec sts sess, setupOf I t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      spec.retainRounds = true ∧ ∀ s ∈ sess.streams.toList, ∀ r ∈ s.rounds.toList, ∃ b, r.bytes = some b := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, -, hd, -⟩ := verify_ok_tables hv
  have hsr : spec.retainRounds = true := (setupTables_retain hh hs).trans hr
  exact ⟨spec, sts, sess, hs, hd, hsr, decode_retained hsr hd⟩

/-- `decode_retained` as `Refine/Frames.lean`'s `AllRetained`, for every stream. -/
theorem decode_allRetained {text : String} {sess : Session} (hr : spec.retainRounds = true)
    (h : Record.decode spec text = .ok sess) : ∀ s ∈ sess.streams.toList, AllRetained s.rounds := fun s hs i hi => by
  rw [getElem!_pos s.rounds i hi]
  exact decode_retained hr h s hs _ (Array.getElem_mem_toList hi)

/-- **flock-verify's acceptance under retaining tags gives `AllRetained`** (`Refine/Frames.lean`) for every stream of
the record it decodes: the hypothesis `squeeze_bindA`, `sample_bindA` and `forIn_answers` take. -/
theorem accepts_allRetained {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false)
    (hr : I.tags.retainRounds = true) (h : Accepts I t) :
    ∃ spec sts sess, setupOf I t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds := by
  obtain ⟨spec, sts, sess, hs, hd, hsr, -⟩ := accepts_retained hh hr h
  exact ⟨spec, sts, sess, hs, hd, decode_allRetained hsr hd⟩

end FlockSoundness.Discharge.Exec
