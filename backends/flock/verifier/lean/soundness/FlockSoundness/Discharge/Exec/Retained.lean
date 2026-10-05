import FlockSoundness.Discharge.Exec.Setup
import FlockSoundness.Discharge.Exec.Tables
import FlockSoundness.Refine.Frames
import FlockSoundness.Protocol.Discharge.Exec.Retained

/-!
# The production statements retain every round's bytes

`liveTable` (`Refine/Live.lean`) is the coin server that keeps each round's framed bytes (`roundOf`'s
`bytes := some b`), and the alignment `DecodesAll` asks for rests on it: with retained bytes, round `k`'s coins answer
only the framed bytes absorbed since round `k - 1`, exactly (`Transcript.squeeze`, R2). This file shows that every
`hm96Rows` tag pinned today is of that kind, and that a record flock-verify accepts under it has every round's bytes.

* `retainRounds_circuitEb90718f`, `retainRounds_circuit`, `retainRounds_circuit967b8d06`, `retainRounds_circuitTypes`:
  each `hm96Rows` tag has `retainRounds = true`, inherited from `circuit631567f7` (rfl).
* `setupH_tags`, `setupTables_retain`: the session's spec retains rounds when its tags do.
* `decode_retained`: when the spec retains rounds, `Record.decode` gives every round of every stream its bytes (S12).
* `accepts_retained`: so every round of a record flock-verify accepts under such tags carries its bytes.
* `accepts_allRetained`: the same as `Refine.AllRetained` for every stream, the hypothesis of `Refine/Frames.lean`'s
  squeeze and sampling lemmas and of `Refine/FramesPiop.lean`'s `zerocheck_frames`, whose text says "on the unsalted
  path": these tags retain too, so those lemmas apply to the production statements.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

theorem retainRounds_circuitEb90718f : Tags.circuitEb90718f.retainRounds = true := rfl
theorem retainRounds_circuit : Tags.circuit.retainRounds = true := rfl
theorem retainRounds_circuit967b8d06 : Tags.circuit967b8d06.retainRounds = true := rfl
theorem retainRounds_circuitTypes : Tags.circuitTypes.retainRounds = true := rfl

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}

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

/-- **flock-verify's acceptance under retaining tags gives `AllRetained`** (`Refine/Frames.lean`) for every stream of
the record it decodes: the hypothesis `squeeze_bindA`, `sample_bindA`, `forIn_answers` and `zerocheck_frames` take. -/
theorem accepts_allRetained {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false)
    (hr : I.tags.retainRounds = true) (h : Accepts I t) :
    ∃ spec sts sess, setupOf I t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds := by
  obtain ⟨spec, sts, sess, hs, hd, hsr, -⟩ := accepts_retained hh hr h
  exact ⟨spec, sts, sess, hs, hd, decode_allRetained hsr hd⟩

end FlockSoundness.Discharge.Exec
