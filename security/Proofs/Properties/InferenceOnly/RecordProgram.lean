module

public import Proofs.Properties.InferenceOnly.RecordGates
public import Proofs.Properties.InferenceOnly.RecordRead

/-! The Program of a record `read` accepts is the instance's text decoded, and one the record check reads. -/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)

theorem decode_of_read {j tr : Json} {c : InferenceOnlyVerifier.Record.Case} (h : InferenceOnlyVerifier.Record.read j tr = .ok c) :
    Flock.Program.decode c.f.i.program.toUTF8 = .ok c.f.P := by
  have hd := (InferenceOnlyVerifier.Record.SecurityProofs.inst_of_read h).1
  unfold InferenceOnlyVerifier.decodeProgram at hd
  split at hd
  · rename_i P hP
    injection hd with hd
    subst hd
    exact hP
  · cases hd

theorem readProg_of_read {j tr : Json} {c : InferenceOnlyVerifier.Record.Case} (h : InferenceOnlyVerifier.Record.read j tr = .ok c) :
    ReadProg c.f.P := by
  obtain ⟨hov, hl, hin, hwk, -, hkey⟩ := InferenceOnlyVerifier.Record.SecurityProofs.program_of_read h
  exact readProg_of_decode (decode_of_read h) hov hl hin hwk hkey

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
