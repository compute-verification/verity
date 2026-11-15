module

public import Proofs.Properties.InferenceOnly.RecordGates
public import Proofs.Properties.InferenceOnly.RecordRead

/-!
# The Program of a record the check reads

`decode_of_read`: the Program a record `read` accepts keeps is the instance's text, decoded by Flock's decoder.
`readProg_of_read`: it is a Program the record check reads (`ReadProg`), which is what every lemma about a unit takes
of its Program: `read` refuses a reference out of order, a root not laid out, an input node off the root, a body it
cannot walk and a root node of a composite keyed as an input id (`program_of_read`), and the decoder puts no
Definition of the root's key before the root (`readProg_of_decode`).
-/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Lean (Json)

/-- **The Program of a record `read` accepts is the instance's text, decoded.** -/
theorem decode_of_read {j tr : Json} {c : InferenceOnlyCheck.Record.Case} (h : InferenceOnlyCheck.Record.read j tr = .ok c) :
    Flock.Program.decode c.f.i.program.toUTF8 = .ok c.f.P := by
  have hd := (InferenceOnlyCheck.Record.SecurityProofs.inst_of_read h).1
  unfold InferenceOnlyCheck.decodeProgram at hd
  split at hd
  · rename_i P hP
    injection hd with hd
    subst hd
    exact hP
  · cases hd

/-- **The Program of a record `read` accepts is one the record check reads.** -/
theorem readProg_of_read {j tr : Json} {c : InferenceOnlyCheck.Record.Case} (h : InferenceOnlyCheck.Record.read j tr = .ok c) :
    ReadProg c.f.P := by
  obtain ⟨hov, hl, hin, hwk, -, hkey⟩ := InferenceOnlyCheck.Record.SecurityProofs.program_of_read h
  exact readProg_of_decode (decode_of_read h) hov hl hin hwk hkey

end InferenceOnlyCheck.Properties.Record.SecurityProofs
