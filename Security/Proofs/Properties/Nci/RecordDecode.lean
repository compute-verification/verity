module

public import Proofs.Properties.Nci.RecordServed
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.String.Defs
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
import all Init.Data.List.Control
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.WF

/-!
# Two more runs of the kernel, beside `Proofs.Properties.Nci.RecordServed`

A scratch file, never merged. A cut of 1 the text states is read as stated, not as none (`tie_cutOne`): the wiring's
cuts are the text's list, not what `cutsOf` gives of it. And whether the kernel runs the decoder on the Program of no
node (`decode_runs`): where it does not, `Served`'s first clause is shown of no text by running it.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

/-- `{"cuts":{"B":{"x":1}},"wires":[]}` -/
def recordCutOne : List Char :=
  ['{', '"', 'c', 'u', 't', 's', '"', ':', '{', '"', 'B', '"', ':', '{', '"', 'x', '"', ':', '1', '}', '}',
   ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

theorem tie_cutOne : readTie recordCutOne = some ⟨[], [("B", [("x", 1)])], [], none⟩ := by decide +kernel

/-- The decoder reads the Program of no node. -/
theorem decode_runs : (Flock.Program.decode programEmpty.toUTF8).toOption.isSome = true := by decide +kernel

end NciCheck.Properties.Record.Attack
