module

public import Proofs.Properties.Nci.Scratch.RecordAttack.DecodeText

/-!
# The decoder on the Program of no node, asked of the kernel

A scratch file, never merged, and expected to fail where the kernel cannot evaluate the decoder: a closed witness of
a served record needs the decoder's answer on one text with no hypothesis about it.
-/

@[expose] public section

namespace Flock.Program.Closed

open Lean (Json)

/-- The text parses. -/
theorem parse_empty_ok : (Flock.JsonParse.parse programEmpty).toBool = true := by decide +kernel

/-- The decoder accepts the text. -/
theorem decode_empty_ok : (Flock.Program.decode programEmpty.toUTF8).toBool = true := by decide +kernel

end Flock.Program.Closed
