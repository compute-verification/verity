import Mathlib

/-!
# ASCII texts: what the canonical printer prints

`Flock.canon` writes every character outside printable ASCII as a `\u` escape, so every text it prints is ASCII without
DEL (`Json.Ascii`: every character below `0x7f`, so every byte is), the bytes Python's `codec.canonical_json` writes.

Mathlib only, as `Definitions/` is.
-/

namespace Claims

namespace Json

/-- **A text of ASCII without DEL**: every character below `0x7f`, so every byte is. Canonical JSON escapes every other
character, so its texts are. -/
def Ascii (s : String) : Prop := ∀ c ∈ s.toList, c.toNat < 0x7f

end Json

end Claims
