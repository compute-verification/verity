module

public import Proofs.Flock.Recursive.ZkServerJson
public import CoreProtocol.Protocol

/-!
# Canonical JSON at plain values

`Flock.canon` writes a plain value of ASCII strings as its `printJson` text (`canon_plain`; a plain string may hold a
non-ASCII character, which `canon` escapes and `printJson` does not), the same at its parsed form (`canon_norm`), and a
canonical text of ASCII parses back to `norm` of its value (`canonicalJson_canon`). These are the canonical-JSON
reader's facts; this file states them for the honest session as the named proposition `JsonFacts`, which
`HonestSession.JsonCanonFacts` proves with no hypothesis (`JsonCanon.jsonFacts`). The pieces that use it take it as a
hypothesis, and `Witness` supplies the proof.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.JsonCanon

open Lean (Json)
open FlockSoundness.Discharge.Recursion.ZkServerJson

/-- Every character of a value's text is below `0x7f`: its strings and keys are ASCII without DEL. `Flock.canon` escapes
every other character as `\uXXXX`, so only a value of this kind prints as its own canonical text. -/
def Printable (j : Json) : Prop := ∀ c ∈ (Flock.ZkServer.printJson j).toList, c.toNat < 0x7f

/-- **The canonical-JSON reader's three facts** (proved: `JsonCanon.jsonFacts`): `Flock.canon` writes a plain value of ASCII strings as its
`printJson` text, writes a plain value and its parsed form the same, and a canonical text of ASCII parses back to
`norm` of its value. -/
structure JsonFacts : Prop where
  canon_plain : ∀ {j : Json}, Flock.ZkServer.plain j = true → Printable j →
    Flock.canon j = .ok (Flock.ZkServer.printJson j)
  canon_norm : ∀ {j : Json}, Flock.ZkServer.plain j = true → Flock.canon (norm j) = Flock.canon j
  canonicalJson_canon : ∀ {j : Json} {s : String}, Flock.ZkServer.plain j = true →
    (s.toUTF8.data.all (· < 0x7f)) = true → Flock.canon j = .ok s → Core.canonicalJson s.toUTF8 = .ok (norm j)

end Verity.Sampled.Instance.HonestSession.JsonCanon
