module

public import Proofs.Flock.Recursive.FirewallJson
public import CoreProtocol.Protocol

/-!
# Canonical JSON at plain values

`Flock.canon` writes a plain value's `printJson` text (`canon_plain`), the same at its parsed form (`canon_norm`), and a
canonical text of ASCII parses back to `norm` of its value (`canonicalJson_canon`). These are the canonical-JSON
reader's facts; this file states them for the honest session and its proofs are that reader's.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.JsonCanon

open Lean (Json)
open FlockSoundness.Discharge.Recursion.FirewallJson

/-- The three facts about `Flock.canon` at plain values. -/
structure JsonFacts : Prop where
  canon_plain : ∀ {j : Json}, Flock.Firewall.plain j = true → Flock.canon j = .ok (Flock.Firewall.printJson j)
  canon_norm : ∀ {j : Json}, Flock.Firewall.plain j = true → Flock.canon (norm j) = Flock.canon j
  canonicalJson_canon : ∀ {j : Json} {s : String}, Flock.Firewall.plain j = true →
    (s.toUTF8.data.all (· < 0x7f)) = true → Flock.canon j = .ok s → Core.canonicalJson s.toUTF8 = .ok (norm j)

theorem jsonFacts : JsonFacts := by
  sorry -- the canonical-JSON reader's lemmas

end Verity.Sampled.Instance.HonestSession.JsonCanon
