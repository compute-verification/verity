module

public import Proofs.Properties.Nci.Record

/-!
# What the closure digest computes where the form's comparison answers no

A scratch file, never merged. `canonicalEntry` and `closureDigest` decide whether a Definition is a composite by
`Json`'s `==`, whose body a proof cannot open. These lemmas say what the two compute in the reading where that
comparison answers `false`, a reading no proof excludes: the canonical entry is the descriptor's own entry, interning
indices and all, so a Definition's digest does not read the interning tables.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)
open NciCheck.DefinitionDigest (canonicalEntry definitionDigest)

/-- Where the comparison answers no, the canonical entry is the entry itself, whatever the interning tables. -/
theorem canonicalEntry_raw (callees types : Array Json) (e : Json)
    (h : (e.getObjValD "form" == Json.str "composite") = false) : canonicalEntry callees types e = .ok e := by
  unfold canonicalEntry
  first
    | (simp only [h]; rfl)
    | (simp [h]; done)
    | (simp [h]; rfl)

/-- Where the comparison answers no on every pair, a Definition's digest reads no interning table: a node's callee and
its types enter it as indices. -/
theorem definitionDigest_raw (hb : ∀ a b : Json, (a == b) = false) (callees types callees' types' : Array Json)
    (e : Json) : definitionDigest callees types e = definitionDigest callees' types' e := by
  unfold definitionDigest
  rw [canonicalEntry_raw callees types e (hb _ _), canonicalEntry_raw callees' types' e (hb _ _)]

end NciCheck.Properties.Record.Attack
