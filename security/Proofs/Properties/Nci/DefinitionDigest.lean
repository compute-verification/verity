module

public import NciCheck.DefinitionDigest

/-!
# Which branch a Definition's digests take

`NciCheck.DefinitionDigest`'s two digests treat a composite's descriptor entry apart from every other: the canonical
entry rewrites a composite's body and returns any other entry as it is, and the closure digest follows a composite's
callees. Both decide it by `isComposite`, which compares the entry's `form` as a string.

* `isComposite_iff`: `isComposite e` holds exactly when `e`'s `form` is the string `composite`.
* `canonicalEntry_of_not_composite`: the canonical entry of any other entry is the entry itself.
-/

@[expose] public section

namespace NciCheck.DefinitionDigest.SecurityProofs

open Lean (Json)

/-- **Which branch the digests take**: `isComposite e` holds exactly when `e`'s `form` is the string `composite`. -/
theorem isComposite_iff (e : Json) : isComposite e = true ↔ e.getObjValD "form" = .str "composite" := by
  unfold isComposite
  generalize e.getObjValD "form" = f
  cases f <;> simp

/-- **The canonical entry of an entry that is no composite's is the entry itself**, whatever the interning tables:
`canonicalEntry` rewrites a composite's entry only. -/
theorem canonicalEntry_of_not_composite (callees types : Array Json) (e : Json)
    (h : e.getObjValD "form" ≠ .str "composite") : canonicalEntry callees types e = .ok e := by
  have hf : isComposite e = false := by
    rw [← Bool.not_eq_true, isComposite_iff]
    exact h
  unfold canonicalEntry
  simp only [hf]
  rfl

end NciCheck.DefinitionDigest.SecurityProofs
