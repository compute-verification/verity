module

public import InferenceOnlyCheck.DefinitionDigest

/-! Which branch a Definition's two digests take: `isComposite` decides it by the entry's `form` string. -/

@[expose] public section

namespace InferenceOnlyCheck.DefinitionDigest.SecurityProofs

open Lean (Json)

theorem isComposite_iff (e : Json) : isComposite e = true ↔ e.getObjValD "form" = .str "composite" := by
  unfold isComposite
  generalize e.getObjValD "form" = f
  cases f <;> simp

theorem canonicalEntry_of_not_composite (callees types : Array Json) (e : Json)
    (h : e.getObjValD "form" ≠ .str "composite") : canonicalEntry callees types e = .ok e := by
  have hf : isComposite e = false := by
    rw [← Bool.not_eq_true, isComposite_iff]
    exact h
  unfold canonicalEntry
  simp only [hf]
  rfl

end InferenceOnlyCheck.DefinitionDigest.SecurityProofs
