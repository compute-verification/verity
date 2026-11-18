module

public import Definitions.Poub.Guarantees.RemoteScheme
public import Proofs.Core.Scheme
public import Proofs.Poub.RemoteHeld.Sketch
import Proofs.Pouw.Prob

/-!
# The bits a PoUB session reads, through the scheme

`ReadsGetsBack` is the accounting layer's theorem (`Verity.Scheme.accountingLayer`) at `Trace.scheme`: the uniform
probability on the outcomes, read in `ℝ≥0∞`, is monotone and subadditive, and the scheme's break event is
`SketchBound`'s (`breaks_le`, and conversely `sketchBound_of_breaks`, so `SketchBound T δs` is the scheme's break bound
at `δs ≥ 0`). `ReadsGetsBackDeployed` is `ReadsGetsBack` at `SketchBoundDeployed`'s bound.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions Pouw
open scoped ENNReal

variable {Ω : Type} [Fintype Ω]

/-- `SketchBound T δs` bounds the scheme's break event by `δs`. -/
theorem breaks_le {T : Trace Ω} {δs : ℝ} (h : SketchBound T δs) (b : ℕ) (δ' : ℝ≥0∞) :
    ResourceAccounting.Remote.Run.prE (Ω := Ω) (T.scheme b δ').breaks ≤ ENNReal.ofReal δs :=
  ENNReal.ofReal_le_ofReal h

/-- At `δs ≥ 0`, the scheme's break event bounded by `δs` is `SketchBound T δs`. -/
theorem sketchBound_of_breaks {T : Trace Ω} {δs : ℝ} (hδ : 0 ≤ δs) (b : ℕ) (δ' : ℝ≥0∞)
    (h : ResourceAccounting.Remote.Run.prE (Ω := Ω) (T.scheme b δ').breaks ≤ ENNReal.ofReal δs) : SketchBound T δs :=
  (ENNReal.ofReal_le_ofReal_iff hδ).1 h

/-- **The bits read, through the scheme, at any sketch bound**
(`ResourceAccounting.Guarantees.Remote.ReadsGetsBack`). -/
theorem ReadsGetsBack : ResourceAccounting.Guarantees.Remote.ReadsGetsBack := by
  intro Ω _ T b δs D ε haud hk δ
  refine Verity.Scheme.accountingLayer (T.scheme b (ENNReal.ofReal δs)) (fun _ => D)
    (ResourceAccounting.Remote.Run.prE (Ω := Ω)) ε ?_ ?_ haud (breaks_le hk b _) δ
  · exact fun p q h => ENNReal.ofReal_le_ofReal (pr_mono (α := Ω) h)
  · exact fun p q => (ENNReal.ofReal_le_ofReal (pr_or_le (α := Ω) p q)).trans ENNReal.ofReal_add_le

/-- **The bits read, through the scheme, at the deployed receipts**
(`ResourceAccounting.Guarantees.Remote.ReadsGetsBackDeployed`). -/
theorem ReadsGetsBackDeployed : ResourceAccounting.Guarantees.Remote.ReadsGetsBackDeployed := by
  intro β Hs _ _ P N S L q Q G qtot εrank A T b D ε hε hP hR hG hq hH hK hRc hPr haud δ
  exact ReadsGetsBack _ T b _ D ε haud
    (SketchBoundDeployed β Hs P N S L q Q G qtot εrank A T hε hP hR hG hq hH hK hRc hPr) δ

end ResourceAccounting.SecurityProofs.Remote
