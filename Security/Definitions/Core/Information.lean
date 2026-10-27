import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog

/-!
# Entropy and mutual information over a uniform finite space

These are the information-theoretic quantities of a game whose coins are uniform over a finite type `Ω`, with
probabilities counted as `Pous.pr` counts them. A random quantity is a function `X : Ω → α`, and `X = a` has
probability equal to the fraction of `Ω` on which it holds. Entropies are in bits. Mathlib has `Real.negMulLog` and
the binary entropy, but no Shannon entropy of a finite distribution.
-/

namespace Verity.Information

open Classical in
/-- The Shannon entropy `H(X)`, in bits, of `X ω` for `ω` uniform on the finite type `Ω`: `∑ₐ −p(a)·log₂ p(a)`, where
`p(a)` is the fraction of `Ω` on which `X = a` (0 when `Ω` is empty). -/
noncomputable def entropy {Ω α : Type} [Fintype Ω] [Fintype α] (X : Ω → α) : ℝ :=
  (∑ a : α, Real.negMulLog (((Finset.univ.filter fun ω => X ω = a).card : ℝ) / Fintype.card Ω)) / Real.log 2

/-- The conditional entropy `H(X | Y) = H(X, Y) − H(Y)`, in bits, for `ω` uniform on `Ω`. -/
noncomputable def condEntropy {Ω α β : Type} [Fintype Ω] [Fintype α] [Fintype β] (X : Ω → α) (Y : Ω → β) : ℝ :=
  entropy (fun ω => (X ω, Y ω)) - entropy Y

/-- The conditional mutual information `I(X ; Y | Z) = H(X | Z) − H(X | Y, Z)`, in bits, for `ω` uniform on `Ω`: what
`Y` tells about `X` beyond what `Z` tells. -/
noncomputable def condMutualInfo {Ω α β γ : Type} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ]
    (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) : ℝ :=
  condEntropy X Z - condEntropy X (fun ω => (Y ω, Z ω))

end Verity.Information
