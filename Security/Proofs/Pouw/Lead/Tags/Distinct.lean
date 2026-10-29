import Definitions.Pouw.Lead.Gap

/-!
# Tag distinctness at tile grain, and the words' exactness

`repeats_le`: committed tiles repeat rarely, so copying whole tiles saves at most the coin-branch term. `word_exact`:
the served 32-bit register holds each committed word exactly (the served layout's completeness; `G` doesn't read it).
-/

namespace Pouw.SecurityProofs.Lead.Tags

open Pouw.Lead

/-- **Tiles repeat rarely**: for any data, the committed tiles less their distinct values, in words
and averaged over the coins, are at most `coinBranchGap`. Reads `word`, `rowTag`, `colCode`, `rotWeight` and
`tileOf`. -/
theorem repeats_le (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ) :
    ∑ ω : Coins (lead T m n K), (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) := by
  sorry

/-- **The words fit a 32-bit register** for `K ≤ 65,536` and data in range. -/
theorem word_exact (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (hK : K ≤ 65536) (hκ : DataInRange (lead T m n K) κ) :
    ∀ ω i t j, -(2 ^ 31) ≤ word (lead T m n K) κ ω i t j ∧ word (lead T m n K) κ ω i t j < 2 ^ 31 := by
  sorry

end Pouw.SecurityProofs.Lead.Tags
