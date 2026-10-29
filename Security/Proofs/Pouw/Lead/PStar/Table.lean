import Definitions.Pouw.Lead.Gap

/-!
# The coin lanes' guess, and tables against the committed tiles

`pStar_eq`: the exact `p*` of `blc4a-t64-e7`'s three lanes under `-rot`'s weight `(127, 1, 1)` with the tag bit, 130
of `2 · 128³`, so `65 / 2^21`. `table_le`: a memory fixed before the coins, or a coin stream of raw coins, holds a
committed tile rarely: at most `tableGap` in words averaged over the coins.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Pouw.Lead

/-- A coin lane's value from its draw `u`: `2u − 127`. -/
def lane (u : Fin 128) : ℤ := 2 * (u : ℤ) - 127

/-- **The exact `p*`**: a value of `127 s₀ + s₁ + s₂ + 1 + b` is taken by at most 130 of the
`2 · 128³` draws, and one value by exactly 130 (`p* = 65 / 2^21`). -/
theorem pStar_eq :
    (∀ v : ℤ, (Finset.univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card ≤ 130) ∧
      ∃ v : ℤ, (Finset.univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card = 130 := by
  sorry

/-- **Tables rarely hold a committed tile**: in a call of whole tiles, for every strategy's pre-coin
tables and every raw coin stream, the committed tiles its memory starts with, in words and averaged over the coins,
are at most `tableGap`. A step-0 tile is a given value with chance at most `49^-8` (`7^-8` against the at most 32
stream tiles holding its block's `G`), a later step's at most `2^-100` (each column's even and odd rows fix its three
lanes); a union over the `2 · mem + 1` values. Reads `coinTerm`, `word`, `tileOf`, `RawStream` and `bal13`. -/
theorem table_le (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (σ : Strategy (lead T m n K)) (stream : Coins (lead T m n K) → ℕ → Fin tileWords → ℤ)
    (hm : 16 ∣ m) (hn : 8 ∣ n) (hs : RawStream (lead T m n K) stream) :
    ∑ ω : Coins (lead T m n K), (tileWords * tableHits (lead T m n K) κ σ stream ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * tableGap (lead T m n K) := by
  sorry

end Pouw.SecurityProofs.Lead.PStar
