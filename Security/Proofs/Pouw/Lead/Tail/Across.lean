import Proofs.Pouw.Lead.Tail.Blocks
import Proofs.Pouw.Lead.Tail.Fiber

/-!
# Blocks that repeat an earlier tile

`across_tail`: at a call of the lead, `a` or more blocks repeat a tile of an earlier block on at most a `λ_X^a / a!`
share of the coins, `λ_X = ρ² · blocks² / 2^57` (`lamX`).

A block's own coordinates are its step's three coin lanes of its coin tile at its column block's eight columns
(`setLanes`). The lead's columns fill whole column blocks, so every coin lane is one block's, and the coins are the
rest (`b`, `a₁`, `G`), drawn first, and a product over the blocks in block order (`blkIdx`). `count_tail_fiber` then
needs:

* **Reads**: block `β`'s tiles read the lanes of `β`'s coin tile and column block at steps up to `β`'s, and the
  earlier tiles it is compared with read their own blocks' lanes: all blocks at or before `β`.
* **The fiber** (`across_fiber`): with every other coin fixed, `β`'s tile at row block `rb` equals a fixed earlier
  tile only where, at its first row and each of its eight columns, `127 s₀ + s₁ + s₂` hits one value: for each
  `s₁, s₂` at most one `s₀`, a `2^−56` share (`Tags.tile_lane_card_le`'s count, on one fiber). `β` has at most `ρ`
  tiles, and at most `ρ · blkIdx β` tiles come before it.
* **The sum**: `Σ_β ρ² · blkIdx β / 2^56 = ρ² · B (B − 1) / 2^57 ≤ lamX`, `B` the blocks.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `ω` with block `β`'s own coin lanes set to `v`: step `β.1`'s three lanes of coin tile `β.2.1` at column block
`β.2.2`'s columns, column `8 cb + c` taking `v ℓ c`. -/
def setLanes (L : Pouw.Lead.Layout) (ω : Coins L) (β : Blk L) (v : Fin 3 → Fin 8 → Fin (2 ^ L.e)) : Coins L :=
  (ω.1, ω.2.1, ω.2.2.1, fun t ct ℓ j =>
    if t = β.1 ∧ ct = β.2.1 ∧ (j : ℕ) / 8 = β.2.2 then v ℓ ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩
    else ω.2.2.2 t ct ℓ j)

open Classical in
/-- **A block's own lanes rarely repeat an earlier tile**: with every other coin fixed, block `β` repeats an earlier
tile on at most a `ρ² · blkIdx β / 2^56` share of its own coin lanes' values. -/
theorem across_fiber (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (ω : Coins (lead T m n K)) (β : Blk (lead T m n K)) :
    ((Finset.univ.filter fun v => acrossAt _ κ (setLanes _ ω β v) β).card : ℝ) ≤
      (rho (lead T m n K) ^ 2 * blkIdx (lead T m n K) β.1 β.2.1 β.2.2 : ℕ) / 2 ^ 56 *
        Fintype.card (Fin 3 → Fin 8 → Fin (2 ^ (lead T m n K).e)) := by
  sorry

/-- **Blocks that repeat an earlier tile are rarely many**: `a` or more of them on at most a `λ_X^a / a!` share of
the coins. -/
theorem across_tail (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (a : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => a ≤ (across _ κ ω).card) ≤ lamX (lead T m n K) ^ a / a.factorial := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
