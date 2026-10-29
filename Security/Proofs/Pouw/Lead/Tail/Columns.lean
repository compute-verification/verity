import Proofs.Pouw.Lead.Tail.Blocks
import Proofs.Pouw.Lead.Tail.Fiber

/-!
# Column blocks with a tile repeated within a block

`columns_tail`: at a call of the lead, `k` or more column blocks hold a block with two equal tiles on at most a
`λ_G^k / k!` share of the coins, `λ_G = pairs · colBlocks · S · tiles / 7^8` (`lamG`).

A column block's own coordinates are its eight columns' tag weights (`setG`), and the coins are the rest (`b`, `a₁`,
the coin lanes), drawn first, and a product over the column blocks. `count_tail_fiber` then needs:

* **Reads**: a column block's tiles read the tag weights of its own columns only.
* **The fiber** (`columns_fiber`): two tiles at one step, column block and coin tile have rows of one parity in one
  coin tile, so their coin lanes cancel, and at their first rows and each column they differ by the row tags'
  difference against the column's four tag weights, plus terms the tag weights don't move. The tags differ
  (`Tags.rowTag_block_ne`), so with every other coin fixed, each column's weights meet the difference on at most `7³`
  of their `7⁴` values (`Tags.agreeG_card_le`'s count, on one fiber): `7^−8` a pair of tiles, over `pairs` pairs at
  each of the `S · tiles` slots.
* **The sum**: `colBlocks · pairs · S · tiles / 7^8 = lamG`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `ω` with column block `cb`'s tag weights set to `v`: column `8 cb + c` taking `v q c` as its digit `q`. -/
def setG (L : Pouw.Lead.Layout) (ω : Coins L) (cb : Fin L.colBlocks) (v : Fin 4 → Fin 8 → Fin 7) : Coins L :=
  (ω.1, ω.2.1,
    fun q j => if (j : ℕ) / 8 = cb then v q ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ else ω.2.2.1 q j, ω.2.2.2)

open Classical in
/-- **A column block's own tag weights rarely make two tiles of a block agree**: with every other coin fixed, column
block `cb` holds a block with two equal tiles on at most a `pairs · S · tiles / 7^8` share of its tag weights'
values. -/
theorem columns_fiber (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (ω : Coins (lead T m n K)) (cb : Fin (lead T m n K).colBlocks) :
    ((Finset.univ.filter fun v => ∃ t ct, withinAt _ κ (setG _ ω cb v) (t, ct, cb)).card : ℝ) ≤
      (pairs (lead T m n K) * (lead T m n K).S * (lead T m n K).tiles : ℕ) / 7 ^ 8 *
        Fintype.card (Fin 4 → Fin 8 → Fin 7) := by
  sorry

/-- **Column blocks with a repeat inside a block are rarely many**: `k` or more of them on at most a `λ_G^k / k!`
share of the coins. -/
theorem columns_tail (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (k : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => k ≤ (hitCols _ κ ω).card) ≤ lamG (lead T m n K) ^ k / k.factorial := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
