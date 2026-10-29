import Proofs.Pouw.Lead.Tail.Blocks
import Proofs.Pouw.Lead.Tail.Fiber

/-!
# One column block's slots with a tile repeated within a block

`slots_tail`: at a call of the lead whose coin tiles are whole (or which has one row block), `x` or more of one column
block's slots (step, coin tile) hold two equal tiles on at most a `λ_b^x / x!` share of the coins,
`λ_b = pairs · S · tiles / 2^16` (`lamB`).

A slot's own coordinates are 64 fresh bits, one a row of its coin tile (`setBits`): digit 0 of the step-0 bits `a₁` at
step 0, the tag bits `b` of its step at a later step. With whole coin tiles the coins are the rest (`b` at step 0,
`a₁`'s digits 1 to 3, `G`, the coin lanes), drawn first, and a product over the slots in slot order (step, then coin
tile). `count_tail_fiber` then needs:

* **Reads**: a slot's tiles are prefix sums over steps up to its own, of its own coin tile's rows: the bits of slots
  at or before it.
* **The fiber** (`slots_fiber`): two tiles at one step, column block and coin tile differ, at each of their 16 row
  pairs, by a term the slot's bits don't move and the two rows' fresh bits' difference: `b` once, or `a₁`'s digit 0
  against a column code digit 0 that is nonzero at one of the block's first two columns (it is zero only at
  `j ≡ 12 (mod 13)`). With every other coin fixed, each pair of rows meets it on at most half of the first row's bit
  values: `2^−16` a pair of tiles, over `pairs` pairs.
* **The sum**: `S · tiles · pairs / 2^16 = lamB`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `ω` with slot `(t, ct)`'s fresh bits set to `v`, row `i` of coin tile `ct` taking `v (i % 64)`: digit 0 of the
step-0 bits `a₁` at step 0, and the tag bits `b` of step `t` at a later step. -/
def setBits (L : Pouw.Lead.Layout) (ω : Coins L) (t : Fin L.S) (ct : Fin L.tiles) (v : Fin 64 → Fin 2) : Coins L :=
  (fun i τ => if (t : ℕ) ≠ 0 ∧ τ = t ∧ (i : ℕ) / 64 = ct then v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
      else ω.1 i τ,
    fun i q => if (t : ℕ) = 0 ∧ (q : ℕ) = 0 ∧ (i : ℕ) / 64 = ct then v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
      else ω.2.1 i q,
    ω.2.2.1, ω.2.2.2)

open Classical in
/-- **A slot's fresh bits rarely make two tiles of a block agree**: with every other coin fixed, slot `(t, ct)` of
column block `cb` holds two equal tiles on at most a `pairs / 2^16` share of its fresh bits' values. -/
theorem slots_fiber (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (ω : Coins (lead T m n K)) (cb : Fin (lead T m n K).colBlocks) (t : Fin (lead T m n K).S)
    (ct : Fin (lead T m n K).tiles) :
    ((Finset.univ.filter fun v => withinAt _ κ (setBits _ ω t ct v) (t, ct, cb)).card : ℝ) ≤
      (pairs (lead T m n K) : ℝ) / 2 ^ 16 * Fintype.card (Fin 64 → Fin 2) := by
  sorry

/-- **One column block's slots with a repeat inside a block are rarely many**: `x` or more of them on at most a
`λ_b^x / x!` share of the coins, when the coin tiles are whole or there is one row block. -/
theorem slots_tail (T m n K : ℕ) (hm : 64 ∣ m ∨ m ≤ 16)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (cb : Fin (lead T m n K).colBlocks) (x : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => x ≤ (hitSlots _ κ ω cb).card) ≤
      lamB (lead T m n K) ^ x / x.factorial := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
