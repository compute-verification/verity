import Proofs.Pouw.Lead.Tags.Repeats
import Proofs.Pouw.Lead.Tail.Chernoff

/-!
# The coin-branch tail at the three points and decode

`coinTail_le`: at the three points and decode, for every call of `n, K ≥ 4,096` and at most `2^40` tiles, a call's
repeated tiles exceed the coin-branch mean by more than the slack on at most a `2^−42` share of the coins, whatever
its data.

The blocks of one step, coin tile and column block don't read disjoint coins. A word is a prefix sum over steps, so a
tile at step `t` reads every earlier step's coin lanes and tag bits `b`, and step 0's tag weights `G` (one set a
column, shared by every step and coin tile) and `a₁` bits (one set a row, shared by every step and column block). Two
tiles at one step, column block and coin tile have rows of one parity, so their coin lanes cancel, and only `G`,
`a₁` and `b` separate them. The proof splits the repeats in two, each through `Tail.count_tail`:

* **Across blocks** (`Tags.tile_lane_card_le`): order the blocks by step, coin tile and column block, with `G`, `a₁`
  and `b` drawn first. A block's own coin lanes are read by no earlier block, and on every cylinder each of its four
  tiles agrees with any earlier tile on at most `2^−56` of them.
* **Within a block** (`Tags.tile_rows_card_le`): with every coin but `G` fixed, a column block's agreements read only
  its own `G`, on at most `7^−8` a pair, so the column blocks with any agreement are few (`count_tail` over the
  column blocks). For each value of `G`, the steps and coin tiles with an agreement in one column block are few too,
  by `a₁`'s first digit at step 0 and the fresh `b_t` at each later step, at most `2^−16` a pair (`count_tail` over
  steps and coin tiles).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- **The coin-branch tail at the three points and decode.** -/
theorem coinTail_le (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) :
    ∀ L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K, lead T 1 n K],
      L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 40 → coinTail L ≤ 1 / 2 ^ 42 := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
