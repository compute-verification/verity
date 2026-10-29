import Proofs.Pouw.Lead.Tail.Chernoff
import Proofs.Pouw.Lead.Tail.Table
import Proofs.Pouw.Lead.Tail.Repeats
import Proofs.Pouw.Lead.Tail.G
import Proofs.Pouw.Lead.Tail.Gap
import Proofs.Pouw.Lead.Tail.Product
import Proofs.Pouw.Lead.Tail.Hoeffding
import Proofs.Pouw.Lead.Tail.Bennett
import Proofs.Pouw.Lead.Tail.Coins

/-!
# The lead's tail form

* `Chernoff`: sequentially dominated events are rarely many (`count_tail`).
* `Table`: the tables' union bound over tiles and steps (`table_tail`).
* `Repeats`: the coin-branch tail at the three points and decode (`coinTail_le`).
* `G`: `G_tail` from `Compose.perCoin` and the two tails, and the tail at the three points and decode (`tailAt`).
* `Gap`: the fixed gap at the three points and decode (`gapFixedAt`).

Concentration over independent coin blocks, for sums whose terms each read only their own block:

* `Product`: the coins on which the sum reaches `a` number at most `exp (−l·a)` times the product over the blocks of
  each block's exponential sum, and so at most `#coins · exp (−l·a + ∑ i, B i)` given a bound `B i` on each block.
* `Hoeffding`: terms in `[0, c i]` exceed the sum of the blocks' means by `t ≥ 0` on at most an
  `exp (−2t² / ∑ i, c i²)` share, from Mathlib's Hoeffding lemma under each block's uniform measure.
* `Bennett`: terms in `[0, c]` with block means at most `m i` reach `a` on at most an
  `exp (−l·a + (e^{l c} − 1)/c · ∑ i, m i)` share for every `l ≥ 0`: `(e − 1) · ∑ i, m i + s` on at most an
  `exp (−s/c)` share, and `M + s` (`M = ∑ i, m i`) on at most an `exp (−(M/c) · h (s/M))` share.
* `Coins`: each as `prCoin` over coins that are a shared part and blocks, the means bounded at every shared value.

Its proofs go in `Lead/Tail/`, its declarations in `Pouw.SecurityProofs.Lead.Tail`, and each file is imported here.
-/
