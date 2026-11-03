import Proofs.Pouw.Lead.Tail.Chernoff
import Proofs.Pouw.Lead.Tail.Fiber
import Proofs.Pouw.Lead.Tail.Blocks
import Proofs.Pouw.Lead.Tail.Words
import Proofs.Pouw.Lead.Tail.Across
import Proofs.Pouw.Lead.Tail.Columns
import Proofs.Pouw.Lead.Tail.Slots
import Proofs.Pouw.Lead.Tail.Route
import Proofs.Pouw.Lead.Tail.Table
import Proofs.Pouw.Lead.Tail.Repeats
import Proofs.Pouw.Lead.Tail.G
import Proofs.Pouw.Lead.Tail.Gap
import Proofs.Pouw.Lead.Tail.Nontrivial

/-!
# The lead's tail form

* `Chernoff`: sequentially dominated events are rarely many (`count_tail`).
* `Fiber`: the same with the domination read off one coordinate (`count_tail_fiber`).
* `Blocks`: the coin-branch repeats split over the blocks (`repeats_le_blocks`, `within_le`), the three counts'
  Poisson parameters and `cut`.
* `Words`: what the three fiber counts share: a tile's words (`tileOf_at`), a coin tile's pairs of row blocks
  (`pairsAt`, `withinAt_pair`) and functions pinned off one block (`card_le_of_block`).
* `Across`, `Columns`, `Slots`: the three counts' tails (`across_tail`, `columns_tail`, `slots_tail`), each from its
  fiber count (`across_fiber`, `columns_fiber`, `slots_fiber`).
* `Route`: the counts that bound the tail at the three points and decode (`route_le`).
* `Table`: the tables' union bound over tiles and steps (`table_tail`).
* `Repeats`: the coin-branch tail at the three points and decode (`coinTail_le`).
* `G`: `G_tail` from `Compose.perCoin` and the two tails, and the tail at the three points and decode (`tailAt`).
* `Gap`: the fixed gap at the three points and decode (`gapFixedAt`).
* `Nontrivial`: the honest work above the fixed gap and `etaLead < 1` at the three points (`nontrivialTail`).

Its proofs go in `Lead/Tail/`, its declarations in `Pouw.SecurityProofs.Lead.Tail`, and each file is imported here.
-/
