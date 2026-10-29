import Proofs.Pouw.Lead.Tail.Chernoff
import Proofs.Pouw.Lead.Tail.Table
import Proofs.Pouw.Lead.Tail.Repeats
import Proofs.Pouw.Lead.Tail.G

/-!
# The lead's tail form

* `Chernoff`: sequentially dominated events are rarely many (`count_tail`).
* `Table`: the tables' union bound over tiles and steps (`table_tail`).
* `Repeats`: the coin-branch tail at the three points and decode (`coinTail_le`).
* `G`: `G_tail` from `Compose.perCoin` and the two tails, and its figures (`tailAt`, `gapFixedAt`).

Its proofs go in `Lead/Tail/`, its declarations in `Pouw.SecurityProofs.Lead.Tail`, and each file is imported here.
-/
