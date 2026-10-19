import Definitions.Pouw.RelativeSecurity

/-!
# The whole-unit guess, one checkpoint at a time (trusted)

The per-checkpoint condition of route R-CP (note:20261019T0534Z-draft-fpx-luck), beside the whole-unit guess bound
(Compose's `GuessLuckAt`, the headlines' `WholeLuckAt`). CP-W's check pins every one of a unit's `KS + m` checkpoints
(`PouwWarpCheck_v2`: the XOR over the warp's lanes and chains of `y` and of `z` after each set), and a unit is deficient
at the checkpoint of each segment whose product and fold it skips. So a bound per deficient checkpoint, given the
earlier ones, compounds over them (`Accounting.Guarantees.Checkpoint.WholeGuessOfCheckpoints`). Mode whole includes a
unit that skips only its last segment, with one deficient checkpoint, so at one `q` the chain bounds mode whole by `q`,
not `q^KS` (note:20261019T1052Z-draft-rt-luck, RT-LUCK-1).
-/

namespace Accounting.Assumptions.Checkpoint

open Pouw.RelativeSecurity

/-- `accounting-checkpoint-guess` (the per-checkpoint guess bound; a claim, Daniel's to rule on, no card filed).
**A unit keeps each checkpoint it is deficient at, given the earlier ones, only by guessing it**: over a tile's coins
`C`, for every checkpoint `s < n`, `Pr[E_s ∧ E_r for every r < s ∧ whole] ≤ q·Pr[E_r for every r < s ∧ whole]`, where
`E_s` is "the committed checkpoint after set `s` is the honest one" and `whole` is the mode. It is a condition at the
`q`, `n` and `E` it is given, and holds only where every unit in the mode is deficient at each of the `n` checkpoints:
a checkpoint a unit computes is kept with probability `1`. Compose's and #2482's mode whole includes a unit that skips
only its last segment's product and fold (RT-LUCK-1), so for that mode it is read at the one deficient checkpoint
(`n = 1`), or the mode is split by the number of segments skipped.

What it is read at:
* It is stronger than ASK-9's per-unit `½`, not weaker: at `½` over `n ≥ 64` checkpoints it gives the per-unit `2^-64`
  that ASK-9 left ungranted (RT-LUCK-2).
* The live closed line needs a per-unit `p0 ≤ 0.4278` at every unit of positive unfolded share. At `½` the line stays
  at 0.173% (r = 1); 0.151% would need `q ≤ 0.4278` at the one deficient checkpoint, and no card asks for it.
* The seeded whole branch (r6's fix (b), its moment at `θw = K/8`) needs the mode split by skipped segments and
  `q ≤ 2^-7.45` per checkpoint (a single skipped segment's moment at `2^-7` is above 1). Per seed query it is capped:
  a cheater that computes the products and commits the checkpoints of a guessed 1024-bit answer keeps them on at least
  `2^-1024` of the query's answers, so no per-query whole-skip entropy above `1024·ln 2` (709.8 nats) a tile holds.
  At `2^-8` the chain passes that cap above `K = 4096`, and fix (b) fails above `K = 5678` whatever `q` is (RT-LUCK-7);
  Compose's `WholeEntropyAt` at `K/8` reads the same cap.

Where its bound comes from: in the scope `GuessLuckAt` states (no computed word, and no coin read before the skip,
determines a fold state) a unit's committed checkpoints read none of its coins. The served set coins are SHA-512 images
of a seed the earlier sets pin, so the bound is the honest checkpoint's point mass over set `s`'s fresh coins at the
state the earlier sets leave only through a random-oracle bridge, a named premise not yet stated: off an event of
probability about `T_q·2^-512` a round (the cheater queried the oracle at the round's salt), the round's set coins are
uniform and independent of the view (RT-LUCK-3). Relative luck fails on that event, so its probability is an additive
term of the window's ε, outside `q`. That point mass is worst case over the other coins, so the events are the `KS`
segment sets' (fresh `b`, `c`): `O'` and `D_d` read their fresh `z` only through `U·A_d`, which is `0` at `U = 0`. All
of it is for one live salt per round; the served code draws one a window (RT-LUCK-5). It fails where `GuessLuckAt`
fails: under a coin event or an instance on which a checkpoint is cheap to compute without the product and the fold. -/
def CheckpointGuessAt {C : Type} [Fintype C] (q : ℝ) (n : ℕ) (E : ℕ → C → Prop) (whole : C → Prop) : Prop :=
  ∀ s < n, prob (fun c => E s c ∧ (∀ r < s, E r c) ∧ whole c) ≤ q * prob (fun c => (∀ r < s, E r c) ∧ whole c)

end Accounting.Assumptions.Checkpoint
