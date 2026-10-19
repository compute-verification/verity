import Definitions.Accounting.TS.Receipt
import Models.Accounting.Assumptions.Checkpoint

/-!
# The whole-unit guess from its checkpoints (trusted)

The statement, over `Accounting.TS.ReceiptGame` and `Accounting.Assumptions.Checkpoint.CheckpointGuessAt`. The proof is
`Security/Proofs/Accounting/Checkpoint.lean`.

* `WholeGuessOfCheckpoints`: a tile's stage as a one-round game split at the receipt, with share `1` where the tile is in
  mode `s` and check `same`, is (L) at `q^n` (`Accounting.TS.RoundGame.LuckAt`) when `same` forces `n` checkpoint
  events and each of them, given the earlier ones and the mode, holds with probability at most `q`. The game is
  Compose's `tileGame mode right same s t` at `fun c => mode c t`, `fun c => right c t` and `fun c => same c t`, so at
  `s = .whole` the conclusion is its `GuessLuckAt (q^n)`. At a constant mode it bounds `Pr[same]` by `q^n`, the
  headlines' `WholeLuck` at `q^n` for a round whose unfolded share is positive. `n` counts the checkpoints every unit
  in the mode is deficient at: mode whole includes a unit that skips only its last segment, so unsplit it is `n = 1`
  and the bound is `q`, not `q^KS` (note:20261019T1052Z-draft-rt-luck, RT-LUCK-1). The theorem doesn't move a headline
  line by itself; what it gives is the chain rule.
* `CheckpointGuessOfFresh`: in the guess bound's scope, `CheckpointGuessAt` at `q` is a theorem once each set's fresh
  coins move its checkpoint with point mass at most `q`: on coins split into the view and each set's fresh coins, a
  checkpoint event that reads only the view and the coins of the sets up to its own holds, given the earlier ones, with
  probability at most the largest probability set `s`'s fresh coins give it with everything else fixed.
-/

namespace Accounting.Guarantees.Checkpoint

open Accounting.TS Accounting.Assumptions.Checkpoint
open Pouw.RelativeSecurity (prob)

/-- **The whole-unit guess compounds over its checkpoints.** On finite coins `C`, a tile in mode `mode c` whose check
`same` holds only if each of the checkpoint events `E 0, …, E (n − 1)` does: if each checkpoint, given the earlier ones
and the mode `s`, holds with probability at most `q` (`CheckpointGuessAt`), the tile's one-round game at mode `s` is (L)
at `q^n`, so `Pr[same ∧ mode = s] ≤ q^n·Pr[mode = s]`. -/
def WholeGuessOfCheckpoints : Prop :=
  ∀ (C M : Type) [Fintype C] [DecidableEq M] (mode : C → M) (s : M) (right same : C → Prop) (E : ℕ → C → Prop)
    (n : ℕ) (q : ℝ),
    (∀ c, same c → ∀ r < n, E r c) → CheckpointGuessAt q n E (fun c => mode c = s) →
      Accounting.TS.RoundGame.LuckAt (q ^ n)
        (ReceiptGame.round (fun c => if mode c = s then 1 else 0) same right same fun _ => .done).game

/-- **In the guess bound's scope, each checkpoint's luck is its set's point mass.** On coins `V × (Fin n → X)`, uniform:
`v`, the round's other coins, from which the cheater's mode and committed checkpoints are computed, and `x s`, set `s`'s
fresh coins, all independent (`GuessLuckAt`'s scope: no computed word and no coin read before the skip reads the unit's
noise, which is its own). The served coins are this product only through a random-oracle bridge, a named premise not
yet stated, and only off an additive event of about `T_q·2^-512` a round, on which relative luck fails
(note:20261019T1052Z-draft-rt-luck, RT-LUCK-3), for one live salt per round (RT-LUCK-5). Over a round's whole tiles,
their sets enumerated tile by tile in an order fixed before the round's coins in which no tile's commitments read a
later tile's coins, it gives their joint bound; without such an order it doesn't
(`Accounting.SecurityProofs.Checkpoint.crossRead`), and an order read from the coins excludes nothing (RT-LUCK-4). If
checkpoint `s`'s event `E s` (the committed checkpoint after set `s` is the honest one) reads only `v` and the coins of
the sets `≤ s`, and for every `v` in mode `whole` and every value of the other sets' coins, set `s`'s fresh coins make
it hold with probability at most `q`, then `CheckpointGuessAt q n E` holds at that mode. For an honest run whose state
enters set `s` as a function of `v` and the earlier sets' coins, the second hypothesis is the point mass of set `s`'s
checkpoint over its fresh coins at a known state, a property of the fold and the noise alone. It fails at a checkpoint
the unit computes (point mass `1`), so for a mode that includes a suffix skip it applies at the deficient sets only
(RT-LUCK-1). -/
def CheckpointGuessOfFresh : Prop :=
  ∀ (V X : Type) [Fintype V] [Fintype X] [Nonempty X] (n : ℕ) (E : ℕ → V × (Fin n → X) → Prop) (whole : V → Prop)
    (q : ℝ),
    (∀ s < n, ∀ v (x x' : Fin n → X), (∀ r : Fin n, r.val ≤ s → x r = x' r) → (E s (v, x) ↔ E s (v, x'))) →
    (∀ s (hs : s < n), ∀ v, whole v → ∀ x, prob (fun ξ : X => E s (v, Function.update x ⟨s, hs⟩ ξ)) ≤ q) →
      CheckpointGuessAt q n E (fun c => whole c.1)

end Accounting.Guarantees.Checkpoint
