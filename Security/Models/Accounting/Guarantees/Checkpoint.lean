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
  `s = .whole` the conclusion is its `GuessLuckAt (q^n)`. At a constant mode it bounds `Pr[same]` by `q^n`, which is
  the headlines' `WholeLuck` at `q^n` for a round whose unfolded share is positive.
-/

namespace Accounting.Guarantees.Checkpoint

open Accounting.TS Accounting.Assumptions.Checkpoint

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

end Accounting.Guarantees.Checkpoint
