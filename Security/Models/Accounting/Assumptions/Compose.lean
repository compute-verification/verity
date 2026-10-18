import Definitions.Accounting.Compose
import Models.Accounting.Assumptions.TS

/-!
# The compiled circuit's named assumptions (trusted)

The premises the composed theorem (`Models.Accounting.Guarantees.Compose`,
note:compute-accounting/20261018T1230Z-draft-theorem-compose-r3) adds to the tensor–scalar model's, the data sheet's,
sampled (I8)'s and the headlines' (#2482), each a `Prop` taken as a hypothesis, never an `axiom`. None is about a
hash: under live coins the statement has no random-oracle premise, binding is charged to `δ_bind` per committed
position (RT-I8's B1, decision 114), and its Fiat–Shamir form reads proofs' `FiatShamirFloor`, stated in the
random-oracle model (decision 112).

The logical layer (decision 113; `CircuitWork`) rests on four claims, the headlines' (#2482), sampled (I8)'s (#2466)
and its own:

* (H-μ), `Accounting.Assumptions.TS.OperandHMu`, and B′, `Accounting.Assumptions.TS.ReceiptLuckAt`, taken per tile at
  its stage folded (`PerTile`): a folded tile keeps its honest checked values at `μ + (1 − μ)·r_B`, `7/8` at `¾` and
  `½` (`Accounting.Guarantees.TS.LuckOfReceipt`), as the headlines' `p`;
* `GuessLuckAt`: the whole-unit guess bound, per tile at its stage whole;
* the fixed-words claim, #2466's `Accounting.Assumptions.TS.UnitFixed` (Lemma E1(c) per unit with its fold clause),
  at the run's modes (`Accounting.Compose.tileMode`) and the tiles' words. Its good units are the ones the run paid
  for, so a receipt-lucky skipped tile isn't credited (RT-COMPOSE r2's R2-B1), and its fold clause is round 2's
  `UnitFolded`.

The physical layer's (`CreditOfWork`):

* `FoldOptFolded`: FOLD-OPT per word folded (B1). `FoldOpt` is it at `N_w` words, which a sampled cheater that skips
  whole units falsifies.
-/

namespace Accounting.Assumptions.Compose

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

/-- `accounting-compose-guess-luck` (the whole-unit guess bound; a claim, Daniel's to rule on). **A unit skipped whole
keeps its honest checked values only by guessing them**: in the ideal game split at the receipt, at every round,
whatever the past, a deficient round passes with probability at most `p0` (`Accounting.TS.RoundGame.LuckAt` of its
game). Taken per tile at its stage whole (`tileGame … .whole`, whose check is `same`): a tile whose product and fold
are both skipped keeps its checked values, the fold state its proof pins, with probability at most `p0`, the guess
bound `2^-(state bits)`, `2^-64` for a 64-bit fold state. It replaces round 2's `luck0 = 0`, which a guess falsifies
(RT-COMPOSE r2's R2-B3). It fails if the cheater's view pins the fold state without the fold: a state that a computed
word, or a coin read before the skip, determines. -/
def GuessLuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p0 : ℝ) (G : ReceiptGame Ω S) : Prop :=
  Accounting.TS.RoundGame.LuckAt p0 G.game

/-- `accounting-compose-fold-opt` (assumed; FOLD-OPT(fx1)'s mechanical fact, at least 4 issue slots a word folded).
**FOLD-OPT_μ per word folded**: at every certificate, a run that folds `folded` committed words has its fold priced at
least `(1 − μ_f)·φ*(π)` a word, `(1 − μ_f)·folded·φ*(π)`. -/
def FoldOptFolded {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (folded : ℝ) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * folded * Z.phiStar lam ≤ Z.G.price lam A.nfold

end Accounting.Assumptions.Compose
