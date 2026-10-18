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

The logical layer (decision 113; `CircuitWork`) rests on four claims, the headlines' (#2482) and its own:

* (H-μ), `Accounting.Assumptions.TS.OperandHMu`, and B′, `Accounting.Assumptions.TS.ReceiptLuckAt`, taken per tile at
  its stage folded (`PerTile`): a folded tile keeps its honest checked values at `μ + (1 − μ)·r_B`, `7/8` at `¾` and
  `½` (`Accounting.Guarantees.TS.LuckOfReceipt`), as the headlines' `p`;
* `GuessLuckAt`: the whole-unit guess bound, per tile at its stage whole;
* `RightFixed`: the fixed-words claim in share form, Lemma E1(c) for the sampled window, with the fold as its second
  clause. It replaces round 2's `UnitFixed` at the tiles' words, whose good tiles were right ones, so a receipt-lucky
  skipped tile counted as right with wrong words (RT-COMPOSE r2's R2-B1), and round 2's `UnitFolded`.

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

/-- `accounting-compose-right-fixed` (Lemma E1(c) for the sampled window in share form, proved on paper; the fixed-words
claim, as the headlines' `Accounting.Assumptions.TS.RightFixed` is under full checking). **What isn't fixed is in the
skipped tiles' cones, and what isn't folded is in the whole tiles**: on every accepted path, along the window's coins,
* the run's owned right fixed words are at least `(1 − (D_F + D₀))·N_w`, `D_F` and `D₀` the late cones' weights
  (`tileCone`) of the tiles folded and whole: a word of a tile that no skipped tile reaches late (`Declared.Reaches`)
  is computed from values fixed before its noise's coin, so it is not adaptive (E1(c): an adaptive word lies below a
  value chosen after its coin);
* the run folds at least `(1 − D₀')·N_w` committed words, `D₀'` the whole tiles' own weight: a paid tile folds its
  words and a folded tile its committed ones, and only a whole tile skips its fold (round 2's `UnitFolded`).
It reads the run's modes, never a tile's rightness or the proofs: a skipped tile counts as skipped however lucky its
receipt, so this holds for the cheater that skips one unit and guesses its receipt (R2-B1). The cones are the IR's
(`Declared.Late`, at the cuts `Declared.cutOf`), through glue, run inputs, the KV cache and decode feedback (W1); a
tile of no words is charged its cone too (W4). -/
def RightFixed {ι K T L U C B : Type} [Fintype C] {Np M : ℕ} (Sc : Scheme K) (Cd : Declared ι)
    (P : Play (Paths C M × B) 1 K T L U) (W : SplitStages C (Fin Np) M) (folded : Paths (Paths C M × B) 1 → ℝ) :
    Prop :=
  ∀ x, P.accepts x →
    (1 - (W.share (tileStage Cd Np) (tileCone Sc Cd Np) (coinAt x 0).1 .folded +
        W.share (tileStage Cd Np) (tileCone Sc Cd Np) (coinAt x 0).1 .whole)) * (wordsOf Sc Cd : ℝ) ≤
      (P.run x).fixed ∧
    (1 - W.share (tileStage Cd Np) (tileWeight Sc Cd Np) (coinAt x 0).1 .whole) * (wordsOf Sc Cd : ℝ) ≤ folded x

/-- `accounting-compose-fold-opt` (assumed; FOLD-OPT(fx1)'s mechanical fact, at least 4 issue slots a word folded).
**FOLD-OPT_μ per word folded**: at every certificate, a run that folds `folded` committed words has its fold priced at
least `(1 − μ_f)·φ*(π)` a word, `(1 − μ_f)·folded·φ*(π)`. -/
def FoldOptFolded {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (folded : ℝ) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * folded * Z.phiStar lam ≤ Z.G.price lam A.nfold

end Accounting.Assumptions.Compose
