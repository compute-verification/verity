import Definitions.Accounting.Compose
import Models.Accounting.Assumptions.TS

/-!
# The compiled circuit's named assumptions (trusted)

The premises the composed theorem (`Models.Accounting.Guarantees.Compose`,
note:compute-accounting/20261018T0618Z-draft-theorem-compose-r2) adds to the tensor–scalar model's, the data sheet's and
sampled (I8)'s (`RightFixedDrawn`), each a `Prop` taken as a hypothesis, never an `axiom`. None is about a hash: under
live coins the statement has no random-oracle premise, and its Fiat–Shamir form reads proofs' `FiatShamirFloor`, stated
in the random-oracle model (decision 112).

The logical layer's (decision 113; `CircuitWork`):

* `HMu`: (H-μ), per template, the one computational claim (decision 110). It replaces round 1's `TemplateNoise`, which
  restricted the cheater's play and was false at the deployed degree-2 template (RT-COMPOSE B2).
* `RightFoldedDrawn`: Lemma E1(c) for the fold, `RightFixedDrawn`'s companion (B1: the fold carried in (I8)'s
  conjunct).

The physical layer's (`CreditOfWork`):

* `FoldOptFolded`: FOLD-OPT per word folded (B1). `FoldOpt` is it at `N_w` words, which a sampled cheater that skips
  whole units falsifies.
-/

namespace Accounting.Assumptions.Compose

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

/-- `accounting-compose-h-mu` ((H-μ), per template; a claim, Daniel's to rule on). **A cheater that hasn't computed a
product can't predict its honest receipt above its template's luck, from a view that includes the public seed**: in the
window's stages, at each tile's stage, whatever the past, a folded tile is right with probability at most its kind's
`luck` times that of being folded, and a whole one with its `luck0` (`SplitStages.LuckAt`). A tile that carries no
work is held to `1`, so this asks nothing of a filler or of a kind credited `0` (decision 109: a kind whose noise a zero
or sparse operand defeats, as attention's does today, is credited its proved floor, `0`, and is never lucky against
it).

What the template's coins give is proved, and `Scheme.LuckProved` holds the luck to at least it: `¾` for degree-2 cube
operands within the receipt's binding (`Accounting.Guarantees.TS.LuckCaseACube`, `OperandLuckCube`, #2447's
`DuoWindowNoKernel`), `Z·C(b, ⌊b/2⌋)/2^b` for `b` fresh bits at fibre `Z` (`Accounting.Guarantees.HMu.CoinPointMass`,
`ReceiptLuckOfCoins`) and `0` for a pinned value (`Accounting.Guarantees.FillerLuck.PinnedFillLuckAt`). Those are
stated for the template's own coins under full checking; that a skipping cheater's view, which holds the public seed
(decisions 101 and 105) and every word it did compute, leaves them so is the claim. It fails if computing one word pins the coin of
another a skipped word bets on (a dither bit shared across a row segment's columns, `HMu`'s notes). -/
def HMu {ι K C : Type} [Fintype C] {Np M : ℕ} (Sc : Scheme K) (Cd : Declared ι) (W : SplitStages C (Fin Np) M) :
    Prop :=
  W.LuckAt (tileLuck Sc Cd Np) (tileLuck0 Sc Cd Np) (tileStage Cd Np)

/-- `accounting-compose-right-folded` (Lemma E1(c) for the fold, proved on paper; a sampled window). **With the drawn
units right, the fold covers all but the whole share**: on every accepted path whose drawn units' committed values
(each read before its round's draws, `DrawnLayer.committed`) are right, the run folds at least `(1 − D₀/S)·N_w` of its
committed words, `D₀` its whole-skipped work share: a paid unit folds its words and a folded unit its committed ones,
and the rounds carry equal shares. `RightFixedDrawn` bounds the fixed words by the skipped share `d`, folded and whole;
this bounds the folded words by its whole part, so a folded unit's skip costs its words and a whole unit's its words and
its fold. -/
def RightFoldedDrawn {K T L U A B : Type} [Fintype A] [Fintype B] {S s : ℕ} {J V : Type}
    (P : Play (A × B) S K T L U) (pf : DrawnLayer A B S s J V) (Nw : ℝ) (D0 folded : Paths (A × B) S → ℝ) : Prop :=
  ∀ x, P.accepts x → (∀ w i, pf.right w (pf.drawnAt x w i) (pf.committed x w (pf.drawnAt x w i))) →
    (1 - D0 x / S) * Nw ≤ folded x

/-- `accounting-compose-fold-opt` (assumed; FOLD-OPT(fx1)'s mechanical fact, at least 4 issue slots a word folded).
**FOLD-OPT_μ per word folded**: at every certificate, a run that folds `folded` committed words has its fold priced at
least `(1 − μ_f)·φ*(π)` a word, `(1 − μ_f)·folded·φ*(π)`. -/
def FoldOptFolded {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (folded : ℝ) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * folded * Z.phiStar lam ≤ Z.G.price lam A.nfold

end Accounting.Assumptions.Compose
