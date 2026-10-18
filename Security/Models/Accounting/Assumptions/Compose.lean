import Definitions.Accounting.Compose
import Models.Accounting.Assumptions.TS

/-!
# The compiled circuit's named assumptions (trusted)

The premises the composed theorem (`Models.Accounting.Guarantees.Compose`,
note:compute-accounting/20261018T0618Z-draft-theorem-compose-r2) adds to the tensor–scalar model's, the data sheet's and
sampled (I8)'s, each a `Prop` taken as a hypothesis, never an `axiom`. None is about a hash: under live
coins the statement has no random-oracle premise, binding is charged to `δ_bind` per committed position (RT-I8's B1,
decision 114), and its Fiat–Shamir form reads proofs' `FiatShamirFloor`, stated in the random-oracle model (decision
112).

The logical layer's (decision 113; `CircuitWork`):

* `HMu`: (H-μ), per template, the one computational claim (decision 110). It replaces round 1's `TemplateNoise`, which
  restricted the cheater's play and was false at the deployed degree-2 template (RT-COMPOSE B2).
* `UnitFolded`: Lemma E1(c) for the fold, per tile, sampled (I8)'s `UnitFixed`'s companion (B1: a whole tile's skip
  costs its fold too). Like `UnitFixed`, it reads neither the draw nor the proofs.

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
work is held to `1`, so this asks nothing of a filler or of a kind credited `0` (decision 109: a kind whose noise proves
no floor, as one-sided noise proves none, is credited `0` and is never lucky against it; decision 117: H1's attention
matmuls are credited at their floors, `QKᵀ`'s from a coin per entry on both operands and `P·V`'s from `U·b`, `Scheme`).
It is per tile and unchanged by decision 119: the share charges a skipped tile its late cone, so a tile computed
honestly downstream of a skipped one is paid, and this asks nothing of it.

What the template's coins give is proved, and `Scheme.LuckProved` holds the luck to at least it: `¾` for degree-2 cube
operands within the receipt's binding (`Accounting.Guarantees.TS.LuckCaseACube`, `OperandLuckCube`, #2447's
`DuoWindowNoKernel`), `Z·C(b, ⌊b/2⌋)/2^b` for `b` fresh bits at fibre `Z` (`Accounting.Guarantees.HMu.CoinPointMass`,
`ReceiptLuckOfCoins`) and `0` for a pinned value (`Accounting.Guarantees.FillerLuck.PinnedFillLuckAt`). Those are
stated for the template's own coins under full checking; that a skipping cheater's view, which holds the public seed
(decisions 101 and 105) and every word it did compute, leaves them so is the claim. It fails if computing one word pins
the coin of another a skipped word bets on (a dither bit shared across a row segment's columns, `HMu`'s notes). -/
def HMu {ι K C : Type} [Fintype C] {Np M : ℕ} (Sc : Scheme K) (Cd : Declared ι) (W : SplitStages C (Fin Np) M) :
    Prop :=
  W.LuckAt (tileLuck Sc Cd Np) (tileLuck0 Sc Cd Np) (tileStage Cd Np)

/-- `accounting-compose-unit-folded` (Lemma E1(c) for the fold, per tile, proved on paper). **A tile that isn't whole
has its words folded**: on every accepted path, the run folds at least the committed words of the tiles that aren't
whole along the window's coins: a paid tile folds its words and a folded tile its committed ones, and only a whole tile
skips its fold. With the tiles' words summing to `N_w` it gives `(1 − D₀)·N_w`, `D₀` the whole share. It reads neither
the draw nor the proofs: which tiles are whole is the run's, and the proofs only vouch for the drawn ones
(`Accounting.Guarantees.TS.DrawnRightOff`). -/
def UnitFolded {ι K T L U C B : Type} [Fintype C] {Np M : ℕ} (Sc : Scheme K) (Cd : Declared ι)
    (P : Play (Paths C M × B) 1 K T L U) (W : SplitStages C (Fin Np) M) (folded : Paths (Paths C M × B) 1 → ℝ) :
    Prop :=
  ∀ x, P.accepts x →
    (∑ t, if W.modeAt (tileStage Cd Np) (coinAt x 0).1 t = .whole then 0 else tileWords Sc Cd Np t) ≤ folded x

/-- `accounting-compose-fold-opt` (assumed; FOLD-OPT(fx1)'s mechanical fact, at least 4 issue slots a word folded).
**FOLD-OPT_μ per word folded**: at every certificate, a run that folds `folded` committed words has its fold priced at
least `(1 − μ_f)·φ*(π)` a word, `(1 − μ_f)·folded·φ*(π)`. -/
def FoldOptFolded {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (folded : ℝ) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * folded * Z.phiStar lam ≤ Z.G.price lam A.nfold

end Accounting.Assumptions.Compose
