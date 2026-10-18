import Definitions.Accounting.Compose
import Models.Accounting.Assumptions.TS

/-!
# The compiled circuit's named assumptions (trusted)

The premises the composed theorem (`Models.Accounting.Guarantees.Compose`,
note:compute-accounting/20261018T1600Z-draft-theorem-compose-r5) adds to the tensor–scalar model's, the data sheet's,
sampled (I8)'s and the headlines' (#2482), each a `Prop` taken as a hypothesis, never an `axiom`. Under live coins
none is about a hash: binding is charged to `δ_bind` per committed position (RT-I8's B1, decision 114), a per-play
term. Under seeded coins each round's coin is SHA-512's answers to the cheater's queries for its seed, read as uniform,
and that reading is one named premise, `SeedExtractedAt` (`random-oracle`, card 9cd57182), which GRIND's premise
(`SeedLuckAt`) reads at every stage; neither assumes the hash collision resistant.

The logical layer (decision 113; `CircuitWork`) rests on four claims, the headlines' (#2482), sampled (I8)'s (#2466)
and its own:

* (H-μ), `Accounting.Assumptions.TS.OperandHMu`, and B′, `Accounting.Assumptions.TS.ReceiptLuckAt`, taken per tile at
  its stage folded (`PerTile`): a folded tile keeps its honest checked values at `μ + (1 − μ)·r_B`, `7/8` at `¾` and
  `½` (`Accounting.Guarantees.TS.LuckOfReceipt`), as the headlines' `p`. Compose states no H-μ of its own (round 2's
  `HMu` is gone): it takes the headlines' definition, so the premise is one Lean statement, counted once;
* `GuessLuckAt`: the whole-unit guess bound, per tile at its stage whole, a condition at whatever bound a kind's
  `Scheme.guess` names;
* the fixed-words claim, #2466's `Accounting.Assumptions.TS.UnitFixed` (Lemma E1(c) per unit with its fold clause),
  at the run's modes (`Accounting.Compose.tileMode`) and the tiles' words. Its good units are the ones the run paid
  for, so a receipt-lucky skipped tile isn't credited (RT-COMPOSE r2's R2-B1), and its fold clause is round 2's
  `UnitFolded`.

One premise is protocol structure, named apart because today's code doesn't do what it says (RT-COMPOSE r4's check 1):

* `LayerCommitted`: the service commits each value the prover writes under a root before its IR cut's coin, as
  `Declared.layer` cuts it; on the live path each round's values and run inputs under its root before its coin.

The seeded statements (`CircuitWorkSeeded`, `CircuitCreditSeeded`) take GRIND's premise in place of the four claims'
luck halves:

* `SeedExtractedAt`: the random-oracle premise, proofs' `Pouw.Assumptions.SeedExtraction` at one seed group: the seed
  a stage's accepted proofs derive is the oracle's answer to one of the cheater's queries at the input its commitment
  names, except with probability `ε_id`;
* `SeedLuckAt`: per stage, whatever the past, the stage's gain at exchange rate `κ` is at most the gain of the query
  its proofs' seed answers (`SeedExtractedAt`), whose moment at the seed group's rate is at most `1`, off a binding
  event of probability `δ_b`: at `κ = 4` for folded tiles (GRIND's `TemplateMoment` point) and at `κ₀` for whole ones.

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
are both skipped keeps its checked values, the fold state its proof pins, with probability at most `p0`. It is a
condition at the `p0` it is given: `2^-64` for a 64-bit fold state is the proposed value (card 8b00536a, not yet
decided), and nothing here takes it as a fact. It replaces round 2's `luck0 = 0`, which a guess falsifies
(RT-COMPOSE r2's R2-B3). It fails if the cheater's view pins the fold state without the fold: a state that a computed
word, or a coin read before the skip, determines. -/
def GuessLuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p0 : ℝ) (G : ReceiptGame Ω S) : Prop :=
  Accounting.TS.RoundGame.LuckAt p0 G.game

/-- `accounting-compose-layer-committed` (assumed; protocol structure, which today's code doesn't meet). **The service
commits as `Declared.layer` says**: every value `ℓ` the prover writes is committed under a root, which every opening of
it is checked against, before coin `Declared.commitRound ℓ` of the window's `rounds`, and that coin is no later than
its IR cut (`Declared.cutOf ℓ`). The layer cuts `ℓ` at the later of the two, so this is what makes its cuts the IR's
(`StageCuts`, `Proofs.Accounting.Compose.layer_stageCuts`): if it fails, the layer has late edges the IR's cones
don't charge, and `ServiceCovers` doesn't hold. It is also the order the binding card needs: `δ_bind`
(`DrawnLayer.MovedIn`; WindowEps's `Σ_c δlink_c`, through C-Flock's extractor and `cr/sha-512`) bounds an opening
that differs from a root fixed before the cut, and nothing for a value committed after it.

On the live path the order is: the registered inputs (weights, a codec's parameters) before the first coin; each
round's served values, and each run input (a request's tokens, decode feedback) under the root of its first reader's
round, before that round's coin; and the values that read the coin (a noised word, a receipt) before the next round's.
A run input committed after its reader's coin lets the prover choose tokens or decode feedback to match the reader's
noise: that breaks (H-μ)'s salt-independence (E3) for the reader, and it is a binding failure the `δ_bind` card
doesn't cover.

Under seeded coins round `r`'s coin is its seed, the oracle's answer at a preimage that names the roots its operands
are committed under (GRIND's `SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B))`, decision 112). Fiat–Shamir over the
roots binds them before the seed, and a root re-rolled after it is a new query, which the premium prices
(`premiumOf`). The order the seeded statements need is every root fixed before the service's live draw, each named by
its reader's seed preimage: `PROTOCOL.md`'s lifecycle describes it, and whether the code's preimage names every
operand's root is for `Properties.Compose.UnitsOfLayout`, widened to commit rounds, to show.

Today's code doesn't meet the live order: PoUW's coins are one salt a window and the draw
(`verity/core/protocols/pouw/coins.py`'s `LABELS = ("salt", "draw")`), with no per-round coin or root, and the salt
is opened before the activations are committed (RT-COMPOSE r3). And `commitRound` is the service's schedule only
across the faithfulness gap: nothing yet builds a `Declared` from a compiled Program. -/
def LayerCommitted {ι : Type} (Cd : Declared ι) : Prop :=
  ∀ ℓ, Cd.Writes ℓ → Cd.commitRound ℓ ≤ Cd.cutOf ℓ

/-- `random-oracle` (the random-oracle card, 9cd57182: PoUW only, each use a named premise; proofs'
`Pouw.Assumptions.SeedExtraction` at one seed group). **The seed a stage's accepted proofs derive is the oracle's
answer at a committed input**: `sq a`, at the stage's coin `a` (the oracle's answers to the cheater's `T_q` queries for
the round's seed, `Fin T_q → B`, uniform), is the query whose answer the stage's drawn units derive their seed from
in-circuit (decision 105), at the input `SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B))` that names the roots the
round committed, and it exists except on an event `Off` of probability at most `ε_id`.

Its content is gap (4)'s straight-line extraction, with no rewinding: each drawn unit's knowledge soundness extracts
the roots and position its seed is derived from, and the extractor reads the query off the cheater's query log. The
oracle stands for SHA-512 domain-separated for the commitments' trees as well as the seeds, one oracle, so the
witnessed input differs from every query's only if the queries hold a collision or open a node no query answered:
`ε_id ≈ (T_q² + T_q)/2^512` a stage, over the oracle. SHA-512 is no random oracle, and this premise is where the
seeded statements read it as one. It is false for a service whose proofs don't bind the seed to the committed roots
and position: a position the prover picks inside the unit picks the seed (decision 106's condition). -/
def SeedExtractedAt {B : Type} [Fintype B] {Tq : ℕ} (sq : (Fin Tq → B) → Option (Fin Tq)) (εid : ℝ) : Prop :=
  ∃ Off : (Fin Tq → B) → Prop, prob Off ≤ εid ∧ ∀ a, ¬ Off a → (sq a).isSome = true

/-- `accounting-compose-seed-luck` (proposed; GRIND's premise, note:compute-accounting/20261018T0700Z-draft-grind).
**A seed group's template luck at exchange rate `κ`, per oracle query**: at every stage `j` of a window whose rounds'
coins are the oracle's `T_q` answers (`Fin T_q → B`, uniform), whatever the past, the stage's proofs' seed is query
`sq a`'s answer off `ε_id` (`SeedExtractedAt`, the random-oracle premise), and there are per-query gains `g q` whose
exponential moment at rate `θ j` over query `q`'s own answer, every other answer fixed, is at most `1` (GRIND's
per-query moment at `M = 1`), and a binding event `Break` of probability at most `δ_b` (a root opened two ways after
its seed), such that off `Break` the stage's gain in mode `s` (`stageGain`: `L − κ·U` at weights `w`) is at most the
gain `g q` of the query `q = sq a` its seed answers. With `Accounting.Guarantees.TS.GrindPremiumAt` it puts the gain
below `ln(T_q/δ_g)/θ j` except with probability `δ_g + δ_b + ε_id`.

`δ_b` is per prover, as `δ_bind` is: about `T_q²/2^513` for a prover whose fork finder holds no SHA-512 collision at
about `2·T_q` queries (`cr/sha-512` on that finder), and `1` for a prover that holds one. `Break` is an event of the
play's own stage coin, so a prover holding a collision fails this premise and gets no bound; no constant bounds it for
every prover.

At `s = folded`, `κ = 4` and `θ` from `Scheme.seedRate` (`seedTheta`) it is the template luck GRIND proves:
`Accounting.Guarantees.TS.TemplateMoment` (32 columns, a word dead with probability at most `2^-w` over the row bits,
then lucky at most `½` over the column bits, `θ = 1/(256·r)` a product) and `PrivateMoment`, the `½` from
`Accounting.Guarantees.HMu.CoinPointMassHalf`. What stays assumed: that a seed group's gain decomposes into its
segments' (argued in GRIND, not stated), that the query fixes the skip decision (E3), the binding term, and the
oracle (`SeedExtractedAt`). At
`s = whole` and `κ₀` it is the guess bound per query (`LuckMoment`, `PrivateMoment` at the fold state's guess), at the
same rate. It is taken at `κ = 4`, not at the live luck `p = 7/8`: from `p = 7/8` alone no `θ > 0` has moment at most
`1` at `κ = 4`, which needs `κ > p/(1 − p) = 7` (RT-COMPOSE r4). -/
def SeedLuckAt {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (stg : V → ℕ)
    (W : SplitStages (Fin Tq → B) V M) (w : V → ℝ) (s : Mode) (κ : ℝ) (θ : ℕ → ℝ) (εid δb : ℝ) : Prop :=
  W.EveryStage fun j mode _ same => ∃ (sq : (Fin Tq → B) → Option (Fin Tq)) (g : Fin Tq → (Fin Tq → B) → ℝ)
    (Break : (Fin Tq → B) → Prop), SeedExtractedAt sq εid ∧
    (∀ q a, ∑ b : B, Real.exp (θ j * g q (Function.update a q b)) ≤ Fintype.card B) ∧ prob Break ≤ δb ∧
      ∀ a q, ¬ Break a → sq a = some q → stageGain stg j mode same w s κ a ≤ g q a

/-- `accounting-compose-fold-opt` (assumed; FOLD-OPT(fx1)'s mechanical fact, at least 4 issue slots a word folded).
**FOLD-OPT_μ per word folded**: at every certificate, a run that folds `folded` committed words has its fold priced at
least `(1 − μ_f)·φ*(π)` a word, `(1 − μ_f)·folded·φ*(π)`. -/
def FoldOptFolded {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (folded : ℝ) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * folded * Z.phiStar lam ≤ Z.G.price lam A.nfold

end Accounting.Assumptions.Compose
