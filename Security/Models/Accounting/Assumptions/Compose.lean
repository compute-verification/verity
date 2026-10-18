import Definitions.Accounting.Compose
import Models.Accounting.Assumptions.TS

/-!
# The compiled circuit's named assumptions (trusted)

The premises the composed theorem (`Models.Accounting.Guarantees.Compose`,
note:compute-accounting/20261018T1758Z-draft-compose-r6) adds to the tensor–scalar model's, the data sheet's,
sampled (I8)'s and the headlines' (#2482), each a `Prop` taken as a hypothesis, never an `axiom`. Under live coins
none is about a hash: binding is charged to `δ_bind` per committed position (RT-I8's B1, decision 114), a per-play
term. Under seeded coins the window's coins are SHA-512's answers to the cheater's queries for its seeds, read as one
uniform oracle, and that reading is one named premise, `SeedExtractedAt` (`random-oracle`, card 9cd57182), taken once
for the window beside GRIND's premise (`SeedLuckAt`), which reads its queries per round; neither assumes the hash
collision resistant.

The logical layer (decision 113; `CircuitWork`) rests on four claims, the headlines' (#2482), sampled (I8)'s (#2466)
and its own:

* (H-μ), `Accounting.Assumptions.TS.OperandHMu`, and B′, `Accounting.Assumptions.TS.ReceiptLuckAt`, taken per tile at
  its stage folded (`PerTile`): a folded tile keeps its honest checked values at `μ + (1 − μ)·r_B`, `7/8` at `¾` and
  `½` (`Accounting.Guarantees.TS.LuckOfReceipt`), as the headlines' `p`. Compose states no H-μ of its own (round 2's
  `HMu` is gone): it takes the headlines' definition, so the premise is one Lean statement, counted once;
* `GuessLuckAt`: the whole-unit guess bound, per tile at its stage whole, a condition at whatever bound a kind's
  `Scheme.guess` names;
* the fixed-words claim, #2466's `Accounting.Assumptions.TS.UnitFixed` (Lemma E1(c) per unit with its fold clause),
  at the run's modes (`Accounting.Compose.tileMode`; `jointMode` in the seeded statements, on the IR-cut layer
  `Declared.atIR`) and the tiles' words. Its good units are the ones the run paid for, so a receipt-lucky skipped tile
  isn't credited (RT-COMPOSE r2's R2-B1), and its fold clause is round 2's `UnitFolded`.

The service's commit order (`Accounting.Compose.LayerCommitted`, RT-COMPOSE r4's check 1) is protocol structure, a
fact of the declared circuit, and lives in the live statements' `Scope`, not here: today's code doesn't meet it, so a
live window it serves is out of scope.

The seeded statements (`CircuitWorkSeeded`, `CircuitCreditSeeded`) take GRIND's premise in place of the four claims'
luck halves, over the whole window's oracle answers (a `JointRun`, no stage order: a cheater may choose round `r`'s
operands and skips after querying later rounds' seeds):

* `SeedExtractedAt`: the random-oracle premise, proofs' `Pouw.Assumptions.SeedExtraction` at the window's seeds: the
  seed each round's accepted proofs derive is the oracle's answer to one of the cheater's queries at the input its
  commitment names, except on one event of probability `ε_id(rounds·T_q, rounds)`, per prover;
* `SeedLuckAt`: per round, round `r`'s gain at exchange rate `κ` is at most the gain of the query its proofs' seed
  answers, whose moment over that one answer, every other answer of the window fixed, is at most `1` at the seed
  group's rate, off a binding event of probability `δ_b`: at `κ = 4` for folded tiles (GRIND's `TemplateMoment` point)
  and at `κ₀` for whole ones.

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

/-- `random-oracle` (the random-oracle card, 9cd57182: PoUW only, each use a named premise; proofs'
`Pouw.Assumptions.SeedExtraction` at the window's seeds). **The seed each round's accepted proofs derive is the
oracle's answer at a committed input**: on the window's oracle answers `a` (round `r`'s `T_q` seed queries' answers its
`r`-th coin, `Fin T_q → B`, all uniform and independent), `sq a r` is the query of round `r` whose answer its drawn
units derive their seed from in-circuit (decision 105), at the input `SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B))`
that names the roots the round committed, and it exists for every round except on one event `Off` of probability at
most `ε_id(rounds·T_q, rounds)`.

The seeded statements are stated in the random-oracle model: the window's seeds are uniform answers to the prover's
queries naming its operands' roots (the coin type `Paths (Fin T_q → B) rounds`), and this premise is where that model
is named; SHA-512 is no random oracle. The probability is over the window's seed answers only: the hashes of the
commitments' trees, the live draw and the proofs' coins are not in this space. So `ε_id` is per prover: `Off` is an
event of the play's own answers, taken for the prover at hand at its own `T_q` queries a round, and the collision
part of its number is a SHA-512 collision in that prover's query log, priced by `ecr/sha-512` (`Claims.ExpectedCR`) on
a finder that runs the play and reads its log, as `δ_b` is. A prover whose queries hold a collision, or open a node no
query answered, can make `Off` likely; it fails the premise at any small `ε_id` and gets no bound.

Alone it says only that `sq` names a query of every round off `Off` (`sq := fun _ _ => some 0` meets it): what ties
`sq` to the proofs' seeds is `SeedLuckAt`'s last conjunct, which reads the same `sq`.

Its content is gap (4)'s straight-line extraction, with no rewinding: each drawn unit's knowledge soundness extracts
the roots and position its seed is derived from, and the extractor reads the query off the cheater's query log. The
oracle stands for SHA-512 domain-separated for the commitments' trees as well as the seeds, one oracle, so the
witnessed input differs from every query's only if the queries hold a collision or open a node no query answered.
`ε_id` is proofs' formula, `εid T |J|` at `T = rounds·T_q` evaluations under the window's salt and `|J| = rounds`
seeds, whose expected instance is `(T² + T·|J|)/2^512`. It is false for a service whose proofs don't bind the seed to
the committed roots and position: a position the prover picks inside the unit picks the seed (decision 106's
condition). -/
def SeedExtractedAt {B : Type} [Fintype B] {Tq M : ℕ} (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq))
    (εid : ℕ → ℕ → ℝ) : Prop :=
  ∃ Off : Paths (Fin Tq → B) M → Prop, prob Off ≤ εid (M * Tq) M ∧ ∀ a, ¬ Off a → ∀ r, (sq a r).isSome = true

/-- `accounting-compose-seed-luck` (proposed; GRIND's premise, note:compute-accounting/20261018T0700Z-draft-grind).
**A seed group's template luck at exchange rate `κ`, per oracle query, over the whole window's oracle**: on the
window's oracle answers `a` (round `r`'s `T_q` seed queries' answers its `r`-th coin, uniform), for a joint run `R`
whose modes and checked values may read every answer of every round, at every round `r` there are per-query gains
`g q` whose exponential moment at rate `θ r` over query `q`'s own answer (`redraw`), every other answer of the window
fixed, is at most `1` (GRIND's per-query moment at `M = 1`), and a binding event `Break` of probability at most `δ_b`
(a root opened two ways after its seed), such that off `Break` round `r`'s gain in mode `s` (`roundGain` over the
tiles of round `rd v = r`: `L − κ·U` at weights `w`) is at most the gain `g q` of the query `q = sq a r` its proofs'
seed answers (`SeedExtractedAt`, the random-oracle premise, which names `sq`). With
`Accounting.Guarantees.TS.GrindPremiumAt` it puts the gain below `ln(T_q/δ_g)/θ r` except with probability `δ_g + δ_b`,
off `SeedExtractedAt`'s `Off`.

It has no stage order: a cheater may choose round `r`'s operands and skips after querying the oracle for every later
round's seed (look-ahead: try several of round `r`'s skip patterns, query round `r + 1`'s seed for each, keep the
best), or re-roll round `r`'s values after its seed. Each try is a query of its own, and GRIND's per-query union over
the `T_q` queries is what pays for it, so the premium's number is the staged one's. The binding the seeded statements
check is at the draw (`Declared.atDraw`), after every seed, where Fiat–Shamir has fixed the roots each seed's
preimage names and every round's receipts.

`δ_b` is per prover, as `δ_bind` is: `ecr/sha-512` (`Claims.ExpectedCR`, the record PoUW's binding reads) on that
prover's fork finder at about `2·T_q` SHA-512 calls a round gives about `2·T_q/2^256` (the strict birthday bound,
`cr/sha-512`, implies it, at `T_q²/2^513`), and a prover that holds a collision fails this premise and gets no
bound. `Break` is an event of the play's own answers, so no constant bounds it for every prover.

At `s = folded`, `κ = 4` and `θ` from `Scheme.seedRate` (`seedTheta`) it is the template luck GRIND proves:
`Accounting.Guarantees.TS.TemplateMoment` (32 columns, a word dead with probability at most `2^-w` over the row bits,
then lucky at most `½` over the column bits, `θ = 1/(256·r)` a product) and `PrivateMoment`, the `½` from
`Accounting.Guarantees.HMu.CoinPointMassHalf`. What stays assumed: that a seed group's gain decomposes into its
segments' (argued in GRIND, not stated), that the query fixes the skip decision (E3), the binding term, and the
oracle (`SeedExtractedAt`). At `s = whole` and `κ₀` it is the guess bound per query (`LuckMoment`, `PrivateMoment` at
the fold state's guess), at the same rate. It is taken at `κ = 4`, not at the live luck `p = 7/8`: from `p = 7/8`
alone no `θ > 0` has moment at most `1` at `κ = 4`, which needs `κ > p/(1 − p) = 7` (RT-COMPOSE r4). So the seeded
statements read a stronger luck than the live ones: `TemplateMoment` sees each word's row and column bits, where the
live statements' (H-μ) and B′ see a folded tile's one point mass. -/
def SeedLuckAt {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (rd : V → ℕ)
    (R : JointRun (Paths (Fin Tq → B) M) V) (w : V → ℝ) (s : Mode) (κ : ℝ) (θ : ℕ → ℝ)
    (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq)) (δb : ℝ) : Prop :=
  ∀ r : Fin M, ∃ (g : Fin Tq → Paths (Fin Tq → B) M → ℝ) (Break : Paths (Fin Tq → B) M → Prop),
    (∀ q a, ∑ b : B, Real.exp (θ r.val * g q (redraw a r q b)) ≤ Fintype.card B) ∧ prob Break ≤ δb ∧
      ∀ a q, ¬ Break a → sq a r = some q → roundGain rd r.val R w s κ a ≤ g q a

/-- `accounting-compose-fold-opt` (assumed; FOLD-OPT(fx1)'s mechanical fact, at least 4 issue slots a word folded).
**FOLD-OPT_μ per word folded**: at every certificate, a run that folds `folded` committed words has its fold priced at
least `(1 − μ_f)·φ*(π)` a word, `(1 − μ_f)·folded·φ*(π)`. -/
def FoldOptFolded {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (folded : ℝ) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * folded * Z.phiStar lam ≤ Z.G.price lam A.nfold

end Accounting.Assumptions.Compose
