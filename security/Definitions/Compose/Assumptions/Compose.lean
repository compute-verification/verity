module

public import Definitions.Pouw.Assumptions.TS
public import Definitions.Compose.Compose
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! The compiled circuit's named assumptions, each a `Prop` taken as a hypothesis, never an `axiom`; none assumes the hash
collision resistant. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.Compose

open ResourceAccounting.TS ResourceAccounting.Compose Pouw.RelativeSecurity

/-- `accounting-compose-guess-luck` (the whole-unit guess bound; a claim, to be ruled on). **A unit skipped whole keeps
its honest checked values only by guessing them**: in the ideal game split at the receipt, at every round, whatever the
past, a deficient round passes with probability at most `p0` (`ResourceAccounting.TS.RoundGame.LuckAt` of its game). Taken per
tile at its stage whole (`tileGame … .whole`, whose check is `same`): a tile whose product and fold are both skipped
keeps its checked values, the fold state its proof pins, with probability at most `p0`. It is a condition at the `p0` it
is given, and nothing here takes it as a fact. The live line reads it at B′'s `½` until a smaller value survives attack:
`2^-64`, one set's checkpoint (two 32-bit receipt words of the tile's `2·(K/32 + m)`), was proposed and isn't granted,
and no statement reads it. A guess falsifies `luck0 = 0`. It fails if the cheater's
view pins the fold state without the fold: a state that a computed word, or a coin read before the skip, determines. -/
def GuessLuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p0 : ℝ) (G : ReceiptGame Ω S) : Prop :=
  ResourceAccounting.TS.RoundGame.LuckAt p0 G.game

/-- `random-oracle` (the random-oracle ruling: PoUW only, each use a named premise; this one at the window's
seeds). **The seed each round's accepted proofs derive is the
oracle's answer at a committed input**: on the window's oracle answers `a` (round `r`'s `T_q` seed queries' answers its
`r`-th coin, `Fin T_q → B`, all uniform and independent), `sq a r` is the query of round `r` whose answer its sampled
proof blocks derive their seed from in-circuit, at the input `SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B))`
that names the roots the round committed, and it exists for every round except on one event `Off` of probability at
most `ε_id(T, |J|)`.

The seeded statements are stated as Fiat–Shamir in the random-oracle model (`fiat-shamir`): the window's seeds are
uniform answers to the prover's queries naming its operands' roots (the coin type `Paths (Fin T_q → B) rounds`), and
this premise is where that model is named; SHA-512 is no random oracle. The probability is over the window's seed
answers only: the hashes of the commitments' trees, the live sample and the proofs' coins are not in this space. So `ε_id`
is per prover: `Off` is an event of the play's own answers, taken for the prover at hand at its own `T_q` queries a
round, and the collision part of its number is a SHA-512 collision in that prover's query log, priced by `ecr/sha-512`
(`Claims.ExpectedCR`) on a finder that runs the play and reads its log, as `δ_b` is. A prover whose queries hold a
collision, or open a node no query answered, can make `Off` likely; it fails the premise at any small `ε_id` and gets no
bound.

Alone it says only that `sq` names a query of every round off `Off` (`sq := fun _ _ => some 0` meets it): what ties
`sq` to the proofs' seeds is `SeedLuckAt`'s last conjunct, which reads the same `sq`.

Its content is gap (4)'s straight-line extraction, with no rewinding: each sampled proof block's knowledge soundness extracts
the roots and position its seed is derived from, and the extractor reads the query off the cheater's query log. The
oracle stands for SHA-512 domain-separated for the commitments' trees as well as the seeds, one oracle, so the
witnessed input differs from every query's only if the queries hold a collision or open a node no query answered.

`ε_id` is proofs' formula `εid T |J|`, and the statement takes `T` and `|J|` as given, counted so:
* a query of round `r` (one of its `T_q`) is one try of the round's roots, and its answer (`B`) is every position seed
  of the round's units at those roots, so one query is `units_r` SHA-512 calls;
* `T` counts the cheater's SHA-512 calls in the window under the one oracle: its `T_q` tries a round times the round's
  units' position seeds, plus every call of the commitments' trees, so `T ≥ Σ_r T_q·units_r`, not `rounds·T_q`;
* `|J|` counts the seed inputs the proofs name, one a sampled proof block's seed, at most `Σ_r units_r`, not `rounds`.
The random-oracle instance of `ε_id` is `(T² + T·|J|)/2^512` (`2^-256` at `T = 2^128`), and it is kept apart from
the reading of SHA-512 outside the model: the collision part read through `ecr/sha-512` on a finder that runs the play
(`E[cost]/2^256`, about `T/2^256`, `2^-128` at `T = 2^128`) prices collisions of SHA-512 itself, which the model has
none of. That is the weaker figure, and the one a reader quoting `ε_id` per prover outside the model uses. It is false for a
service whose proofs don't bind the seed to the committed roots and position: a position the prover picks inside the
unit picks the seed. -/
def SeedExtractedAt {B : Type} [Fintype B] {Tq M : ℕ} (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq))
    (εid : ℕ → ℕ → ℝ) (T J : ℕ) : Prop :=
  ∃ Off : Paths (Fin Tq → B) M → Prop, prob Off ≤ εid T J ∧ ∀ a, ¬ Off a → ∀ r, (sq a r).isSome = true

/-- `accounting-compose-seed-luck` (proposed; GRIND's premise). **A seed group's template luck at exchange rate `κ`, per
oracle query, over the whole window's oracle**: on the window's oracle answers `a` (round `r`'s `T_q` seed queries'
answers its `r`-th coin, uniform), for a joint run `R` whose modes and checked values may read every answer of every
round, at every round `r` there are per-query gains `g q` whose exponential moment at rate `θ r` over query `q`'s own
answer (`resample`), every other answer of the window fixed, is at most `1` (GRIND's per-query moment at `M = 1`), and a
binding event `Break` of probability at most `δ_b` (a root opened two ways after its seed), such that off `Break` round
`r`'s gain in mode `s` (`roundGain` over the tiles of round `rd v = r`: `L − κ·U` at weights `w`) is at most the gain
`g q` of the query `q = sq a r` its proofs' seed answers (`SeedExtractedAt`, the random-oracle premise, which names
`sq`). With `ResourceAccounting.Guarantees.TS.GrindPremiumAt` it puts the gain below `ln(T_q/δ_g)/θ r` except with probability
`δ_g + δ_b`, off `SeedExtractedAt`'s `Off`.

It has no stage order: a cheater may choose round `r`'s operands and skips after querying the oracle for every later
round's seed (look-ahead: try several of round `r`'s skip patterns, query round `r + 1`'s seed for each, keep the
best), or re-roll round `r`'s values after its seed. Each try is a query of its own, and GRIND's per-query union over
the `T_q` queries is what pays for it, so the premium's number is the staged one's. The binding the seeded statements
check is at the sample (`Declared.atSample`), after every seed, where Fiat–Shamir has fixed the roots each seed's
preimage names and every round's receipts.

`δ_b` is per prover, as `δ_bind` is: `ecr/sha-512` (`Claims.ExpectedCR`, the record PoUW's binding reads) on that
prover's fork finder at about `2·T_q` SHA-512 calls a round gives about `2·T_q/2^256` (the strict birthday bound,
`cr/sha-512`, implies it, at `T_q²/2^513`), and a prover that holds a collision fails this premise and gets no
bound. `Break` is an event of the play's own answers, so no constant bounds it for every prover.

At `s = folded`, `κ = 4` and `θ` from `Scheme.seedRate` (`seedTheta`) it is the template luck GRIND proves:
`ResourceAccounting.Guarantees.TS.TemplateMoment` (32 columns, a word dead with probability at most `2^-w` over the row bits,
then lucky at most `½` over the column bits, `θ = 1/(256·r)` a product) and `PrivateMoment`, the `½` from
`ResourceAccounting.Guarantees.HMu.CoinPointMassHalf`. What stays assumed: that a seed group's gain decomposes into its
segments' (argued in GRIND, not stated); E3 in its one-oracle form, that query `q`'s own answer reaches round `r`'s
gain only through `same` (the round's skips and the values the run commits for its tiles may read every other answer
of the window, but not `q`'s, which decides only whether they are the honest ones); the binding term; and the oracle
(`SeedExtractedAt`). A query is one try of the round's roots, whose answer is every position seed of the round's
units (`SeedExtractedAt`'s count).

At `s = whole` and `κ₀` it is the guess bound per query (`LuckMoment`, `PrivateMoment` at the fold state's guess) at the
whole rate (`wholeTheta`, from `Scheme.wholeRate`), not the folded one: at the folded rate a whole tile's moment is
`guess·e^{K/8}` (`e^{467}` at `K = 4096`), past `1` for `K ≥ 355` at `guess = 2^-64`. At the guess bound's rate,
`x₀ = 44.34` nats a tile at `2^-64` and `κ₀ = 0.092`, it follows from `WholeEntropyAt` at `h = 64·ln 2` a tile
(`Guarantees.Compose.WholeLuckOfEntropy`); at the folded rate it would need `WholeEntropyAt` at `K/8` nats a tile (to
within `e^{−κ₀·K/8}`; proposed, not taken). Neither is granted; the live line reads `½`. Read per query, `½`
(`WholeEntropyAt` at `h = ln 2`) has moment `½e^{x₀} + ½e^{−κ₀x₀} ≤ 1` only for `κ₀ > 1`, and then `x₀ < ln 2` a tile,
about 177 tiles a seed group at `T_q = 2^128`, which leaves the seeded pair's small-batch decode line above 50%.

It is taken at `κ = 4`, not at the live luck `p = 7/8`: from `p = 7/8` alone no `θ > 0` has moment at most `1` at
`κ = 4`, which needs `κ > p/(1 − p) = 7`. So the seeded statements read a stronger luck than the live ones:
`TemplateMoment` sees each word's row and column bits, where the live statements' (H-μ) and B′ see a folded tile's one
point mass.

The premium's `T_q` and the binding's `δ_b` and `δ_bind` are priced at one prover budget `T`, the same cheater's: at
`T = 2^128` a fork finder of that expected cost gives `δ_bind ≈ |L|·2^-64` (`Guarantees.Compose.BindingOfECR` at its
best `δ'` and at `κ = 0`; the finder there runs the window twice, which is `√2` more), not negligible beside
`δ_s = 2^-40` for `|L|` past `2^24`, and at `T = 2^80` it gives `|L|·2^-88`, where the premium `ln(T_q/δ_g)/θ` is
`0.73` of its `2^128` value at `δ_g = 2^-40/(2·rounds)` over 47 to 4,488 rounds. -/
def SeedLuckAt {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (rd : V → ℕ)
    (R : JointRun (Paths (Fin Tq → B) M) V) (w : V → ℝ) (s : Mode) (κ : ℝ) (θ : ℕ → ℝ)
    (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq)) (δb : ℝ) : Prop :=
  ∀ r : Fin M, ∃ (g : Fin Tq → Paths (Fin Tq → B) M → ℝ) (Break : Paths (Fin Tq → B) M → Prop),
    (∀ q a, ∑ b : B, Real.exp (θ r.val * g q (resample a r q b)) ≤ Fintype.card B) ∧ prob Break ≤ δb ∧
      ∀ a q, ¬ Break a → sq a r = some q → roundGain rd r.val R w s κ a ≤ g q a

open Classical in
/-- Optional, read only by `Guarantees.Compose.WholeLuckOfEntropy`. At `h = 64·ln 2` a tile it rests on one set's checkpoint
being near-uniform; at `h = K/8` nats it needs fx1's set-to-set map to behave as a keyed random permutation, a
heuristic, not proved. -/
def WholeEntropyAt {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (rd : V → ℕ)
    (R : JointRun (Paths (Fin Tq → B) M) V) (h : V → ℝ)
    (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq)) (δb : ℝ) : Prop :=
  ∀ r : Fin M, ∃ (Wh Lk : Fin Tq → Paths (Fin Tq → B) M → Finset V) (Break : Paths (Fin Tq → B) M → Prop),
    (∀ q a b, Wh q (resample a r q b) = Wh q a) ∧ (∀ q a v, v ∈ Wh q a → rd v = r.val) ∧
    (∀ q a (S : Finset V), S ⊆ Wh q a →
      ((Finset.univ.filter fun b : B => S ⊆ Lk q (resample a r q b)).card : ℝ) ≤
        Fintype.card B * Real.exp (-∑ v ∈ S, h v)) ∧
    prob Break ≤ δb ∧
    ∀ a q, ¬ Break a → sq a r = some q → ∀ v, rd v = r.val →
      ((R.mode a v = .whole ↔ v ∈ Wh q a) ∧ (R.mode a v = .whole → R.same a v → v ∈ Lk q a))

end ResourceAccounting.Assumptions.Compose
