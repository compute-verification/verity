import Definitions.Accounting.Compose
import Definitions.Accounting.Assumptions.TS

/-!
# The compiled circuit's named assumptions (trusted)

The premises the composed theorem (`Definitions.Accounting.Guarantees.Compose`) adds to the tensor–scalar model's, the
data sheet's, sampled (I8)'s and the headlines', each a `Prop` taken as a hypothesis, never an `axiom`. Under live coins
none is about a hash: binding is charged to `δ_bind` per committed position, a per-play term. Under seeded coins the
window's coins are SHA-512's answers to the cheater's queries for its seeds, read as one uniform oracle, and that
reading is one named premise, `SeedExtractedAt` (`random-oracle`), taken once for the window beside GRIND's premise
(`SeedLuckAt`), which reads its queries per round; neither assumes the hash collision resistant.

The logical layer (`CircuitWork`) rests on four claims, the headlines', sampled (I8)'s and its own:

* (H-μ), `Accounting.Assumptions.TS.OperandHMu`, and B′, `Accounting.Assumptions.TS.ReceiptLuckAt`, taken per tile at
  its stage folded (`PerTile`): a folded tile keeps its honest checked values at `μ + (1 − μ)·r_B`, `7/8` at `¾` and `½`
  (`Accounting.Guarantees.TS.LuckOfReceipt`), as the headlines' `p`. Compose states no H-μ of its own: it takes the
  headlines' definition, so the premise is one Lean statement, counted once;
* `GuessLuckAt`: the whole-unit guess bound, per tile at its stage whole, a condition at whatever bound a kind's
  `Scheme.guess` names: the live line reads it at B′'s `½`, and `2^-64` isn't granted;
* the fixed-words claim, `Accounting.Assumptions.TS.UnitFixed` (Lemma E1(c) per unit with its fold clause),
  at the run's modes (`Accounting.Compose.tileMode`; `jointMode` in the seeded statements, on the IR-cut layer
  `Declared.atIR`) and the tiles' words. Its good units are the ones the run paid for, so a receipt-lucky skipped tile
  isn't credited.

The service's commit order (`Accounting.Compose.LayerCommitted`) is protocol structure, a fact of the declared circuit,
and lives in the live statements' `Scope`, not here: today's code doesn't meet it, so a live window it serves is out of
scope.

The seeded statements (`CircuitWorkSeeded`, `CircuitCreditSeeded`) take GRIND's premise in place of the four claims'
luck halves, over the whole window's oracle answers (a `JointRun`, no stage order: a cheater may choose round `r`'s
operands and skips after querying later rounds' seeds):

* `SeedExtractedAt`: the random-oracle premise at the window's seeds: the
  seed each round's accepted proofs derive is the oracle's answer to one of the cheater's queries at the input its
  commitment names, except on one event of probability `ε_id(T, |J|)`, per prover, `T` the window's SHA-512 calls
  and `|J|` the seeds the proofs name;
* `SeedLuckAt`: per round, round `r`'s gain at exchange rate `κ` is at most the gain of the query its proofs' seed
  answers, whose moment over that one answer, every other answer of the window fixed, is at most `1` at the seed
  group's rate, off a binding event of probability `δ_b`: at `κ = 4` and the folded rate for folded tiles (GRIND's
  `TemplateMoment` point), and at `κ₀` and the whole rate for whole ones, read off a whole-tile luck a query that no
  ruling grants, so the seeded statements' numbers are conditional on it;
* `WholeEntropyAt` (optional, read only by `Guarantees.Compose.WholeLuckOfEntropy`): a tile skipped whole keeps its
  checked values with probability at most `e^{−h}` per seed query, jointly over the round's whole tiles, given the
  cheater's view. At `h = 64·ln 2` it gives `SeedLuckAt`'s whole half at the guess bound's rate; at `K/8` nats it would
  give it at the folded rate.

The physical layer's (`CreditOfWork`):

* `Accounting.Assumptions.TS.FoldOptFoldedAt`: FOLD-OPT per word folded at one certificate, the headlines' premise,
  asked at the certificate `lam` each credit statement prices at. `FoldOptFolded` is it at every certificate, which a
  bit-exact fold falsifies, and `FoldOpt` is it at `N_w` words, which a sampled cheater that skips whole units
  falsifies.
-/

namespace Accounting.Assumptions.Compose

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

/-- `accounting-compose-guess-luck` (the whole-unit guess bound; a claim, to be ruled on). **A unit skipped whole keeps
its honest checked values only by guessing them**: in the ideal game split at the receipt, at every round, whatever the
past, a deficient round passes with probability at most `p0` (`Accounting.TS.RoundGame.LuckAt` of its game). Taken per
tile at its stage whole (`tileGame … .whole`, whose check is `same`): a tile whose product and fold are both skipped
keeps its checked values, the fold state its proof pins, with probability at most `p0`. It is a condition at the `p0` it
is given, and nothing here takes it as a fact. The live line reads it at B′'s `½` until a smaller value survives attack:
`2^-64` for a 64-bit fold state was proposed and isn't granted. A guess falsifies `luck0 = 0`. It fails if the cheater's
view pins the fold state without the fold: a state that a computed word, or a coin read before the skip, determines. -/
def GuessLuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p0 : ℝ) (G : ReceiptGame Ω S) : Prop :=
  Accounting.TS.RoundGame.LuckAt p0 G.game

/-- `random-oracle` (the random-oracle ruling: PoUW only, each use a named premise; this one at the window's
seeds). **The seed each round's accepted proofs derive is the
oracle's answer at a committed input**: on the window's oracle answers `a` (round `r`'s `T_q` seed queries' answers its
`r`-th coin, `Fin T_q → B`, all uniform and independent), `sq a r` is the query of round `r` whose answer its drawn
units derive their seed from in-circuit, at the input `SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B))`
that names the roots the round committed, and it exists for every round except on one event `Off` of probability at
most `ε_id(T, |J|)`.

The seeded statements are stated as Fiat–Shamir in the random-oracle model (`fiat-shamir`): the window's seeds are
uniform answers to the prover's queries naming its operands' roots (the coin type `Paths (Fin T_q → B) rounds`), and
this premise is where that model is named; SHA-512 is no random oracle. The probability is over the window's seed
answers only: the hashes of the commitments' trees, the live draw and the proofs' coins are not in this space. So `ε_id`
is per prover: `Off` is an event of the play's own answers, taken for the prover at hand at its own `T_q` queries a
round, and the collision part of its number is a SHA-512 collision in that prover's query log, priced by `ecr/sha-512`
(`Claims.ExpectedCR`) on a finder that runs the play and reads its log, as `δ_b` is. A prover whose queries hold a
collision, or open a node no query answered, can make `Off` likely; it fails the premise at any small `ε_id` and gets no
bound.

Alone it says only that `sq` names a query of every round off `Off` (`sq := fun _ _ => some 0` meets it): what ties
`sq` to the proofs' seeds is `SeedLuckAt`'s last conjunct, which reads the same `sq`.

Its content is gap (4)'s straight-line extraction, with no rewinding: each drawn unit's knowledge soundness extracts
the roots and position its seed is derived from, and the extractor reads the query off the cheater's query log. The
oracle stands for SHA-512 domain-separated for the commitments' trees as well as the seeds, one oracle, so the
witnessed input differs from every query's only if the queries hold a collision or open a node no query answered.

`ε_id` is proofs' formula `εid T |J|`, and the statement takes `T` and `|J|` as given, counted so:
* a query of round `r` (one of its `T_q`) is one try of the round's roots, and its answer (`B`) is every position seed
  of the round's units at those roots, so one query is `units_r` SHA-512 calls;
* `T` counts the cheater's SHA-512 calls in the window under the one oracle: its `T_q` tries a round times the round's
  units' position seeds, plus every call of the commitments' trees, so `T ≥ Σ_r T_q·units_r`, not `rounds·T_q`;
* `|J|` counts the seed inputs the proofs name, one a drawn unit's seed, at most `Σ_r units_r`, not `rounds`.
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
answer (`redraw`), every other answer of the window fixed, is at most `1` (GRIND's per-query moment at `M = 1`), and a
binding event `Break` of probability at most `δ_b` (a root opened two ways after its seed), such that off `Break` round
`r`'s gain in mode `s` (`roundGain` over the tiles of round `rd v = r`: `L − κ·U` at weights `w`) is at most the gain
`g q` of the query `q = sq a r` its proofs' seed answers (`SeedExtractedAt`, the random-oracle premise, which names
`sq`). With `Accounting.Guarantees.TS.GrindPremiumAt` it puts the gain below `ln(T_q/δ_g)/θ r` except with probability
`δ_g + δ_b`, off `SeedExtractedAt`'s `Off`.

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
`T = 2^128` a fork finder of that cost gives `δ_bind ≈ |L|·2^-64` (`Guarantees.Compose.BindingOfECR` at its best
`δ'`), not negligible beside `δ_s = 2^-40` for `|L|` past `2^24`, and at `T = 2^80` it gives `|L|·2^-88`, where the
premium is `0.73` of its `2^128` value. -/
def SeedLuckAt {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (rd : V → ℕ)
    (R : JointRun (Paths (Fin Tq → B) M) V) (w : V → ℝ) (s : Mode) (κ : ℝ) (θ : ℕ → ℝ)
    (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq)) (δb : ℝ) : Prop :=
  ∀ r : Fin M, ∃ (g : Fin Tq → Paths (Fin Tq → B) M → ℝ) (Break : Paths (Fin Tq → B) M → Prop),
    (∀ q a, ∑ b : B, Real.exp (θ r.val * g q (redraw a r q b)) ≤ Fintype.card B) ∧ prob Break ≤ δb ∧
      ∀ a q, ¬ Break a → sq a r = some q → roundGain rd r.val R w s κ a ≤ g q a

open Classical in
/-- `accounting-compose-whole-entropy` (optional, proposed at `h = K/8`, the seeded statements' reading at
`h = 64·ln 2`; read only by `Guarantees.Compose.WholeLuckOfEntropy`). **A tile skipped whole keeps its checked values
with probability at most `e^{−h}` per seed query, jointly over the round's whole tiles, given the cheater's view**: on
the window's oracle answers `a`, at every round `r` there are, per query `q`, a set `Wh q a` of round `r`'s tiles and a
set `Lk q a` of lucky ones, `Wh q` not reading query `q`'s own answer (E3: the whole skips are chosen before it), such
that for every `S ⊆ Wh q a` at most a share `e^{−Σ_{v∈S} h v}` of `q`'s answers puts all of `S` in `Lk q`, and a binding
event `Break` of probability at most `δ_b`, off which, where the proofs' seed answers `q` (`sq a r = some q`), round
`r`'s whole tiles are `Wh q a` and a whole tile whose checked values are honest (`same`) is in `Lk q a`.

What a whole-skipped H1 + fx1 tile must reproduce (`verity/core/protocols/pouw/circuit/h1tile.py`): its served `z`,
each cell's exact product, which reads no salt, and the CP-W receipt, `2·(K/32 + m)` 32-bit words (`256` served at
`K = 4096`, then `O′` and `D`), each set's pair the XOR over the tile's 32 lanes and 8 duo chains of `PouwXinFinal`
of the fx1 state, which starts from the lane's 512 seed-derived key bits (16,384 a tile). The fold state itself is
not compared. Given a view that doesn't read the query's answer (E3), `z` is fixed and no receipt bit is: fx1's word
step is a bijection of the state (`Pouw.Guarantees.Receipt.Fx1StepBijective`), so each set's 64-bit checkpoint is
near-uniform over the keys (uniform if `PouwXinFinal` is balanced), and tiles read disjoint key blocks. So
`h = 64·ln 2` a tile, one set's checkpoint, is the rigorous core, a fold-state size rather than a margin: it is
conservative against the receipt's `64·(K/32 + m)` bits, about 11 times `K/(8·ln 2)` at every `K`. With it
`WholeLuckOfEntropy` gives `SeedLuckAt`'s whole half at the guess bound's rate (`Scheme.wholeRate`, `x₀ = 44.34`
nats a tile).

At `h = K/8` nats a tile (`K/(8·ln 2)` bits, 739 at `K = 4096`) it would give it at the folded rate (`Scheme.seedRate`),
with the card "A tile skipped whole keeps its checked values with probability at most `e^{−K/8}` per seed query given
the cheater's view." That needs about `K/355` sets' checkpoints (11.5 at `K = 4096`, 9% of the served ones) jointly
near-uniform, so fx1's set-to-set map behaving as a keyed random permutation, which is a heuristic, not proved (duo has
a differential trail at `2^-32.05`), and E3 for whole skips under seeded coins. It fails if the cheater's view pins the
receipt without the fold: a key block read before the skip, or checkpoints that one computed set determines. -/
def WholeEntropyAt {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (rd : V → ℕ)
    (R : JointRun (Paths (Fin Tq → B) M) V) (h : V → ℝ)
    (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq)) (δb : ℝ) : Prop :=
  ∀ r : Fin M, ∃ (Wh Lk : Fin Tq → Paths (Fin Tq → B) M → Finset V) (Break : Paths (Fin Tq → B) M → Prop),
    (∀ q a b, Wh q (redraw a r q b) = Wh q a) ∧ (∀ q a v, v ∈ Wh q a → rd v = r.val) ∧
    (∀ q a (S : Finset V), S ⊆ Wh q a →
      ((Finset.univ.filter fun b : B => S ⊆ Lk q (redraw a r q b)).card : ℝ) ≤
        Fintype.card B * Real.exp (-∑ v ∈ S, h v)) ∧
    prob Break ≤ δb ∧
    ∀ a q, ¬ Break a → sq a r = some q → ∀ v, rd v = r.val →
      ((R.mode a v = .whole ↔ v ∈ Wh q a) ∧ (R.mode a v = .whole → R.same a v → v ∈ Lk q a))

end Accounting.Assumptions.Compose
