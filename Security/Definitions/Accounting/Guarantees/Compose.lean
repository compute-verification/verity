import Definitions.Claims
import Definitions.Accounting.Guarantees.Sheet
import Definitions.Accounting.Assumptions.Compose

/-!
# PoUW's credit on an arbitrary declared circuit (trusted)

THEOREM's composed statement. Each `def G : Prop` is proved as `Accounting.SecurityProofs.Compose.G` in
`Proofs.Accounting.Compose`, whose module doc gives each proof's steps. The circuit is any `Declared` one, a `Gates.Dag`
with its call sites, its rounds, their coin modes, the service's commit coins and its proof units, its words, kinds,
cuts and late cones read off the IR.
Nothing about its form is a hypothesis beyond `Declared.Wf` (call sites, rounds and units in range; one unit's gates in
one round; each coin input read in its own round; one unit's words of one kind).

The live statements (`CircuitWork`, `CircuitCredit`) need every round that holds a credited tile live and the service
committing by the IR's cuts, a computed value only its own unit reads (a receipt) by the draw (`Scope`'s
`Declared.Live` and `LayerCommitted`), and describe the designed protocol.
PoUW's code today has one salt a window and derives every noise in the scheme, so every window it runs is seeded, and
its credit is the seeded statements' (`CircuitWorkSeeded`, `CircuitCreditSeeded`), over the same circuit, at the
grinding premium per seed group. The certificate records which (`Certificate.coins`). Nothing yet builds a
`Declared` from a compiled Program, and `Properties.Compose.UnitsOfLayout`, which would tie its units, its commit
coins and its draw to the service's, is unproved: until both exist, these speak of the declared circuit, not of a
served workload.

The credit splits in two, and `CircuitCredit` is their composition:

* (L) `CircuitWork`, the logical layer: an accepted window's run has all but a share of its committed words fixed
  right, at costs `a, b` below `Λ·ln(1/δ_s) + min(a, b)·D₀` (`WorkAt`, `rOf`). It rests on four claims: (H-μ)
  (`OperandHMu`) and B′ (`ReceiptLuckAt`) per tile folded, the whole-unit guess bound (`GuessLuckAt`) per tile whole,
  at B′'s `½` (`2^-64` isn't granted), and the fixed-words claim, `UnitFixed` with
  its fold clause at the run's modes (`tileMode`), with the templates' proved luck (`Scheme.LuckProved`), the service's commit order (`LayerCommitted`, in `Scope`), the
  service's draw and sampled (I8)'s pieces on the window's proofs (`DrawnLayer.Of`, binding at
  `δ_bind`, joint proof soundness, `Pins`). The proofs' layer is the IR's (`Declared.layer`): its slots, writers, cuts, coins and draw are
  defined from the circuit, so under `LayerCommitted` it is bound at the IR's cuts (a value only its own unit reads,
  committed later, at its commit coin, which adds no late edge out of its unit), reads each round's coin and has
  every late edge the IR has (`StageCuts`, `StageCoins`, `StageSlots`, `StageComplete`, proved of it), and only its
  run-dependent half (`ProofRun`) is a parameter. No machine model. Its crypto half is `DrawnRightOff`, one
  event of probability `δ_bind + δ_ks` off which every drawn tile is right.
* (P) `CreditOfWork`, the physical layer's step: on the tensor–scalar model of a data sheet, a run with that work takes
  at least `(1 − ε)·T_cred`. Its premises are exactly the machine model's: the model the circuit's on the sheet
  (`Fits`, `ModelScope`), the data sheet's A1–A7, FOLD-OPT per word folded at the certificate it prices at
  (`FoldOptFoldedAt`) and FILL-OPT.

The seeded pair is (L) and its composition with (P) under seeded coins, stated as Fiat–Shamir in the random-oracle model
(`fiat-shamir`): the window's seeds are uniform answers to the prover's queries naming its operands' roots, and
`SeedExtractedAt` (`random-oracle`) is where that model is named, at `ε_id` once a window. Its run is a `JointRun` on
every round's answers at once, with no stage order. Its binding is checked at the draw (`Declared.atDraw`: `Of`,
`MovedIn`, `JointSound`), after every seed, since a seeded round's coin is a hash the cheater can evaluate before it
commits; its cones are counted at the IR's cuts (`Declared.atIR`: `UnitFixed`, `ServiceCovers`), which read no committed
value, and its scope needs each seeded round's seed to name what the round reads (`SeedNamed`). GRIND's premise
(`SeedLuckAt`) at the template's `κ = 4` for folded tiles and the guess bound's `κ₀` for whole ones stands in for the
live luck claims, and the bound is `rOfSeeded`, the live draw's charge on the unlucky share plus each seed group's
premiums `π* = ln(T_q/δ_g)/θ` from the IR (`premiumOf`), folded and whole at their own rates (`seedTheta`,
`wholeTheta`). No ruling grants a whole-tile luck per seed query, so the seeded pair's numbers are each labelled by the
premise its whole rate reads: B′'s `½` a query (the ruled live value, which leaves its small-batch decode line above 50%),
`GuessLuckAt` at `2^-64` or `WholeEntropyAt` at `K/8` nats a tile, neither granted.

The other guarantees are (L)'s lemmas:

* `ServiceBudget`: the window's budget `RoundGame.BudgetAt Λ` for the service's one draw, from per-stage luck and the
  stage-share bound (W3), with folded and whole shares priced apart (B1);
* `SplitRateClosed`: `SplitRate` at `Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b(1 + Kρw_max)/(ρK(1 − p0)))` for whole units
  lucky at the guess bound `p0 ∈ [0, 1)`;
* `SplitRateExp`: `SplitRateAt` at the sharp `Λ`, with its `ν, ν₀` in closed form;
* `SplitRateMerged`: `SplitRate` at `Λ = b(1 + Kρw_max)/(ρK(1 − q))` for any luck at most `q`;
* `ServiceCovers`: the service's share, each skipped tile's late cone from the IR (`tileCone`), covers every tile that
  isn't good at any joint run's modes on the IR's layer (`DrawnLayer.Covers`, from `DrawnLayer.covers_of_cone`);
* `BindingOfECR`: a prover's `δ_bind` from `DrawnBindingOfFork`, AM–GM and `ecr/sha-512` on its fork finder,
  the one reading of binding's number PoUW's per-prover floors share;
* `WholeLuckOfEntropy`: `SeedLuckAt`'s whole half from `WholeEntropyAt` (the optional entropy premise), at any rate
  whose per-tile moment at that entropy is at most `1`.

`CircuitCreditFS`, a generic Fiat–Shamir form over an abstract service, and the abstract Fiat–Shamir floor it
was to meet (proofs' gap (7), `FiatShamirFloor`) are retired: the seeded pair takes GRIND's lemmas directly on the
circuit's layer.

`Pouw.Guarantees.EndToEnd` is not subsumed and stays: it is the TT cost model with Fiat–Shamir in the random-oracle
model, concluding `(1 − ε_s)^t + ε(q, N)`, where `CircuitCredit` is the TS model under live coins.
-/

namespace Accounting.Guarantees.Compose

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-! ## The logical layer -/

/-- **The service's window budget.** For a window of `M` rounds' stages over `N` tiles with share weights `w ≥ 0` (the
tiles' cones in `CircuitWork`), each tile at least `ρ` times its share of the `N` (`ρ·N·w_v ≤ 1`; at grid padding
`ρ = 1/(F·r)`, `r = (N/N_real)·(w_max/w̄)`), every stage's share at most `w_max`, per-stage luck `lk` folded and `lk0`
whole on the tiles' checked values (`SplitStages.LuckAt`), at most `p` and `p0` on every tile that carries work, and
costs `a, b ≥ 0` per unit of folded and whole share: if `SplitRate K ρ p p0 a b w_max Λ`, the window's one round, its
draw a uniform `K`-subset (`serviceGame`), has its saving per unit of detection bounded by `Λ`. Stages are independent
given the past, so the cheater's best is to concentrate its share in one stage, and `Λ` falls as `1/K` until the stage
share binds (W3). -/
def ServiceBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M)
    (w lk lk0 : Fin N → ℝ) (K : ℕ) (ρ p p0 a b wmax Λ : ℝ),
    (∀ v, 0 ≤ w v) → 0 < ρ → (∀ v, ρ * N * w v ≤ 1) → (∀ j, stageShare stg w j ≤ wmax) →
    (∀ v, 0 < w v → lk v ≤ p ∧ lk0 v ≤ p0) → W.LuckAt lk lk0 stg → 0 ≤ a → 0 ≤ b →
    SplitRate K ρ p p0 a b wmax Λ → RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W w a b K)

/-- **The rate in closed form, whole units lucky at the guess bound.** For `K ≥ 1` draws, `ρ > 0`, `0 ≤ ρ·w_max ≤ 1`,
folded luck `p ∈ [0, 1)`, whole luck `p0 ∈ [0, 1)` (the guess bound, at any value) and costs `a, b > 0`, `SplitRate`
holds at `Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b(1 + Kρw_max)/(ρK(1 − p0)))`, the larger of the folded and the whole
branch: B1's `max(κ/ln(1/p), (σ_w + σ_f)/s)` for the service's one draw, with `1/(K(1 − p))` in place of
`1/(K·ln(1/p))`. The folded branch binds when `b·(1 − p) ≤ a·(1 − p0)`, the whole one otherwise. A whole luck `p0 = 0`
is one a guess exceeds. At `K_d = 96,000`, `ln(1/δ_s) = ln 2^40`, `a = 0.218`, `b = 0.998`, `p = 7/8`, `F = 1` and
`Kρw_max = 2/r`, with whole units at `p0 = ½` (B′'s, the live line's value), `b/a = 4.58` exceeds
`(1 − p0)/(1 − p) = 4`, so the whole branch binds (the folded one would for `p0 ≤ 0.428`), and the live `ε`'s logical
term is 0.173%, 0.231% and 0.346% at `r = 1, 2, 4`. At `p0 = 2^-64` (proposed, not granted) the folded branch binds at
0.151%, 0.201% and 0.302%. Proved from `SplitRateExp`: each closed branch is at least the sharp one. -/
def SplitRateClosed : Prop :=
  ∀ (K : ℕ) (ρ p p0 a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ p → p < 1 → 0 ≤ p0 → p0 < 1 →
    0 < a → 0 < b →
    SplitRate K ρ p p0 a b wmax
      (max (a * (1 + K * ρ * wmax) / (ρ * K * (1 - p))) (b * (1 + K * ρ * wmax) / (ρ * K * (1 - p0))))

/-- **The rate, sharp, with its `ν` in closed form**. For `K ≥ 1` draws, `ρ > 0`, `0 < ρ·w_max ≤ 1`, lucks
`p, p0 ∈ [0, 1)` and costs `a, b > 0`, `SplitRateAt` holds at
`Λ = max(a·w_max/−ln(p + (1 − p)·e^{−Kρw_max}), b·w_max/−ln(p0 + (1 − p0)·e^{−Kρw_max}))` with
`ν = (e^{a·w_max/Λ} − 1)/((1 − p)·w_max)` and `ν₀ = (e^{b·w_max/Λ} − 1)/((1 − p0)·w_max)`. The proof: the left side is
convex in `(L_F, L_0)` and the right affine, so their vertices suffice; `(1 − t)^K ≤ e^{−Kt}`; the bound is then the
exponential of an affine function of `(W_F, W_0)`, so the simplex's vertices suffice; at `(w_max, 0)` with `L_F = 0` it
is `Λ` at least the folded branch, with `L_F = w_max` an identity by `ν`'s definition, and the same at `(0, w_max)`. At
`SplitRateClosed`'s parameters, `p0 = ½`, the live `ε`'s logical term is 0.110%, 0.153% and 0.263% at `r = 1, 2, 4`: the
folded branch binds while `b/a ≤ ln(p0 + (1 − p0)e^{−Kρw_max})/ln(p + (1 − p)e^{−Kρw_max})` (4.95, 4.62 and 4.34 there,
against `b/a = 4.58`), so only `r = 4` is the whole branch's, where `p0 = 2^-64` gives 0.250%. Proved. -/
def SplitRateExp : Prop :=
  ∀ (K : ℕ) (ρ p p0 a b wmax Λ : ℝ), 1 ≤ K → 0 < ρ → 0 < wmax → ρ * wmax ≤ 1 →
    0 ≤ p → p < 1 → 0 ≤ p0 → p0 < 1 → 0 < a → 0 < b →
    Λ = max (a * wmax / -Real.log (p + (1 - p) * Real.exp (-(K * ρ * wmax))))
      (b * wmax / -Real.log (p0 + (1 - p0) * Real.exp (-(K * ρ * wmax)))) →
    SplitRateAt K ρ p p0 a b wmax Λ ((Real.exp (a * wmax / Λ) - 1) / ((1 - p) * wmax))
      ((Real.exp (b * wmax / Λ) - 1) / ((1 - p0) * wmax))

/-- **The rate in closed form, merged.** For `K ≥ 1` draws, `ρ > 0`, `0 ≤ ρ·w_max ≤ 1`, luck `p, p0 ≤ q` with
`q ∈ [0, 1)` and costs `0 < a ≤ b`, `SplitRate` holds at `Λ = b(1 + Kρw_max)/(ρK(1 − q))`: every skip priced at the
whole cost and the larger luck. -/
def SplitRateMerged : Prop :=
  ∀ (K : ℕ) (ρ p p0 q a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ q → p ≤ q → p0 ≤ q → q < 1 →
    0 < a → a ≤ b → SplitRate K ρ p p0 a b wmax (b * (1 + K * ρ * wmax) / (ρ * K * (1 - q)))

/-- **The service's share covers every tile that isn't good.** For a declared circuit's window whose ideal game is the
service's at share `1` per unit of each skipped tile's late cone (`tileCone`, from the IR) at any joint run `R` of the
rounds' coins (`jointGame`; a split-stage window's `serviceGame` is one, at `SplitStages.joint`), and its proofs the
IR's layer (`Declared.layer`, at any run-dependent half `pr`), committed as the layer says (`LayerCommitted`; at
`Declared.atIR` always), the share covers the words of every tile that isn't good at the run's modes
(`DrawnLayer.Covers` at `jointMode R`, the tiles' words and `N_w`). The layer is then bound at the IR's cuts (a
computed value only its own unit reads at its later commit coin), reads each round's coin and has its slots and
writers inside the IR (`StageCuts`, `StageCoins`, `StageSlots`), so a late edge of the layer is one of the IR's or
stays in its unit, and a tile that isn't good is in the cone of a tile the run didn't pay for,
which the share charges that cone (`DrawnLayer.covers_of_cone`). It reads no committed value and no rightness, so the
run's modes may read every round's coin. Tiles of no words are charged too: their cones are the bound's `D₀`. -/
def ServiceCovers : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (R : JointRun (Paths C Cd.rounds) (Fin Np)) (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V),
    Cd.Wf → LayerCommitted Cd → Pl.game = jointGame (Ξ := Ξ) R (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.layer hr hK pr).Covers Pl (jointMode R) (fun _ t => (tileWords Sc Cd Np t : ℝ)) (wordsOf Sc Cd : ℝ)

open Classical in
/-- **A prover's `δ_bind`, read through `ecr/sha-512`** (the one reading of binding's number, the one PoUW's
per-prover floors use). For a sampled window's proofs (`DrawnLayer`) whose round-`w` slots lie in `L`, and a SHA-512
collision finder `F` for `H` whose collisions bound the fork event up to `κ` (`DrawnLayer.Forked` at most
`Pr[F finds] + κ`: the prover's fork finder, which runs the window twice from the cut and takes both openings with
C-Flock's knowledge extractor, `κ` its extraction failure), for every `δ' > 0`
`Pr[MovedIn w] ≤ δ' + |L|²/(4δ')·(ecr(F) + κ)`, where `ecr(F)` is `E[cost F]/2^256` if `F` meets `ecr/sha-512`
(`Claims.ExpectedCR`) and `⊤` if it doesn't. It is `DrawnBindingOfFork` (`Pr[MovedIn]² ≤ |L|²·Pr[fork]`)
with AM–GM (`√(γx) ≤ δ' + γx/(4δ')`), and it assumes no hash: a prover whose fork finder beats `ecr` gets `⊤`, so it
is the per-prover `δ_bind` the statements take (`MovedIn ≤ δ_bind`), on the IR's layer under live coins and on the
draw's (`Declared.atDraw`) under seeded ones. `SeedLuckAt`'s `δ_b` is read the same way, at a round's roots, at about
`2·T_q` SHA-512 calls. At the best `δ'` the bound is about `|L|·√(E[cost F]/2^256)`, and a seeded window prices it at
the budget its premium reads (one `T` for the cheater): at `2^128` it is `|L|·2^-64` (`2^-44` to `2^-34` at `|L|` from
`2^20` to `2^30`, not negligible beside `δ_s = 2^-40`), at `2^80` it is `|L|·2^-88` (`2^-68` to `2^-58`). `ecr` is the
weaker of the two collision claims at a fixed cost (`StrictBirthday` implies it) and the one that applies to
expected-time finders such as an extractor. -/
def BindingOfECR : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : DrawnLayer C B S m s J Q Pos V) (w : Fin S) (L : Finset Pos) (H : Claims.Msg → Claims.Msg)
    (F : Claims.Finder) (κ δ' : ℝ), (∀ j k, pf.pos w j k ∈ L) → 0 < δ' →
    prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2) ≤
      F.Pr H + κ →
    ENNReal.ofReal (prob (fun x => pf.MovedIn x w)) ≤
      ENNReal.ofReal δ' + ENNReal.ofReal ((L.card : ℝ) ^ 2 / (4 * δ')) *
        ((if Claims.ExpectedCR H F then ENNReal.ofReal (Claims.Ex (fun c => (F.cost c : ℝ)) / 2 ^ (256 : ℝ))
          else (⊤ : ENNReal)) + ENNReal.ofReal κ)

/-- **The whole skips' seeded luck from their entropy.** For a joint run `R` of a window's oracle answers, weights
`w ≥ 0`, `κ₀ ≥ 0` and rates `θ ≥ 0`, if every tile's moment at its entropy is at most `1`
(`e^{−h}·e^{θ·w} + (1 − e^{−h})·e^{−κ₀·θ·w} ≤ 1`, the guess bound's `LuckMoment` at `e^{−h}`) and `WholeEntropyAt`
holds at `h`, then `SeedLuckAt` holds for whole tiles at `κ₀` and `θ`, off the same binding event. The gain of query
`q` is `Σ_{v ∈ Wh q} w_v·((1 + κ₀)·[v ∈ Lk q] − κ₀)`, which bounds the round's (`same` lands in `Lk`); its
exponential is `Π_v (e^{−κ₀x_v} + [v ∈ Lk q]·(e^{x_v} − e^{−κ₀x_v}))`, `x_v = θ·w_v`, which expands over the subsets
`S` of `Wh q` with nonnegative coefficients, so its mean over `q`'s answer is at most
`Π_v (e^{−h_v}·e^{x_v} + (1 − e^{−h_v})·e^{−κ₀x_v}) ≤ 1`. At `h = 64·ln 2` it is the whole half of the seeded pair's
luck at the guess bound's rate (`x₀ = 44.34` nats a tile at `κ₀ = 0.092`); at `h = K/8` nats it is at the folded
rate. -/
def WholeLuckOfEntropy : Prop :=
  ∀ {B V : Type} [Fintype B] [Fintype V] {Tq M : ℕ} (rd : V → ℕ) (R : JointRun (Paths (Fin Tq → B) M) V)
    (w h : V → ℝ) (κ0 : ℝ) (θ : ℕ → ℝ) (sq : Paths (Fin Tq → B) M → Fin M → Option (Fin Tq)) (δb : ℝ),
    0 ≤ κ0 → (∀ v, 0 ≤ w v) → (∀ r, 0 ≤ θ r) →
    (∀ v, Real.exp (-h v) * Real.exp (θ (rd v) * w v) +
      (1 - Real.exp (-h v)) * Real.exp (-(κ0 * (θ (rd v) * w v))) ≤ 1) →
    WholeEntropyAt rd R h sq δb → SeedLuckAt rd R w .whole κ0 θ sq δb

/-- **(L) The circuit's work, live coins** (the logical layer). For every declared circuit `Cd` and scheme `Sc`, run as
one window: the circuit's rounds in order, each round's noise coin live and released after its operands are committed
(one coin a round), then the service's draw of `K_d` of the `N_p ≥ units` grid-padded tiles, one a proof unit (`drawOf`,
a uniform `K_d`-subset), with the proofs' coins `Ξ`, each drawn tile proved with its noise's generation. Under:
* the logical ranges (`Scope`: every round holding a credited tile live, `Declared.Live`, and the service's commit
  order, `LayerCommitted`: each value the prover writes committed under a root before its IR cut's coin, or, a
  computed value only its own unit reads (a receipt), before the draw; on the live path each round's values and run
  inputs under its root before its coin, which today's code doesn't meet);
* every tile at least `ρ` times its detected weight's share of the `N_p` (`shareCone`: its late cone when it carries
  words; `ρ = 1/(F·r)` at the padding's `r = (N_p/N)·(w_max/w̄)` and `F` the most tiles a late cone holds, per kind,
  `kindF`: `1` for every kind served), every round's detected weight at most `w_max`, and every tile whose cone carries
  work lucky at most `p` folded (`tileLuck`, `μ + (1 − μ)·r_B`) and `p0` whole (`tileLuck0`, the guess bound, `½` as
  ruled);
* costs `a, b ≥ 0`, `Λ > 0` and `SplitRate` at them;
* the play's ideal game the service's at share `1` per unit of each skipped tile's late cone (`tileCone`, tiles of no
  words included), and sampled (I8)'s pieces on the window's proofs, the IR's layer (`Declared.layer` at the run's `pr`:
  positions the IR's gates, its slots, writers, cuts, coins and draw defined from the circuit): `Of`, binding at
  `δ_bind` in the round (`MovedIn`; WindowEps: `Σ_c δlink_c`, which needs `LayerCommitted`'s order), proof soundness
  jointly over the draws at `δ_ks` (`JointSound`; WindowEps: `max_c κ_c + coll`) and `Pins`;
* the four claims: (H-μ) (`OperandHMu` at its kind's `μ`) and B′ (`ReceiptLuckAt` at its kind's `r_B`) for every
  credited tile folded at its stage, the guess bound (`GuessLuckAt` at its kind's guess) for every one whole, and the
  fixed-words claim, `UnitFixed` with its fold clause, on the IR's layer at the run's modes (`tileMode`) and
  the tiles' words;

an accepted window's run has all but a share `d_F + d₀` of its committed words fixed right and all but `d₀` folded,
with `a·d_F + b·d₀ < Λ·ln(1/δ_s) + min(a, b)·D₀` (`WorkAt` at `rOf`; `D₀` the cone weight on tiles of no words,
`zeroCone`, `0` for every Program COMPILE accepts), except with probability `δ_s + δ_bind + δ_ks`. Nothing here reads
a machine model or a hash: equivocation on a committed position is a binding failure, in `δ_bind` (`BindingOfECR`
reads its number per prover). The work is the prover system's: no premise ties it to a device.

Its luck line is the live claims' point lucks through `SplitRate`: `p = μ + (1 − μ)·r_B = 7/8` per folded tile, at
`1/(1 − p) = 8` per unit of unlucky share, and `p0 = ½` per whole tile, at `1/(1 − p0) = 2` per unit at the
whole cost `b = 4.58·a`, so the whole branch binds: 0.173% at `r = 1` closed (`SplitRateClosed`), 0.110% sharp, where
the folded branch still binds. The seeded pair's folded branch reads a stronger luck, GRIND's `TemplateMoment` at
`κ = 4` (each word's row and column bits, `1 + κ = 5` per unit, plus the premium), but its whole skips pay a premium at
a rate no granted premise sets (`Scheme.wholeRate`), so its line is a condition: 56.1% at Llama-3.1-8B decode, `r = 1`,
at the decided `½` a query, 1.120% if `GuessLuckAt` holds at `2^-64`, and 0.101%, below this one, only if
`WholeEntropyAt` holds at `K/8` nats a tile. `TemplateMoment` is a property of the template, not of the coin source, so
it would hold here too; reading it needs a per-round moment premise and a rate lemma in place of `SplitRate`, a
statement of its own, so this line leaves that margin unused.

Nothing yet builds a `Declared` from a compiled Program, and `Properties.Compose.UnitsOfLayout`, which would tie its
units, commit coins and draw to the service's, is unproved: until both exist, this speaks of the declared circuit, not
of a served workload. -/
def CircuitWork : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (_hN : Cd.units ≤ Np) (hr : 0 < Cd.rounds)
    (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.rounds)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 ≤ a → 0 ≤ b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.layer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.layer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.layer hr hK pr).JointSound δks → (Cd.layer hr hK pr).Pins →
    PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G) →
    PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) →
    PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) →
    UnitFixed Pl (Cd.layer hr hK pr) (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (rOf Sc Cd Np a b Λ δs) (Pl.run x).fixed (folded x)) ≤ δs + (δbind + δks)

/-- **(L) The circuit's work, seeded coins** (the logical layer under one seed a call), stated as Fiat–Shamir in the
random-oracle model (`fiat-shamir`): the window's seeds are uniform answers to the prover's queries naming its operands'
roots, and `SeedExtractedAt` is where that model is named. For every declared circuit `Cd` and scheme `Sc`, run as one
window whose rounds are seed groups: each round one call site, its coin the oracle's answers to the cheater's `T_q`
queries for that call's seed (`Fin T_q → B`, uniform), then the service's live draw of `K_d ≥ 1` of the `N_p` tiles
(`drawOf`) with the proofs' coins `Ξ`. The run is any joint run `R` of all the rounds' answers: a cheater may choose
round `r`'s operands, values and skips after querying the oracle for any round's seed. Under:
* the seeded ranges (`ScopeSeeded`: every round holding a credited tile seeded, one call site a round, no credited
  kind GRIND hasn't priced, so H1 + fx1's attention uncredited, and `SeedNamed`: every value a seeded round reads,
  committed before that round, named by its seed's preimage or registered before the salt, which today's code doesn't
  meet) and `δ_g > 0`;
* every tile at least `ρ` times its detected weight's share, costs `a > 0`, `b ≥ 0` and `κ₀ ≥ 0`;
* the play's ideal game the service's on `R` (`jointGame`) at share `1` per unit of each skipped tile's late cone;
* sampled (I8)'s binding pieces on the proofs' layer cut at the draw (`Declared.atDraw`'s `Declared.layer` at the run's
  `pr`): `Of`, `MovedIn` at `δ_bind`, `JointSound` at `δ_ks` and `Pins`. A seeded round's coin is a hash the cheater
  can evaluate before it commits, so nothing it writes, operands or a round's receipts, is bound before the live draw,
  and the draw is where the `δ_bind` card holds: Fiat–Shamir has fixed every root the seeds' preimages name;
* the fixed-words claim, `UnitFixed`, on the proofs' layer cut at the IR (`Declared.atIR`) at `R`'s modes
  (`jointMode`): it reads modes, cuts and writers, never a committed value, so it needs no binding at the IR cut, and
  its late cones are the IR's, as `ServiceCovers` counts them;
* the random-oracle premise once for the window (`SeedExtractedAt`: each round's proofs' seed is the answer to query
  `sq a r`, off one event of probability `ε_id(T, |J|)`, per prover, `T` the window's SHA-512 calls and `|J|` the
  seeds the proofs name, as `SeedExtractedAt` counts them);
* GRIND's premise per round (`SeedLuckAt` at that `sq`): folded tiles at the template's `κ = 4` (`TemplateMoment`'s
  point) and the seed groups' folded rates from the IR (`seedTheta`), whole tiles at `κ₀` and their own rates
  (`wholeTheta`, read off a whole-tile luck a query, which no ruling grants), each query's moment over its own answer
  with every other answer of the window fixed, off a binding event of probability `δ_b`, a per-prover term as
  `δ_bind` is;

an accepted window's run has all but a share `d_F + d₀` of its committed words fixed right and all but `d₀` folded,
with `a·d_F + b·d₀ < max(5a, (1 + κ₀)·b)·ln(1/δ_s)/(ρ·K_d) + a·Π_F + b·Π_0 + min(a, b)·D₀` (`WorkAt` at
`rOfSeeded`, `Π_F` and `Π_0` the seed groups' premiums `π* = ln(T_q/δ_g)/θ` from the IR at the folded and the whole
rates, `premiumOf`), except with probability `δ_s + δ_bind + δ_ks + ε_id(T, |J|) + 2·rounds·(δ_g + δ_b)`. `a > 0`
makes the bound positive, which `WorkAt`'s strict `<` needs. The premium's `T_q` and `δ_bind`, `δ_b` are read at one
prover budget: at `2^128` a fork finder gives `δ_bind ≈ |L|·2^-64` (`BindingOfECR`).

Its luck line is per seed group from the IR at GRIND's shapes (every linear its own round, attention uncredited,
`a = 0.218`, `b = 0.998`, `K_d = 96,000`, `δ_s = 2^-40`, `T_q = 2^128` and `δ_g = 2^-40/(2·rounds)`, so the grinding
events total `2^-40`), and its whole skips' premium is most of it. No ruling grants a whole-tile luck per seed query
(the ruling decided `½` for the live line only), so each number is conditional on the premise its `wholeTheta` reads:
* at `½` a query (B′'s value read per query, `WholeEntropyAt` at `ln 2`; what the decided value gives here), the
  whole moment `½e^{x₀} + ½e^{−κ₀x₀} ≤ 1` needs `κ₀ > 1` and has `x₀ < ln 2`, so a whole skip costs about 177 tiles a
  seed group: at the best `κ₀` (8.5) the line is 56.1%, 56.3% and 56.8% at `r = 1, 2, 4` for Llama-3.1-8B decode at
  batch up to 32, 8.9% at batch 256, 1.27% at a 2K-token prefill and 35.7% for Llama-3.1-70B decode. With only granted
  premises the seeded pair says nothing useful at small-batch decode;
* under `GuessLuckAt` at `2^-64` (proposed, not granted), `κ₀ = 0.092` and `x₀ = 44.34` nats a tile, about 2.8 tiles a
  seed group whatever `K` is: 1.120%, 1.152% and 1.214% at 8B decode up to batch 32 (`Π_F` 0.057%, `Π_0` 1.078%), 0.576%
  at batch 64, 0.168% at 256, 0.048% at prefill and 0.587% for 70B decode (0.824% and 0.437% at `T_q = 2^80`);
* under `WholeEntropyAt` at `K/8` nats a tile (proposed, not granted), the whole skips at the folded rate: 0.101%,
  0.133% and 0.196% at 8B decode, below the live pair's 0.173% closed.

`down_proj`'s tile (`K = 14336`) is 3.5 times a `K = 4096` tile's share, which is why the whole premium dominates at
small batch. Its folded branch reads a stronger luck than the live pair: GRIND's `TemplateMoment` at `κ = 4`, which
sees each word's row and column bits and charges `1 + κ = 5` per unit of unlucky share, where the live pair reads a
folded tile's one point luck `p = 7/8` at `1/(1 − p) = 8`, from which no moment at `κ = 4` follows (`κ > 7` is
needed). The premium `a·Π_F + b·Π_0` is what the seeded line pays for its seeds.

Nothing yet builds a `Declared` from a compiled Program, and `Properties.Compose.UnitsOfLayout`, which would tie its
units, commit coins and draw to the service's, is unproved: until both exist, this speaks of the declared circuit, not
of a served workload. -/
def CircuitWorkSeeded : Prop :=
  ∀ {ι K T L U B Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype B] [Fintype Ξ]
    [Nonempty V] {Tq : ℕ} (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (R : JointRun (Paths (Fin Tq → B) Cd.rounds) (Fin Np))
    (Pl : Play (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun (Fin Tq → B) (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ)
    (sq : Paths (Fin Tq → B) Cd.rounds → Fin Cd.rounds → Option (Fin Tq)) (εid : ℕ → ℕ → ℝ) (T J : ℕ)
    (ρ a b κ0 δs δg δb δbind δks : ℝ),
    ScopeSeeded Sc Cd δs → 0 < δg → 0 < Kd →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) → 0 < a → 0 ≤ b → 0 ≤ κ0 →
    Pl.game = jointGame (Ξ := Ξ) R (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.atDraw.layer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.atDraw.layer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.atDraw.layer hr hK pr).JointSound δks → (Cd.atDraw.layer hr hK pr).Pins →
    SeedExtractedAt sq εid T J →
    SeedLuckAt (tileRound Cd Np) R (shareCone Sc Cd Np) .folded 4 (seedTheta Sc Cd) sq δb →
    SeedLuckAt (tileRound Cd Np) R (shareCone Sc Cd Np) .whole κ0 (wholeTheta Sc Cd) sq δb →
    UnitFixed Pl (Cd.atIR.layer hr hK pr) (jointMode R) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (rOfSeeded Sc Cd Np Kd Tq a b κ0 ρ δs δg) (Pl.run x).fixed (folded x)) ≤
      δs + (δbind + δks) + εid T J + 2 * (Cd.rounds : ℝ) * (δg + δb)

/-! ## The physical layer's step -/

/-- **(P) Credit of the work, on a data sheet** (the physical layer: the tensor–scalar model and its vendor axioms,
attribution's unit grade). For a declared circuit `Cd` and scheme `Sc` whose model `Z` is the circuit's on sheet `Sh`
(`Fits`, `ModelScope`), every run `A` that meets the data sheet's A1–A7, FOLD-OPT per word folded at its `folded` words
and the certificate `lam`, and FILL-OPT, and has the work `WorkAt` states at the circuit's costs `aOf`, `bOf` and bound
`r`, takes at least `(1 − ε)·T_cred`, `ε = ε_♭ + ε_μ + r` (`epsOf`) and `T_cred = B(♭)/f_max` at the circuit's floor
`♭ = Σ_words ♭_{kind}`. It reads no coin: it is `SheetDIdealFloor`'s chain on one run, with the fold priced at the words
folded. -/
def CreditOfWork : Prop :=
  ∀ {ι K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X]
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X)
    (A : Anatomy K T L U) (folded : ℝ) (lam : SheetRow P → ℝ) (r : ℝ),
    Fits Sc Cd Sh Z → ModelScope Sc Cd Z lam → A.Split →
    SheetClock Sh A.run → SheetDispatch A.run → SheetTensor Sh A.run → SheetPipe Sh A.run → SheetSmem Sh A.run →
    SheetWords Z A → SheetSemantics Z A → FoldOptFoldedAt Z A folded lam → FillOpt Z A →
    WorkAt (wordsOf Sc Cd : ℝ) (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) r A.fixed folded →
    (1 - epsOf Sc Cd Z lam r) * creditOf Sc Cd Z ≤ A.run.time

/-! ## The composed credit -/

/-- **The credit theorem, live coins**: `CreditOfWork ∘ CircuitWork` at the circuit's costs `a = aOf`, `b = bOf` (both
`≥ 0` under `ModelScope`). Under `CircuitWork`'s premises at those costs and `CreditOfWork`'s on every run, the window's
certificate in device window `win` holds (`certOf` at `.live`): it is accepted with time below `(1 − ε)·T_cred` with
probability at most `δ_s + δ_bind + δ_ks`, where `T_cred = B(♭)/f_max` at the circuit's floor and
`ε = ε_♭ + ε_μ + Λ·ln(1/δ_s) + min(a, b)·D₀` (`rOf`). The certificate also reports the words per kind, `γ_♭` (W1) and
the coin mode, `live` under `Scope`'s `Declared.Live`: a seeded window's credit is `CircuitCreditSeeded`'s. Its `Scope`
holds the service's commit order (`LayerCommitted`), which today's code doesn't meet. Its luck line is `CircuitWork`'s,
the live claims' point lucks `p = 7/8` folded and `p0 = ½` whole (as ruled; 0.173% at `r = 1` closed, 0.110% sharp), and
with `ε_♭ = 0.571%` (NOTATION's sm_120 sheet, H1 + fx1) and `ε_μ = 0` under FOLD-OPT its `ε` is 0.744%, 0.801% and
0.917% at `r = 1, 2, 4` closed (0.681%, 0.724% and 0.834% sharp), below EPS1's 1%, with every premise it reads granted
or sampled (I8)'s (0.722%, 0.772% and 0.873% closed at the ungranted `p0 = 2^-64`). The seeded pair's is above it at
small-batch decode under every premise but `WholeEntropyAt`'s (`CircuitCreditSeeded`). Work is the prover system's, and
the certificate is what the physical layer's ledger consumes: what device time the work took, and whose, is the
ledger's, which credits one device's time once. Nothing yet builds a `Declared` from a compiled Program, and
`Properties.Compose.UnitsOfLayout`, which would tie its units, commit coins and draw to the service's, is unproved:
until both exist, this speaks of the declared circuit, not of a served workload. -/
def CircuitCredit : Prop :=
  ∀ {ι K P T L U F X C Ξ V : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] [Fintype C] [Fintype Ξ] [Nonempty V]
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (H : Honest K)
    {Np Kd : ℕ} (_hN : Cd.units ≤ Np) (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.rounds)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (lam : SheetRow P → ℝ)
    (ρ p p0 wmax Λ δs δbind δks : ℝ) (win : Accounting.Window),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 < Λ → SplitRate Kd ρ p p0 (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) wmax Λ →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.layer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.layer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.layer hr hK pr).JointSound δks → (Cd.layer hr hK pr).Pins →
    PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G) →
    PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) →
    PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) →
    UnitFixed Pl (Cd.layer hr hK pr) (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    Fits Sc Cd Sh Z → ModelScope Sc Cd Z lam → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Z (Pl.run x) (folded x) lam) → (∀ x, FillOpt Z (Pl.run x)) →
    (certOf Sc Cd Z H lam (rOf Sc Cd Np (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) Λ δs) (δs + (δbind + δks)) win
      CoinMode.live).Holds Pl

/-- **The credit theorem, seeded coins**: `CreditOfWork ∘ CircuitWorkSeeded` at the circuit's costs `a = aOf > 0`,
`b = bOf` (`≥ 0` under `ModelScope`). Under `CircuitWorkSeeded`'s premises at those costs and `CreditOfWork`'s on every
run, the window's certificate in device window `win` holds at `.seeded` (`certOf`): it is accepted with time below
`(1 − ε)·T_cred` with probability at most `δ_s + δ_bind + δ_ks + ε_id(T, |J|) + 2·rounds·(δ_g + δ_b)`, where
`ε = ε_♭ + ε_μ + max(5a, (1 + κ₀)·b)·ln(1/δ_s)/(ρ·K_d) + a·Π_F + b·Π_0 + min(a, b)·D₀` (`rOfSeeded`, `Π_F` and `Π_0` the
seed groups' premiums from the IR at the folded and the whole rates). It is stated as Fiat–Shamir in the random-oracle
model (`fiat-shamir`; `SeedExtractedAt` names it), its binding checked at the draw (`Declared.atDraw`) and its cones at
the IR (`Declared.atIR`), as `CircuitWorkSeeded`'s, and its luck line is that statement's, conditional as its numbers
are on the whole-tile luck a query its `wholeTheta` reads. With `ε_♭ = 0.571%` (NOTATION's sm_120 sheet, H1 + fx1)
and `ε_μ = 0` under FOLD-OPT, its `ε` at `r = 1` and `T_q = 2^128` is:
* at `½` a query (the ruled live value; no premise per query is granted): 56.6% for Llama-3.1-8B decode at batch up to
  32, 9.5% at batch 256, 1.85% at a 2K-token prefill and 36.3% for 70B decode, so with only granted premises a seeded
  window's credit is vacuous at small-batch decode and above 1% even at prefill;
* under `GuessLuckAt` at `2^-64` (not granted): 1.691% for 8B decode at batch up to 32 (1.395% at `T_q = 2^80`),
  1.158% for 70B decode, 0.738% for 8B decode at batch 256 and about 0.62% at prefill;
* under `WholeEntropyAt` at `K/8` nats a tile (not granted): at most 0.672%.

The live pair's is 0.744% closed and 0.681% sharp. It is the credit of every window PoUW's code runs, whose noise it
derives from the salt, once the code's seed preimages name what each round reads (`SeedNamed`, in `ScopeSeeded`) and
its binding is checked at the draw. Nothing yet builds a `Declared` from a compiled Program, and `Properties.Compose.UnitsOfLayout`, which would
tie its units, commit coins, seed preimages and draw to the service's, is unproved: until both exist, this speaks of
the declared circuit, not of a served workload. -/
def CircuitCreditSeeded : Prop :=
  ∀ {ι K P T L U F X B Ξ V : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] [Fintype B] [Fintype Ξ] [Nonempty V] {Tq : ℕ}
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (H : Honest K)
    {Np Kd : ℕ} (_hN : Cd.units ≤ Np) (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (R : JointRun (Paths (Fin Tq → B) Cd.rounds) (Fin Np))
    (Pl : Play (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun (Fin Tq → B) (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (lam : SheetRow P → ℝ)
    (sq : Paths (Fin Tq → B) Cd.rounds → Fin Cd.rounds → Option (Fin Tq)) (εid : ℕ → ℕ → ℝ) (T J : ℕ)
    (ρ κ0 δs δg δb δbind δks : ℝ) (win : Accounting.Window),
    ScopeSeeded Sc Cd δs → 0 < δg → 0 < Kd →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) → 0 < aOf Sc Cd Z lam → 0 ≤ κ0 →
    Pl.game = jointGame (Ξ := Ξ) R (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.atDraw.layer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.atDraw.layer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.atDraw.layer hr hK pr).JointSound δks → (Cd.atDraw.layer hr hK pr).Pins →
    SeedExtractedAt sq εid T J →
    SeedLuckAt (tileRound Cd Np) R (shareCone Sc Cd Np) .folded 4 (seedTheta Sc Cd) sq δb →
    SeedLuckAt (tileRound Cd Np) R (shareCone Sc Cd Np) .whole κ0 (wholeTheta Sc Cd) sq δb →
    UnitFixed Pl (Cd.atIR.layer hr hK pr) (jointMode R) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    Fits Sc Cd Sh Z → ModelScope Sc Cd Z lam → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Z (Pl.run x) (folded x) lam) → (∀ x, FillOpt Z (Pl.run x)) →
    (certOf Sc Cd Z H lam (rOfSeeded Sc Cd Np Kd Tq (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) κ0 ρ δs δg)
      (δs + (δbind + δks) + εid T J + 2 * (Cd.rounds : ℝ) * (δg + δb)) win .seeded).Holds Pl

end Accounting.Guarantees.Compose
