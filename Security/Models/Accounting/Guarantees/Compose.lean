import Models.Accounting.Guarantees.Sheet
import Models.Accounting.Assumptions.Compose

/-!
# PoUW's credit on an arbitrary declared circuit (trusted)

THEOREM's composed statement, round 5 (note:compute-accounting/20261018T1600Z-draft-theorem-compose-r5), restated after
RT-COMPOSE r4 (note:compute-accounting/20261018T1522Z-draft-rt-compose-r4) under decisions 107, 109–114, 117, 119 and
122. Each `def G : Prop` is to be proved as `Accounting.SecurityProofs.Compose.G` in `Proofs.Accounting.Compose`, where
each is a `sorry` stub for now. The circuit is any `Declared` one, a `Gates.Dag` with its call sites, its rounds, their
coin modes, the service's commit coins and its proof units, its words, kinds, cuts and late cones read off the IR.
Nothing about its form is a hypothesis beyond `Declared.Wf` (call sites, rounds and units in range; one unit's gates in
one round; each coin input read in its own round; one unit's words of one kind).

The live statements (`CircuitWork`, `CircuitCredit`) need every round that holds a credited tile live (`Scope`'s
`Declared.Live`), and describe the designed protocol. PoUW's code today has one salt a window and derives every noise
in the scheme, so every window it runs is seeded, and its credit is the seeded statements' (`CircuitWorkSeeded`,
`CircuitCreditSeeded`), over the same circuit and the same proofs' layer, at decision 122's grinding premium per seed
group. The certificate records which (`Certificate.coins`). Nothing yet builds a `Declared` from a compiled Program,
and `Properties.Compose.UnitsOfLayout`, which would tie its units to the service's layout, is unproved: until
both exist, these speak of the declared circuit, not of a served workload.

Decision 113 splits the credit in two, and `CircuitCredit` is their composition:

* (L) `CircuitWork`, the logical layer: an accepted window's run has all but a share of its committed words fixed
  right, at costs `a, b` below `Λ·ln(1/δ_s) + min(a, b)·D₀` (`WorkAt`, `rOf`). It rests on four claims: (H-μ)
  (`OperandHMu`) and B′ (`ReceiptLuckAt`) per tile folded, the whole-unit guess bound (`GuessLuckAt`) per tile whole,
  and the fixed-words claim, #2466's `UnitFixed` with its fold clause at the run's modes (`tileMode`), with the
  templates' proved luck (`Scheme.LuckProved`), the service's commit order (`LayerCommitted`), the service's draw and
  sampled (I8)'s pieces as #2466 states them on the window's proofs (`DrawnLayer.Of`, binding at `δ_bind`, joint proof
  soundness, `Pins`). The proofs' layer is the IR's (`Declared.layer`): its slots, writers, cuts, coins and draw are
  defined from the circuit, so under `LayerCommitted` it is bound at the IR's cuts, reads each round's coin and has
  every late edge the IR has (`StageCuts`, `StageCoins`, `StageSlots`, `StageComplete`, proved of it), and only its
  run-dependent half (`ProofRun`) is a parameter. No machine model. Its crypto half is #2466's `DrawnRightOff`, one
  event of probability `δ_bind + δ_ks` off which every drawn tile is right (RT-I8's `compose_sampled` shape).
* (P) `CreditOfWork`, the physical layer's step: on the tensor–scalar model of a data sheet, a run with that work takes
  at least `(1 − ε)·T_cred`. Its premises are exactly the machine model's: the model the circuit's on the sheet
  (`Fits`, `ModelScope`), the data sheet's A1–A7, FOLD-OPT per word folded (`FoldOptFolded`) and FILL-OPT.

The seeded pair is (L) and its composition with (P) under seeded coins: the same premises on the proofs' layer, GRIND's
premise (`SeedLuckAt`) at the template's `κ = 4` in place of the live luck claims, read through the random-oracle
premise (`SeedExtractedAt`, `random-oracle`) at `ε_id` a stage, and the bound `rOfSeeded`, the live draw's charge on
the unlucky share plus each seed group's premium `π* = ln(T_q/δ_g)/θ` from the IR (`premiumOf`).

The other guarantees are (L)'s lemmas:

* `ServiceBudget`: the window's budget `RoundGame.BudgetAt Λ` for the service's one draw, from per-stage luck and the
  stage-share bound (W3), with folded and whole shares priced apart (B1);
* `SplitRateClosed`: `SplitRate` at `Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b(1 + Kρw_max)/(ρK(1 − p0)))` for whole units
  lucky at the guess bound `p0 ∈ [0, 1)`;
* `SplitRateExp`: `SplitRateAt` at the sharp `Λ`, with its `ν, ν₀` in closed form;
* `SplitRateMerged`: `SplitRate` at `Λ = b(1 + Kρw_max)/(ρK(1 − q))` for any luck at most `q`;
* `ServiceCovers`: the service's share, each skipped tile's late cone from the IR (`tileCone`), covers every tile that
  isn't good at the run's modes on the IR's layer (#2466's `DrawnLayer.Covers`, from `DrawnLayer.covers_of_cone`).

Round 4's `CircuitCreditFS`, a generic Fiat–Shamir form over an abstract service against proofs' `FiatShamirFloor`, is
retired: the seeded pair takes GRIND's lemmas directly on the circuit's layer. Proofs' Fiat–Shamir layer
(`FiatShamirFloor`, `ServiceFloorFS`, `SeedExtraction`) is theirs to land, and nothing here reads it.

`Pouw.Guarantees.EndToEnd` is not subsumed and stays: it is the TT cost model with Fiat–Shamir in the random-oracle
model, concluding `(1 − ε_s)^t + ε(q, N)`, where `CircuitCredit` is the TS model under live coins (RT-COMPOSE §6).
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
`1/(K·ln(1/p))`. If the guess bound holds at `2^-64` (card 8b00536a, not decided), the whole branch is
`b(1 + Kρw_max)/(ρK)` to 20 digits; round 2's `b/(ρK)` at `p0 = 0` was for a luck a guess exceeds (R2-B3). At
`K_d = 96,000`, `ln(1/δ_s) = ln 2^40`, `a = 0.218`, `b = 0.998`, `p = 7/8`, `p0 = 2^-64`, `F = 1` and `Kρw_max = 2/r`
the live `ε`'s logical term is 0.151%, 0.201% and 0.302% at `r = 1, 2, 4`: stated, not proved, while this and
`CircuitWork` are `sorry`. -/
def SplitRateClosed : Prop :=
  ∀ (K : ℕ) (ρ p p0 a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ p → p < 1 → 0 ≤ p0 → p0 < 1 →
    0 < a → 0 < b →
    SplitRate K ρ p p0 a b wmax
      (max (a * (1 + K * ρ * wmax) / (ρ * K * (1 - p))) (b * (1 + K * ρ * wmax) / (ρ * K * (1 - p0))))

/-- **The rate, sharp, with its `ν` in closed form** (RT-COMPOSE r4's check 3). For `K ≥ 1` draws, `ρ > 0`,
`0 < ρ·w_max ≤ 1`, lucks `p, p0 ∈ [0, 1)` and costs `a, b > 0`, `SplitRateAt` holds at
`Λ = max(a·w_max/−ln(p + (1 − p)·e^{−Kρw_max}), b·w_max/−ln(p0 + (1 − p0)·e^{−Kρw_max}))` with
`ν = (e^{a·w_max/Λ} − 1)/((1 − p)·w_max)` and `ν₀ = (e^{b·w_max/Λ} − 1)/((1 − p0)·w_max)`. The proof: the left side is
convex in `(L_F, L_0)` and the right affine, so their vertices suffice; `(1 − t)^K ≤ e^{−Kt}`; the bound is then the
exponential of an affine function of `(W_F, W_0)`, so the simplex's vertices suffice; at `(w_max, 0)` with `L_F = 0`
it is `Λ` at least the folded branch, with `L_F = w_max` an identity by `ν`'s definition, and the same at `(0, w_max)`.
At `SplitRateClosed`'s parameters the live `ε`'s logical term is 0.110%, 0.153% and 0.250% at `r = 1, 2, 4`: numeric
(a grid check of this inequality, tight to rounding), until this is proved. -/
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
service's at share `1` per unit of each skipped tile's late cone (`tileCone`, from the IR), and its proofs the IR's
layer (`Declared.layer`, at any run-dependent half `pr`), committed as the layer says (`LayerCommitted`), the share
covers the words of every tile that isn't good at the run's modes (`DrawnLayer.Covers` at `tileMode`, the tiles' words
and `N_w`). The layer is then bound at the IR's cuts, reads each round's coin and has its slots and writers inside the
IR (`StageCuts`, `StageCoins`, `StageSlots`), so a late edge of the layer is one of the IR's, and a tile that isn't
good is in the cone of a tile the run didn't pay for, which the share charges that cone (`DrawnLayer.covers_of_cone`).
Tiles of no words are charged too: their cones are the bound's `D₀`. -/
def ServiceCovers : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (W : SplitStages C (Fin Np) Cd.rounds) (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V),
    Cd.Wf → LayerCommitted Cd → Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.layer hr hK pr).Covers Pl (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ)) (wordsOf Sc Cd : ℝ)

/-- **(L) The circuit's work, live coins** (decision 113's logical layer). For every declared circuit `Cd` and scheme
`Sc`, run as one window: the circuit's rounds in order, each round's noise coin live and released after its operands
are committed (one coin a round, decision 122), then the service's draw of `K_d` of the `N_p ≥ units` grid-padded
tiles, one a proof unit (`drawOf`, a uniform `K_d`-subset), with the proofs' coins `Ξ`, each drawn tile proved with its
noise's generation. Under:
* the logical ranges (`Scope`: every round holding a credited tile live, `Declared.Live`), and the service's commit
  order (`LayerCommitted`: each value the prover writes committed under a root before its IR cut's coin; on the live
  path each round's values and run inputs under its root before its coin, which today's code doesn't meet);
* every tile at least `ρ` times its detected weight's share of the `N_p` (`shareCone`: its late cone when it carries
  words, decision 119; `ρ = 1/(F·r)` at the padding's `r = (N_p/N)·(w_max/w̄)` and `F` the most tiles a late cone
  holds, per kind, `kindF`: `1` for every kind served), every round's detected weight at most `w_max`, and every tile
  whose cone carries work lucky at most `p` folded (`tileLuck`, `μ + (1 − μ)·r_B`) and `p0` whole (`tileLuck0`, the
  guess bound);
* costs `a, b ≥ 0`, `Λ > 0` and `SplitRate` at them;
* the play's ideal game the service's at share `1` per unit of each skipped tile's late cone (`tileCone`, tiles of no
  words included), and sampled (I8)'s pieces (decisions 114 and 119) on the window's proofs, the IR's layer
  (`Declared.layer` at the run's `pr`: positions the IR's gates, its slots, writers, cuts, coins and draw defined from
  the circuit): `Of`, binding at `δ_bind` in the round (`MovedIn`; WindowEps: `Σ_c δlink_c`, which needs
  `LayerCommitted`'s order), proof soundness jointly over the draws at `δ_ks` (`JointSound`; WindowEps:
  `max_c κ_c + coll`) and `Pins`;
* the four claims: (H-μ) (`OperandHMu` at its kind's `μ`) and B′ (`ReceiptLuckAt` at its kind's `r_B`) for every
  credited tile folded at its stage, the guess bound (`GuessLuckAt` at its kind's guess) for every one whole, and the
  fixed-words claim, #2466's `UnitFixed` with its fold clause, on the IR's layer at the run's modes (`tileMode`) and
  the tiles' words;

an accepted window's run has all but a share `d_F + d₀` of its committed words fixed right and all but `d₀` folded,
with `a·d_F + b·d₀ < Λ·ln(1/δ_s) + min(a, b)·D₀` (`WorkAt` at `rOf`; `D₀` the cone weight on tiles of no words,
`zeroCone`, `0` for every Program COMPILE accepts), except with probability `δ_s + δ_bind + δ_ks`. Nothing here reads
a machine model or a hash: equivocation on a committed position is a binding failure, in `δ_bind`. The work is the
prover system's: no premise ties it to a device. Nothing yet builds a `Declared` from a compiled Program, and
`Properties.Compose.UnitsOfLayout`, which would tie its units to the service's layout, is unproved: until
both exist, this speaks of the declared circuit, not of a served workload. -/
def CircuitWork : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (_hN : Cd.units ≤ Np) (hr : 0 < Cd.rounds)
    (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.rounds)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs → LayerCommitted Cd →
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

/-- **(L) The circuit's work, seeded coins** (decision 113's logical layer under decision 122's one seed a call). For
every declared circuit `Cd` and scheme `Sc`, run as one window whose rounds are seed groups: each round one call site,
its coin the oracle's answers to the cheater's `T_q` queries for that call's seed (`Fin T_q → B`, uniform), then the
service's live draw of `K_d ≥ 1` of the `N_p` tiles (`drawOf`) with the proofs' coins `Ξ`. Under:
* the seeded ranges (`ScopeSeeded`: every round holding a credited tile seeded, one call site a round, and no credited
  kind GRIND hasn't priced, so H1 + fx1's attention uncredited), the service's commit order (`LayerCommitted`, whose
  seeded reading is Fiat–Shamir over the roots its seed preimages name), and `δ_g > 0`;
* every tile at least `ρ` times its detected weight's share, costs `a > 0`, `b ≥ 0` and `κ₀ ≥ 0`;
* the same play and the same proofs' layer as `CircuitWork` (`Declared.layer` at the run's `pr`), with `Of`,
  `MovedIn` at `δ_bind`, `JointSound` at `δ_ks`, `Pins` and `UnitFixed`;
* GRIND's premise per stage (`SeedLuckAt`): folded tiles at the template's `κ = 4` (`TemplateMoment`'s point) and
  whole tiles at `κ₀`, both at the seed groups' rates from the IR (`seedTheta`), each read through the random-oracle
  premise (`SeedExtractedAt`: the proofs' seed is a query's answer, off `ε_id`) and off a binding event of
  probability `δ_b`, a per-prover term as `δ_bind` is;

an accepted window's run has all but a share `d_F + d₀` of its committed words fixed right and all but `d₀` folded,
with `a·d_F + b·d₀ < max(5a, (1 + κ₀)·b)·ln(1/δ_s)/(ρ·K_d) + (a + b)·Π + min(a, b)·D₀` (`WorkAt` at `rOfSeeded`, `Π`
the seed groups' premiums `π* = ln(T_q/δ_g)/θ` from the IR, `premiumOf`), except with probability
`δ_s + δ_bind + δ_ks + 2·rounds·(δ_g + δ_b + ε_id)`. `a > 0` makes the bound positive, which `WorkAt`'s strict `<`
needs.
Nothing yet builds a `Declared` from a compiled Program, and `Properties.Compose.UnitsOfLayout`, which would
tie its units to the service's layout, is unproved: until both exist, this speaks of the declared circuit, not of a
served workload. -/
def CircuitWorkSeeded : Prop :=
  ∀ {ι K T L U B Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype B] [Fintype Ξ]
    [Nonempty V] {Tq : ℕ} (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (W : SplitStages (Fin Tq → B) (Fin Np) Cd.rounds)
    (Pl : Play (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun (Fin Tq → B) (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ)
    (ρ a b κ0 δs δg δb εid δbind δks : ℝ),
    ScopeSeeded Sc Cd δs → LayerCommitted Cd → 0 < δg → 0 < Kd →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) → 0 < a → 0 ≤ b → 0 ≤ κ0 →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.layer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.layer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.layer hr hK pr).JointSound δks → (Cd.layer hr hK pr).Pins →
    SeedLuckAt (tileStage Cd Np) W (shareCone Sc Cd Np) .folded 4 (seedTheta Sc Cd) εid δb →
    SeedLuckAt (tileStage Cd Np) W (shareCone Sc Cd Np) .whole κ0 (seedTheta Sc Cd) εid δb →
    UnitFixed Pl (Cd.layer hr hK pr) (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (rOfSeeded Sc Cd Np Kd Tq a b κ0 ρ δs δg) (Pl.run x).fixed (folded x)) ≤
      δs + (δbind + δks) + 2 * (Cd.rounds : ℝ) * (δg + δb + εid)

/-! ## The physical layer's step -/

/-- **(P) Credit of the work, on a data sheet** (decision 113's physical layer: the tensor–scalar model and its
vendor axioms, attribution's unit grade). For a declared circuit `Cd` and scheme `Sc` whose model `Z` is the circuit's
on sheet `Sh` (`Fits`, `ModelScope`), every run `A` that meets the data sheet's A1–A7, FOLD-OPT per word folded at
its `folded` words and FILL-OPT, and has the work `WorkAt` states at the circuit's costs `aOf`, `bOf` and bound `r`,
takes at least `(1 − ε)·T_cred`, `ε = ε_♭ + ε_μ + r` (`epsOf`) and `T_cred = B(♭)/f_max` at the circuit's floor
`♭ = Σ_words ♭_{kind}`. It reads no coin: it is `SheetDIdealFloor`'s chain on one run, with the fold priced at the
words folded. -/
def CreditOfWork : Prop :=
  ∀ {ι K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X]
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X)
    (A : Anatomy K T L U) (folded : ℝ) (lam : SheetRow P → ℝ) (r : ℝ),
    Fits Sc Cd Sh Z → ModelScope Sc Cd Z lam → A.Split →
    SheetClock Sh A.run → SheetDispatch A.run → SheetTensor Sh A.run → SheetPipe Sh A.run → SheetSmem Sh A.run →
    SheetWords Z A → SheetSemantics Z A → FoldOptFolded Z A folded → FillOpt Z A →
    WorkAt (wordsOf Sc Cd : ℝ) (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) r A.fixed folded →
    (1 - epsOf Sc Cd Z lam r) * creditOf Sc Cd Z ≤ A.run.time

/-! ## The composed credit -/

/-- **The credit theorem, live coins**: `CreditOfWork ∘ CircuitWork` at the circuit's costs `a = aOf`, `b = bOf` (both
`≥ 0` under `ModelScope`). Under `CircuitWork`'s premises at those costs and `CreditOfWork`'s on every run, the
window's certificate in device window `win` holds (`certOf` at `.live`): it is accepted with time below
`(1 − ε)·T_cred` with probability at most `δ_s + δ_bind + δ_ks`, where `T_cred = B(♭)/f_max` at the circuit's floor and
`ε = ε_♭ + ε_μ + Λ·ln(1/δ_s) + min(a, b)·D₀` (`rOf`). The certificate also reports the words per kind, `γ_♭` (W1) and
the coin mode, `live` under `Scope`'s `Declared.Live`: a seeded window's credit is `CircuitCreditSeeded`'s. It takes
the service's commit order (`LayerCommitted`), which today's code doesn't meet. Work is the prover system's, and the
certificate is what the physical layer's ledger consumes: what device time the work took, and whose, is the ledger's,
which credits one device's time once (decisions 111 and 113). Nothing yet builds a `Declared` from a compiled Program,
and `Properties.Compose.UnitsOfLayout`, which would tie its units to the service's layout, is unproved:
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
    Scope Sc Cd δs → LayerCommitted Cd →
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
    (∀ x, FoldOptFolded Z (Pl.run x) (folded x)) → (∀ x, FillOpt Z (Pl.run x)) →
    (certOf Sc Cd Z H lam (rOf Sc Cd Np (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) Λ δs) (δs + (δbind + δks)) win
      CoinMode.live).Holds Pl

/-- **The credit theorem, seeded coins**: `CreditOfWork ∘ CircuitWorkSeeded` at the circuit's costs `a = aOf > 0`,
`b = bOf` (`≥ 0` under `ModelScope`). Under `CircuitWorkSeeded`'s premises at those costs and `CreditOfWork`'s on
every run, the window's certificate in device window `win` holds at `.seeded` (`certOf`): it is accepted with time
below `(1 − ε)·T_cred` with probability at most `δ_s + δ_bind + δ_ks + 2·rounds·(δ_g + δ_b + ε_id)`, where
`ε = ε_♭ + ε_μ + max(5a, (1 + κ₀)·b)·ln(1/δ_s)/(ρ·K_d) + (a + b)·Π + min(a, b)·D₀` (`rOfSeeded`, `Π` the seed groups'
premiums from the IR). It is the credit of every window PoUW's code runs, whose noise it derives from the salt, once
the code's seed preimages name the roots `LayerCommitted`'s seeded reading needs. Nothing yet builds a `Declared` from
a compiled Program, and `Properties.Compose.UnitsOfLayout`, which would tie its units to the service's
layout, is unproved: until both exist, this speaks of the declared circuit, not of a served workload. -/
def CircuitCreditSeeded : Prop :=
  ∀ {ι K P T L U F X B Ξ V : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] [Fintype B] [Fintype Ξ] [Nonempty V] {Tq : ℕ}
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (H : Honest K)
    {Np Kd : ℕ} (_hN : Cd.units ≤ Np) (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (W : SplitStages (Fin Tq → B) (Fin Np) Cd.rounds)
    (Pl : Play (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun (Fin Tq → B) (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (lam : SheetRow P → ℝ)
    (ρ κ0 δs δg δb εid δbind δks : ℝ) (win : Accounting.Window),
    ScopeSeeded Sc Cd δs → LayerCommitted Cd → 0 < δg → 0 < Kd →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) → 0 < aOf Sc Cd Z lam → 0 ≤ κ0 →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd Np) 1 1 Kd →
    (Cd.layer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.layer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.layer hr hK pr).JointSound δks → (Cd.layer hr hK pr).Pins →
    SeedLuckAt (tileStage Cd Np) W (shareCone Sc Cd Np) .folded 4 (seedTheta Sc Cd) εid δb →
    SeedLuckAt (tileStage Cd Np) W (shareCone Sc Cd Np) .whole κ0 (seedTheta Sc Cd) εid δb →
    UnitFixed Pl (Cd.layer hr hK pr) (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    Fits Sc Cd Sh Z → ModelScope Sc Cd Z lam → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOptFolded Z (Pl.run x) (folded x)) → (∀ x, FillOpt Z (Pl.run x)) →
    (certOf Sc Cd Z H lam (rOfSeeded Sc Cd Np Kd Tq (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) κ0 ρ δs δg)
      (δs + (δbind + δks) + 2 * (Cd.rounds : ℝ) * (δg + δb + εid)) win .seeded).Holds Pl

end Accounting.Guarantees.Compose
