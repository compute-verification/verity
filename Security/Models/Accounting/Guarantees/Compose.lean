import Models.Accounting.Guarantees.Sheet
import Models.Accounting.Assumptions.Compose
import Models.Pouw.Assumptions.FiatShamir

/-!
# PoUW's credit on an arbitrary declared circuit (trusted)

THEOREM's composed statement, round 3 (note:compute-accounting/20261018T1230Z-draft-theorem-compose-r3), restated after
RT-COMPOSE r2 (note:compute-accounting/20261018T1200Z-draft-rt-compose-r2) under decisions 107, 109–114, 117, 119 and
122. Each `def G : Prop` is to be proved as `Accounting.SecurityProofs.Compose.G` in `Proofs.Accounting.Compose`, where
each is a `sorry` stub for now. The circuit is any `Declared` one, a `Gates.Dag` with its call sites, its rounds and its
proof units, its words, kinds, cuts and late cones read off the IR. Nothing about its form is a hypothesis beyond
`Declared.Wf` (call sites, rounds and units in range; one unit's gates in one round and its words of one kind).

Decision 113 splits the credit in two, and `CircuitCredit` is their composition:

* (L) `CircuitWork`, the logical layer: an accepted window's run has all but a share of its committed words fixed
  right, at costs `a, b` below `Λ·ln(1/δ_s) + max(a, b)·D₀` (`WorkAt`, `rOf`). It rests on the four claims the
  headlines (#2482) rest on: (H-μ) (`OperandHMu`) and B′ (`ReceiptLuckAt`) per tile folded, the whole-unit guess bound
  (`GuessLuckAt`) per tile whole, and the fixed-words claim in share form (`Assumptions.Compose.RightFixed`), with the
  templates' proved luck (`Scheme.LuckProved`), the service's draw and sampled (I8)'s pieces as #2466 states them
  (`DrawnLayer.Of`, binding per committed position at its IR cut, `StageCuts`, each tile's noise at its round's coin,
  `StageCoins`, joint proof soundness, `Pins`). No machine model. Its crypto half is #2466's `DrawnRightOff`, one event
  of probability `δ_bind + δ_ks` off which every drawn tile is right (RT-I8's `compose_sampled` shape).
* (P) `CreditOfWork`, the physical layer's step: on the tensor–scalar model of a data sheet, a run with that work takes
  at least `(1 − ε)·T_cred`. Its premises are exactly the machine model's: the model the circuit's on the sheet
  (`Fits`, `ModelScope`), the data sheet's A1–A7, FOLD-OPT per word folded (`FoldOptFolded`) and FILL-OPT.

The other guarantees are (L)'s lemmas:

* `ServiceBudget`: the window's budget `RoundGame.BudgetAt Λ` for the service's one draw, from per-stage luck and the
  stage-share bound (W3), with folded and whole shares priced apart (B1);
* `SplitRateClosed`: `SplitRate` at `Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b(1 + Kρw_max)/(ρK(1 − p0)))` for whole units
  lucky at the guess bound `p0 ∈ [0, 1)`;
* `SplitRateMerged`: `SplitRate` at `Λ = b(1 + Kρw_max)/(ρK(1 − q))` for any luck at most `q`;

and `CircuitCreditFS` is the composed credit's Fiat–Shamir form, against proofs' `FiatShamirFloor` (decision 112).

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
folded luck `p ∈ [0, 1)`, whole luck `p0 ∈ [0, 1)` (the guess bound, `2^-64`) and costs `a, b > 0`, `SplitRate` holds at
`Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b(1 + Kρw_max)/(ρK(1 − p0)))`, the larger of the folded and the whole branch:
B1's `max(κ/ln(1/p), (σ_w + σ_f)/s)` for the service's one draw, with `1/(K(1 − p))` in place of `1/(K·ln(1/p))`. At
`p0 = 2^-64` the whole branch is `b(1 + Kρw_max)/(ρK)` to 20 digits; round 2's `b/(ρK)` at `p0 = 0` was for a luck a
guess exceeds (R2-B3). -/
def SplitRateClosed : Prop :=
  ∀ (K : ℕ) (ρ p p0 a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ p → p < 1 → 0 ≤ p0 → p0 < 1 →
    0 < a → 0 < b →
    SplitRate K ρ p p0 a b wmax
      (max (a * (1 + K * ρ * wmax) / (ρ * K * (1 - p))) (b * (1 + K * ρ * wmax) / (ρ * K * (1 - p0))))

/-- **The rate in closed form, merged.** For `K ≥ 1` draws, `ρ > 0`, `0 ≤ ρ·w_max ≤ 1`, luck `p, p0 ≤ q` with
`q ∈ [0, 1)` and costs `0 < a ≤ b`, `SplitRate` holds at `Λ = b(1 + Kρw_max)/(ρK(1 − q))`: every skip priced at the
whole cost and the larger luck. -/
def SplitRateMerged : Prop :=
  ∀ (K : ℕ) (ρ p p0 q a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ q → p ≤ q → p0 ≤ q → q < 1 →
    0 < a → a ≤ b → SplitRate K ρ p p0 a b wmax (b * (1 + K * ρ * wmax) / (ρ * K * (1 - q)))

/-- **(L) The circuit's work, live coins** (decision 113's logical layer). For every declared circuit `Cd` and scheme
`Sc`, run as one window: the circuit's rounds in order, each round's noise coin released after its operands are
committed (one coin a round, decision 122), then the service's draw of `K_d` of the `N_p ≥ units` grid-padded tiles,
one a proof unit (`drawOf`, a uniform `K_d`-subset), with the proofs' coins `Ξ`, each drawn tile proved with its
noise's generation. Under:
* the logical ranges (`Scope`);
* every tile at least `ρ` times its share weight's share of the `N_p` (`shareCone`: its late cone, decision 119;
  `ρ = 1/(F·r)` at the padding's `r = (N_p/N)·(w_max/w̄)` and `F` the most tiles a late cone holds, per kind, `kindF`:
  `1` for whole coordinates, `K/2048` for K-chunks of 2048), every round's cones at most `w_max`, and every tile whose
  cone carries work lucky at most `p` folded (`tileLuck`, `μ + (1 − μ)·r_B`) and `p0` whole (`tileLuck0`, the guess
  bound);
* costs `a, b ≥ 0`, `Λ > 0` and `SplitRate` at them;
* the play's ideal game the service's at share `1` per unit of a skipped tile's cone, its draws the service's, and
  sampled (I8)'s pieces (decisions 114 and 119): the window's proofs with values per committed position, the IR's
  gates (`DrawnLayer` over the rounds' coins and the draw), each position bound at its IR cut (`StageCuts`), each
  tile's noise at its round's coin (`StageCoins`), `Of`, binding at `δ_bind` in the round (`MovedIn`; WindowEps:
  `Σ_c δlink_c`), proof soundness jointly over the draws at `δ_ks` (`JointSound`; WindowEps: `max_c κ_c + coll`) and
  `Pins`;
* the four claims: (H-μ) (`OperandHMu` at its kind's `μ`) and B′ (`ReceiptLuckAt` at its kind's `r_B`) for every
  credited tile folded at its stage, the guess bound (`GuessLuckAt` at its kind's guess) for every one whole, and the
  fixed-words claim in share form with its fold clause (`Assumptions.Compose.RightFixed`);

an accepted window's run has all but a share `d_F + d₀` of its committed words fixed right and all but `d₀` folded,
with `a·d_F + b·d₀ < Λ·ln(1/δ_s) + max(a, b)·D₀` (`WorkAt` at `rOf`; `D₀` the cone weight on tiles of no words,
`zeroCone`), except with probability `δ_s + δ_bind + δ_ks`. Nothing here reads a machine model or a hash:
equivocation on a committed position is a binding failure, in `δ_bind`. The work is the prover system's: no premise
ties it to a device. -/
def CircuitWork : Prop :=
  ∀ {ι K T L U C Ξ Q V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (_hN : Cd.units ≤ Np) (hK : Kd ≤ Np)
    (W : SplitStages C (Fin Np) Cd.rounds) (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pf : DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) 1 Cd.rounds Kd (Fin Np) Q ℕ V)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 ≤ a → 0 ≤ b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (shareCone Sc Cd Np) 1 1 Kd →
    pf.draw = drawOf hK → StageCuts Cd pf → StageCoins Cd pf →
    pf.Of Pl → (∀ w, prob (fun x => pf.MovedIn x w) ≤ δbind) → pf.JointSound δks → pf.Pins →
    PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G) →
    PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) →
    PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) →
    Accounting.Assumptions.Compose.RightFixed Sc Cd Pl W folded →
    prob (fun x => Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (rOf Sc Cd Np a b Λ δs) (Pl.run x).fixed (folded x)) ≤ δs + (δbind + δks)

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
window's certificate in device window `win` holds (`certOf`): it is accepted with time below `(1 − ε)·T_cred` with
probability at most `δ_s + δ_bind + δ_ks`, where `T_cred = B(♭)/f_max` at the circuit's floor and
`ε = ε_♭ + ε_μ + Λ·ln(1/δ_s) + max(a, b)·D₀` (`rOf`). The certificate also reports the words per kind and `γ_♭` (W1).
Work is the prover system's, and the certificate is what the physical layer's ledger consumes: what device time the
work took, and whose, is the ledger's, which credits one device's time once (decisions 111 and 113). -/
def CircuitCredit : Prop :=
  ∀ {ι K P T L U F X C Ξ Q V : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] [Fintype C] [Fintype Ξ] [Nonempty V]
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (H : Honest K)
    {Np Kd : ℕ} (_hN : Cd.units ≤ Np) (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.rounds)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pf : DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) 1 Cd.rounds Kd (Fin Np) Q ℕ V)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (lam : SheetRow P → ℝ)
    (ρ p p0 wmax Λ δs δbind δks : ℝ) (win : Accounting.Window),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 < Λ → SplitRate Kd ρ p p0 (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) wmax Λ →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (shareCone Sc Cd Np) 1 1 Kd →
    pf.draw = drawOf hK → StageCuts Cd pf → StageCoins Cd pf →
    pf.Of Pl → (∀ w, prob (fun x => pf.MovedIn x w) ≤ δbind) → pf.JointSound δks → pf.Pins →
    PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G) →
    PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) →
    PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) →
    Accounting.Assumptions.Compose.RightFixed Sc Cd Pl W folded →
    Fits Sc Cd Sh Z → ModelScope Sc Cd Z lam → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOptFolded Z (Pl.run x) (folded x)) → (∀ x, FillOpt Z (Pl.run x)) →
    (certOf Sc Cd Z H lam (rOf Sc Cd Np (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) Λ δs) (δs + (δbind + δks)) win).Holds Pl

/-- **The credit theorem's Fiat–Shamir form** (decision 112), against proofs' `FiatShamirFloor`. For a service whose
stage draws are a uniform oracle's answers at the inputs its commitment names (SHA-512 at the window's salt, read as a
random oracle: the statement's probability model, not a premise), an analysis's steer events `Steer`, `T_q` oracle
calls and a saving `w ≥ 0` of the credit: if a try steers with probability at most `μ^m`, `0 ≤ μ ≤ 1`, with `m ≥ h·w`
(`h ≥ 0` bits of matched noise per unit of saved work, which the GRIND workstream proves per template), and the live
coins alone bound accepted runs charged below `(1 − ε − w)·T_cred` at unsteered draws by `ε_live`, `ε` at the logical
layer's bound `r`, then every adversary within `T_q` calls has its accepted runs charged below `(1 − ε − w)·T_cred`
with probability at most `ε_live + T_q·μ^{h·w}`. At `μ = ½` that term is at most `δ_FS` once `w ≥ log₂(T_q/δ_FS)/h`,
the premium GRIND prices a seed. Steering is the logical layer's (a seed's coins); it is stated at the composed charge
because `FiatShamirFloor` is. The open link is `LiveFloorUnsteered` for the compiled circuit, from `CircuitCredit`: it
needs a map from the service's adversaries to the circuit's plays, as `RunMapFS` is for the staged service. -/
def CircuitCreditFS : Prop :=
  ∀ {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) (r : ℝ)
    (J Nd Live : Type) [Fintype J] [Fintype Nd] [Fintype Live] (Mc : Pouw.RelativeSecurity.Machine)
    (S : FSService J Nd Live) (Tq : ℕ) (Steer : SteerMap Mc S) (εlive μ h w : ℝ) (m : ℕ),
    0 ≤ εlive → 0 ≤ μ → μ ≤ 1 → 0 ≤ h → 0 ≤ w → h * w ≤ m → Pouw.Assumptions.SteerBound Mc S Tq Steer μ m →
    LiveFloorUnsteered Mc S Tq Steer ((1 - epsOf Sc Cd Z lam r - w) * creditOf Sc Cd Z) εlive →
    ServiceFloorFS Mc S Tq ((1 - epsOf Sc Cd Z lam r - w) * creditOf Sc Cd Z) (εlive + Tq * μ ^ (h * w))

end Accounting.Guarantees.Compose
