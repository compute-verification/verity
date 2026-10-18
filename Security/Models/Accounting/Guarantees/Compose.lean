import Models.Accounting.Guarantees.Sheet
import Models.Accounting.Assumptions.Compose
import Models.Pouw.Assumptions.FiatShamir

/-!
# PoUW's credit on an arbitrary declared circuit (trusted)

THEOREM's composed statement, round 2 (note:compute-accounting/20261018T0618Z-draft-theorem-compose-r2), restated after
RT-COMPOSE (note:compute-accounting/20261018T0501Z-draft-rt-compose) and RT-I8
(note:compute-accounting/20261018T0728Z-draft-rt-i8) under decisions 107, 109–114, 117 and 119. Each
`def G : Prop` is to be proved as `Accounting.SecurityProofs.Compose.G` in `Proofs.Accounting.Compose`, where each is a
`sorry` stub for now. The circuit is any `Declared` one, a `Gates.Dag` with its call sites, its units and their kinds
read off the IR. Nothing about its form is a hypothesis beyond `Declared.Wf` (every gate's call site is one of its
calls).

Decision 113 splits the credit in two, and `CircuitCredit` is their composition:

* (L) `CircuitWork`, the logical layer: an accepted window's run has the noised products' committed words right, all
  but a share `Λ·ln(1/δ_s)` of the work at costs `a, b` (`WorkAt`), for values fixed before their seeds. Its premises
  are the templates' luck, (H-μ) per template (`HMu`, with `Scheme.LuckProved`), the service's draw, sampled (I8)'s
  pieces as #2466 states them after RT-I8 and decision 119 (`DrawnLayer.Of`, binding per committed position at the
  coin after its call site, `StageCuts`, each tile's noise at its call site's coin, `StageCoins`, `WriterReads`, joint
  proof soundness, `Pins`), the modes the run's (`PaidSound`: a paid tile is held and right, and the share charges a
  skipped tile its late cone, so it covers every tile that isn't good), and E1(c) per tile for the fixed words
  (`UnitFixed`) and the fold (`UnitFolded`). No machine model. Its crypto half is #2466's `DrawnRightOff`, one event of probability `δ_bind + δ_ks` off which every drawn
  tile is right (RT-I8's `compose_sampled` shape).
* (P) `CreditOfWork`, the physical layer's step: on the tensor–scalar model of a data sheet, a run with that work takes
  at least `(1 − ε)·T_cred`. Its premises are exactly the machine model's: the model the circuit's on the sheet
  (`Fits`, `ModelScope`), the data sheet's A1–A7, FOLD-OPT per word folded (`FoldOptFolded`) and FILL-OPT.

The other guarantees are (L)'s lemmas:

* `ServiceBudget`: the window's budget `RoundGame.BudgetAt Λ` for the service's one draw, from per-stage luck and the
  stage-share bound (W3), with folded and whole shares priced apart (B1);
* `SplitRateClosed`: `SplitRate` at `Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b/(ρK))` when whole units are luckless;
* `SplitRateMerged`: `SplitRate` at `Λ = b(1 + Kρw_max)/(ρK(1 − q))` for any luck at most `q`;
* `ServiceCovers`: the service's share, a skipped tile's late cone, covers every tile that isn't good once a paid tile
  is held and right (#2466's `DrawnLayer.Covers`, proved for this game from `DrawnLayer.covers_of_cone`);

and `CircuitCreditFS` is the composed credit's Fiat–Shamir form, against proofs' `FiatShamirFloor` (decision 112).

`Pouw.Guarantees.EndToEnd` is not subsumed and stays: it is the TT cost model with Fiat–Shamir in the random-oracle
model, concluding `(1 − ε_s)^t + ε(q, N)`, where `CircuitCredit` is the TS model under live coins (round 1 said
otherwise; RT-COMPOSE §6).
-/

namespace Accounting.Guarantees.Compose

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-! ## The logical layer -/

/-- **The service's window budget.** For a window of `M` call sites' stages over `N` tiles with share weights `w ≥ 0`
(the tiles' cones in `CircuitWork`), each tile at least `ρ` times its share of the `N` (`ρ·N·w_v ≤ 1`; at grid padding
`ρ = 1/(F·r)`, `r = (N/N_real)·(w_max/w̄)`), every stage's share at most `w_max`, per-stage luck `lk` folded and `lk0` whole
(`SplitStages.LuckAt`), at most `p` and `p0` on every tile that carries work, and costs `a, b ≥ 0` per unit of folded
and whole share: if `SplitRate K ρ p p0 a b w_max Λ`, the window's one round, its draw a uniform `K`-subset
(`serviceGame`), has its saving per unit of detection bounded by `Λ`. Stages are independent given the past, so the
cheater's best is to concentrate its share in one stage, and `Λ` falls as `1/K` until the stage share binds (W3). -/
def ServiceBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M)
    (w lk lk0 : Fin N → ℝ) (K : ℕ) (ρ p p0 a b wmax Λ : ℝ),
    (∀ v, 0 ≤ w v) → 0 < ρ → (∀ v, ρ * N * w v ≤ 1) → (∀ j, stageShare stg w j ≤ wmax) →
    (∀ v, 0 < w v → lk v ≤ p ∧ lk0 v ≤ p0) → W.LuckAt lk lk0 stg → 0 ≤ a → 0 ≤ b →
    SplitRate K ρ p p0 a b wmax Λ → RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W w a b K)

/-- **The rate in closed form, whole units luckless.** For `K ≥ 1` draws, `ρ > 0`, `0 ≤ ρ·w_max ≤ 1`, folded luck
`p ∈ [0, 1)`, whole luck `0`, `a > 0` and `b ≥ 0`, `SplitRate` holds at
`Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b/(ρK))`, the larger of the folded and the whole branch: B1's
`max(κ/ln(1/p), (σ_w + σ_f)/s)` for the service's one draw, with `1/(K(1 − p))` in place of `1/(K·ln(1/p))`. -/
def SplitRateClosed : Prop :=
  ∀ (K : ℕ) (ρ p a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ p → p < 1 → 0 < a → 0 ≤ b →
    SplitRate K ρ p 0 a b wmax
      (max (a * (1 + K * ρ * wmax) / (ρ * K * (1 - p))) (b / (ρ * K)))

/-- **The rate in closed form, merged.** For `K ≥ 1` draws, `ρ > 0`, `0 ≤ ρ·w_max ≤ 1`, luck `p, p0 ≤ q` with
`q ∈ [0, 1)` and costs `0 < a ≤ b`, `SplitRate` holds at `Λ = b(1 + Kρw_max)/(ρK(1 − q))`: every skip priced at the
whole cost and the larger luck, which needs no ruling on whole units' luck. -/
def SplitRateMerged : Prop :=
  ∀ (K : ℕ) (ρ p p0 q a b wmax : ℝ), 1 ≤ K → 0 < ρ → 0 ≤ wmax → ρ * wmax ≤ 1 → 0 ≤ q → p ≤ q → p0 ≤ q → q < 1 →
    0 < a → a ≤ b → SplitRate K ρ p p0 a b wmax (b * (1 + K * ρ * wmax) / (ρ * K * (1 - q)))

/-- **The service's share covers every tile that isn't good** (decision 119). For the window as the service's one round
at share `1` per unit of a skipped tile's late cone (`serviceGame … (tileCone …) 1 1`) and its proofs (`DrawnLayer`,
one round of the call sites' coins and the draw), if a paid tile is held and right (`PaidSound`), then on every
accepted path the words of the tiles that aren't good are at most the game's deficiency times `N_w`
(`DrawnLayer.Covers`): a tile that isn't good has a tile of its late cone that isn't held and right, which is folded or
whole, and whose late cone holds it (`DrawnLayer.covers_of_cone`). -/
def ServiceCovers : Prop :=
  ∀ {ι K T L U C Ξ Q Pos V : Type} [Fintype C] [Fintype Ξ] [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ}
    (W : SplitStages C (Fin Np) Cd.calls) (Pl : Play (Paths C Cd.calls × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pf : DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) 1 Cd.calls Kd (Fin Np) Q Pos V),
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd pf) 1 1 Kd →
    PaidSound (tileStage Cd Np) W pf → pf.Covers Pl (fun _ => tileWords Sc Cd Np) (wordsOf Sc Cd : ℝ)

/-- **(L) The circuit's work, live coins** (decision 113's logical layer). For every declared circuit `Cd` and scheme
`Sc`, run as one window: the circuit's call sites in order, each's noise coin released after its operands are
committed, then the service's draw of `K_d` of the `N_p ≥ N` grid-padded tiles (`drawOf`, a uniform `K_d`-subset) with
the proofs' coins `Ξ`, each drawn tile proved with its noise's generation. Under:
* the logical ranges (`Scope`);
* every tile at least `ρ` times its late cone's share of the `N_p` (`tileCone`, decision 119: `ρ = 1/(F·r)` at the
  padding's `r = (N_p/N)·(w_max/w̄)`, the coordinator's ruling of 18 Oct 05:52Z, and `F` tiles a late chain within a
  call site), every call site's cones at most `w_max`, and every tile whose cone carries work lucky at most `p` folded
  and `p0` whole;
* costs `a, b ≥ 0`, `Λ > 0` and `SplitRate` at them;
* the play's ideal game the service's at share `1` per unit of a skipped tile's late cone, its draws the service's, and
  sampled (I8)'s pieces after RT-I8 (decisions 114 and 119): the window's proofs with values per
  committed position (`DrawnLayer` over the call sites' `M` coins and the draw), each position bound before the coin
  after the call site that wrote it (`StageCuts`), each tile's noise at its call site's coin (`StageCoins`), each
  position among its writer's slots (`WriterReads`), `Of`, binding at `δ_bind` in the round (`MovedIn`; WindowEps:
  `Σ_c δlink_c`), proof soundness jointly over the draws at `δ_ks` (`JointSound`; WindowEps: `max_c κ_c + coll`) and
  `Pins`;
* the modes the run's (`PaidSound`), (H-μ) (`HMu`), and E1(c) per tile for the fixed words (`UnitFixed`, at the tiles'
  words) and the fold (`UnitFolded`);

an accepted window's run has all but a share `d_F + d₀` of its committed words fixed right and all but `d₀` folded,
with `a·d_F + b·d₀ < Λ·ln(1/δ_s)` (`WorkAt`), except with probability `δ_s + δ_bind + δ_ks`. Nothing here reads a
machine model or a hash: equivocation on a committed position is a binding failure, in `δ_bind`. The work is the
prover system's: no premise ties it to a device. -/
def CircuitWork : Prop :=
  ∀ {ι K T L U C Ξ Q Pos V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ} (_hN : Cd.N ≤ Np) (hK : Kd ≤ Np)
    (W : SplitStages C (Fin Np) Cd.calls) (Pl : Play (Paths C Cd.calls × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pf : DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) 1 Cd.calls Kd (Fin Np) Q Pos V)
    (folded : Paths (Paths C Cd.calls × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t, ρ * Np * tileCone Sc Cd pf t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (tileCone Sc Cd pf) j ≤ wmax) →
    (∀ t, 0 < tileCone Sc Cd pf t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 ≤ a → 0 ≤ b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd pf) 1 1 Kd →
    pf.draw = drawOf hK → StageCuts (tileStage Cd Np) pf → StageCoins (tileStage Cd Np) pf → pf.WriterReads →
    pf.Of Pl → (∀ w, prob (fun x => pf.MovedIn x w) ≤ δbind) → pf.JointSound δks → pf.Pins →
    PaidSound (tileStage Cd Np) W pf → HMu Sc Cd W → UnitFixed Pl pf (fun _ => tileWords Sc Cd Np) →
    UnitFolded Sc Cd Pl W folded →
    prob (fun x => Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (Λ * Real.log (1 / δs)) (Pl.run x).fixed (folded x)) ≤ δs + (δbind + δks)

/-! ## The physical layer's step -/

/-- **(P) Credit of the work, on a data sheet** (decision 113's physical layer: the tensor–scalar model and its
vendor axioms, attribution's unit grade). For a declared circuit `Cd` and scheme `Sc` whose model `Z` is the circuit's
on sheet `Sh` (`Fits`, `ModelScope`), every run `A` that meets the data sheet's A1–A7, FOLD-OPT per word folded at
its `folded` words and FILL-OPT, and has the work `WorkAt` states at the circuit's costs `aOf`, `bOf` and bound `r`,
takes at least `(1 − ε)·T_cred`, `ε = ε_♭ + ε_μ + r` (`epsOf`) and `T_cred = B(♭)/f_max` at the circuit's floor
`♭ = Σ_u ♭_{kind u}`. It reads no coin: it is `SheetDIdealFloor`'s chain on one run, with the fold priced at the words
folded. -/
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
`ε = ε_♭ + ε_μ + Λ·ln(1/δ_s)`. The certificate also reports the work per kind and `γ_♭` (W1). Work is the prover
system's, and the certificate is what the physical layer's ledger consumes: what device time the work took, and whose,
is the ledger's, which credits one device's time once (decisions 111 and 113). -/
def CircuitCredit : Prop :=
  ∀ {ι K P T L U F X C Ξ Q Pos V : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] [Fintype C] [Fintype Ξ] [Nonempty V]
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (H : Honest K)
    {Np Kd : ℕ} (_hN : Cd.N ≤ Np) (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.calls)
    (Pl : Play (Paths C Cd.calls × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pf : DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) 1 Cd.calls Kd (Fin Np) Q Pos V)
    (folded : Paths (Paths C Cd.calls × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (lam : SheetRow P → ℝ)
    (ρ p p0 wmax Λ δs δbind δks : ℝ) (win : Accounting.Window),
    Scope Sc Cd δs →
    0 < ρ → (∀ t, ρ * Np * tileCone Sc Cd pf t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (tileCone Sc Cd pf) j ≤ wmax) →
    (∀ t, 0 < tileCone Sc Cd pf t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 < Λ → SplitRate Kd ρ p p0 (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) wmax Λ →
    Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd pf) 1 1 Kd →
    pf.draw = drawOf hK → StageCuts (tileStage Cd Np) pf → StageCoins (tileStage Cd Np) pf → pf.WriterReads →
    pf.Of Pl → (∀ w, prob (fun x => pf.MovedIn x w) ≤ δbind) → pf.JointSound δks → pf.Pins →
    PaidSound (tileStage Cd Np) W pf → HMu Sc Cd W → UnitFixed Pl pf (fun _ => tileWords Sc Cd Np) →
    UnitFolded Sc Cd Pl W folded →
    Fits Sc Cd Sh Z → ModelScope Sc Cd Z lam → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOptFolded Z (Pl.run x) (folded x)) → (∀ x, FillOpt Z (Pl.run x)) →
    (certOf Sc Cd Z H lam Λ δs (δs + (δbind + δks)) win).Holds Pl

/-- **The credit theorem's Fiat–Shamir form** (decision 112), against proofs' `FiatShamirFloor`. For a service whose
stage draws are a uniform oracle's answers at the inputs its commitment names (SHA-512 at the window's salt, read as a
random oracle: the statement's probability model, not a premise), an analysis's steer events `Steer`, `T_q` oracle
calls and a saving `w ≥ 0` of the credit: if a try steers with probability at most `μ^m`, `0 ≤ μ ≤ 1`, with `m ≥ h·w`
(`h ≥ 0` bits of matched noise per unit of saved work, which the GRIND workstream proves per template), and the live
coins alone bound accepted runs charged below `(1 − ε − w)·T_cred` at unsteered draws by `ε_live`, then every
adversary within `T_q` calls has its accepted runs charged below `(1 − ε − w)·T_cred` with probability at most
`ε_live + T_q·μ^{h·w}`. At `μ = ½` that term is at most `δ_FS` once `w ≥ log₂(T_q/δ_FS)/h`, the premium GRIND prices a
seed. Steering is the logical layer's (a seed's coins); it is stated at the composed charge because `FiatShamirFloor`
is. The open link is `LiveFloorUnsteered` for the compiled circuit, from `CircuitCredit`: it needs a map from the
service's adversaries to the circuit's plays, as `RunMapFS` is for the staged service. -/
def CircuitCreditFS : Prop :=
  ∀ {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) (Λ δs : ℝ)
    (J Nd Live : Type) [Fintype J] [Fintype Nd] [Fintype Live] (Mc : Pouw.RelativeSecurity.Machine)
    (S : FSService J Nd Live) (Tq : ℕ) (Steer : SteerMap Mc S) (εlive μ h w : ℝ) (m : ℕ),
    0 ≤ εlive → 0 ≤ μ → μ ≤ 1 → 0 ≤ h → 0 ≤ w → h * w ≤ m → Pouw.Assumptions.SteerBound Mc S Tq Steer μ m →
    LiveFloorUnsteered Mc S Tq Steer ((1 - epsOf Sc Cd Z lam (Λ * Real.log (1 / δs)) - w) * creditOf Sc Cd Z) εlive →
    ServiceFloorFS Mc S Tq ((1 - epsOf Sc Cd Z lam (Λ * Real.log (1 / δs)) - w) * creditOf Sc Cd Z)
      (εlive + Tq * μ ^ (h * w))

end Accounting.Guarantees.Compose
