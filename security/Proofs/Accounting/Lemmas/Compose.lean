module

public import Definitions.Accounting.Guarantees.Compose

/-!
# PoUW's credit on an arbitrary declared circuit: lemmas

The statements of `Definitions.Accounting.Guarantees.Compose` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.

* `ServiceBudget`: the window's budget `RoundGame.BudgetAt Λ` for the service's one sample, from per-stage luck and the
  stage-share bound (W3), with folded and whole shares priced apart (B1);
* `SplitRateClosed`: `SplitRate` at `Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b(1 + Kρw_max)/(ρK(1 − p0)))` for whole units
  lucky at the guess bound `p0 ∈ [0, 1)`;
* `SplitRateExp`: `SplitRateAt` at the sharp `Λ`, with its `ν, ν₀` in closed form;
* `SplitRateMerged`: `SplitRate` at `Λ = b(1 + Kρw_max)/(ρK(1 − q))` for any luck at most `q`;
* `ServiceCovers`: the service's share, each skipped tile's late cone from the IR (`tileCone`), covers every tile that
  isn't good at any joint run's modes on the IR's layer (`SampledLayer.Covers`, from `SampledLayer.covers_of_cone`);
* `BindingOfECR`: a prover's `δ_bind` from `SampledBindingOfFork`, AM–GM and `ecr/sha-512` on its fork finder,
  the one reading of binding's number PoUW's per-prover floors share;
* `WholeLuckOfEntropy`: `SeedLuckAt`'s whole half from `WholeEntropyAt` (the optional entropy premise), at any rate
  whose per-tile moment at that entropy is at most `1`.
-/

@[expose] public section

namespace Accounting.Guarantees.Compose

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The service's window budget.** For a window of `M` rounds' stages over `N` tiles with share weights `w ≥ 0` (the
tiles' cones in `CircuitWork`), each tile at least `ρ` times its share of the `N` (`ρ·N·w_v ≤ 1`; at grid padding
`ρ = 1/(F·r)`, `r = (N/N_real)·(w_max/w̄)`), every stage's share at most `w_max`, per-stage luck `lk` folded and `lk0`
whole on the tiles' checked values (`SplitStages.LuckAt`), at most `p` and `p0` on every tile that carries work, and
costs `a, b ≥ 0` per unit of folded and whole share: if `SplitRate K ρ p p0 a b w_max Λ`, the window's one round, its
sample a uniform `K`-subset (`serviceGame`), has its saving per unit of detection bounded by `Λ`. Stages are independent
given the past, so the cheater's best is to concentrate its share in one stage, and `Λ` falls as `1/K` until the stage
share binds (W3). -/
def ServiceBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M)
    (w lk lk0 : Fin N → ℝ) (K : ℕ) (ρ p p0 a b wmax Λ : ℝ),
    (∀ v, 0 ≤ w v) → 0 < ρ → (∀ v, ρ * N * w v ≤ 1) → (∀ j, stageShare stg w j ≤ wmax) →
    (∀ v, 0 < w v → lk v ≤ p ∧ lk0 v ≤ p0) → W.LuckAt lk lk0 stg → 0 ≤ a → 0 ≤ b →
    SplitRate K ρ p p0 a b wmax Λ → RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W w a b K)

/-- **The rate in closed form, whole units lucky at the guess bound.** For `K ≥ 1` samples, `ρ > 0`, `0 ≤ ρ·w_max ≤ 1`,
folded luck `p ∈ [0, 1)`, whole luck `p0 ∈ [0, 1)` (the guess bound, at any value) and costs `a, b > 0`, `SplitRate`
holds at `Λ = max(a(1 + Kρw_max)/(ρK(1 − p)), b(1 + Kρw_max)/(ρK(1 − p0)))`, the larger of the folded and the whole
branch: B1's `max(κ/ln(1/p), (σ_w + σ_f)/s)` for the service's one sample, with `1/(K(1 − p))` in place of
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

/-- **The rate, sharp, with its `ν` in closed form**. For `K ≥ 1` samples, `ρ > 0`, `0 < ρ·w_max ≤ 1`, lucks
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

/-- **The rate in closed form, merged.** For `K ≥ 1` samples, `ρ > 0`, `0 ≤ ρ·w_max ≤ 1`, luck `p, p0 ≤ q` with
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
(`SampledLayer.Covers` at `jointMode R`, the tiles' words and `N_w`). The layer is then bound at the IR's cuts (a
computed value only its own unit reads at its later commit coin), reads each round's coin and has its slots and
writers inside the IR (`StageCuts`, `StageCoins`, `StageSlots`), so a late edge of the layer is one of the IR's or
stays in its unit, and a tile that isn't good is in the cone of a tile the run didn't pay for,
which the share charges that cone (`SampledLayer.covers_of_cone`). It reads no committed value and no rightness, so the
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
per-prover floors use). For a sampled window's proofs (`SampledLayer`) whose round-`w` slots lie in `L`, and a SHA-512
collision finder `F` for `H` whose collisions bound the fork event up to `κ` (`SampledLayer.Forked` at most
`Pr[F finds] + κ`: the prover's fork finder, which runs the window twice from the cut and takes both openings with
Flock's knowledge extractor, `κ` its extraction failure), for every `δ' > 0`
`Pr[MovedIn w] ≤ δ' + |L|²/(4δ')·(ecr(F) + κ)`, where `ecr(F)` is `E[cost F]/2^256` if `F` meets `ecr/sha-512`
(`Claims.ExpectedCR`) and `⊤` if it doesn't. It is `SampledBindingOfFork` (`Pr[MovedIn]² ≤ |L|²·Pr[fork]`)
with AM–GM (`√(γx) ≤ δ' + γx/(4δ')`), and it assumes no hash: a prover whose fork finder beats `ecr` gets `⊤`, so it
is the per-prover `δ_bind` the statements take (`MovedIn ≤ δ_bind`), on the IR's layer under live coins and on the
sample's (`Declared.atSample`) under seeded ones. `SeedLuckAt`'s `δ_b` is read the same way, at a round's roots, at about
`2·T_q` SHA-512 calls. At the best `δ'`, `|L|·√(ecr(F) + κ)/2`, the bound is `|L|·√(ecr(F) + κ)`, and a seeded
window prices it at the budget its premium reads (one `T` for the cheater), with `κ = 0` and `E[cost F] = T` (a finder
that runs the window twice costs `2T`, which is `√2` more): at `2^128` it is `|L|·2^-64` (`2^-44` to `2^-34` at `|L|`
from `2^20` to `2^30`, not negligible beside `δ_s = 2^-40`), at `2^80` it is `|L|·2^-88` (`2^-68` to `2^-58`). Each
figure takes `L` as one window's positions. `ecr`
is the weaker of the two collision claims at a fixed cost (`StrictBirthday` implies it) and the one that applies to
expected-time finders such as an extractor. -/
def BindingOfECR : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : SampledLayer C B S m s J Q Pos V) (w : Fin S) (L : Finset Pos) (H : Claims.Msg → Claims.Msg)
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

end Accounting.Guarantees.Compose
