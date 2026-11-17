module

public import Properties.Pouw.MonitoringPeriodJ10R
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

/-! A period's J10-R credit over several tile strata of the sampling block, the union of the one-stratum bounds. -/

@[expose] public section

namespace Properties.Pouw.MonitoringPeriodJ10RStrata

open Properties.Pouw.MonitoringPeriodJ10R
open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The sampling block's sample**: a permutation of each stratum `s` of `σ`, independently; stratum `s`'s sample is the
proof blocks of its first `k s` images (`tileDraw`). -/
abbrev BlockDraw {nB mB : ℕ} (σ : Fin nB → Fin mB) : Type := (s : Fin mB) → Equiv.Perm (Fin (tileCount σ s))

/-- **The rest of the block's sample beside stratum `s`**: the other strata's permutations. -/
abbrev Rest {nB mB : ℕ} (σ : Fin nB → Fin mB) (s : Fin mB) : Type :=
  (j : {j : Fin mB // j ≠ s}) → Equiv.Perm (Fin (tileCount σ j.1))

/-- **A window's coins over the block's strata**: its `m` rounds' coins and the block's sample. -/
abbrev StrataCoin (C : Type) (m : ℕ) {nB mB : ℕ} (σ : Fin nB → Fin mB) : Type := Paths C m × BlockDraw σ

/-- **Stratum `s`'s view of a window's coins**: the one-stratum statement's `Coin` at `s_T = s`, the rounds' coins (`r`
of them, `r = m`), then the block's sample split at `s` into the stratum's permutation and the rest. -/
def atStratum {C : Type} {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB) :
    StrataCoin C m σ ≃ Coin C (Rest σ s) r σ s :=
  (Equiv.cast (congrArg (Paths C) h.symm)).prodCongr (Equiv.piSplitAt s _)

/-- **A relabelling of each window's coins, on the period's coins**, window by window. -/
def pathsCongr {α β : Type} (e : α ≃ β) : (n : ℕ) → Paths α n ≃ Paths β n
  | 0 => Equiv.refl Unit
  | n + 1 => e.prodCongr (pathsCongr e n)

/-- **Stratum `s`'s view of the period's coins**: `atStratum` in every window. -/
def pathsAt (C : Type) {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB) (n : ℕ) :
    Paths (StrataCoin C m σ) n ≃ Paths (Coin C (Rest σ s) r σ s) n :=
  pathsCongr (atStratum h σ s) n

/-- **A period-wide input read on stratum `s`'s view** of the period's coins. -/
def pullAt {C β : Type} {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB) {n : ℕ}
    (f : Paths (StrataCoin C m σ) n → β) : Paths (Coin C (Rest σ s) r σ s) n → β :=
  fun y => f ((pathsAt C h σ s n).symm y)

/-- **The period's `Credit.window` inputs on stratum `s`'s view**: each window's registration and the rest of its
inputs, the period's at the coins the view relabels. -/
def creditAt {C : Type} {r m nB mB n : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB)
    (ws : CreditExecutionPhases (StrataCoin C m σ) n) : CreditExecutionPhases (Coin C (Rest σ s) r σ s) n where
  reg := pullAt h σ s ws.reg
  ev := pullAt h σ s ws.ev

/-- **Stratum `s`'s proof block relation**: `pinnedAt` at the window's salt and the place of the stratum's tiles, on its view. -/
def relAt {C V : Type} {r m nB mB n : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB)
    (ws : CreditExecutionPhases (StrataCoin C m σ) n) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    Paths (Coin C (Rest σ s) r σ s) n → Fin n → Fin (tileCount σ s) → (ℕ → V) → Prop :=
  pinnedAt (saltOf (creditAt h σ s ws)) (pullAt h σ s place) tile

/-- **Stratum `s`'s tile rightness**: `j10rAtCoin` at the place of the stratum's tiles, on its view. -/
def rightAt {C V : Type} {r m nB mB n : ℕ} (h : r = m) (hr : 0 < r) (σ : Fin nB → Fin mB) (s : Fin mB)
    (enc : C → ByteArray) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    Paths (Coin C (Rest σ s) r σ s) n → Fin n → Fin (tileCount σ s) → (ℕ → V) → Prop :=
  j10rAtCoin hr enc (pullAt h σ s place) tile

/-- **Stratum `s`'s proofs with `Credit.window`'s verdict**: `tileRun` at the stratum, on its view, with the period's
`Credit.window` inputs and the service's openings. -/
noncomputable def runAt {ι C V : Type} (Cd : Declared ι) (hr : 0 < Cd.rounds) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (s : Fin mB) (unitOf : Fin nB → ℕ)
    (ws : CreditExecutionPhases (StrataCoin C m σ) n) (opened : Paths (StrataCoin C m σ) n → Fin n → ℕ → ℕ → V)
    (enc : C → ByteArray) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    Outcomes C (Equiv.Perm (Fin (tileCount σ s)) × Rest σ s) Cd.rounds (k s) (tileCount σ s) n V :=
  tileRun σ k hk s unitOf (creditAt hM σ s ws) (pullAt hM σ s opened) (relAt hM σ s ws place tile)
    (rightAt hM hr σ s enc place tile)

/-- **Stratum `s`'s proofs as the IR defines them**: `tileLayer` at the stratum, on its view. -/
noncomputable def layerAt {ι C V : Type} (Cd : Declared ι) (hr : 0 < Cd.rounds) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (s : Fin mB) (unitOf : Fin nB → ℕ)
    (ws : CreditExecutionPhases (StrataCoin C m σ) n) (opened : Paths (StrataCoin C m σ) n → Fin n → ℕ → ℕ → V)
    (enc : C → ByteArray) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    SampledLayer C (Equiv.Perm (Fin (tileCount σ s)) × Rest σ s) n Cd.rounds (k s) (Fin (tileCount σ s)) ℕ (Fin n × ℕ) V :=
  tileLayer Cd hr σ k hk s unitOf (creditAt hM σ s ws) (pullAt hM σ s opened) (relAt hM σ s ws place tile)
    (rightAt hM hr σ s enc place tile)

/-- **`Credit.window`'s sample is the block's sample at the tile strata**: on every path and every window `Credit.window`
credits, the proof blocks its sample demands proofs of (`creditDrawn`) are the window proof blocks (`unitOf`) of the tile strata's
coordinates of the block's sample at the window's coins, `tileDraw`'s `k s` proof blocks of stratum `s` for each `s ∈ S`, and
no others. At one stratum it is `TileSampleIsCoin`'s body on the block's sample; read per stratum it would be false at a
window over two non-empty strata. A code fact once its code exists. Code change: the verifier samples the tiles
inside the sampling block's stratified sample, as one subcircuit of the integrity registration, one stratum a `K`, and
reads `k s` from it; today `Credit.window` samples by its own sampling design, one stratum a `K` template. Where the sample's
stream is expanded from the live coin by SHA-512, reading it as the uniform permutations uses SHA-512 as a random
oracle (R-ACCEL-3). -/
def StrataSampleIsCoin {C : Type} {m nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (S : Finset (Fin mB)) (unitOf : Fin nB → ℕ)
    (ws : CreditExecutionPhases (StrataCoin C m σ) n) : Prop :=
  ∀ (x : Paths (StrataCoin C m σ) n) (w : Fin n), ExecutionPhaseAccepts (ws.reg x w) (ws.ev x w) →
    ∃ d : Array ℕ, creditDrawn (ws.reg x w) (ws.ev x w) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ s ∈ S, ∃ b ∈ tileDraw σ k hk s ((coinAt x w).2 s), unitOf b = u

open Classical in
/-- **Stratum `s`'s live words over the accepted prefix**: the words of the stratum's live tiles (`ExecutionPhasePlay.liveAt` of its
period, on its view) in each window before the first window `Credit.window` refuses. -/
noncomputable def stratumLive {ι K C : Type} (Sc : Scheme K) (Cd : Declared ι) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (s : Fin mB)
    (Wp : ExecutionPhasePlay C (Coin C (Rest σ s) Cd.rounds σ s) (Fin (tileCount σ s)) Cd.rounds n)
    (ws : CreditExecutionPhases (StrataCoin C m σ) n) (x : Paths (StrataCoin C m σ) n) : ℝ :=
  ∑ w : Fin n, if ws.AcceptedUpTo x w then
    ∑ t : Fin (tileCount σ s), (liveWords Sc Cd (tileCount σ s) (Wp.liveAt (pathsAt C hM σ s n x) w) t : ℝ) else 0

/-- **(L)'s credit over the windows served H1's verdict credits, at J10-R's tile proof block, over several tile strata.** For
every scheme `Sc` and period of at most `n ≥ 1` windows, each window's `Credit.window` inputs any function of the
period's coins (`ws`), over a finite set `S` of the sampling block's tile strata of `σ`, each `s ∈ S` with its own
registered circuit `Cd s` (`(Cd s).units ≤ N_s = |stratum σ s|`, one round a window), sample `k s ≥ 1`, live share
`0 < ρ s ≤ 1` and rate `Λ_s = j10rRate (k s) (ρ s)`: under the period-wide premises `RegisteredBeforeDraw`,
`StrataSampleIsCoin` and `SaltIsLiveCoin`, and at each `s ∈ S` the premises `H1J10RCheckedCreditMonitoringPeriod` takes at its one
stratum, at stratum `s`'s view (`layerAt`, `runAt`, `relAt`), with `TileSampleIsCoin` replaced by `StrataSampleIsCoin`,
except with probability `∑_{s ∈ S} (δ_s(s) + n·(δ_bind(s) + δ_ks(s)))` an accepted period's fixed words summed over
`S` are at least the sum over `S` of `(1 − ε_s(j))₊` of stratum `s`'s live words in the windows before the first
refusal, `ε_s` at the stratum's `N_s`, `Cd s`, `Λ_s` and `δ_s(s)`. -/
def H1J10RCheckedCreditMonitoringPeriodStrata : Prop :=
  ∀ {ι K T L U C V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Nonempty V]
    {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Finset (Fin mB)) (unitOf : Fin nB → ℕ) (Sc : Scheme K) (Cd : Fin mB → Declared ι) {n : ℕ} (_hn : 0 < n)
    (_hN : ∀ s ∈ S, (Cd s).units ≤ tileCount σ s) (hr : ∀ s, 0 < (Cd s).rounds) (hM : ∀ s, (Cd s).rounds = 1)
    (_hKd : ∀ s ∈ S, 1 ≤ k s)
    (ws : CreditExecutionPhases (StrataCoin C 1 σ) n) (opened : Paths (StrataCoin C 1 σ) n → Fin n → ℕ → ℕ → V)
    (enc : C → ByteArray) (place : (s : Fin mB) → Paths (StrataCoin C 1 σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : (s : Fin mB) → Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop)
    (Wp : (s : Fin mB) → ExecutionPhasePlay C (Coin C (Rest σ s) (Cd s).rounds σ s) (Fin (tileCount σ s)) (Cd s).rounds n)
    (Pl : (s : Fin mB) → Play (Coin C (Rest σ s) (Cd s).rounds σ s) n K T L U)
    (folded : (s : Fin mB) → Paths (Coin C (Rest σ s) (Cd s).rounds σ s) n → ℝ) (ρ δs δbind δks : Fin mB → ℝ),
    RegisteredBeforeDraw ws → StrataSampleIsCoin σ k hk S unitOf ws →
    Properties.Pouw.MonitoringPeriod.SaltIsLiveCoin Nat.one_pos (saltOf ws) enc →
    (∀ s ∈ S, Scope Sc (Cd s) (δs s)) →
    (∀ s ∈ S, 0 < ρ s ∧ ρ s ≤ 1) →
    (∀ s ∈ S, ∀ t : Fin (tileCount σ s), ρ s * (tileCount σ s) * shareCone Sc (Cd s) (tileCount σ s) t ≤ 1) →
    (∀ s ∈ S, ∀ j, stageShare (tileStage (Cd s) (tileCount σ s)) (shareCone Sc (Cd s) (tileCount σ s)) j ≤ 1) →
    (∀ s ∈ S, ∀ t : Fin (tileCount σ s), 0 < shareCone Sc (Cd s) (tileCount σ s) t →
      tileLuck Sc (Cd s) (tileCount σ s) t ≤ 7 / 8 ∧ tileLuck0 Sc (Cd s) (tileCount σ s) t ≤ 1 / 2) →
    (∀ s ∈ S, (Pl s).game =
      monitoringPeriodGame (tileStage (Cd s) (tileCount σ s)) (liveCone Sc (Cd s) (tileCount σ s)) 1 1 (k s) (Wp s)) →
    (∀ s ∈ S, (Wp s).EveryExecutionPhase (fun live _ => DeadConeDead Sc (Cd s) (tileCount σ s) live)) →
    (∀ s ∈ S, PrefixOf (layerAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s) (tile s)) (Pl s).game) →
    (∀ s ∈ S, ∀ w, prob (fun y => (layerAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s)
      (tile s)).MovedIn y w) ≤ δbind s) →
    (∀ s ∈ S, TileServiceSound σ s unitOf (creditAt (hM s) σ s ws) (pullAt (hM s) σ s opened)
      (relAt (hM s) σ s ws (place s) (tile s)) (δks s)) →
    (∀ s ∈ S, SharedNoiseLucks Sc (Cd s) (Wp s)) →
    (∀ s ∈ S, UnitFixed (Pl s) (layerAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s) (tile s))
      (monitoringPeriodMode (Cd s) (Wp s) (runAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s) (tile s)))
      (fun _ t => (tileWords Sc (Cd s) (tileCount σ s) t : ℝ)) (folded s)) →
    prob (fun x : Paths (StrataCoin C 1 σ) n => (∀ s ∈ S, (Pl s).accepts (pathsAt C (hM s) σ s n x)) ∧
      n * ∑ s ∈ S, ((Pl s).run (pathsAt C (hM s) σ s n x)).fixed <
        ∑ s ∈ S, creditPrefix Sc (Cd s) (tileCount σ s) (ws.prefixLen x)
          (stratumLive Sc (Cd s) (hM s) σ s (Wp s) ws x) 1 1 (j10rRate (k s) (ρ s)) (δs s)) ≤
      ∑ s ∈ S, (δs s + n * (δbind s + δks s))

end Properties.Pouw.MonitoringPeriodJ10RStrata
