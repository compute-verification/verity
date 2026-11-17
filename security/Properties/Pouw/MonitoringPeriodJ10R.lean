module

public import Definitions.Pouw.TS.J10R
public import Properties.Pouw.MonitoringPeriod
public import PouwCheck.Credit
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

/-! A period's credit over the windows served H1's verdict credits (`Credit.executionPhase`), at J10-R's unit,
sampled at the sampling block's tile stratum. -/

@[expose] public section

namespace Properties.Pouw.MonitoringPeriodJ10R

open Lean (Json)
open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **Stratum `s` of the sampling block**: the units its assignment `σ` puts in `s`, `FlockSoundness.Audit.SamplingDesign.stratum`
restated (the sampling design is `security_proofs`', which `security_properties` may not require). -/
def stratum {nB mB : ℕ} (σ : Fin nB → Fin mB) (s : Fin mB) : Finset (Fin nB) := Finset.univ.filter (σ · = s)

/-- **The tile stratum's size**, the period's population `N_p`. -/
abbrev tileCount {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) : ℕ := (stratum σ sT).card

/-- **The tile stratum's `t`-th unit** of the sampling block, in ascending order. -/
noncomputable def tileAt {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) (t : Fin (tileCount σ sT)) : Fin nB :=
  (stratum σ sT).orderEmbOfFin rfl t

/-- **The tile stratum's coordinate of the sampling block's sample** at a permutation `π` of the stratum: the units of
its first `k s_T` images (`drawOf`'s samples, as units of the block). -/
noncomputable def tileDraw {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (π : Equiv.Perm (Fin (tileCount σ sT))) : Finset (Fin nB) :=
  Finset.univ.image fun i : Fin (k sT) => tileAt σ sT (π (Fin.castLE (hk sT) i))

/-- **The tile coordinate at a uniform permutation is the stratum's sample**: it misses a set `B` of the sampling
block's proof blocks with the stratum's hypergeometric, `C(n_s − m_s, k_s)/C(n_s, k_s)`. -/
def TileStratumAcceptProb : Prop :=
  ∀ {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
    (B : Finset (Fin nB)),
    prob (fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      (((tileCount σ sT - (B.filter (σ · = sT)).card).choose (k sT) : ℕ) : ℝ) /
        ((tileCount σ sT).choose (k sT) : ℝ)

/-- **A window's coins over the tile stratum**: its rounds' coins, then its sampling coin, the tile stratum's permutation
and the rest of the sampling block's sample `Ξ`. -/
abbrev Coin (C Ξ : Type) (m : ℕ) {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) : Type :=
  Paths C m × (Equiv.Perm (Fin (tileCount σ sT)) × Ξ)

/-- **What a window commits before its sampling coin**, of `Credit.executionPhase`'s inputs: its open root and window record
(`window`), the circuit in hex (`descriptor`), its committed values (`entries`, θ's among them by reference), its commit,
its manifest (`calls`), its served rows (`served`), its salt, its period's committed `k` and its T table rows (`rows`).
Its pre-draw inputs: the window's commitment, not the registration log, whose θ and builder these only reference. -/
structure PreDraw where
  window : Json
  descriptor : Json
  entries : Json
  commit : Json
  calls : Json
  served : Json
  salt : Json
  k : Json
  rows : Json

/-- **The rest of `Credit.executionPhase`'s inputs**: the challenger's key, the auditor's ledger, the pins, the main and
session records, the round log, the sample (its coin, root and receipts), the proofs and the live evidence. -/
structure Answered where
  key : Json
  ledger : Json
  pins : Json
  main : Json
  session : Json
  rounds : Json
  draw : Json
  proofs : Json
  live : Json

/-- **Served H1's verdict credits the window**: `Credit.executionPhase` on its inputs gives credited calls, not a refusal. -/
def ExecutionPhaseAccepts (r : PreDraw) (e : Answered) : Prop :=
  ∃ got : List ℕ, PouwCheck.Credit.executionPhase e.key e.ledger e.pins e.main r.window e.session r.salt r.calls r.served
    r.commit e.rounds e.draw r.descriptor r.entries e.proofs e.live r.k r.rows = .ok got

/-- **The units the window's sample demands proofs of**: `Credit.descriptorDrawn`, the verifier's `Credit.executionPhaseDrawn`
(the sample `Credit.executionPhaseLayout` holds the proofs to) over the strata of the descriptor's layout, at the window's root,
the sample's coin, root and receipts and the period's `k`. The `window-drawn` difftest holds it to the Python verifier's
`credit._window_drawn`. -/
def creditDrawn (r : PreDraw) (e : Answered) : PouwCheck.R (Array ℕ) :=
  PouwCheck.Credit.descriptorDrawn r.window e.draw r.descriptor r.k

/-- **A period's windows as `Credit.executionPhase` reads them**: on the period's coins, window `w`'s registration and the
rest of its inputs, each any function of the coins. -/
structure CreditExecutionPhases (Ω : Type) (n : ℕ) where
  reg : Paths Ω n → Fin n → PreDraw
  ev : Paths Ω n → Fin n → Answered

namespace CreditExecutionPhases

variable {Ω : Type} {n : ℕ}

/-- **Window `w` and every earlier one are credited** by `Credit.executionPhase` on the period's coins `x`. -/
def AcceptedUpTo (ws : CreditExecutionPhases Ω n) (x : Paths Ω n) (w : Fin n) : Prop :=
  ∀ w' : Fin n, w' ≤ w → ExecutionPhaseAccepts (ws.reg x w') (ws.ev x w')

open Classical in
/-- **The accepted prefix**: the number of windows credited with every earlier one, the windows before the first
window `Credit.executionPhase` refuses. -/
noncomputable def prefixLen (ws : CreditExecutionPhases Ω n) (x : Paths Ω n) : ℕ :=
  (Finset.univ.filter fun w : Fin n => ws.AcceptedUpTo x w).card

end CreditExecutionPhases

open Classical in
/-- **The accepted prefix's live words**: the words of each window's live tiles (`ExecutionPhasePlay.liveAt`), over the windows
before the first refusal. `Properties.Pouw.MonitoringPeriod.checkedLive` with `Credit.executionPhase`'s prefix. -/
noncomputable def creditLive {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : ExecutionPhasePlay C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (ws : CreditExecutionPhases (Paths C Cd.rounds × B) n)
    (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  ∑ w : Fin n, if ws.AcceptedUpTo x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0

/-- **The period's proofs with `Credit.executionPhase`'s verdict**: window `w`'s proof of each sampled tile is accepted exactly
when `Credit.executionPhase` credits window `w` (it gives no partial credit: `Credit.proved` holds every sampled proof block to one
accepted outcome); the `i`-th sampled tile's opening is the service's opening `opened x w u` at its window unit
`u = unitOf (tileAt σ s_T (drawOf …))`; the tiles' relations and rightness are the instance's. -/
noncomputable def tileRun {C Ξ V : Type} {m nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ)
    (ws : CreditExecutionPhases (Coin C Ξ m σ sT) n) (opened : Paths (Coin C Ξ m σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ m σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop) :
    Outcomes C (Equiv.Perm (Fin (tileCount σ sT)) × Ξ) m (k sT) (tileCount σ sT) n V where
  accepted x w _ := ExecutionPhaseAccepts (ws.reg x w) (ws.ev x w)
  opened x w i := opened x w (unitOf (tileAt σ sT (drawOf (hk sT) (coinAt x w).2 i)))
  rel := rel
  right := right

/-- **The period's proofs as the IR defines them, under `Credit.executionPhase`**: `Declared.monitoringPeriodLayer` at `tileRun`. -/
noncomputable def tileLayer {ι C Ξ V : Type} (Cd : Declared ι) {nB mB n : ℕ} (hr : 0 < Cd.rounds) (σ : Fin nB → Fin mB)
    (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ)
    (ws : CreditExecutionPhases (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop) :
    SampledLayer C (Equiv.Perm (Fin (tileCount σ sT)) × Ξ) n Cd.rounds (k sT) (Fin (tileCount σ sT)) ℕ (Fin n × ℕ) V :=
  Cd.monitoringPeriodLayer hr (hk sT) (tileRun σ k hk sT unitOf ws opened rel right)

/-- **`Credit.executionPhase`'s sample is the tile coordinate of the sampling block's sample**: on every path and every window
`Credit.executionPhase` credits, the units its sample demands proofs of (`creditDrawn`) are the window units (`unitOf`) of the tile
stratum's coordinate at the window's sampling coin (`tileDraw`'s, `drawOf`'s samples), `k s_T` of the stratum's
`|stratum σ s_T|`. A code fact once its code exists. Code change: the verifier samples the tiles inside the
sampling block's stratified sample, as one subcircuit of the integrity registration, and reads `k s_T` from it; today
`Credit.executionPhase` samples by its own work sampling design. Where the sample's stream is expanded from the live coin by SHA-512, reading
it as the uniform permutation uses SHA-512 as a random oracle. -/
def TileSampleIsCoin {C Ξ : Type} {m nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ)
    (ws : CreditExecutionPhases (Coin C Ξ m σ sT) n) : Prop :=
  ∀ (x : Paths (Coin C Ξ m σ sT) n) (w : Fin n), ExecutionPhaseAccepts (ws.reg x w) (ws.ev x w) →
    ∃ d : Array ℕ, creditDrawn (ws.reg x w) (ws.ev x w) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ i : Fin (k sT), unitOf (tileAt σ sT (drawOf (hk sT) (coinAt x w).2 i)) = u

/-- **The service is sound at `δ_ks` against `Credit.executionPhase`**: in every window, `Credit.executionPhase` credits it and some
tile of the stratum whose window unit its sample demanded has an opening that fails its relation, with probability at
most `δ_ks` over the period's coins. Flock's soundness at each sampled proof block under live coins is what it needs. -/
def TileServiceSound {C Ξ V : Type} [Fintype C] [Fintype Ξ] {m nB mB n : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB)
    (unitOf : Fin nB → ℕ) (ws : CreditExecutionPhases (Coin C Ξ m σ sT) n)
    (opened : Paths (Coin C Ξ m σ sT) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Coin C Ξ m σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop) (δks : ℝ) : Prop :=
  ∀ w : Fin n, prob (fun x => ExecutionPhaseAccepts (ws.reg x w) (ws.ev x w) ∧
    ∃ d : Array ℕ, creditDrawn (ws.reg x w) (ws.ev x w) = .ok d ∧
      ∃ t : Fin (tileCount σ sT), unitOf (tileAt σ sT t) ∈ d.toList ∧
        ¬ rel x w t (opened x w (unitOf (tileAt σ sT t)))) ≤ δks

/-- **Each window's commitment is fixed before its sampling coin**: `Properties.Pouw.MonitoringPeriod.RegisteredBeforeDraw` at
`Credit.executionPhase`'s pre-draw inputs (`PreDraw`). It keeps window `w`'s round coins, the salt among them, so it carries
the commitment before the sampling coin and not the order binding needs: H1's anchors (`x`, `sx`, the weight rows, `sw`;
J10-R's root(`a/<K>@<s>`) and root(`b/<K>`), which `seed_c` hashes with the salt) committed before the window's salt
coin, which is released against their root. The check holds a live window to that order: its salt opens on
`anchoredEpochRoot W X k`, `X` the root of the activations the window committed (`executionPhaseAnchors`), and the loop commits
the H1 tiles' A and B rows there, though not yet the served `x` and `sx`. No premise states the order yet (each
anchor's root invariant under `spliceAt w 0`, the salt's coordinate). -/
def RegisteredBeforeDraw {C B : Type} {m n : ℕ} (ws : CreditExecutionPhases (Paths C m × B) n) : Prop :=
  ∀ (x z : Paths (Paths C m × B) n) (w : Fin n), ws.reg (spliceAt w.val m x z) w = ws.reg x w

/-- **Window `w`'s salt as `Credit.executionPhase` reads it**: its registered `salt`, hex-decoded. -/
def saltOf {Ω : Type} {n : ℕ} (ws : CreditExecutionPhases Ω n) (x : Paths Ω n) (w : Fin n) : ByteArray :=
  (PouwCheck.hex? (ws.reg x w).salt).getD ByteArray.empty

/-- **A unit's place in its window**, as the verifier derives it for the unit's public row
(`Credit.registeredFile`'s `j10r`): its position `pos` (`u0 + t`), its call's first unit `call` (`u0`), the call's
slot `m` and contraction `K`, the tile `r × c`, the unit's tile coordinates `(I, J)` (`(t / (n/c), t % (n/c))`) and its
call's registered roots root(`a/<K>@<s>`) and root(`b/<K>`) as bytes. -/
structure Place where
  pos : ℕ
  call : ℕ
  m : ℕ
  K : ℕ
  r : ℕ
  c : ℕ
  I : ℕ
  J : ℕ
  rootA : List ℕ
  rootB : List ℕ

/-- J10-R's noise of one unit: its rows' (dither bits, digit coins) and its columns' (coin bits, `U`). -/
abbrev Noise : Type := List (List ℕ × List (List ℕ)) × List (List ℕ × List ℕ)

namespace Place

/-- The unit's tile at its call's contraction, `r × c` at `K`. -/
def layout (p : Place) : J10R.NoiseLayout := ⟨p.r, p.c, p.K⟩

/-- The unit's first row block and first column block, the verifier's (`Credit.j10rBlocks`). -/
def blocks (p : Place) : ℕ × ℕ := PouwCheck.Credit.j10rBlocks p.m p.K (p.r, p.c) p.I p.J

/-- The unit's noise at salt `s`: its call's `seed_c` at its blocks (`J10R.unitNoise`). -/
def noise (s : ByteArray) (p : Place) : Noise :=
  J10R.unitNoise (s.toList.map (·.toNat)) p.rootA p.rootB p.call p.layout p.blocks

/-- The unit's fold keys at salt `s`: σ_R at its own position (`J10R.unitKeys`). -/
def keys (s : ByteArray) (p : Place) : List (List ℕ) :=
  J10R.unitKeys (s.toList.map (·.toNat)) p.rootA p.rootB p.pos p.layout

end Place

/-- **J10-R's unit relation, its one description**: window `w`'s unit `t` satisfies `j10r_reference`'s relation
(`tile`, the registered J10-R Definition's function of a unit's noise and fold keys) at the noise and keys its public
row gives in-circuit: those of its place under the window's salt as `Credit.executionPhase` reads it. The statement the service
proves is this relation, the pinned form (every word recomputed from committed operands at a pinned start), so no second
description of the tile appears and no premise relates two. That the registered Definition computes `tile` is a code
fact, circuit-check's pin of the Definition to `j10r_reference`. -/
def pinnedAt {Ω V : Type} {n Np : ℕ} (salt : Paths Ω n → Fin n → ByteArray)
    (place : Paths Ω n → Fin n → Fin Np → Place) (tile : Noise → List (List ℕ) → Fin Np → (ℕ → V) → Prop) :
    Paths Ω n → Fin n → Fin Np → (ℕ → V) → Prop :=
  fun x w t => tile ((place x w t).noise (salt x w)) ((place x w t).keys (salt x w)) t

/-- **J10-R's tile rightness**: window `w`'s tile `t` meets `j10r_reference`'s relation (`tile`) at the noise and fold keys of its place
under the window's round-0 coin. -/
def j10rAtCoin {C B V : Type} {m n Np : ℕ} (hm : 0 < m) (enc : C → ByteArray)
    (place : Paths (Paths C m × B) n → Fin n → Fin Np → Place)
    (tile : Noise → List (List ℕ) → Fin Np → (ℕ → V) → Prop) :
    Paths (Paths C m × B) n → Fin n → Fin Np → (ℕ → V) → Prop :=
  fun x w t =>
    tile ((place x w t).noise (enc (coinAt (coinAt x w).1 ⟨0, hm⟩)))
      ((place x w t).keys (enc (coinAt (coinAt x w).1 ⟨0, hm⟩))) t

/-- **The lucks at J10-R's shared noise** (one claim): every live tile's game in every window meets the three per-tile
lucks, `OperandHMu`
(`μ = Sc.mu k`), `ReceiptLuckAt` (`r_B = Sc.rB k`) and `GuessLuckAt` (`p0 = Sc.guess k`), the form
`CircuitCreditMonitoringPeriod`'s rate reads. J10-R derives a call's noise from one seed (`J10R.callSeed`), so the units of a
call row read that row's noise `U_r` and the units of a call column that column's `F_c` (`J10R.j10rNoise` from the
unit's bases), and only the fold keys are a unit's own (`J10R.unitKeys`). The per-tile lucks were argued (H-μ
at `μ = 3/4`, B′ at `r_B = ½`, the guess bound at `½`) of a tile whose noise is its own.

What it claims is the call-level form.
The round is the whole call: every tile that reads the call's `U_r` and `F_c` is in one round, over the
call's joint coins, and the three lucks of each of its tiles are special cases of one moment-form slot at that round,
`E[1_pass · e^{η(c − cost/H)₊}] ≤ 1` (the proposed `Pouw.TheoremO.Slot`). That slot is one expectation over the joint
coins and assumes no independence between tiles that share `U_r` or `F_c`; the period multiplies moments across
rounds (live coins in sequence), never across a call's tiles. Read per tile, each tile's game is its game given the
coins it shares with the call's other tiles (`PerLiveTile`'s "whatever the past"), and the slot is what makes that
reading sound. Sharing changes no constant: each of a tile's terms `A_rB_c + A_rF_c + U_rB_c + U_rF_c` reads its own
data block or its own pair `(U_r, F_c)`. Every coin these games read reaches the prover before it commits, so the
claim is computational, not luck, and the fold's no-cheap-part property is necessary for the slot at
a useful `(c, η)`. The one computational premise covers exact matrix
products only, so this claim is to be proved in the slot's model or designed away (the no-fold receipt),
not assumed; until then it is a named premise. -/
def SharedNoiseLucks {ι K C Ω : Type} [Fintype C] (Sc : Scheme K) (Cd : Declared ι) {Np M n : ℕ}
    (Wp : ExecutionPhasePlay C Ω (Fin Np) M n) : Prop :=
  Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) ∧
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) ∧
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0))

/-- **`SplitRateClosed`'s rate at H1's lucks**: `max(a(1 + Kρw)/(ρK(1 − p)), b(1 + Kρw)/(ρK(1 − p0)))` at `a = b = 1`,
`w_max = 1`, `p = 7/8`, `p0 = ½`, the first, `8(1 + Kρ)/(ρK)`. -/
noncomputable def j10rRate (K : ℕ) (ρ : ℝ) : ℝ := 8 * (1 + K * ρ) / (ρ * K)

/-- **(L)'s credit over the windows served H1's verdict credits, at J10-R's tile unit.** For every registered circuit
`Cd` and scheme `Sc` run as a period of at most `n ≥ 1` windows, each window's `Credit.executionPhase` inputs any function of
the period's coins (`CreditExecutionPhases`), over the sampling block's tile stratum `s_T` of `σ` (`N_p = |stratum σ s_T|`,
`K_d = k s_T ≥ 1`): `CircuitCreditMonitoringPeriod`'s premises at the period's proofs with `Credit.executionPhase`'s verdict (`tileRun`,
`tileLayer`), its `JointSound` replaced by `TileSampleIsCoin` and `TileServiceSound`, with `RegisteredBeforeDraw`, at
one round a window (`Cd.rounds = 1`, `w_max = 1`: every unit's noise reads the window's one salt, through its call's
`seed_c`), a live share `0 < ρ ≤ 1`, H1's lucks `p = 7/8`, `p0 = ½`, the word line `a = b = 1` and
`SplitRateClosed`'s rate `Λ = j10rRate (k s_T) ρ`. The unit's relation is
`j10r_reference`'s at the unit's place under the window's salt as `Credit.executionPhase` reads it (`pinnedAt`), which
`TileServiceSound` is about, and its rightness `j10r_reference`'s at the noise and fold keys of its place under the
window's coin (`j10rAtCoin`); `Pins` is replaced by the window's salt being its live coin
(`Properties.Pouw.MonitoringPeriod.SaltIsLiveCoin`), and the three lucks are `SharedNoiseLucks`. Then, except with probability `δ_s + n·(δ_bind + δ_ks)`, an accepted period's fixed words are at
least `(1 − ε(j))₊` of the live words of the windows before the first refusal. -/
def H1J10RCheckedCreditMonitoringPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (sT : Fin mB) (unitOf : Fin nB → ℕ) (Sc : Scheme K) (Cd : Declared ι) {n : ℕ} (_hn : 0 < n)
    (_hN : Cd.units ≤ tileCount σ sT) (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (_hKd : 1 ≤ k sT)
    (ws : CreditExecutionPhases (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V) (enc : C → ByteArray)
    (place : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (Wp : ExecutionPhasePlay C (Coin C Ξ Cd.rounds σ sT) (Fin (tileCount σ sT)) Cd.rounds n)
    (Pl : Play (Coin C Ξ Cd.rounds σ sT) n K T L U)
    (folded : Paths (Coin C Ξ Cd.rounds σ sT) n → ℝ) (ρ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → ρ ≤ 1 →
    (∀ t : Fin (tileCount σ sT), ρ * (tileCount σ sT) * shareCone Sc Cd (tileCount σ sT) t ≤ 1) →
    (∀ j, stageShare (tileStage Cd (tileCount σ sT)) (shareCone Sc Cd (tileCount σ sT)) j ≤ 1) →
    (∀ t : Fin (tileCount σ sT), 0 < shareCone Sc Cd (tileCount σ sT) t →
      tileLuck Sc Cd (tileCount σ sT) t ≤ 7 / 8 ∧ tileLuck0 Sc Cd (tileCount σ sT) t ≤ 1 / 2) →
    Pl.game = monitoringPeriodGame (tileStage Cd (tileCount σ sT)) (liveCone Sc Cd (tileCount σ sT)) 1 1 (k sT) Wp →
    Wp.EveryExecutionPhase (fun live _ => DeadConeDead Sc Cd (tileCount σ sT) live) →
    PrefixOf (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
      (j10rAtCoin hr enc place tile)) Pl.game →
    (∀ w, prob (fun x => (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
      (j10rAtCoin hr enc place tile)).MovedIn x w) ≤ δbind) →
    RegisteredBeforeDraw ws → TileSampleIsCoin σ k hk sT unitOf ws →
    TileServiceSound σ sT unitOf ws opened (pinnedAt (saltOf ws) place tile) δks →
    Properties.Pouw.MonitoringPeriod.SaltIsLiveCoin hr (saltOf ws) enc → SharedNoiseLucks Sc Cd Wp →
    UnitFixed Pl (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
        (j10rAtCoin hr enc place tile))
      (monitoringPeriodMode Cd Wp (tileRun σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
        (j10rAtCoin hr enc place tile)))
      (fun _ t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x) 1 1 (j10rRate (k sT) ρ) δs) ≤
        δs + n * (δbind + δks)

end Properties.Pouw.MonitoringPeriodJ10R
