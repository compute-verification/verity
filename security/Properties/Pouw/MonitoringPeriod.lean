module

public import Definitions.Compose.ComposeMonitoringPeriod
public import Definitions.Compose.Assumptions.Compose
public import PouwCheck

/-! PoUW's per-period work and credit over the windows its deployed check accepts, with H1's tile unit on the torch
loop as its instance. -/

@[expose] public section

namespace Properties.Pouw.MonitoringPeriod

open PouwCheck
open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The deployed check accepts the window**: `protocol.check` at the verifier's parameters `p`, on the window's
registration `c` and the proof system's answer `e`, accepts with some public items and profile. -/
def Accepts (p : Params) (c : ExecutionPhaseCommitmentRecord) (e : Evidence) : Prop :=
  ∃ (pub : Public) (prof : Core.Integrity), protocol.check p c e = .accept pub prof

/-- **The executions of a period's windows as the check reads them**: on the period's coins `x`, window `w`'s
registration and the proof system's answer to it, each any function of the coins. -/
structure Executions (Ω : Type) (n : ℕ) where
  reg : Paths Ω n → Fin n → ExecutionPhaseCommitmentRecord
  ev : Paths Ω n → Fin n → Evidence

namespace Executions

variable {Ω : Type} {n : ℕ}

/-- **Window `w` and every earlier one are accepted** by the check on the period's coins `x`. -/
def AcceptedUpTo (ws : Executions Ω n) (p : Params) (x : Paths Ω n) (w : Fin n) : Prop :=
  ∀ w' : Fin n, w' ≤ w → Accepts p (ws.reg x w') (ws.ev x w')

open Classical in
/-- **The check's accepted prefix**: the number of windows accepted with every earlier one, the windows before the
check's first refusal. -/
noncomputable def prefixLen (ws : Executions Ω n) (p : Params) (x : Paths Ω n) : ℕ :=
  (Finset.univ.filter fun w : Fin n => ws.AcceptedUpTo p x w).card

end Executions

open Classical in
/-- **The accepted prefix's live words**: the words of each window's live tiles (`ExecutionPhasePlay.liveAt`), over the windows
before the check's first refusal. `creditedLive` with the check's prefix for the run's. -/
noncomputable def checkedLive {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : ExecutionPhasePlay C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (p : Params)
    (ws : Executions (Paths C Cd.rounds × B) n) (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  ∑ w : Fin n, if ws.AcceptedUpTo p x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0

/-- **The period's proofs with the check's verdict**: window `w`'s proof of each sampled tile is accepted exactly when the
check accepts window `w`, since it accepts only when every sampled proof block's proof was (`Properties.Pouw.Check.AcceptMeans`,
no partial credit); the `i`-th sampled tile's opening is the service's opening `opened x w u` of that tile's unit `u`
(`drawOf` at the window's sampling coin); the tiles' relations and rightness are the instance's. -/
def checkedRun {C Ξ V : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) :
    Outcomes C (Equiv.Perm (Fin Np) × Ξ) m Kd Np n V where
  accepted x w _ := Accepts p (ws.reg x w) (ws.ev x w)
  opened x w i := opened x w (drawOf hK (coinAt x w).2 i).val
  rel := rel
  right := right

/-- **The period's proofs as the IR defines them, under the check**: `Declared.monitoringPeriodLayer` at `checkedRun`. -/
def checkedLayer {ι C Ξ V : Type} (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (p : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) :
    SampledLayer C (Equiv.Perm (Fin Np) × Ξ) n Cd.rounds Kd (Fin Np) ℕ (Fin n × ℕ) V :=
  Cd.monitoringPeriodLayer hr hK (checkedRun hK p ws opened rel right)

/-- **The check's sample is the period's sampling coin**: on every path and every window the check accepts, the window's
level has the population `N_p` and the sample size `K_d`, and the units it publishes as sampled (`Public.drawn`, the
service's) are the units of the model's sample at the window's sampling coin (`drawOf`). It holds when the model's
permutation coin is read as the service's `subset:K′` sample from the challenger's sampling coin
(`Spot.Select.select`), at one work level for every window. A named assumption in the lock: a
protocol-code fact to prove. -/
def SampleIsCoin {C Ξ : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) : Prop :=
  ∀ (x : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n) (pub : Public) (prof : Core.Integrity),
    protocol.check p (ws.reg x w) (ws.ev x w) = .accept pub prof →
      pub.work.n = Np ∧ pub.sampling.k = Kd ∧
        ∀ u : ℕ, u ∈ pub.drawn.toList ↔ ∃ i : Fin Kd, (drawOf hK (coinAt x w).2 i).val = u

/-- **The service is sound at `δ_ks` against the check**: in every window, the check accepts and some unit it
publishes as sampled has an opening (`opened x w u`) that fails its relation (`rel`, the registered unit Program's) with
probability at most `δ_ks` over the period's coins. Flock's soundness at each sampled proof block, with one-stage's acceptance
and the check's, is what it needs. A named assumption in the lock: standard crypto, Flock's soundness. -/
def ServiceSound {C Ξ V : Type} [Fintype C] [Fintype Ξ] {m Np n : ℕ} (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) (δks : ℝ) : Prop :=
  ∀ w : Fin n, prob (fun x => ∃ (pub : Public) (prof : Core.Integrity),
    protocol.check p (ws.reg x w) (ws.ev x w) = .accept pub prof ∧
      ∃ u : Fin Np, u.val ∈ pub.drawn.toList ∧ ¬ rel x w u (opened x w u.val)) ≤ δks

/-- **Each window's registration is fixed before its sampling coin**: on every path `x`, window `w`'s registration is the
same on every continuation from `w`'s sample (`spliceAt w m`: `x`'s coins before window `w`'s sampling coin, then any
others'). With the window's rows committed by its registration's roots, it is what makes `MovedIn ≤ δ_bind` the roots'
binding: an opening other than the committed row then opens a root fixed before the sample named the unit. It states
the order only; the commitment is the roots', which on H1's path are tags (the loop's registration has `anchors []`
and `wires []`). Its discharge path is to commit the loop's `x`, `sx`, weight rows and `sw` as
anchors (hm96 rows in frame-v3-sha512 trees) before the window's sampling coin, and to bind the H1 tiles' rows to them. A
named assumption in the lock: a protocol-code fact to prove. -/
def RegisteredBeforeDraw {C Ξ : Type} {m Np n : ℕ} (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) : Prop :=
  ∀ (x z : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n), ws.reg (spliceAt w.val m x z) w = ws.reg x w

/-- **(L) The circuit's work on its live tiles over the windows PoUW's deployed check accepts, live coins.** For every
registered circuit `Cd` and scheme `Sc` run as a period of at most `n ≥ 1` windows, each window's registration and the
proof system's answer any function of the period's coins (`Executions`), with the check's parameters `prm`: under
`CircuitWorkMonitoringPeriod`'s premises at the period's proofs with the check's verdict (`checkedRun`, `checkedLayer`), its
`JointSound` replaced by the check's sample being the period's sampling coin (`SampleIsCoin`) and the service's soundness
against the check (`ServiceSound`), with each registration fixed before its window's sampling coin
(`RegisteredBeforeDraw`), and at `K_d > 0`, an accepted period (`Pl.accepts`) has, over the windows before
the check's first refusal (`Executions.prefixLen`, `j` of them), all but `(d_F + d₀)·N_w` of their live words
(`checkedLive`) fixed right and all but `d₀·N_w` folded, with `a·d_F + b·d₀ < Λ·ln(1/δ_s) + j·min(a, b)·D₀`
(`LiveWorkAt` at `rOfPrefix`), except with probability `δ_s + n·(δ_bind + δ_ks)`. -/
def CheckedWorkMonitoringPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (_hKd : 0 < Kd) (prm : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (Wp : ExecutionPhasePlay C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n K T L U)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 ≤ a → 0 ≤ b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = monitoringPeriodGame (tileStage Cd Np) (liveCone Sc Cd Np) 1 1 Kd Wp →
    Wp.EveryExecutionPhase (fun live _ => DeadConeDead Sc Cd Np live) →
    PrefixOf (checkedLayer Cd hr hK prm ws opened rel right) Pl.game →
    (∀ w, prob (fun x => (checkedLayer Cd hr hK prm ws opened rel right).MovedIn x w) ≤ δbind) →
    RegisteredBeforeDraw ws → SampleIsCoin hK prm ws → ServiceSound prm ws opened rel δks →
    (∀ x w j v, rel x w j v → right x w j v) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (checkedLayer Cd hr hK prm ws opened rel right)
      (monitoringPeriodMode Cd Wp (checkedRun hK prm ws opened rel right)) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ LiveWorkAt (checkedLive Sc Cd Wp prm ws x) (wordsOf Sc Cd : ℝ) a b
        (rOfPrefix Sc Cd Np (ws.prefixLen prm x) a b Λ δs) (n * (Pl.run x).fixed) (n * folded x)) ≤
      δs + n * (δbind + δks)

/-- **(L)'s credit in live words over the windows the deployed check accepts**: `CheckedWorkMonitoringPeriod`'s premises at
`a, b > 0` give, except with probability `δ_s + n·(δ_bind + δ_ks)`, an accepted period's fixed words at least
`(1 − ε(j))₊` of the live words of the windows before the check's first refusal (`creditPrefix` at `Executions.prefixLen`
and `checkedLive`), `CircuitCreditMonitoringPeriod`'s conclusion at the check's prefix. -/
def CheckedCreditMonitoringPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (_hKd : 0 < Kd) (prm : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (Wp : ExecutionPhasePlay C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n K T L U)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 < a → 0 < b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = monitoringPeriodGame (tileStage Cd Np) (liveCone Sc Cd Np) 1 1 Kd Wp →
    Wp.EveryExecutionPhase (fun live _ => DeadConeDead Sc Cd Np live) →
    PrefixOf (checkedLayer Cd hr hK prm ws opened rel right) Pl.game →
    (∀ w, prob (fun x => (checkedLayer Cd hr hK prm ws opened rel right).MovedIn x w) ≤ δbind) →
    RegisteredBeforeDraw ws → SampleIsCoin hK prm ws → ServiceSound prm ws opened rel δks →
    (∀ x w j v, rel x w j v → right x w j v) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (checkedLayer Cd hr hK prm ws opened rel right)
      (monitoringPeriodMode Cd Wp (checkedRun hK prm ws opened rel right)) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd Np (ws.prefixLen prm x) (checkedLive Sc Cd Wp prm ws x) a b Λ δs) ≤ δs + n * (δbind + δks)

/-- **Each window's salt is its live coin** (H1's instance). On every path, window `w`'s salt (`salt x w`: the 64-byte
salt the verifier holds for the window, which every tile's noise reads through its public row, `PouwCheck.publicBc`, and
every sampled proof block's proof is held to) is `enc` of the window's round-0 coin, the model's coin, released after the
round's values are committed. The check holds a live window to it: `PouwCheck.executionPhaseSaltReason` accepts a live
window's salt only as its salt session's live coin (`Properties.Pouw.Check.SaltKept`), opened after the window's
committed activations (`anchoredEpochRoot W X k`). The served windows don't meet it yet: no path that serves H1 windows
takes its salt from such a session, and the proved tiles' salt port is a value the prover records. A named assumption
in the lock: a deployment gap until the salt is a live coin from the auditor's challenger and the proved tiles are tied
to it. -/
def SaltIsLiveCoin {C B : Type} {m n : ℕ} (hm : 0 < m) (salt : Paths (Paths C m × B) n → Fin n → ByteArray)
    (enc : C → ByteArray) : Prop :=
  ∀ x w, salt x w = enc (coinAt (coinAt x w).1 ⟨0, hm⟩)

/-- **The staged statement pins the flat tile**: at every salt `s`, values of a unit `t` that satisfy the statement
the service proves of it (`staged s t`, the window's staged statement, which the subprotocol's proofs are of) satisfy the
pinned flat tile's rightness there (`flat s t`, `PouwH1Tile_v1`'s function at salt `s`, the circuit `core_check.opening`
reads and the lucks are claims of). Nothing proves it yet, and neither side has a Lean function, so both are
parameters of the statements that take it. A named assumption in the lock: a
protocol-code fact to prove. -/
def StagedPins {Np : ℕ} {V : Type} (staged flat : ByteArray → Fin Np → (ℕ → V) → Prop) : Prop :=
  ∀ s t v, staged s t v → flat s t v

/-- **H1's unit relation**: window `w`'s unit `t` satisfies the staged statement at the window's salt. -/
def stagedAt {Ω V : Type} {n Np : ℕ} (salt : Paths Ω n → Fin n → ByteArray)
    (staged : ByteArray → Fin Np → (ℕ → V) → Prop) : Paths Ω n → Fin n → Fin Np → (ℕ → V) → Prop :=
  fun x w => staged (salt x w)

/-- **H1's tile rightness**: window `w`'s tile `t` is the flat tile's function at the window's round-0 coin. -/
def flatAtCoin {C B V : Type} {m n Np : ℕ} (hm : 0 < m) (enc : C → ByteArray)
    (flat : ByteArray → Fin Np → (ℕ → V) → Prop) : Paths (Paths C m × B) n → Fin n → Fin Np → (ℕ → V) → Prop :=
  fun x w => flat (enc (coinAt (coinAt x w).1 ⟨0, hm⟩))

/-- **(L)'s credit over the windows the deployed check accepts, at H1's tile unit on the torch loop.**
`CheckedCreditMonitoringPeriod` at the torch loop's window (module docstring): one round a window (`Cd.rounds = 1`,
`w_max = 1`, every tile's noise reads the window's one salt), a population of `N_p = 144` of which `K_d = 24` are
sampled, every tile's share of the words at most `1/128` (`ρ = 8/9`: 128 equal real tiles, the 16 fillers wordless),
H1's lucks `p = 7/8`, `p0 = ½`, the word line `a = b = 1` and `SplitRateClosed`'s rate `Λ = 67/8`. The unit's relation
is the staged statement at the window's salt (`stagedAt`), which `ServiceSound` is about, and its rightness the flat
tile's at the window's coin (`flatAtCoin`); `Pins` is replaced by the window's salt being its live coin
(`SaltIsLiveCoin`) and the staged statement pinning the flat tile (`StagedPins`), with each registration fixed before
its window's sampling coin (`RegisteredBeforeDraw`). Then, except with probability
`δ_s + n·(δ_bind + δ_ks)`, an accepted period's fixed words are at least `(1 − ε(j))₊` of the live words of the
windows before the check's first refusal, `ε(j) = (67/8)·ln(1/δ_s)·N_w/N_live` (`creditPrefix`). -/
def H1CheckedCreditMonitoringPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ 144)
    (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (salt : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ByteArray) (enc : C → ByteArray)
    (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop)
    (Wp : ExecutionPhasePlay C (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) (Fin 144) Cd.rounds n)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ) (δs δbind δks : ℝ),
    Scope Sc Cd δs →
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Sc Cd 144 t ≤ 1) →
    (∀ j, stageShare (tileStage Cd 144) (shareCone Sc Cd 144) j ≤ 1) →
    (∀ t : Fin 144, 0 < shareCone Sc Cd 144 t → tileLuck Sc Cd 144 t ≤ 7 / 8 ∧ tileLuck0 Sc Cd 144 t ≤ 1 / 2) →
    Pl.game = monitoringPeriodGame (tileStage Cd 144) (liveCone Sc Cd 144) 1 1 24 Wp →
    Wp.EveryExecutionPhase (fun live _ => DeadConeDead Sc Cd 144 live) →
    PrefixOf (checkedLayer Cd hr hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat)) Pl.game →
    (∀ w, prob (fun x =>
      (checkedLayer Cd hr hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat)).MovedIn x w) ≤ δbind) →
    RegisteredBeforeDraw ws → SampleIsCoin hK prm ws → ServiceSound prm ws opened (stagedAt salt staged) δks →
    SaltIsLiveCoin hr salt enc → StagedPins staged flat →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryExecutionPhase (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (checkedLayer Cd hr hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat))
      (monitoringPeriodMode Cd Wp (checkedRun hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat)))
      (fun _ t => (tileWords Sc Cd 144 t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd 144 (ws.prefixLen prm x) (checkedLive Sc Cd Wp prm ws x) 1 1 (67 / 8) δs) ≤
        δs + n * (δbind + δks)

end Properties.Pouw.MonitoringPeriod
