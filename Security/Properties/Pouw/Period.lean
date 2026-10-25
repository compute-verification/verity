import Definitions.Accounting.ComposePeriod
import Models.Accounting.Assumptions.Compose
import PouwCheck

/-!
# A period's work over the windows PoUW's deployed check accepts (`PouwCheck.protocol.check`)

PoUW's per-period statement about its deployed Lean code, for the all-protocols period (one loop audited every period by
each protocol, composed in `Accounting.Guarantees.JointPeriod`). `Accounting.Guarantees.ComposePeriod.CircuitWorkPeriod`
is stated over the accounting model's abstract proofs (`PeriodRun`): whether window `w`'s proof of its `i`-th drawn tile
was accepted is a parameter. Here it is the deployed check's verdict on the window: `protocol.check` (`verity/pouw/v1`,
`core_check.py`'s port, the `Core.Protocol` that `CoreOnsite` composes) on the window's registration and the proof
service's answer. The check gives no partial credit (`Properties.Pouw.Check.AcceptMeans`: it accepts only when one-stage
accepted the window's every drawn unit), so its verdict is every drawn proof's (`checkedRun`). The period is credited up
to its first window the check refuses (`Windows.prefixLen`), on those windows' live words (`checkedLive`).

It is parametric in the deployed unit: the scheme `Sc`, the declared circuit `Cd`, the population `N_p` and the draw size
`K_d` are arguments, as are the check's parameters `prm` (its check item names the scheme and its tile work). An instance
(H1's tile unit, or ncp-v2's on the loop's linears) fixes them.

* `Accepts`: the check accepts the window.
* `Windows`, `Windows.AcceptedUpTo`, `Windows.prefixLen`, `checkedLive`: each window's registration and the service's
  answer on the period's coins; it and every earlier window accepted; the number of such windows, the windows before
  the first refusal; their live tiles' words.
* `checkedRun`, `checkedLayer`: the period's proofs with the check's verdict as every drawn proof's acceptance, each
  drawn tile's opening the service's opening of that unit; `Declared.periodLayer` at that run.
* `DrawIsCoin`, `ServiceSound`: on every window the check accepts, the units it publishes as drawn (`Public.drawn`) are
  the model's draw at the window's draw coin (`drawOf`), of the population `N_p`, `K_d` of them; and the service's
  soundness against the check: a window is accepted with a published unit whose opening fails its relation with
  probability at most `δ_ks`.
* `CheckedWorkPeriod`, `CheckedCreditPeriod`: `CircuitWorkPeriod`'s and `CircuitCreditPeriod`'s conclusions at the
  check's accepted prefix, under their premises at `checkedRun`, with `JointSound` replaced by `DrawIsCoin` and
  `ServiceSound`.

*What the check discharges.* The accepted prefix (`PeriodRun.Credited`, `prefixLen`) and the credited words are the
check's, defined from `protocol.check` and nothing else. `JointSound` at the checked run is about the model's drawn
tiles; the service proves the units the check publishes, so it follows from `ServiceSound` about those units and
`DrawIsCoin`, which says they are the model's. Every other premise is `CircuitWorkPeriod`'s at the checked run, and the
check doesn't discharge it:

* `Scope Sc Cd δs` and the static numbers (`ρ`, `w_max`, the lucks `p, p0`, `SplitRate`), and `DeadConeDead`: facts of the
  declared circuit and the scheme. They need a `Declared` built from the registered Program, which nothing builds yet
  (`Properties.Compose.UnitsOfLayout`, `Properties.Served.IsServed`, the extractor). `Scope`'s `LayerCommitted` also needs
  the live schedule, a coin a round after its root (`PouwCheck.roundsCheck`, `PouwCheck.saltCheck`), which
  `protocol.check` doesn't read: under PoUW's seeded salt-and-draw schedule it fails on every circuit an activation
  feeds (`Properties.Served.SeededScheduleNotLive`, proved).
* The game identity (`Pl.game = periodGame …`) and `PrefixOf`: protocol structure, the play's ideal game and its pass at
  the drawn tiles' rightness, held by construction of the play.
* `MovedIn ≤ δ_bind` in each window: binding of the service's openings to the registered roots (SHA-512 Merkle trees,
  `Accounting.Guarantees.Compose.BindingOfECR`), per prover.
* `ServiceSound δks`: C-Flock's soundness at each drawn unit under live coins (`Flock.Guarantees.EndToEndRegisteredRows`
  for one session) with one-stage's acceptance (`Properties.SampledProofs.AcceptMeans`) and the check's
  (`Properties.Pouw.Check.AcceptMeans`), both proved; `rel` the registered unit Program's relation.
* `DrawIsCoin`: the service's `subset:K′` draw from the challenger's draw coin (`SampledProofs.Select.select`) read as
  the model's uniform permutation coin, with `K′ = K_d` and `N = N_p` (`Properties.Pouw.Check.AcceptedDraw` gives the
  published draw `K′` distinct units below `N`; `Properties.Served.DrawMissesSubset` bounds its misses by the model's).
* `Pins` (`rel` gives `right`): the registered unit Program's relation pins the declared tile's rightness, a fact of the
  Program and its Definition, with `UnitsOfLayout`'s tie of `Cd` to it.
* `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt` per live tile, and `UnitFixed` at `periodMode`: the named claims, as in
  `CircuitWorkPeriod`. On H1's unit they are (H-μ) at `μ = 3/4` (#2482), B′ at `r_B = 1/2` and the guess bound at
  `1/2`; ncp-v2 has no Compose `Scheme`, and its receipt bound is a time-tradeoff form
  (`Accounting.Guarantees.TileUnpredictableOfNCPU`), not lucks.

`PouwCheck.Credit.periodCredit`, served H1's period credit, is another verdict: its windows' (`PouwCheck.Credit.window`),
refusing the whole period on any refused window. A loop credited through it takes these statements with `Accepts` at
`Credit.window`'s verdict; a period it credits is one whose windows are all accepted, `Windows.prefixLen = n`.

It is stated, not proved. Its proof is `CircuitWorkPeriod` at `checkedRun`, `Credited` there being `AcceptedUpTo` at
`K_d > 0`, and `JointSound` from `ServiceSound` and `DrawIsCoin` by monotonicity.
-/

namespace Properties.Pouw.Period

open PouwCheck
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The deployed check accepts the window**: `protocol.check` at the verifier's parameters `p`, on the window's
registration `c` and the proof service's answer `e`, accepts with some public items and profile. -/
def Accepts (p : Params) (c : Registration) (e : Evidence) : Prop :=
  ∃ (pub : Public) (prof : Core.Integrity), protocol.check p c e = .accept pub prof

/-- **A period's windows as the check reads them**: on the period's coins `x`, window `w`'s registration and the proof
service's answer to it, each any function of the coins. -/
structure Windows (Ω : Type) (n : ℕ) where
  reg : Paths Ω n → Fin n → Registration
  ev : Paths Ω n → Fin n → Evidence

namespace Windows

variable {Ω : Type} {n : ℕ}

/-- **Window `w` and every earlier one are accepted** by the check on the period's coins `x`. -/
def AcceptedUpTo (ws : Windows Ω n) (p : Params) (x : Paths Ω n) (w : Fin n) : Prop :=
  ∀ w' : Fin n, w' ≤ w → Accepts p (ws.reg x w') (ws.ev x w')

open Classical in
/-- **The check's accepted prefix**: the number of windows accepted with every earlier one, the windows before the
check's first refusal. -/
noncomputable def prefixLen (ws : Windows Ω n) (p : Params) (x : Paths Ω n) : ℕ :=
  (Finset.univ.filter fun w : Fin n => ws.AcceptedUpTo p x w).card

end Windows

open Classical in
/-- **The accepted prefix's live words**: the words of each window's live tiles (`Period.liveAt`), over the windows
before the check's first refusal. `creditedLive` with the check's prefix for the run's. -/
noncomputable def checkedLive {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (p : Params)
    (ws : Windows (Paths C Cd.rounds × B) n) (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  ∑ w : Fin n, if ws.AcceptedUpTo p x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0

/-! ## The period's proofs under the check -/

/-- **The period's proofs with the check's verdict**: window `w`'s proof of each drawn tile is accepted exactly when the
check accepts window `w`, since it accepts only when every drawn unit's proof was (`Properties.Pouw.Check.AcceptMeans`,
no partial credit); the `i`-th drawn tile's opening is the service's opening `opened x w u` of that tile's unit `u`
(`drawOf` at the window's draw coin); the tiles' relations and rightness are the instance's. -/
def checkedRun {C Ξ V : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (p : Params)
    (ws : Windows (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) :
    PeriodRun C (Equiv.Perm (Fin Np) × Ξ) m Kd Np n V where
  accepted x w _ := Accepts p (ws.reg x w) (ws.ev x w)
  opened x w i := opened x w (drawOf hK (coinAt x w).2 i).val
  rel := rel
  right := right

/-- **The period's proofs as the IR defines them, under the check**: `Declared.periodLayer` at `checkedRun`. -/
def checkedLayer {ι C Ξ V : Type} (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (p : Params)
    (ws : Windows (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) :
    DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) n Cd.rounds Kd (Fin Np) ℕ (Fin n × ℕ) V :=
  Cd.periodLayer hr hK (checkedRun hK p ws opened rel right)

/-- **The check's draw is the period's draw coin**: on every path and every window the check accepts, the window's
level has the population `N_p` and the draw size `K_d`, and the units it publishes as drawn (`Public.drawn`, the
service's) are the units of the model's draw at the window's draw coin (`drawOf`). It holds when the model's
permutation coin is read as the service's `subset:K′` draw from the challenger's draw coin
(`SampledProofs.Select.select`), at one work level for every window. -/
def DrawIsCoin {C Ξ : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (p : Params)
    (ws : Windows (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) : Prop :=
  ∀ (x : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n) (pub : Public) (prof : Core.Integrity),
    protocol.check p (ws.reg x w) (ws.ev x w) = .accept pub prof →
      pub.work.n = Np ∧ pub.sampling.k = Kd ∧
        ∀ u : ℕ, u ∈ pub.drawn.toList ↔ ∃ i : Fin Kd, (drawOf hK (coinAt x w).2 i).val = u

/-- **The service is sound at `δ_ks` against the check**: in every window, the check accepts and some unit it
publishes as drawn has an opening (`opened x w u`) that fails its relation (`rel`, the registered unit Program's) with
probability at most `δ_ks` over the period's coins. C-Flock's soundness at each drawn unit, with one-stage's acceptance
and the check's, is what it needs. -/
def ServiceSound {C Ξ V : Type} [Fintype C] [Fintype Ξ] {m Np n : ℕ} (p : Params)
    (ws : Windows (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) (δks : ℝ) : Prop :=
  ∀ w : Fin n, prob (fun x => ∃ (pub : Public) (prof : Core.Integrity),
    protocol.check p (ws.reg x w) (ws.ev x w) = .accept pub prof ∧
      ∃ u : Fin Np, u.val ∈ pub.drawn.toList ∧ ¬ rel x w u (opened x w u.val)) ≤ δks

/-! ## The period's conclusion at the check's accepted prefix -/

/-- **(L) The circuit's work on its live tiles over the windows PoUW's deployed check accepts, live coins.** For every
declared circuit `Cd` and scheme `Sc` run as a period of at most `n ≥ 1` windows, each window's registration and the
proof service's answer any function of the period's coins (`Windows`), with the check's parameters `prm`: under
`CircuitWorkPeriod`'s premises at the period's proofs with the check's verdict (`checkedRun`, `checkedLayer`), its
`JointSound` replaced by the check's draw being the period's draw coin (`DrawIsCoin`) and the service's soundness
against the check (`ServiceSound`), and at `K_d > 0`, an accepted period (`Pl.accepts`) has, over the windows before
the check's first refusal (`Windows.prefixLen`, `j` of them), all but `(d_F + d₀)·N_w` of their live words
(`checkedLive`) fixed right and all but `d₀·N_w` folded, with `a·d_F + b·d₀ < Λ·ln(1/δ_s) + j·min(a, b)·D₀`
(`LiveWorkAt` at `rOfPrefix`), except with probability `δ_s + n·(δ_bind + δ_ks)`. -/
def CheckedWorkPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (_hKd : 0 < Kd) (prm : Params)
    (ws : Windows (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n K T L U)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 ≤ a → 0 ≤ b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = periodGame (tileStage Cd Np) (liveCone Sc Cd Np) 1 1 Kd Wp →
    Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd Np live) →
    PrefixOf (checkedLayer Cd hr hK prm ws opened rel right) Pl.game →
    (∀ w, prob (fun x => (checkedLayer Cd hr hK prm ws opened rel right).MovedIn x w) ≤ δbind) →
    DrawIsCoin hK prm ws → ServiceSound prm ws opened rel δks →
    (∀ x w j v, rel x w j v → right x w j v) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (checkedLayer Cd hr hK prm ws opened rel right)
      (periodMode Cd Wp (checkedRun hK prm ws opened rel right)) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ LiveWorkAt (checkedLive Sc Cd Wp prm ws x) (wordsOf Sc Cd : ℝ) a b
        (rOfPrefix Sc Cd Np (ws.prefixLen prm x) a b Λ δs) (n * (Pl.run x).fixed) (n * folded x)) ≤
      δs + n * (δbind + δks)

/-- **(L)'s credit in live words over the windows the deployed check accepts**: `CheckedWorkPeriod`'s premises at
`a, b > 0` give, except with probability `δ_s + n·(δ_bind + δ_ks)`, an accepted period's fixed words at least
`(1 − ε(j))₊` of the live words of the windows before the check's first refusal (`creditPrefix` at `Windows.prefixLen`
and `checkedLive`), `CircuitCreditPeriod`'s conclusion at the check's prefix. -/
def CheckedCreditPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (_hKd : 0 < Kd) (prm : Params)
    (ws : Windows (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n K T L U)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 < a → 0 < b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = periodGame (tileStage Cd Np) (liveCone Sc Cd Np) 1 1 Kd Wp →
    Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd Np live) →
    PrefixOf (checkedLayer Cd hr hK prm ws opened rel right) Pl.game →
    (∀ w, prob (fun x => (checkedLayer Cd hr hK prm ws opened rel right).MovedIn x w) ≤ δbind) →
    DrawIsCoin hK prm ws → ServiceSound prm ws opened rel δks →
    (∀ x w j v, rel x w j v → right x w j v) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (checkedLayer Cd hr hK prm ws opened rel right)
      (periodMode Cd Wp (checkedRun hK prm ws opened rel right)) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd Np (ws.prefixLen prm x) (checkedLive Sc Cd Wp prm ws x) a b Λ δs) ≤ δs + n * (δbind + δks)

end Properties.Pouw.Period
