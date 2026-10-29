import Definitions.Accounting.ComposePeriod
import Definitions.Accounting.Assumptions.Compose
import PouwCheck

/-!
# A period's work over the windows PoUW's deployed check accepts (`PouwCheck.protocol.check`)

PoUW's per-period statement about its deployed Lean code, for the all-protocols period (one loop audited every period by
each protocol, composed in the causal game, `Accounting.Guarantees.JointPeriodLive`). `Accounting.Guarantees.ComposePeriod.CircuitWorkPeriod`
is stated over the accounting model's abstract proofs (`Outcomes`): whether window `w`'s proof of its `i`-th drawn tile
was accepted is a parameter. Here it is the deployed check's verdict on the window: `protocol.check` (`verity/pouw/v1`,
`core_check.py`'s port, the `Core.Protocol` that `CoreRemote` composes; onsite composes the period's,
`PouwCheck.Period.protocol`) on the window's registration and the proof
service's answer. The check gives no partial credit (`Properties.Pouw.Check.AcceptMeans`: it accepts only when one-stage
accepted the window's every drawn unit), so its verdict is every drawn proof's (`checkedRun`). The period is credited up
to its first window the check refuses (`Executions.prefixLen`), on those windows' live words (`checkedLive`).

It is parametric in the deployed unit and names no device: the scheme `Sc`, the declared circuit `Cd`, the population
`N_p` and the draw size `K_d` are arguments, as are the check's parameters `prm` (its check item names the scheme and its
tile work), and the relation `rel` is the statement the service proves of each unit, whatever it is. An instance fixes
them: `H1CheckedCreditPeriod` is H1's tile unit on the torch loop (below).

* `Accepts`: the check accepts the window.
* `Executions`, `Executions.AcceptedUpTo`, `Executions.prefixLen`, `checkedLive`: each window's registration and the
  service's answer on the period's coins; it and every earlier window accepted; the number of such windows, the
  windows before the first refusal; their live tiles' words.
* `checkedRun`, `checkedLayer`: the period's proofs with the check's verdict as every drawn proof's acceptance, each
  drawn tile's opening the service's opening of that unit; `Declared.periodLayer` at that run.
* `DrawIsCoin`, `ServiceSound`: on every window the check accepts, the units it publishes as drawn (`Public.drawn`) are
  the model's draw at the window's draw coin (`drawOf`), of the population `N_p`, `K_d` of them; and the service's
  soundness against the check: a window is accepted with a published unit whose opening fails its relation with
  probability at most `δ_ks`.
* `RegisteredBeforeDraw`: each window's registration is fixed before the window's draw coin.
* `CheckedWorkPeriod`, `CheckedCreditPeriod`: `CircuitWorkPeriod`'s and `CircuitCreditPeriod`'s conclusions at the
  check's accepted prefix, under their premises at `checkedRun`, with `JointSound` replaced by `DrawIsCoin` and
  `ServiceSound`, and with `RegisteredBeforeDraw`.
* `SaltIsLiveCoin`, `StagedPins`, `stagedAt`, `flatAtCoin`, `H1CheckedCreditPeriod`: H1's instance. Each window's salt
  is its round's live coin; the staged statement the leg proves pins the pinned flat tile at one salt; the unit's
  relation, the staged statement at the window's salt; its rightness, the flat tile's at the window's coin; and
  `CheckedCreditPeriod` at the torch loop's counts with those two in place of `Pins`.

*What the check discharges.* The accepted prefix (`Outcomes.Credited`, `prefixLen`) and the credited words are the
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
* `MovedIn ≤ δ_bind` in each window: binding of the service's openings to the values committed before each position's
  cut, which the proof reads. It is SHA-512's binding at the trees' sizes (`Accounting.Guarantees.Compose.BindingOfECR`)
  only when the window's rows are committed by its registration, under roots the openings are checked against, and the
  registration is fixed before the window's draw coin (`RegisteredBeforeDraw`, which states the order; the commitment
  is the roots'). On today's path the registered roots are tags, not the rows' commitments (below): the loop's
  registration has `anchors []` and `wires []`, so nothing commits the rows before the draw. Its discharge path is
  to commit the loop's `x`, `sx`, weight rows and `sw` as anchors (hm96 rows in frame-v3-sha512 trees) before the
  window's draw coin, and to bind the H1 tiles' rows to those anchors.
* `RegisteredBeforeDraw`: protocol structure. The check reads only that the service's answer cites the registration's
  digest (`Evidence.commitment`), not when the draw coin was released against it, so it doesn't discharge it.
* `ServiceSound δks`: C-Flock's soundness at each drawn unit under live coins (`Flock.Guarantees.EndToEndRegisteredRows`
  for one session) with one-stage's acceptance (`Properties.SampledProofs.AcceptMeans`) and the check's
  (`Properties.Pouw.Check.AcceptMeans`), both proved; `rel` the relation of the statement the service proves (H1's
  leg: the window's staged statement, `stagedAt`).
* `DrawIsCoin`: the service's `subset:K′` draw from the challenger's draw coin (`SampledProofs.Select.select`) read as
  the model's uniform permutation coin, with `K′ = K_d` and `N = N_p` (`Properties.Pouw.Check.AcceptedDraw` gives the
  published draw `K′` distinct units below `N`; `Properties.Served.DrawMissesSubset` bounds its misses by the model's).
* `Pins` (`rel` gives `right`): the relation of the statement the service proves pins the declared tile's rightness, a
  fact of that statement and the tile's Definition, with `UnitsOfLayout`'s tie of `Cd` to it. Where the service proves
  a statement other than the registered tile's (H1's leg proves the window's staged statement), this is the link
  between the two, and H1's instance splits it into `StagedPins` and `SaltIsLiveCoin`.
* `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt` per live tile, and `UnitFixed` at `periodMode`: the named claims, as in
  `CircuitWorkPeriod`. On H1's unit they are (H-μ) at `μ = 3/4`, B′ at `r_B = 1/2` and the guess bound at
  `1/2`; ncp-v2 has no Compose `Scheme`, and its receipt bound is a time-tradeoff form
  (`Accounting.Guarantees.TileUnpredictableOfNCPU`), not lucks.

`PouwCheck.Credit.periodCredit`, served H1's period credit, is another verdict: its windows' (`PouwCheck.Credit.window`),
refusing the whole period on any refused window. A loop credited through it takes these statements with `Accepts` at
`Credit.window`'s verdict; a period it credits is one whose windows are all accepted, `Executions.prefixLen = n`.

*H1's tile unit on the torch loop* (`H1CheckedCreditPeriod`, PoUW's leg of the period above). The loop's W8A8
two-layer MLP at `H = 128` serves 4 requests × 2 decode steps a window, and each linear's `K = 128` is two pinned H1
tiles `PouwH1Tile_v1` (`K = 64`, `R = 4`, `C = 8`, `core_check.TILES`): 2 steps × 2 linears × 16 column blocks × 2
halves of `K` = 128 real tiles a window, in a population of `N_p = 144` (the grid's 16 fillers, of zero rows and no
words, so `ρ = 128/144 = 8/9`), `K_d = 24` of them drawn at the leg's check (`ε = 1/2`, `δ = 10⁻⁶`, `window.k_prime`).
Every tile's noise reads the window's one salt (its public row, `PouwCheck.publicBc`), so the window is one round
(`Cd.rounds = 1`, `w_max = 1`): a coin a decode step would need a salt a step, which the pinned tile and the check don't
take. At H1's lucks (`p = 7/8`, `p0 = ½`, the floors `OperandHMu`, `ReceiptLuckAt` and `GuessLuckAt` claim), `a = b = 1`
and `SplitRateClosed`'s rate there, `Λ = max(67/8, 67/32) = 67/8`, an accepted period is credited `(1 − ε(j))₊` of its
prefix's live words with `ε(j) = Λ·ln(1/δ_s)/j` (`D₀ = 0`, every real tile live): `ε(j) ≥ 1` through 38 accepted windows
at `δ_s = 1%` and through 115 at `10⁻⁶`, and `ε(j) < 1%` from 3,857 and 11,571 windows (`SplitRateExp`'s sharp
`Λ = 7.489`: 34, 103, 3,449 and 10,347). Pooled over a monitoring period of any length, the uncredited words are at most
`Λ·ln(1/δ_s)` windows' worth, 38.6 at `1%` and 115.7 at `10⁻⁶`.

What H1's instance takes that the deployed path doesn't give, each a named premise:
* `RegisteredBeforeDraw`, with the window's rows committed by its registration: the binding premise reads both. The
  tiles' registered roots are tags (below), so the registration commits no row (`anchors []`, `wires []`), and an
  opening the service gives is bound to nothing fixed before the draw. Discharge path: the loop's anchors
  committed before the draw coin with ALLP-H1LOOP's binding of the tiles' rows to them (above).
* `SaltIsLiveCoin`: each window's salt is its round's live coin. The check reads it: a live window (`coins` live in its
  check item) carries its salt in its service answer, and `PouwCheck.windowSaltReason`
  (`Properties.Pouw.Check.SaltKept`) accepts it only as the round-0 salt of a salt-only live session's coin, opened on
  the salt epoch's root `anchoredEpochRoot W X k` of the registration's weights' root `W` and the root `X` of the
  activations its window committed before the salt (`windowAnchors`). That session counts as a live Pearl-C pass's
  does (`PouwCheck.liveSaltCheck`, `Properties.Pouw.LiveSalt`). A window whose check item names no coin mode is seeded,
  its answer carries no salt, and `Scope`'s `Declared.Live` and `LayerCommitted` fail of the circuit it runs. No path
  that serves H1 windows takes its salt from such a session yet: `window_salt.issue` opens one, on the auditor's
  challenger (a `trusted-host` machine) or on a development gate on the issuing machine, and the H200 per-call tiles
  still stand in a SHA-512 of the step's salt key for the salt. What remains: the served windows' salt from the
  auditor's challenger, and a tie of the proved tiles to that salt (their salt port is still a value the prover
  records).
* `StagedPins`: the leg's proofs are of the window's staged statement, not the pinned flat circuit `core_check.opening`
  reads. The bridge is stated over the staged statement (`ServiceSound` is the service's soundness for it, `stagedAt`),
  and `StagedPins` is what links it to the flat tile's rightness, which the lucks are claims of (`flatAtCoin`). No Lean
  function of either side exists yet, so both are parameters.

What it doesn't give: any link from the H1 window to the values integrity's window commits. The tiles' roots are tags
(SHA-512 of the run root and of the layer's weights), not the rows' commitments, and that the tiles' A rows are the
loop's activations rests on the loop's own assertion (`H1Loop.run`). Its credit is the H1 window's committed tiles' words;
tying them to integrity's values is the joint statement's.

The proofs are in `Proofs.Pouw.Period`: `CircuitWorkPeriod` at `checkedRun`, `Credited` there being `AcceptedUpTo` at
`K_d > 0`, and `JointSound` from `ServiceSound` and `DrawIsCoin` by monotonicity; H1's instance is `CheckedCreditPeriod`
at its numbers with `SplitRateClosed` for `SplitRate` and `Pins` from `SaltIsLiveCoin` and `StagedPins`.
`Proofs.Pouw.Period.Accept` runs the deployed check to an accept at H1's parameters (`N_p = 144`, `K_d = 24` on every
accepted window), and `Proofs.Pouw.Period.Honest` is a live-salt period of those windows meeting every premise of
`H1CheckedCreditPeriod`.
-/

namespace Properties.Pouw.Period

open PouwCheck
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The deployed check accepts the window**: `protocol.check` at the verifier's parameters `p`, on the window's
registration `c` and the proof service's answer `e`, accepts with some public items and profile. -/
def Accepts (p : Params) (c : Registration) (e : Evidence) : Prop :=
  ∃ (pub : Public) (prof : Core.Integrity), protocol.check p c e = .accept pub prof

/-- **The executions of a period's windows as the check reads them**: on the period's coins `x`, window `w`'s
registration and the proof service's answer to it, each any function of the coins. -/
structure Executions (Ω : Type) (n : ℕ) where
  reg : Paths Ω n → Fin n → Registration
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
/-- **The accepted prefix's live words**: the words of each window's live tiles (`Period.liveAt`), over the windows
before the check's first refusal. `creditedLive` with the check's prefix for the run's. -/
noncomputable def checkedLive {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (p : Params)
    (ws : Executions (Paths C Cd.rounds × B) n) (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  ∑ w : Fin n, if ws.AcceptedUpTo p x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0

/-! ## The period's proofs under the check -/

/-- **The period's proofs with the check's verdict**: window `w`'s proof of each drawn tile is accepted exactly when the
check accepts window `w`, since it accepts only when every drawn unit's proof was (`Properties.Pouw.Check.AcceptMeans`,
no partial credit); the `i`-th drawn tile's opening is the service's opening `opened x w u` of that tile's unit `u`
(`drawOf` at the window's draw coin); the tiles' relations and rightness are the instance's. -/
def checkedRun {C Ξ V : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) :
    Outcomes C (Equiv.Perm (Fin Np) × Ξ) m Kd Np n V where
  accepted x w _ := Accepts p (ws.reg x w) (ws.ev x w)
  opened x w i := opened x w (drawOf hK (coinAt x w).2 i).val
  rel := rel
  right := right

/-- **The period's proofs as the IR defines them, under the check**: `Declared.periodLayer` at `checkedRun`. -/
def checkedLayer {ι C Ξ V : Type} (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (p : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) :
    DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) n Cd.rounds Kd (Fin Np) ℕ (Fin n × ℕ) V :=
  Cd.periodLayer hr hK (checkedRun hK p ws opened rel right)

/-- **The check's draw is the period's draw coin**: on every path and every window the check accepts, the window's
level has the population `N_p` and the draw size `K_d`, and the units it publishes as drawn (`Public.drawn`, the
service's) are the units of the model's draw at the window's draw coin (`drawOf`). It holds when the model's
permutation coin is read as the service's `subset:K′` draw from the challenger's draw coin
(`SampledProofs.Select.select`), at one work level for every window. A named assumption in the lock: a
protocol-code fact to prove. -/
def DrawIsCoin {C Ξ : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) : Prop :=
  ∀ (x : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n) (pub : Public) (prof : Core.Integrity),
    protocol.check p (ws.reg x w) (ws.ev x w) = .accept pub prof →
      pub.work.n = Np ∧ pub.sampling.k = Kd ∧
        ∀ u : ℕ, u ∈ pub.drawn.toList ↔ ∃ i : Fin Kd, (drawOf hK (coinAt x w).2 i).val = u

/-- **The service is sound at `δ_ks` against the check**: in every window, the check accepts and some unit it
publishes as drawn has an opening (`opened x w u`) that fails its relation (`rel`, the registered unit Program's) with
probability at most `δ_ks` over the period's coins. C-Flock's soundness at each drawn unit, with one-stage's acceptance
and the check's, is what it needs. A named assumption in the lock: standard crypto, C-Flock's soundness. -/
def ServiceSound {C Ξ V : Type} [Fintype C] [Fintype Ξ] {m Np n : ℕ} (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop) (δks : ℝ) : Prop :=
  ∀ w : Fin n, prob (fun x => ∃ (pub : Public) (prof : Core.Integrity),
    protocol.check p (ws.reg x w) (ws.ev x w) = .accept pub prof ∧
      ∃ u : Fin Np, u.val ∈ pub.drawn.toList ∧ ¬ rel x w u (opened x w u.val)) ≤ δks

/-- **Each window's registration is fixed before its draw coin**: on every path `x`, window `w`'s registration is the
same on every continuation from `w`'s draw (`spliceAt w m`: `x`'s coins before window `w`'s draw coin, then any
others'). With the window's rows committed by its registration's roots, it is what makes `MovedIn ≤ δ_bind` the roots'
binding: an opening other than the committed row then opens a root fixed before the draw named the unit. It states
the order only; the commitment is the roots', which on H1's path are tags (the loop's registration has `anchors []`
and `wires []`). Its discharge path is to commit the loop's `x`, `sx`, weight rows and `sw` as
anchors (hm96 rows in frame-v3-sha512 trees) before the window's draw coin, and to bind the H1 tiles' rows to them. A
named assumption in the lock: a protocol-code fact to prove. -/
def RegisteredBeforeDraw {C Ξ : Type} {m Np n : ℕ} (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) : Prop :=
  ∀ (x z : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n), ws.reg (spliceAt w.val m x z) w = ws.reg x w

/-! ## The period's conclusion at the check's accepted prefix -/

/-- **(L) The circuit's work on its live tiles over the windows PoUW's deployed check accepts, live coins.** For every
declared circuit `Cd` and scheme `Sc` run as a period of at most `n ≥ 1` windows, each window's registration and the
proof service's answer any function of the period's coins (`Executions`), with the check's parameters `prm`: under
`CircuitWorkPeriod`'s premises at the period's proofs with the check's verdict (`checkedRun`, `checkedLayer`), its
`JointSound` replaced by the check's draw being the period's draw coin (`DrawIsCoin`) and the service's soundness
against the check (`ServiceSound`), with each registration fixed before its window's draw coin
(`RegisteredBeforeDraw`), and at `K_d > 0`, an accepted period (`Pl.accepts`) has, over the windows before
the check's first refusal (`Executions.prefixLen`, `j` of them), all but `(d_F + d₀)·N_w` of their live words
(`checkedLive`) fixed right and all but `d₀·N_w` folded, with `a·d_F + b·d₀ < Λ·ln(1/δ_s) + j·min(a, b)·D₀`
(`LiveWorkAt` at `rOfPrefix`), except with probability `δ_s + n·(δ_bind + δ_ks)`. -/
def CheckedWorkPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (_hKd : 0 < Kd) (prm : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
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
    RegisteredBeforeDraw ws → DrawIsCoin hK prm ws → ServiceSound prm ws opened rel δks →
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
`(1 − ε(j))₊` of the live words of the windows before the check's first refusal (`creditPrefix` at `Executions.prefixLen`
and `checkedLive`), `CircuitCreditPeriod`'s conclusion at the check's prefix. -/
def CheckedCreditPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (_hKd : 0 < Kd) (prm : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
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
    RegisteredBeforeDraw ws → DrawIsCoin hK prm ws → ServiceSound prm ws opened rel δks →
    (∀ x w j v, rel x w j v → right x w j v) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (checkedLayer Cd hr hK prm ws opened rel right)
      (periodMode Cd Wp (checkedRun hK prm ws opened rel right)) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd Np (ws.prefixLen prm x) (checkedLive Sc Cd Wp prm ws x) a b Λ δs) ≤ δs + n * (δbind + δks)

/-! ## H1's tile unit on the torch loop -/

/-- **Each window's salt is its live coin** (H1's instance). On every path, window `w`'s salt (`salt x w`: the 64-byte
salt the verifier holds for the window, which every tile's noise reads through its public row, `PouwCheck.publicBc`, and
the proof of every unit in the sample is held to) is `enc` of the window's round-0 coin, the model's coin, released
after the round's values are committed. The check holds a live window to it: `PouwCheck.windowSaltReason` accepts a
live window's salt only as its salt session's live coin (`Properties.Pouw.Check.SaltKept`), opened after the window's
committed activations (`anchoredEpochRoot W X k`). The served windows don't meet it yet: no path that serves H1 windows
takes its salt from such a session, and the proved tiles' salt port is a value the prover records. A named assumption
in the lock: a deployment gap until the salt is a live coin from the auditor's challenger and the proved tiles are tied
to it. -/
def SaltIsLiveCoin {C B : Type} {m n : ℕ} (hm : 0 < m) (salt : Paths (Paths C m × B) n → Fin n → ByteArray)
    (enc : C → ByteArray) : Prop :=
  ∀ x w, salt x w = enc (coinAt (coinAt x w).1 ⟨0, hm⟩)

/-- **The staged statement pins the flat tile**: at every salt `s`, values of a unit `t` that satisfy the statement
the service proves of it (`staged s t`, the window's staged statement, which the leg's proofs are of) satisfy the
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
`CheckedCreditPeriod` at the torch loop's window (module docstring): one round a window (`Cd.rounds = 1`,
`w_max = 1`, every tile's noise reads the window's one salt), a population of `N_p = 144` of which `K_d = 24` are
drawn, every tile's share of the words at most `1/128` (`ρ = 8/9`: 128 equal real tiles, the 16 fillers wordless),
H1's lucks `p = 7/8`, `p0 = ½`, the word line `a = b = 1` and `SplitRateClosed`'s rate `Λ = 67/8`. The unit's relation
is the staged statement at the window's salt (`stagedAt`), which `ServiceSound` is about, and its rightness the flat
tile's at the window's coin (`flatAtCoin`); `Pins` is replaced by the window's salt being its live coin
(`SaltIsLiveCoin`) and the staged statement pinning the flat tile (`StagedPins`), with each registration fixed before
its window's draw coin (`RegisteredBeforeDraw`). Then, except with probability
`δ_s + n·(δ_bind + δ_ks)`, an accepted period's fixed words are at least `(1 − ε(j))₊` of the live words of the
windows before the check's first refusal, `ε(j) = (67/8)·ln(1/δ_s)·N_w/N_live` (`creditPrefix`). -/
def H1CheckedCreditPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ 144)
    (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : Params)
    (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (salt : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ByteArray) (enc : C → ByteArray)
    (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) (Fin 144) Cd.rounds n)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ) (δs δbind δks : ℝ),
    Scope Sc Cd δs →
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Sc Cd 144 t ≤ 1) →
    (∀ j, stageShare (tileStage Cd 144) (shareCone Sc Cd 144) j ≤ 1) →
    (∀ t : Fin 144, 0 < shareCone Sc Cd 144 t → tileLuck Sc Cd 144 t ≤ 7 / 8 ∧ tileLuck0 Sc Cd 144 t ≤ 1 / 2) →
    Pl.game = periodGame (tileStage Cd 144) (liveCone Sc Cd 144) 1 1 24 Wp →
    Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd 144 live) →
    PrefixOf (checkedLayer Cd hr hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat)) Pl.game →
    (∀ w, prob (fun x =>
      (checkedLayer Cd hr hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat)).MovedIn x w) ≤ δbind) →
    RegisteredBeforeDraw ws → DrawIsCoin hK prm ws → ServiceSound prm ws opened (stagedAt salt staged) δks →
    SaltIsLiveCoin hr salt enc → StagedPins staged flat →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (checkedLayer Cd hr hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat))
      (periodMode Cd Wp (checkedRun hK prm ws opened (stagedAt salt staged) (flatAtCoin hr enc flat)))
      (fun _ t => (tileWords Sc Cd 144 t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd 144 (ws.prefixLen prm x) (checkedLive Sc Cd Wp prm ws x) 1 1 (67 / 8) δs) ≤
        δs + n * (δbind + δks)

end Properties.Pouw.Period
