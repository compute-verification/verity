import Definitions.Accounting.ComposePeriod
import Models.Accounting.Assumptions.Compose

/-!
# PoUW's work over an accounting period of `n` windows (trusted)

INT-PERIOD's statement (note:compute-accounting/20261019T0739Z-draft-int-period), RT-INT's fix (c)
(note:compute-accounting/20261019T0528Z-draft-rt-int): `CircuitWork` restated over a period of `n` windows of one
declared circuit, each window with its own live coins and its own draw, its words counted on its live tiles only. It
changes what the live guarantee says, so it is for Daniel to take or leave (the card in the note), and it is stated,
not proved: the red team and the card come first.

RT-INT-1: both live lines are quoted at 48,000 equal live rounds, and each has its own round floor for `ε < 1%`. On
Compose's line (`CircuitWork`, 0.744% at #2638, `a = 0.218`, `b = 0.998`) it is at least 10,552 equal live rounds a
window, under marginal per-tile luck. On #2482's sm_120 bound (0.656% sheet, 0.591% measured) it is 9,496 rounds (sheet)
and 9,652 (measured) (HEADLINE-FIX, note:compute-accounting/20261019T0730Z-draft-headline-fix). SERVE-H1's windows
have 18. Over a period the rounds are the period's, `n·M`, so served windows reach it by being many. This statement
is Compose's line, and every number below is on it.

RT2-INT-1 (note:compute-accounting/20261019T1100Z-draft-rt2-int): on SERVE-H1's masked unit (`PouwH1Tile_v2`, #2639)
a dead tile, whose rows are all the canonical empty row, has the canonical empty receipt at every coin, so its whole
skip always passes and no guess bound holds of it; 52.8% of a served window's units are dead (r20261019-012935-2c52).
So each window names its live tiles (`Period`), the per-tile claims are asked of live tiles only (`PerLiveTile`), and
a dead tile's words count for nothing (`liveCone`, `periodMode`, `periodLive`).

* `PeriodBudget`: the composition lemma. If every window's `serviceGame` has its saving per unit of detection bounded by
  `Λ` (`RoundGame.BudgetAt`, which `ServiceBudget` gives per window), whatever the earlier windows did, so does the
  period's game (`periodGame`). Proved (`Proofs.Accounting.ComposePeriod`): a window's game is one round, and
  `BudgetAt` over rounds is that condition at each round given the past.
* `CircuitWorkPeriod`: (L), the circuit's work on live tiles, over the period; not its credit (`CircuitCredit`'s
  `(1 − ε)·T_cred`), which no statement here gives. Not proved; it follows from `CircuitWork`'s lemmas, not from its
  conclusion (a per-window tail at `δ` can't be averaged into `ε/n`): `ServiceBudget` per window at the window's
  detected live weights, `PeriodBudget`, `budget_tail` once at `t = ln(1/δ_s)`, `DrawnRightOff` at `n` rounds, the
  period layer's `Covers` on live tiles and `WriterReads` (`ServiceCovers`' argument with a window index) and
  `UnitFixed`'s per-window averages.

`PeriodTailMax`, the maximal form of `budget_tail` (Ville's inequality), which would credit an accepted prefix of a
period that stops early, is named in the note and not stated.
-/

namespace Accounting.Guarantees.ComposePeriod

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The period's budget from its windows'.** For a period of `n` windows of one circuit of `M` rounds over `N` tiles,
any play (`Period`: each window's live tiles and split stages chosen after the earlier windows' coins), share weights
`w live` at each window's live tiles `live`, costs `a, b` and `K` draws a window: if every window's one round
(`serviceGame`) has its saving per unit of detection bounded by `Λ` (`RoundGame.BudgetAt`), whatever the past, then so
has the period's game (`periodGame`). With `budget_tail` at `n` rounds, an accepted period's total share is below
`Λ·ln(1/δ_s)` except with probability `δ_s`. -/
def PeriodBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M n : ℕ} (stg : Fin N → ℕ) (w : (Fin N → Bool) → Fin N → ℝ) (a b : ℝ)
    (K : ℕ) (Λ : ℝ) (Wp : Period C (Paths C M × (Equiv.Perm (Fin N) × Ξ)) (Fin N) M n),
    Wp.EveryWindow (fun live W => RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W (w live) a b K)) →
    RoundGame.BudgetAt Λ (periodGame stg w a b K Wp)

/-- **(L) The circuit's work on its live tiles over an accounting period, live coins.** For every declared circuit
`Cd` and scheme `Sc`, run as a period of `n ≥ 1` windows in order, each window `CircuitWork`'s: the circuit's rounds
in order, each round's noise coin live and released after its operands are committed, then the service's draw of
`K_d` of the `N_p ≥ units` tiles (`drawOf`, uniform over every tile, dead ones included) with the proofs' coins `Ξ`,
all fresh; window `w + 1`'s first coin is released after window `w`'s draw. The cheater's play in each window, its
live tiles included, may be any function of every earlier window's coins (`Period`). Under:
* `CircuitWork`'s premises on the circuit, unchanged: `Scope`, `ρ`, `w_max`, the lucks `p, p0`, `a, b ≥ 0`, `Λ > 0` and
  `SplitRate` at them. They read the static `shareCone`, which bounds every window's live weights, so none reads the
  load;
* the period's ideal game the windows' `serviceGame`s at share `1` per unit of each skipped tile's live late cone
  (`periodGame` at `liveCone`);
* in every window, no dead tile's late cone holds live words (`DeadConeDead`, which every Program COMPILE accepts has),
  so a dead tile's `liveCone` is `0`;
* sampled (I8)'s pieces on the period's proofs, `Declared.layer` in every window (`Declared.periodLayer` at the run's
  `pr`): `Of`, binding at `δ_bind` in each window (`MovedIn`), proof soundness jointly over each window's draws at
  `δ_ks` (`JointSound`) and `Pins`;
* the four claims: (H-μ), B′ and the guess bound per live credited tile of every window, whatever the past
  (`Period.EveryWindow` of `PerLiveTile`), and `UnitFixed` on the period's proofs at the run's modes with each dead
  tile skipped whole (`periodMode`) and the tiles' words. `UnitFixed` at `periodMode` follows from `UnitFixed` at the
  windows' own modes, since reading a tile as skipped whole only shrinks both its sums;

an accepted period (every drawn tile of every one of its `n` windows accepted) has, averaged over its windows, all
but `(d_F + d₀)·N_w` of a window's live words fixed right and all but `d₀·N_w` folded, `N_w` the declared circuit's
words, with `a·d_F + b·d₀ < Λ·ln(1/δ_s)/n + min(a, b)·D₀` (`LiveWorkAt` at `periodLive` and `rOfPeriod`), except with
probability `δ_s + n·(δ_bind + δ_ks)`. At `n = 1` with every tile live it is `CircuitWork` with positions tagged by
window `0`.

*Which run, and what is credited* (RT2-INT-3). `Pl.run` is the period's one run. `UnitFixed` bounds its `fixed` and
`folded x` below by the period's good and unskipped words over `n`, so the conclusion is about the period's mean
window: not the period's total, and not any one window. It states work, not credit: the credit `(1 − ε)·n·T_cred`
would be `CircuitCredit` over the period (`CircuitCreditPeriod`, not stated), whose sheet premises read one window's
run, not a mean.

*Live tiles on the served gate.* `live` is the gate's. SERVE-GATE's `PouwQueueAdmit_v3` (#2731,
`cursor/ca-serve-gate-ea24` at 166e5e573d) admits a slot iff the request queue, committed before the window's first
coin, assigns it a request live at its step and its row's scale is nonzero. Its always-proved admit unit refuses an
unadmitted slot whose row isn't the canonical empty row (`junk`), a selected slot whose row is (`hollow`, RT-ALGO-4)
and zero bytes under a nonzero scale (`scale_only`). So on an accepted window a tile is live iff its row block holds an
admitted slot, and that is fixed before the window's first coin. A `live` that names a dead tile live makes the guess
premise false (RT2-INT's `guess_false_dead`), and one that names a live tile dead only loses its words. The gate credits
admitted rows, which lie in live tiles, so as a share of the gate's credit the deficit is at most
`(d_F + d₀)·N_w/N_adm`, `N_adm/N_w = u` the period's admitted rows over its declared rows.

*The period's order* (RT2-INT-4). `Period` makes the order the model's. The service's record must show the period's
`n` and circuit committed before window 0's first coin, each window's first coin opened after the previous window's
draw, and `n` windows, each accepted, none replaced or dropped. No check of the record does this today. The
challenger's `order_reason` (`SampledProofs.Challenger.orderReason`, #2731, not on main) checks one family of an
epoch's coins: coin `i` opened once, after the `i`-th recorded root, and every recorded root's coin opened. That is the
order within a window. Run over one family holding every window's round coins and draw in order, with `n` and the
circuit in the epoch's commitment, it would be the period's check.

Its line, with equal rounds (`w_max = 1/M`), `ρ = 1/r` (`F = 1`), `x = K_d/(r·M)`: `Λ` is `SplitRateExp`'s,
`max(a/(M·−ln(p + (1 − p)e^{−x})), b/(M·−ln(p0 + (1 − p0)e^{−x})))`, and the logical term is `Λ·ln(1/δ_s)/n`, or
`Λ·ln(1/δ_s)/(u·n)` of the gate's credit. At `a = 0.218`, `b = 0.998`, `p = 7/8`, `p0 = ½`, `δ_s = 2^-40` and
`ε_♭ = 0.571%` (the sm_120 sheet, `ε_μ = 0`, `D₀ = 0`), SERVE-H1's window shape (`M = 18`, `r = 2.87`) with W2's draw
(`K_d = 8`) has `Λ = 0.745` (`Λ·ln(1/δ_s) = 20.65`). With every tile live (`u = 1`) `ε` is below 1% from 4,815
windows. At r20261019-012935-2c52's load it is below 1% of the live tiles' words (17/36 of units live) from 10,195
windows, and of the gate's credit (`u = 119/1152`, 10.3% of rows admitted) from 46,604 windows: 838,872 live rounds,
372,832 drawn tiles, 237 days at the measured 439 s a window (r20261018-234409-19cd, the masked circuit's own). For
small `x` the term is about `2b·r·ln(1/δ_s)/(u·n·K_d)`: the period's total draws `n·K_d` and its load set it, not how the
period is cut into windows.

What it doesn't say: no window's own `ε` (an accepted period may hold windows skipped whole, `Λ·ln(1/δ_s)` of
declared-word share in all, about 200 windows' worth of the gate's credit at `u = 0.103`); no attribution inside the
period (one certificate for the period's device interval); no credit for a period that stops early (`n` is fixed
before the period, and a window that fails fails the period, with no retry); no link between windows beyond each
window's inputs, cut where its IR cuts them; no check that the service's record ran the windows in order (above); and,
as `CircuitWork`, nothing of a served workload until a `Declared` is built from a compiled Program and
`Properties.Compose.UnitsOfLayout` is proved. -/
def CircuitWorkPeriod : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ Np)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n K T L U)
    (pr : PeriodRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → ℝ) (ρ p p0 a b wmax Λ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) →
    (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) →
    (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) →
    0 ≤ a → 0 ≤ b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = periodGame (tileStage Cd Np) (liveCone Sc Cd Np) 1 1 Kd Wp →
    Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd Np live) →
    (Cd.periodLayer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.periodLayer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.periodLayer hr hK pr).JointSound δks → (Cd.periodLayer hr hK pr).Pins →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (Cd.periodLayer hr hK pr) (periodMode Cd Wp) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ LiveWorkAt (periodLive Sc Cd Wp x) (wordsOf Sc Cd : ℝ) a b (rOfPeriod Sc Cd Np n a b Λ δs)
        (Pl.run x).fixed (folded x)) ≤
        δs + n * (δbind + δks)

end Accounting.Guarantees.ComposePeriod
