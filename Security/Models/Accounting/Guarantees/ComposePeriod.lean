import Definitions.Accounting.ComposePeriod
import Models.Accounting.Assumptions.Compose

/-!
# PoUW's credit over an accounting period of `n` windows (trusted)

INT-PERIOD's statement (note:compute-accounting/20261019T0739Z-draft-int-period), RT-INT's fix (c)
(note:compute-accounting/20261019T0528Z-draft-rt-int): `CircuitWork` restated over a period of `n` windows of one
declared circuit, each window with its own live coins and its own draw. It changes what the live guarantee says, so
it is for Daniel to take or leave (the card in the note), and it is stated, not proved: the red team and the card come
first.

RT-INT-1: `CircuitWork`'s quoted live `ε` (0.744%, #2638; 0.656%, #2482) is for a window of 48,000 equal live rounds.
Under marginal per-tile luck `ε < 1%` needs at least 10,552 equal live rounds a window, and SERVE-H1's windows have 18.
Over a period the rounds are the period's, `n·M`, so served windows reach it by being many.

* `PeriodBudget`: the composition lemma. If every window's `serviceGame` has its saving per unit of detection bounded by
  `Λ` (`RoundGame.BudgetAt`, which `ServiceBudget` gives per window), whatever the earlier windows did, so does the
  period's game (`periodGame`). Proved (`Proofs.Accounting.ComposePeriod`): a window's game is one round, and
  `BudgetAt` over rounds is that condition at each round given the past.
* `CircuitWorkPeriod`: (L) over the period. Not proved; it follows from `CircuitWork`'s lemmas, not from its
  conclusion (a per-window tail at `δ` can't be averaged into `ε/n`): `ServiceBudget` per window, `PeriodBudget`,
  `budget_tail` once at `t = ln(1/δ_s)`, `DrawnRightOff` at `n` rounds, the period layer's `Covers` and `WriterReads`
  (`ServiceCovers`' argument with a window index) and `UnitFixed`'s per-window averages.

`PeriodTailMax`, the maximal form of `budget_tail` (Ville's inequality), which would credit an accepted prefix of a
period that stops early, is named in the note and not stated.
-/

namespace Accounting.Guarantees.ComposePeriod

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The period's budget from its windows'.** For a period of `n` windows of one circuit of `M` rounds over `N` tiles,
any play (`Period`: each window's split stages chosen after the earlier windows' coins), share weights `w`, costs
`a, b` and `K` draws a window: if every window's one round (`serviceGame`) has its saving per unit of detection
bounded by `Λ` (`RoundGame.BudgetAt`), whatever the past, then so has the period's game (`periodGame`). With
`budget_tail` at `n` rounds, an accepted period's total share is below `Λ·ln(1/δ_s)` except with probability `δ_s`. -/
def PeriodBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M n : ℕ} (stg : Fin N → ℕ) (w : Fin N → ℝ) (a b : ℝ) (K : ℕ) (Λ : ℝ)
    (Wp : Period C (Paths C M × (Equiv.Perm (Fin N) × Ξ)) (Fin N) M n),
    Wp.EveryWindow (fun W => RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W w a b K)) →
    RoundGame.BudgetAt Λ (periodGame stg w a b K Wp)

/-- **(L) The circuit's work over an accounting period, live coins.** For every declared circuit `Cd` and scheme `Sc`,
run as a period of `n ≥ 1` windows in order, each window `CircuitWork`'s: the circuit's rounds in order, each round's
noise coin live and released after its operands are committed, then the service's draw of `K_d` of the `N_p ≥ units`
tiles (`drawOf`) with the proofs' coins `Ξ`, all fresh; window `w + 1`'s first coin is released after window `w`'s
draw. The cheater's play in each window may be any function of every earlier window's coins (`Period`). Under:
* `CircuitWork`'s premises on the circuit, unchanged: `Scope`, `ρ`, `w_max`, the lucks `p, p0`, `a, b ≥ 0`, `Λ > 0` and
  `SplitRate` at them;
* the period's ideal game the windows' `serviceGame`s at share `1` per unit of each skipped tile's late cone
  (`periodGame` at `tileCone`);
* sampled (I8)'s pieces on the period's proofs, `Declared.layer` in every window (`Declared.periodLayer` at the run's
  `pr`): `Of`, binding at `δ_bind` in each window (`MovedIn`), proof soundness jointly over each window's draws at
  `δ_ks` (`JointSound`) and `Pins`;
* the four claims: (H-μ), B′ and the guess bound per credited tile of every window, whatever the past
  (`Period.EveryWindow` of `PerTile`), and `UnitFixed` on the period's proofs at the run's modes (`periodMode`) and
  the tiles' words, whose fixed and folded words are per-window averages (`/n`);

an accepted period (every drawn tile of every one of its `n` windows accepted) has, averaged over its windows, all but
a share `d_F + d₀` of a window's committed words fixed right and all but `d₀` folded, with
`a·d_F + b·d₀ < Λ·ln(1/δ_s)/n + min(a, b)·D₀` (`WorkAt` at `rOfPeriod`), except with probability
`δ_s + n·(δ_bind + δ_ks)`. At `n = 1` it is `CircuitWork` with positions tagged by window `0`.

Its line, with equal rounds (`w_max = 1/M`), `ρ = 1/r` (`F = 1`), `x = K_d/(r·M)`: `Λ` is `SplitRateExp`'s,
`max(a/(M·−ln(p + (1 − p)e^{−x})), b/(M·−ln(p0 + (1 − p0)e^{−x})))`, and the logical term is `Λ·ln(1/δ_s)/n`. At
`a = 0.218`, `b = 0.998`, `p = 7/8`, `p0 = ½`, `δ_s = 2^-40` and `ε_♭ = 0.571%` (the sm_120 sheet, `ε_μ = 0`, `D₀ = 0`),
SERVE-H1's window shape (`M = 18`, `r = 2.87`) with W2's draw (`K_d = 8`) has `Λ = 0.745`, so `ε` is 115% at
`n = 18`, 2.64% at 1,000 and below 1% from `n = 4,815` windows (86,670 live rounds, 38,520 drawn tiles): 24.5 days at
the measured 439 s a window (r20261018-234409-19cd). For small `x` the term is about `2b·r·ln(1/δ_s)/(n·K_d)`: the
period's total draws `n·K_d` set it, not how the period is cut into windows.

What it doesn't say: no window's own `ε` (an accepted period may hold about 20 windows skipped whole, `Λ·ln(1/δ_s)` of
share in all); no attribution inside the period (one certificate for the period's device interval); no credit for a
period that stops early (`n` is fixed before the period, and a window that fails fails the period, with no retry); no
link between windows beyond each window's inputs, cut where its IR cuts them; and, as `CircuitWork`, nothing of a
served workload until a `Declared` is built from a compiled Program and `Properties.Compose.UnitsOfLayout` is proved. -/
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
    Pl.game = periodGame (tileStage Cd Np) (tileCone Sc Cd Np) 1 1 Kd Wp →
    (Cd.periodLayer hr hK pr).Of Pl → (∀ w, prob (fun x => (Cd.periodLayer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.periodLayer hr hK pr).JointSound δks → (Cd.periodLayer hr hK pr).Pins →
    Wp.EveryWindow (fun W => PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun W => PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun W => PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (Cd.periodLayer hr hK pr) (periodMode Cd Wp) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (rOfPeriod Sc Cd Np n a b Λ δs) (Pl.run x).fixed (folded x)) ≤
        δs + n * (δbind + δks)

end Accounting.Guarantees.ComposePeriod
