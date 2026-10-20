import Definitions.Accounting.ComposePeriod
import Models.Accounting.Assumptions.Compose

/-!
# PoUW's work over an accounting period of `n` windows (trusted)

INT-PERIOD's statement (note:compute-accounting/20261019T0739Z-draft-int-period), RT-INT's fix (c)
(note:compute-accounting/20261019T0528Z-draft-rt-int): `CircuitWork` restated over a period of `n` windows of one
declared circuit, each window with its own live coins and its own draw, its words counted on its live tiles only. It
changes what the live guarantee says, and Daniel took it (ASK-64 = A, 19 Oct: `ε` quoted per accounting period,
credited on live tiles up to the first refused window) after the red teams' rounds; its credit in live words is
`CircuitCreditPeriod` (RT4-INT-8, note:compute-accounting/20261019T2300Z-draft-rt4-int).

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
a dead tile's words count for nothing (`liveCone`, `periodMode`, `creditedLive`).

RT3-INT-7 (note:compute-accounting/20261019T1700Z-draft-rt3-int): an honest period of 46,604 windows completes with
probability `(1 − f)^n`, 63% at a per-window failure rate `f = 10^-5`, and dropping failed windows instead hides about
`0.26` of the live words per dropped share. So a period is credited up to its first window with a refused proof
(`PeriodRun.Credited`), a stopping time, and the tail is taken at it (`PeriodTailMax`).

* `PeriodBudget`: the composition lemma. If every window's `serviceGame` has its saving per unit of detection bounded by
  `Λ` (`RoundGame.BudgetAt`, which `ServiceBudget` gives per window), whatever the earlier windows did, so does the
  period's game (`periodGame`). Proved (`Proofs.Accounting.ComposePeriod`): a window's game is one round, and
  `BudgetAt` over rounds is that condition at each round given the past.
* `PeriodTailMax`: `budget_tail` at every prefix at once (Ville's inequality, finite form): when `Λ` bounds the play,
  some prefix of its rounds is accepted with share at least `Λ·t` with probability at most `e^{−t}`. So a prefix
  that ends at a stopping time the cheater steers pays `ln(1/δ_s)` once, as a fixed one does. Proved
  (`Proofs.Accounting.ComposePeriod`): by induction on the game, conditioning on the first round's coins as
  `expect_weight_le` does, with the bound for every threshold `D` in the induction.
* `CircuitWorkPeriod`: (L), the circuit's work on live tiles, over the period's accepted prefix. Proved
  (`Proofs.Accounting.ComposePeriod`) from `CircuitWork`'s lemmas, not from its conclusion (a per-window tail at `δ`
  can't be averaged into `ε/n`): `ServiceBudget` per window at the window's detected live weights, `PeriodBudget`,
  `PeriodTailMax` once at `t = ln(1/δ_s)`, `DrawnRightOff` at `n` rounds, the period layer's `Covers` on the credited
  windows' live tiles and `WriterReads` (`PeriodCoversStmt`, `ServiceCovers`' argument with a window index, proved as
  `periodCovers` in `Proofs.Accounting.ComposePeriod.Covers`) and `UnitFixed`'s sums.
* `CircuitCreditPeriod`: (L)'s credit in live words, `(1 − ε(j))₊` of the accepted prefix's live words with `ε` at the
  realized prefix `j` (`creditPrefix`). Proved (`Proofs.Accounting.ComposePeriod`) from `CircuitWorkPeriod`'s
  assembly at `a, b > 0`. Not the time credit (`CircuitCredit`'s `(1 − ε)·T_cred`), which no statement here gives.
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

/-- **SAMPLING's budget at every prefix at once** (Ville's inequality, finite form): for any play of `n` rounds on
coins `Ω` with its saving per unit of detection bounded by `Λ > 0` (`RoundGame.BudgetAt`), the probability that some
prefix of its rounds passes (`RoundGame.acceptsUpTo`) with share at least `Λ·t` (`RoundGame.deficiencyUpTo`) is at most
`e^{−t}`. Its last prefix is `budget_tail`. A period credited up to its first failed window, a stopping time the
cheater may steer, reads it at that window, so it pays `ln(1/δ_s)` once, and neither a fixed `n` nor a rule against
dropping windows is needed for the tail. -/
def PeriodTailMax : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {n : ℕ} (g : RoundGame Ω n) (Λ : ℝ), 0 < Λ → RoundGame.BudgetAt Λ g → ∀ t : ℝ,
    prob (fun x => ∃ j : ℕ, RoundGame.acceptsUpTo g j x ∧ Λ * t ≤ RoundGame.deficiencyUpTo g j x) ≤ Real.exp (-t)

/-- **(L) The circuit's work on its live tiles over an accounting period's accepted prefix, live coins.** For every
declared circuit `Cd` and scheme `Sc`, run as a period of at most `n ≥ 1` windows in order, each window
`CircuitWork`'s: the circuit's rounds in order, each round's noise coin live and released after its operands are
committed, then the service's draw of `K_d` of the `N_p ≥ units` tiles (`drawOf`, uniform over every tile, dead ones
included) with the proofs' coins `Ξ`, all fresh; window `w + 1`'s first coin is released after window `w`'s draw. The
cheater's play in each window, its live tiles included, may be any function of every earlier window's coins
(`Period`). The period is credited up to its first window with a refused proof (`PeriodRun.Credited`): its accepted
prefix, of `j = PeriodRun.prefixLen` windows. Under:
* `CircuitWork`'s premises on the circuit, unchanged: `Scope`, `ρ`, `w_max`, the lucks `p, p0`, `a, b ≥ 0`, `Λ > 0` and
  `SplitRate` at them. They read the static `shareCone`, which bounds every window's live weights, so none reads the
  load;
* the period's ideal game the windows' `serviceGame`s at share `1` per unit of each skipped tile's live late cone
  (`periodGame` at `liveCone`);
* in every window, no dead tile's late cone holds live words (`DeadConeDead`, which every Program COMPILE accepts has),
  so a dead tile's `liveCone` is `0`;
* sampled (I8)'s pieces on the period's proofs, `Declared.layer` in every window (`Declared.periodLayer` at the run's
  `pr`): `Of`'s game clause on every prefix (`PrefixOf`), binding at `δ_bind` in each window (`MovedIn`), proof
  soundness jointly over each window's draws at `δ_ks` (`JointSound`) and `Pins`. Binding is per window, so the period
  composes the windows' binding; it takes the window's binding premise as I8-COIN restates it (RT3-LOGIC-1) when that
  lands;
* the four claims: (H-μ), B′ and the guess bound per live credited tile of every window, whatever the past
  (`Period.EveryWindow` of `PerLiveTile`), and `UnitFixed` on the period's proofs at the run's modes with each dead
  tile and every window past the accepted prefix skipped whole (`periodMode`) and the tiles' words. `UnitFixed` at
  `periodMode` follows from `UnitFixed` at the windows' own modes, since reading a tile as skipped whole only shrinks
  both its sums;

an accepted period (`Pl.accepts`, which may hold on every path: the record credits whatever prefix was accepted, the
empty one included) has, over its accepted prefix of `j` windows, all but `(d_F + d₀)·N_w` of the prefix's live words
fixed right and all but `d₀·N_w` folded, `N_w` the declared circuit's words a window, with
`a·d_F + b·d₀ < Λ·ln(1/δ_s) + j·min(a, b)·D₀` (`LiveWorkAt` at `creditedLive` and `rOfPrefix`, against `n·fixed` and
`n·folded`), except with probability `δ_s + n·(δ_bind + δ_ks)`. Per credited window the bound is
`Λ·ln(1/δ_s)/j + min(a, b)·D₀`. At `n = 1` with every tile live, on a path whose window is accepted, it is
`CircuitWork` with positions tagged by window `0`.

*Which run, and what is credited* (RT2-INT-3, RT3-INT-7). `Pl.run` is the period's one run. `UnitFixed` bounds its
`fixed` and `folded x` below by the accepted prefix's good and unskipped words over `n`, so the conclusion is about
the prefix's total (`n·fixed` against `creditedLive`), and divided by `j` about its mean credited window: not any one
window. Windows past the prefix count as skipped whole, their words neither credited nor charged, so a failure costs
the windows after it, not the period. The prefix ends at a stopping time the cheater may steer, which `PeriodTailMax`
prices: `ln(1/δ_s)` once, whatever `j` is, so ending a period early only shortens what the budget is spread over. It
states work, not credit. The credit in live words is `CircuitCreditPeriod`'s, `(1 − ε(j))₊` of the prefix's live words
at the realized `j`, never at the cap: undiscounted, an all-skip window that passes with `p0^K_d = 2^-8` earns 1.95
times an honest stream's credit per unit of work (RT4-INT-8). The time credit `(1 − ε)·j·T_cred` would be
`CircuitCredit` over the period, whose sheet premises read one window's run, not a prefix's total: not stated.

*Live tiles on the served gate.* `live` is the gate's. SERVE-GATE's `PouwQueueAdmit_v3` (#2731,
`cursor/ca-serve-gate-ea24` at 166e5e573d) admits a slot iff the request queue, committed before the window's first
coin, assigns it a request live at its step and its row's scale is nonzero. Its always-proved admit unit refuses an
unadmitted slot whose row isn't the canonical empty row (`junk`), a selected slot whose row is (`hollow`, RT-ALGO-4)
and zero bytes under a nonzero scale (`scale_only`). So on an accepted window a tile is live iff its row block holds an
admitted slot, and that is fixed before the window's first coin. A `live` that names a dead tile live makes the guess
premise false (RT2-INT's `guess_false_dead`), and one that names a live tile dead only loses its words. The gate credits
admitted rows, which lie in live tiles, so as a share of the gate's credit the deficit is at most
`(d_F + d₀)·N_w/N_adm`, `N_adm/N_w = u` the prefix's admitted rows over its declared rows.

*The period's order and its record* (RT2-INT-4, RT3-INT-13, RT4-INT-6). `Period` makes the order the model's. The
statement needs a period record, kept by the service, from which the credit is read: the period's cap `n` and circuit
committed before window 0's first coin; each window's first coin opened after the previous window's draw (the
challenger's `order_reason` over one family, below); and the record closed at the first window with a refused proof
or at the `n`-th window, every window up to its close recorded, none replaced or dropped. A stream's committed periods
must be disjoint: each window in at most one period, and one period open at a time, the next committed only after the
last closes. Two periods that share windows count those windows twice, and a prover that commits `P` overlapping
periods and claims whichever is accepted pays `P·δ_s`, not `δ_s`. No check of the record does this today. The served
path has no period: `Credits` (SALT-WIRE's, #2847) registers a window once when it is received, before its draw coin
is open or any proof, and nothing closes a stream at its first refused window, so the running ledger credits each
window on its own, which is dropping failed windows (*Liveness*, below). The challenger's `order_reason`
(`SampledProofs.Challenger.orderReason`, #2731, not on main) checks one family of an epoch's coins: coin `i` opened
once, after the `i`-th recorded root, and every recorded root's coin opened. That is the order within a window. Run
over one family holding every window's round coins and draw in order, with `n` and the circuit in the epoch's
commitment, it would be the period's check.

*One stream* (RT3-INT-10). `Period` is a sequence: each window's first coin is opened after the previous window's
draw. Windows run at once on several GPUs are not one period, so a period's windows are one serving stream's, and a
fleet runs one period per stream: adding GPUs doesn't shorten a period.

*Live coins only* (RT3-INT-2). Every number below is for live windows, and no served path runs one yet. A certified
window's noise comes from the certifier's salt, which reaches the prover before its operands are committed, so the
window is seeded, and Compose says every PoUW window it runs is seeded. #2654 (on main) has the live challenger sign
its coin commitment with an LMS key and the verifier read that record, the mechanism a live served window needs; no
served path uses it yet (it touches `challenger.py`, not `served_commit.py` or `serve.py`), and SERVE-H1's
`--coins live` challenger runs in the prover's process, so its coins are seeded too. The seeded pair has no period
form: its premium is per window, so it would not divide by `j`.

Its line, with equal rounds (`w_max = 1/M`), `ρ = 1/r` (`F = 1`), `x = K_d/(r·M)`: `Λ` is `SplitRateExp`'s,
`max(a/(M·−ln(p + (1 − p)e^{−x})), b/(M·−ln(p0 + (1 − p0)e^{−x})))`, and the logical term per credited window is
`Λ·ln(1/δ_s)/j`, or `Λ·ln(1/δ_s)/(u·j)` of the gate's credit. At `a = 0.218`, `b = 0.998`, `p = 7/8`, `p0 = ½`,
`δ_s = 2^-40` and `ε_♭ = 0.571%` (the sm_120 sheet, `ε_μ = 0`, `D₀ = 0`), SERVE-H1's window shape (`M = 18`,
`r = 2.87`) with W2's draw (`K_d = 8`) has `Λ = 0.745` (`Λ·ln(1/δ_s) = 20.65`). With every tile live (`u = 1`) `ε` is
below 1% from 4,815 credited windows. At r20261019-012935-2c52's load it is below 1% of the live tiles' words (17/36 of
units live) from 10,195 windows, and of the gate's credit (`u = 119/1152`, 10.3% of rows admitted) from 46,604
windows: 838,872 live rounds, 372,832 drawn tiles, 237 days of one stream at the measured 439 s a window
(r20261018-234409-19cd, the masked circuit's own). They are live windows, which no served path runs yet (above). For
small `x` the term is about `2b·r·ln(1/δ_s)/(u·j·K_d)`: the prefix's total draws `j·K_d` and its load set it, not how
it is cut into windows.

*Liveness* (RT3-INT-7). An honest stream whose windows fail independently at rate `f` (a late proof, a certifier or a
node fault; no rate is measured) completes a 46,604-window period with probability `(1 − f)^n`: 63% at `f = 10^-5`,
39% at `2·10^-5` and 0.95% at `10^-4`; half fail from `f = 1.5·10^-5` (`1.4·10^-4` at 4,815 windows). A failed
period is credited for its prefix and the stream opens the next, each paying `Λ·ln(1/δ_s)` once, so over a stream the
gate's logical term is `Λ·ln(1/δ_s)/(u·E[j])`, `E[j]` the mean credited prefix: under 1% of the gate's credit only
while `f < 1/46,604 = 2.1·10^-5`, with a cap of 62,743 windows at `f = 10^-5` (1.108% at a cap of 46,604). A window
that fails ends its period; dropping it and running on would hide about `0.26` of the live words per dropped share,
and a period picked from `N` windows after the fact would pay `ln C(N, n)` in place of `ln(1/δ_s)`.

*The whole δ* (RT3-INT-12). The period fails with probability `δ_s + n·(δ_bind + δ_ks)`, `n` the cap, and binding is
priced at one prover budget `T` (`BindingOfECR`): `|L|·2^-88` a window at `T = 2^80` and `|L|·2^-64` at `2^128`, `|L|`
from `2^20` to `2^30`. At `n = 46,604` (`2^15.5`) the binding term is `2^-52.5` to `2^-42.5` at `T = 2^80`, so
`δ_s + n·δ_bind` is `2^-40.0` to `2^-39.8`; at `T = 2^128` it is `2^-28.5` to `2^-18.5`, set by binding, not by `δ_s`
(`2^-31.8` to `2^-21.8` at 4,815). `δ_ks` has no number on main (`WindowEps`'s `max_c κ_c + coll` is unstated), and
`n·δ_ks` stays under `2^-40` at 46,604 windows only if `δ_ks ≤ 2^-55.5` a window.

What it doesn't say: no window's own `ε` (an accepted prefix may hold windows skipped whole, `Λ·ln(1/δ_s)` of
declared-word share in all, about 200 windows' worth of the gate's credit at `u = 0.103`); no attribution inside the
period (one certificate for the prefix's device interval); no credit for windows past the first failure (no retry,
and none dropped); no link between windows beyond each window's inputs, cut where its IR cuts them; no period over
windows run at once (above); no check that the service's record ran the windows in order or kept its periods disjoint
(above); nothing of seeded windows, which every served window is today (above); and, as `CircuitWork`, nothing of a
served workload until a `Declared` is built from a compiled Program and `Properties.Compose.UnitsOfLayout` is
proved. -/
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
    PrefixOf (Cd.periodLayer hr hK pr) Pl.game →
    (∀ w, prob (fun x => (Cd.periodLayer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.periodLayer hr hK pr).JointSound δks → (Cd.periodLayer hr hK pr).Pins →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (Cd.periodLayer hr hK pr) (periodMode Cd Wp pr) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧
      ¬ LiveWorkAt (creditedLive Sc Cd Wp pr x) (wordsOf Sc Cd : ℝ) a b
        (rOfPrefix Sc Cd Np (pr.prefixLen x) a b Λ δs) (n * (Pl.run x).fixed) (n * folded x)) ≤
        δs + n * (δbind + δks)

/-- **(L)'s credit in live words over an accounting period's accepted prefix, live coins** (RT4-INT-8,
note:compute-accounting/20261019T2300Z-draft-rt4-int). Under `CircuitWorkPeriod`'s premises with `a, b > 0`, an
accepted period has at least its credit `(1 − ε(j))₊·N_live` (`creditPrefix`) of its prefix's live words fixed right
(`n·fixed`), `N_live` the prefix's live words (`creditedLive`) and `ε(j) = r(j)·N_w/(min(a, b)·N_live)` at the realized
prefix `j = PeriodRun.prefixLen` (`epsPrefix`), except with probability `δ_s + n·(δ_bind + δ_ks)`. So a period may be
credited `(1 − ε(j))₊` of its prefix's live words, and a prefix's deficit is at most what the discount takes. `ε` is
read at the prefix the period realized, never at its cap `n`: a short prefix, whose `ε` is at least `1`, earns
nothing, and where the prefix ends, which the cheater may steer, is priced by `PeriodTailMax`.

It counts words, each one unit, so its best line is at `a = b = 1` (`r(j)/min(a, b)` is homogeneous in `(a, b)` and
least at `a = b`): `Λ = 3.070` at SERVE-H1's window shape with W2's draw (`Λ·ln(1/δ_s) = 85.1` windows of declared
words, `D₀ = 0`). Then `ε(j) ≥ 1` up to 85 windows with every tile live, 180 at r20261019-012935-2c52's live words
(17/36 of units) and 824 of the gate's credit (`u = 119/1152`), and `ε(j) < 1%` from 8,513, 18,026 and 82,403 windows.
The headline's line is the time credit, at the physical costs `a = 0.218`, `b = 0.998` with `ε_♭ = 0.571%`, where
`ε(j) ≥ 1` up to 20, 43 and 201 windows and `ε(j) < 1%` from 4,815, 10,195 and 46,604: that credit,
`(1 − ε_♭ − ε_μ − r(j)·N_w/N_live)₊` of the prefix's live words' `T_cred`, needs `CreditOfWork` on each credited
window's run, which the period's one `Pl.run` doesn't give, and is not stated. Everything `CircuitWorkPeriod` doesn't
say, this doesn't either: in particular no served path keeps the period record it reads (*The period's order and
its record*). -/
def CircuitCreditPeriod : Prop :=
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
    0 < a → 0 < b → 0 < Λ → SplitRate Kd ρ p p0 a b wmax Λ →
    Pl.game = periodGame (tileStage Cd Np) (liveCone Sc Cd Np) 1 1 Kd Wp →
    Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd Np live) →
    PrefixOf (Cd.periodLayer hr hK pr) Pl.game →
    (∀ w, prob (fun x => (Cd.periodLayer hr hK pr).MovedIn x w) ≤ δbind) →
    (Cd.periodLayer hr hK pr).JointSound δks → (Cd.periodLayer hr hK pr).Pins →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)) →
    Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) →
    UnitFixed Pl (Cd.periodLayer hr hK pr) (periodMode Cd Wp pr) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded →
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd Np (pr.prefixLen x) (creditedLive Sc Cd Wp pr x) a b Λ δs) ≤ δs + n * (δbind + δks)

end Accounting.Guarantees.ComposePeriod
