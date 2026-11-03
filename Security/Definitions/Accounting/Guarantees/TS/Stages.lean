import Definitions.Accounting.Guarantees.TS
import Definitions.Accounting.Assumptions.Stages

/-!
# Sampled proofs under per-stage coins (trusted)

STEER-PROOF's route A, each `def G : Prop` proved as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS.Stages`:

* `SampledBudget`: a sampled window with `s ≥ 2` draws a round whose share is at most `r` times the draws' mass on saved
  work, and whose draws are lucky at `p ∈ [1/e, 1)` (`Pr[hit ∧ ok] ≤ p·Pr[hit]`), has `Λ = r/ln(1/p)`
  (`RoundGame.BudgetAt`). The saving may depend on the coins before the draws, the luck's correlation across units is
  arbitrary, and no closure, LIN-STEER, STEER-1 or menu is read.
* `StageBudget`: the same for per-stage coins: under `StageLuck` at `p` and work weights at most `r` times the draw law,
  `Λ = r/ln(1/p)`.
* `CompilerStage`: Lemma C under per-stage coins: (2.7) with `ε_sample = (r − 1)·ε_jl`.
* `StageIdeal`: with work-weighted draws (`r = 1`), D-ideal's bound itself: `ε_sample = 0`, so the luck's term
  `ε_jl` is all sampling costs.

`s ≥ 2` is needed: at `s = 1` and `Λ = 1/ln(1/p)`, skipping a small draw mass `m` whose units are lucky together with
probability `p` gives `E[e^{d/Λ}·1{pass}] = p^{−m}·(1 − (1 − p)·m) > 1`. PoUW's staged bound
(`Pouw.RelativeSecurity.RoundGame.Staged`) covers one draw a round, at `Λ = 1/(1 − p)`, for skips fixed before the
draw.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **The sampled budget.** For every sampled window with `s ≥ 2` draws a round, if every round's share is at most `r`
times the probability that a draw lands on saved work (`SampledWindow.Priced r`) and the draws are lucky at
`p ∈ [1/e, 1)` (`SampledWindow.LuckAt p`), then `Λ = r/ln(1/p)` bounds the window's saving per unit of detection. -/
def SampledBudget : Prop :=
  ∀ {A D : Type} [Fintype A] [Fintype D] {s n : ℕ} (W : SampledWindow A D s n) (p r : ℝ),
    2 ≤ s → Real.exp (-1) ≤ p → p < 1 → 0 < r → W.Priced r → W.LuckAt p →
    RoundGame.BudgetAt (r / Real.log (1 / p)) W.game

/-- **The budget under per-stage coins.** For every window of rounds of `M` stages with per-stage coins and `s ≥ 2`
draws a round over whole units (tickets `tk`, any law with rational weights), with work weights `w ≥ 0` at most `r`
times each unit's draw probability, (L) at `p ∈ [1/e, 1)` per unit at its stage's coin (`StageLuck`) gives
`Λ = r/ln(1/p)`. -/
def StageBudget : Prop :=
  ∀ {C V D : Type} [Fintype C] [Fintype V] [Fintype D] {M s n : ℕ} (W : StageWindow C V D M s n) (stg : V → ℕ)
    (tk : D → V) (w : V → ℝ) (p r : ℝ),
    2 ≤ s → Real.exp (-1) ≤ p → p < 1 → 0 < r → (∀ u, 0 ≤ w u) → (∀ u, w u ≤ r * prob (fun t => tk t = u)) →
    StageLuck p stg W → RoundGame.BudgetAt (r / Real.log (1 / p)) (W.game stg tk w)

/-- **Lemma C under per-stage coins.** In 𝒫_samp with per-stage coins, `s ≥ 2` draws a round over whole units and work
weights at most `r` times the draw law, whose ideal game is the window's (`StageWindow.game`; its share is the
skipped units' work, the premise (I8) `ProofsSound` reads at that share), under (I1)–(I6) as in `DIdeal`, (I8) and
`StageLuck` at `p ∈ [1/e, 1)` in place of (I7): for every certificate `π`, outside an event of probability at most `δ`,
every accepted run satisfies (2.7) with `ε_sample = (r/ln(1/p) − 1/ln(1/p))·(ln(1/δ′)/S)·κ(π)` subtracted. -/
def CompilerStage : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X] {C V D : Type} [Fintype C] [Fintype V] [Fintype D] {M s : ℕ}
    (Z : Model K R T L U F X) (H : Honest K) (S : ℕ) (P : Play (Paths C M × (Fin s → D)) S K T L U)
    (W : StageWindow C V D M s S) (stg : V → ℕ) (tk : D → V) (w : V → ℝ) (lam : R → ℝ) (p r δ δbind δzk : ℝ)
    (nπ : ℕ),
    IsCert lam → Z.Wf H.prog → 0 < H.CH → 0 < H.fH → 0 < S → 2 ≤ s → Real.exp (-1) ≤ p → p < 1 → 0 < r →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → P.game = W.game stg tk w → (∀ u, 0 ≤ w u) →
    (∀ u, w u ≤ r * prob (fun t => tk t = u)) → StageLuck p stg W → ProofsSound P Z.Nw δbind δzk nπ →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / H.time <
      (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
        Z.epsJl H.prog lam p (δ - S * (δbind + nπ * δzk)) S -
        Z.epsSample H.prog lam (r / Real.log (1 / p)) p (δ - S * (δbind + nπ * δzk)) S) *
        (1 - Z.epsSched H) * (1 - Z.epsClock H)) ≤ δ

/-- **D-ideal under per-stage coins and sampled proofs.** As `CompilerStage` with work-weighted draws (each unit's work
weight at most its draw probability): every accepted run satisfies (2.7), D-ideal's bound, with no `ε_sample`. -/
def StageIdeal : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X] {C V D : Type} [Fintype C] [Fintype V] [Fintype D] {M s : ℕ}
    (Z : Model K R T L U F X) (H : Honest K) (S : ℕ) (P : Play (Paths C M × (Fin s → D)) S K T L U)
    (W : StageWindow C V D M s S) (stg : V → ℕ) (tk : D → V) (w : V → ℝ) (lam : R → ℝ) (p δ δbind δzk : ℝ)
    (nπ : ℕ),
    IsCert lam → Z.Wf H.prog → 0 < H.CH → 0 < H.fH → 0 < S → 2 ≤ s → Real.exp (-1) ≤ p → p < 1 →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → P.game = W.game stg tk w → (∀ u, 0 ≤ w u) →
    (∀ u, w u ≤ prob (fun t => tk t = u)) → StageLuck p stg W → ProofsSound P Z.Nw δbind δzk nπ →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / H.time <
      (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
        Z.epsJl H.prog lam p (δ - S * (δbind + nπ * δzk)) S) * (1 - Z.epsSched H) * (1 - Z.epsClock H)) ≤ δ

end Accounting.Guarantees.TS
