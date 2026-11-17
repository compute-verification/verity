module

public import Definitions.Pouw.RelativeSecurity
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

/-! Per-stage coins with SPOT, STEER-PROOF's route A: stage `c`'s coin is released after stage `c`'s operands are committed. -/

@[expose] public section

namespace ResourceAccounting.TS

open Pouw.RelativeSecurity

/-- **A round's stages under per-stage coins**: per stage, the units the cheater skips and the units whose proof
passes, as functions of the stage's coin `C` (the stage's outputs are computed after it is released), then the rest of
the round, which may depend on it. -/
inductive CoinStages (C U : Type) : ℕ → Type
  | done : CoinStages C U 0
  | stage {m : ℕ} (skip ok : C → U → Prop) (next : C → CoinStages C U m) : CoinStages C U (m + 1)

namespace CoinStages

variable {C U : Type}

/-- Unit `u` is skipped along the round's coins: the stage it belongs to (`stg u`, stage `m` the one with `m` stages
after it) skips it. -/
def skipped (stg : U → ℕ) : {m : ℕ} → CoinStages C U m → Paths C m → U → Prop
  | _, .done, _, _ => False
  | _, .stage (m := m) skip _ next, x, u => (stg u = m ∧ skip x.1 u) ∨ skipped stg (next x.1) x.2 u

/-- Unit `u`'s proof passes along the round's coins: its stage's check holds. -/
def passes (stg : U → ℕ) : {m : ℕ} → CoinStages C U m → Paths C m → U → Prop
  | _, .done, _, _ => True
  | _, .stage (m := m) _ ok next, x, u => (stg u = m → ok x.1 u) ∧ passes stg (next x.1) x.2 u

/-- **(L) at `p`, per unit at its stage's coin**: at every stage, whatever the past, a unit of the stage that is
skipped passes with probability at most `p` over the stage's coin, `Pr[skip_u ∧ ok_u] ≤ p·Pr[skip_u]`. -/
def LuckAt [Fintype C] (p : ℝ) (stg : U → ℕ) : {m : ℕ} → CoinStages C U m → Prop
  | _, .done => True
  | _, .stage (m := m) skip ok next =>
      (∀ u, stg u = m → prob (fun c => skip c u ∧ ok c u) ≤ p * prob (fun c => skip c u)) ∧
        ∀ c, LuckAt p stg (next c)

end CoinStages

/-- **A window of sampled rounds under per-stage coins**: per round, its `M` stages, then the rest of the window, which
may depend on the round's stage coins and its `s` samples (tickets on `D`). -/
inductive StageExecutionPhase (C U D : Type) (M s : ℕ) : ℕ → Type
  | done : StageExecutionPhase C U D M s 0
  | round {n : ℕ} (T : CoinStages C U M) (next : Paths C M × (Fin s → D) → StageExecutionPhase C U D M s n) :
      StageExecutionPhase C U D M s (n + 1)

namespace StageExecutionPhase

variable {C U D : Type} {M s : ℕ}

/-- (L) at `p` per unit in every round (`CoinStages.LuckAt`). -/
def LuckAt [Fintype C] (p : ℝ) (stg : U → ℕ) : {n : ℕ} → StageExecutionPhase C U D M s n → Prop
  | _, .done => True
  | _, .round T next => T.LuckAt p stg ∧ ∀ a, LuckAt p stg (next a)

/-- **The window's ideal game**: round `w`'s share is the work weights `w` of its skipped units, and its check passes
when the proof of every unit its samples name (`tk`) passes. -/
noncomputable def game [Fintype U] (stg : U → ℕ) (tk : D → U) (w : U → ℝ) :
    {n : ℕ} → StageExecutionPhase C U D M s n → RoundGame (Paths C M × (Fin s → D)) n
  | _, .done => .done
  | _, .round T next =>
    .round (fun a => ∑ u, (by classical exact if T.skipped stg a.1 u then w u else 0))
      (fun a => ∀ i, T.passes stg a.1 (tk (a.2 i))) fun a => (next a).game stg tk w

end StageExecutionPhase

structure SampledRound (A D : Type) where
  share : A → ℝ
  hit : A → D → Prop
  ok : A → D → Prop

inductive SampledExecutionPhase (A D : Type) (s : ℕ) : ℕ → Type
  | done : SampledExecutionPhase A D s 0
  | round {n : ℕ} (R : SampledRound A D) (next : A × (Fin s → D) → SampledExecutionPhase A D s n) :
      SampledExecutionPhase A D s (n + 1)

namespace SampledExecutionPhase

variable {A D : Type} {s : ℕ}

def game : {n : ℕ} → SampledExecutionPhase A D s n → RoundGame (A × (Fin s → D)) n
  | _, .done => .done
  | _, .round R next => .round (fun a => R.share a.1) (fun a => ∀ i, R.ok a.1 (a.2 i)) fun a => (next a).game

def LuckAt [Fintype A] [Fintype D] (p : ℝ) : {n : ℕ} → SampledExecutionPhase A D s n → Prop
  | _, .done => True
  | _, .round R next =>
      prob (fun z : A × D => R.hit z.1 z.2 ∧ R.ok z.1 z.2) ≤ p * prob (fun z : A × D => R.hit z.1 z.2) ∧
        ∀ a, LuckAt p (next a)

def Priced [Fintype D] (r : ℝ) : {n : ℕ} → SampledExecutionPhase A D s n → Prop
  | _, .done => True
  | _, .round R next => (∀ a, 0 ≤ R.share a ∧ R.share a ≤ r * prob (R.hit a)) ∧ ∀ a, Priced r (next a)

end SampledExecutionPhase

end ResourceAccounting.TS
