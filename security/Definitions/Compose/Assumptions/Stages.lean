module

public import Definitions.Pouw.TS.Stages
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

/-! Per-stage coins' named assumption `StageLuck`, which is false under one σ for a steered consumer. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.TS

open ResourceAccounting.TS Pouw.RelativeSecurity

/-- `accounting-ts-stage-luck` ((L) per unit, per-stage coins). **(L) at `p` under per-stage coins**: in every round,
at every stage, whatever the past, a unit of the stage that the cheater skips (does not pay for) passes its proof with
probability at most `p` over the stage's coin: `Pr[skip_u ∧ ok_u] ≤ p·Pr[skip_u]`. The unit's operands are committed
before the coin, so its value is paid for or is a bet on the coin: none is made cheap and right by a choice of earlier
values, which under one σ is ε_adapt. -/
def StageLuck {C U D : Type} [Fintype C] {M s n : ℕ} (p : ℝ) (stg : U → ℕ) (W : StageExecutionPhase C U D M s n) : Prop :=
  W.LuckAt p stg

end ResourceAccounting.Assumptions.TS
