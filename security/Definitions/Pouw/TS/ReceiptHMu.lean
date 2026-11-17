module

public import Definitions.Pouw.TS.Receipt
public import Definitions.Pouw.HMu
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

/-! (H-μ) at the receipt's operands, which Lemma B at `Z = 1` turns into the operand luck at `μ`; a cheater that reads a coin pinning the skipped value fails it. -/

@[expose] public section

namespace ResourceAccounting.TS

open Pouw.RelativeSecurity ResourceAccounting.HMu

namespace ReceiptGame

variable {Ω : Type}

/-- **(H-μ) at the operands, at `μ`**: at every round, whatever the past, its share lies in `[0, 1]`, and the round is
a word-level `SkipRound` (the deficient words' values skipped, the cheater's values for them committed from its view,
the share decided from the view, the words all right only where the two agree) whose skipped value has point mass at
most `μ` given the view (`SkipRound.PointMass`). -/
def OperandPointMass [Fintype Ω] (μ : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧
        (∃ (O α : Type) (R : SkipRound Ω O α α), R.PointMass μ ∧ (∀ o x, R.honest o x = x) ∧
          (∀ ω, d ω = R.d (R.view ω)) ∧ ∀ ω, right ω → R.committed (R.view ω) = R.skipped ω) ∧
        ∀ ω, OperandPointMass μ (next ω)

end ReceiptGame

end ResourceAccounting.TS
