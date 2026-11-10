module

public import Proofs.Accounting.CoinFlow.Flow
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

/-!
# The coin rules' guarantees

Each statement of `Definitions/Accounting/Guarantees/CoinFlow.lean`, proved in `CoinFlow/Flow.lean`: a noise injection's
fresh coins reach only its forward cone (rule (b)), a non-sink value reads the coins only through noise injections
(rule (a)), the forward cone is the one reader's, and so `CoinPointMass` holds of every view built from the rest of an
admitted Program.
-/

@[expose] public section

namespace Accounting.SecurityProofs.CoinFlow

theorem ViewIndependent : Accounting.Guarantees.CoinFlow.ViewIndependent := Accounting.CoinFlowProofs.viewIndependent
theorem CoinsThroughNoise : Accounting.Guarantees.CoinFlow.CoinsThroughNoise :=
  Accounting.CoinFlowProofs.coinsThroughNoise
theorem OneReader : Accounting.Guarantees.CoinFlow.OneReader := Accounting.CoinFlowProofs.oneReader
theorem PointMass : Accounting.Guarantees.CoinFlow.PointMass := Accounting.CoinFlowProofs.pointMass

end Accounting.SecurityProofs.CoinFlow
