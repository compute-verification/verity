module

public import Definitions.Pouw.TS.Fixed
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

/-! The two premises `UnitFixed` rests on under option A, `ModesRun` and `CountsFixed`. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.TS

open ResourceAccounting.TS Pouw.RelativeSecurity

/-- **The modes are the run's**, a premise per unit. It holds only where the layer's cuts are no later than the
protocol's commitments: "computed from the committed values" is true of the real run only then. -/
def ModesRun {C B : Type} {S m s : ℕ} {J Q Pos V N W : Type} [Fintype C] [Fintype B] [Nonempty V] [AddCommGroup W]
    (pf : WordLayer C B S m s J Q Pos V N W) (mode : Paths (Paths C m × B) S → Fin S × J → Mode) : Prop :=
  ∀ x u, (mode x u = .paid → pf.Computed x u) ∧ (mode x u ≠ .whole → pf.Folds x u)

open Classical in
/-- **The labelling counts fixed words as fixed**, a conjectured premise at unit grain. It fails if the instruction-level
labelling counts as adaptive, or as not owned or not right, a word that is a fixed word here. -/
def CountsFixed {K T L U : Type} {C B : Type} {S m s : ℕ} {J Q Pos V N W : Type} [Fintype C] [Fintype B] [Fintype J]
    [Nonempty V] [AddCommGroup W] (P : Play (Paths C m × B) S K T L U) (pf : WordLayer C B S m s J Q Pos V N W)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) : Prop :=
  ∀ x, P.accepts x →
    (∑ w, ∑ j, if pf.FixedWords wt x (w, j) then wt w j else 0) / S ≤ (P.run x).fixed ∧
      (∑ w, ∑ j, if pf.Folds x (w, j) then wt w j else 0) / S ≤ folded x

end ResourceAccounting.Assumptions.TS
