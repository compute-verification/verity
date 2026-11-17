module

public import Definitions.Poub.Remote.Fork
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

/-! The receipts' binding bound in its fork form, the step of `ReceiptsBindingOfCR` that assumes no hash; no record
pins it. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Sample

def ReceiptsBindingOfFork : Prop :=
  ∀ {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] (D : Remote.Sample.Run n m s κ X)
    (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ),
    (∀ t ω, D.opened prf rest t ω ⊆ L t) →
    (R.binding D prf rest).toReal ≤
      ∑ t, √((L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal))

end ResourceAccounting.Guarantees.Remote
