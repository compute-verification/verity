module

public import Definitions.Poub.Remote.Sample
public import Poub.Line
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

/-! The verifier's pooled bound covers the opening sample's (`PooledCovers`, `PooledCoversRun`, `LineCoversRun`). -/

@[expose] public section

open Poub

namespace Properties.Poub

/-- Whenever `pooled` gives a bound for windows of sizes `U` sampling `d` at `δ`, its wrong receipts are at least the
sample's pooled bound of the same windows (`ResourceAccounting.Remote.Sample.pooled`) at `δ`. -/
def PooledCovers : Prop :=
  ∀ (n : ℕ) (U d : Fin n → ℕ) (δ : ℚ) (b : Pooled.Bound),
    Pooled.pooled (Array.ofFn fun t => (U t, d t)) δ = .ok b →
      ResourceAccounting.Remote.Sample.pooled n U d δ ≤ b.wrong

/-- Whenever `pooled` gives a bound for a session's windows at outcome `ω`, each window's size and sample count
(`Sample.Run.samples`, `min(s_t, U_t)`), its wrong receipts are at least the session's pooled bound at `δ`
(`Sample.Run.pooled`). -/
def PooledCoversRun : Prop :=
  ∀ (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X : Type) (D : ResourceAccounting.Remote.Sample.Run n m s κ X)
    (ω : ResourceAccounting.Remote.Sample.SampleCoins n κ X) (δ : ℚ) (b : Pooled.Bound),
    Pooled.pooled (Array.ofFn fun t => ((D.size t ω : ℕ), D.samples t ω)) δ = .ok b →
      D.pooled δ ω ≤ b.wrong

/-- Whenever the memory line (`Poub.Line.line`) gives a line for a session's windows at outcome `ω` at `δ`, the
wrong receipts it reads are at least the session's pooled bound at `δ` (`Sample.Run.pooled`). -/
def LineCoversRun : Prop :=
  ∀ (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X : Type) (D : ResourceAccounting.Remote.Sample.Run n m s κ X)
    (ω : ResourceAccounting.Remote.Sample.SampleCoins n κ X) (δ : ℚ) (gs : Array Line.Group)
    (per blocks blockBytes chipBytes nodeBytes : ℕ) (rates : Line.Rates) (l : Line.Line),
    Line.line gs per blocks blockBytes chipBytes nodeBytes (Array.ofFn fun t => ((D.size t ω : ℕ), D.samples t ω)) δ
      rates = .ok l →
      D.pooled δ ω ≤ l.wrong

end Properties.Poub
