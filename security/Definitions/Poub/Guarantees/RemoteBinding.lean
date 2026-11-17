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

/-! The receipts' binding bound, by the forking argument; that the roots read only the cut is not stated, so `split_t`
is not bounded. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Sample

/-- **The receipts' binding term under `cr/sha-512`.** `ReceiptsBindingOfFork` with each window's fork finder
(`Receipts.finder`, at `q_t` SHA-512 calls) meeting `Claims.StrictBirthday`: the binding term is at most
`Σ_t √(|L_t|·s_t·(q_t²/2^513 + split_t))`. -/
def ReceiptsBindingOfCR : Prop :=
  ∀ {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] [Nonempty X]
    (D : Remote.Sample.Run n m s κ X)
    (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ)
    (L : Fin n → Finset ℕ) (q : Fin n → ℕ),
    (∀ t ω, D.opened prf rest t ω ⊆ L t) →
    (∀ t, Claims.StrictBirthday (fun u => (R.hash u).bytes) (R.finder D prf rest t (q t))) →
    (R.binding D prf rest).toReal ≤
      ∑ t, √((L t).card * s t * ((q t : ℝ) ^ 2 / 2 ^ (513 : ℝ) + (R.split D prf rest t).toReal))

end ResourceAccounting.Guarantees.Remote
