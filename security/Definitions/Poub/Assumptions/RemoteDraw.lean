module

public import Definitions.Poub.Remote.Sample
public import Definitions.Poub.Remote.Held
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

/-! What PoUB's opening sample assumes (`Live`, `Refused`, `Counted`); the sample itself is the deployed one, not a
premise. -/

@[expose] public section

namespace ResourceAccounting.Remote.Sample.Assumptions

open ResourceAccounting.Remote.Held

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

/-- **Live coins, as the verifier releases them**: two outcomes that agree on the other coins and on the earlier
windows' openings coins give window `t` the same size and the same correct receipts, whatever its own openings coin and
the later ones. The guarantees read it of the session the prover plays, which holds no coin commitment while it runs;
the memory line over the prover's records reads it of `Committed.view c` at every commitment `c`. -/
def Live (D : Run n m s κ X) : Prop :=
  ∀ t (ω ω' : SampleCoins n κ X), ω.1 = ω'.1 → (∀ j, j < t → ω.2 j = ω'.2 j) →
    D.size t ω = D.size t ω' ∧ D.correct t ω = D.correct t ω'

/-- **Refused**: on every accepting outcome, every receipt a window opens, on the first blocks `prf` and the
fallbacks `rest`, is among its correct ones or moved: presented, accepted at its root and rank, with a row other than
the one its root holds. -/
def Refused [Fintype X] (D : Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X) : Prop :=
  ∀ ω, T.accepts ω → ∀ t, ∀ p ∈ D.opened prf rest t ω, p ∈ D.correct t ω ∨ R.Moved D prf rest t ω p

/-- **Counted**: on every accepting outcome, the credited passes' wrong receipts of registered blocks are at most the
windows' wrong receipts. -/
def Counted (D : Run n m s κ X) (T : Trace (SampleCoins n κ X)) : Prop :=
  ∀ ω, T.accepts ω → T.incorrect ω ≤ D.incorrect ω

end ResourceAccounting.Remote.Sample.Assumptions
