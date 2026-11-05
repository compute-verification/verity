module

public import Definitions.Hash.Guarantees
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
import Proofs.Hash.RowBinding

/-!
# `Hash.SecurityProofs.EmbeddingReadBinding : Hash.Guarantees.EmbeddingReadBinding`

For a token in the vocabulary the read's ok is the witness's acceptance at rank `tok` and its row is the witness's, so
`RowBinding` at rank `tok` against the registered opening is the statement.
-/

@[expose] public section

open Hash in
/-- **The embedding's row read binds its row** (`Hash.Guarantees.EmbeddingReadBinding`). -/
theorem Hash.SecurityProofs.EmbeddingReadBinding : Hash.Guarantees.EmbeddingReadBinding := by
  intro H G domain key root vocab words tok o hok o₀ hin h₀
  have ho : RowLeaf.accepts H G (EmbeddingRead.verifier domain key) root tok o :=
    Or.resolve_left (show vocab ≤ tok ∨ _ from hok) (Nat.not_le.mpr hin)
  rw [EmbeddingRead.h, if_pos hin]
  exact Hash.SecurityProofs.RowBinding H G _ root tok o o₀ ho h₀
