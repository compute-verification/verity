module

public import Definitions.Accounting.Guarantees.RemoteSketchP2
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
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
import Proofs.Accounting.RemoteHeld.Sketch

/-!
# The receipts' sketch bound over PoUS's P2 store: the proof

`SketchBoundSession` at P2's receipts: a unit of four 2,056-byte blocks, `m = 257` sectors, 16 sign rows, `h = 512`,
`r = 12`, at most `Q + P` candidate seeds and `Q` queries, at `ε_ro = (Q + P)²/2^257`, as `SketchBoundDeployed` is at
`m = 128`.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Sketch

/-- **The sketch bound at P2's receipts** (`Accounting.Guarantees.Remote.SketchBoundP2`). -/
theorem SketchBoundP2 : Accounting.Guarantees.Remote.SketchBoundP2 := by
  intro β Hs _ _ P N S L q Q G qtot ε A T hε hP hRank hG hqtot hH hq hrec hpres
  have h := SketchBoundSession β Hs P N 257 512 S L q 16 12 G qtot ε (((Q : ℝ) + P) ^ 2 / 2 ^ 257) A T (by norm_num)
    hε (by positivity) hP hRank hH hq hrec hpres
  unfold SketchBound Trace.Holds at h ⊢
  rw [show (8 * 12 : ℕ) = 96 from rfl] at h
  refine h.trans ?_
  have hG' : ((∑ j, G j : ℕ) : ℝ) ≤ (Q : ℝ) + P := by exact_mod_cast hG
  have hq' : ((∑ j, qtot j : ℕ) : ℝ) ≤ Q := by exact_mod_cast hqtot
  have hnn : (0 : ℝ) ≤ ε + N / 2 ^ 512 := add_nonneg hε (by positivity)
  have h1 : ((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ 512) ≤ ((Q : ℝ) + P) * (ε + N / 2 ^ 512) :=
    mul_le_mul_of_nonneg_right hG' hnn
  have h2 : ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ 96 ≤ (Q : ℝ) / 2 ^ 96 := by gcongr
  exact (add_le_add (add_le_add h1 h2) le_rfl).trans (le_of_eq (by ring))

end Accounting.SecurityProofs.Remote
