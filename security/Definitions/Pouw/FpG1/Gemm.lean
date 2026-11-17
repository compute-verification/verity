module

public import Definitions.Pouw.Basic.Mat
public import Definitions.Pouw.FpG1.Encoding
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

/-! FP8-g1 and W4A8-g1's int8 GEMM, the coin's two rank-one terms, and the epilogue that takes them off the exact product. -/

@[expose] public section

namespace Pouw.FpG1

open Finset

def rowSum (k : ℕ) (A : Mat) (i : ℕ) : ℤ := ∑ l ∈ range k, A i l

def colSum (k : ℕ) (W : Mat) (j : ℕ) : ℤ := ∑ l ∈ range k, W l j

def centred (cx cw : ℚ) (k : ℕ) (A W : Mat) (i j : ℕ) : ℚ := ∑ l ∈ range k, ((A i l : ℚ) - cx) * ((W l j : ℚ) - cw)

def rowTerm (cx cw : ℚ) (k : ℕ) (A : Mat) (i : ℕ) : ℚ := cw * rowSum k A i - cx * cw * k

def colTerm (cx : ℚ) (k : ℕ) (W : Mat) (j : ℕ) : ℚ := cx * colSum k W j

def model (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) : ℚ :=
  ∑ l ∈ range k, (sa i * (EA i l).served) * (sw j * (EW l j).served)

def ints (E : ℕ → ℕ → Entry) : Mat := fun i j => (E i j).int

def epilogue (ex ew : ℤ) (k : ℕ) (A W : Mat) (i j : ℕ) : ℤ :=
  4 * mm k A W i j - 2 * ew * rowSum k A i - 2 * ex * colSum k W j + ex * ew * k

/-- Wrapping once at the end (`wrap32`) is wrapping after each operation, as int32 arithmetic is arithmetic mod `2^32`. -/
def epilogueW (ex ew : ℤ) (k : ℕ) (A W : Mat) (i j : ℕ) : ℤ :=
  wrap32 (4 * mmW k A W i j - 2 * ew * accW (fun l => A i l) k - 2 * ex * accW (fun l => W l j) k + ex * ew * k)

end Pouw.FpG1
