module

public import Definitions.Pouw.FpG1.Encoding
public import Definitions.Pouw.FpG1.Gemm
public import Definitions.Pouw.FpG1.H1
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

/-! FP8-g1 and W4A8-g1: the operand encodings on E4M3's and E2M1's integer grids with a live coin
(`FpG1/Encoding.lean`), the int8 GEMM, the coin's rank-one terms and the epilogue (`FpG1/Gemm.lean`), and g1 on sm_120's
record H1: its grids and a round's operand leaves (`FpG1/H1.lean`). `Proofs.Pouw.FpG1` proves their arithmetic.
`Definitions.Accounting.Guarantees.FpG1`'s headlines read H1's leaves and, through them, the grids; they are the
record's bound at g1's windows, which the record's headlines already cover by their own quantifier over committed values
(`Proofs.Pouw.FpG1`'s module docstring). -/

@[expose] public section
