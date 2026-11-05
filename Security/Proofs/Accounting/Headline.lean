module

public import Proofs.Accounting.Headline.Encoding
public import Proofs.Accounting.Headline.EpsJl
public import Proofs.Accounting.Headline.FpG1
public import Proofs.Accounting.Headline.FpG1Measured
public import Proofs.Accounting.Headline.Luck
public import Proofs.Accounting.Headline.Sm120
public import Proofs.Accounting.Headline.Sm120Measured
public import Proofs.Accounting.Headline.Whole
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

/-! The families' headline guarantees (`Definitions.Accounting.Guarantees.Headline`): the encoding's lemmas
(`Headline/Encoding.lean`), (I7) from the receipt split (`Headline/Luck.lean`), Theorem D-ideal with fold skippers
charged (`Headline/Whole.lean`: FOLD-OPT per word folded and the whole-unit luck) and one file per family and reading
(`Headline/Sm120.lean`: sm_120 at its data sheet;
`Headline/Sm120Measured.lean`: sm_120 on VF3s-16); and FP8-g1's and W4A8-g1's headlines on sm_120
(`Definitions.Accounting.Guarantees.FpG1`), with an honest g1 window that meets every premise, at the data sheet
(`Headline/FpG1.lean`) and on the measured vector (`Headline/FpG1Measured.lean`). -/

@[expose] public section
