module

public import Definitions.Accounting.HeadlineDc.Encoding
public import Definitions.Accounting.HeadlineDc.Sm90
public import Definitions.Accounting.HeadlineDc.Sm90Measured
public import Definitions.Accounting.HeadlineDc.Sm90MeasuredI3
public import Definitions.Accounting.HeadlineDc.Sm100
public import Definitions.Accounting.HeadlineDc.Sm100Measured
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

/-! The data-centre families' headline instances' exact data: the encoding beside `Headline.Encoding`'s
(`HeadlineDc/Encoding.lean`: a filler, a certificate on several facets) and one file per family and reading, which
`benchmarks/pouw/lean_headline_dc.py` writes (`HeadlineDc/Sm90.lean`: the H200, DS-90 and Dither-2S LI-CP-W with
ffill; `HeadlineDc/Sm90Measured.lean`: the H200 on DC-90-ri; `HeadlineDc/Sm90MeasuredI3.lean`: the same reading at
(I3)'s `q = q_T = 3/5`, its twin; `HeadlineDc/Sm100.lean`: the B200, DS-100 and Dither-2S LI-CP-W heavy with ♭
crediting no LDTM; `HeadlineDc/Sm100Measured.lean`: the B200 on DC-100-ri). -/

@[expose] public section
