module

public import Definitions.Accounting.TS.Machine
public import Definitions.Accounting.TS.Routes
public import Definitions.Accounting.TS.Play
public import Definitions.Accounting.TS.Receipt
public import Definitions.Accounting.TS.Slots
public import Definitions.Accounting.TS.ProofLayer
public import Definitions.Accounting.TS.Drawn
public import Definitions.Accounting.TS.FanIn
public import Definitions.Accounting.TS.Stages
public import Definitions.Accounting.TS.Steer
public import Definitions.Accounting.TS.Sheet
public import Definitions.Accounting.TS.Fold
public import Definitions.Accounting.TS.ReceiptHMu
public import Definitions.Accounting.TS.Unfolded
public import Definitions.Accounting.TS.SharedNoise
public import Definitions.Accounting.TS.J10R
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

/-! The tensor–scalar model (EPS1): the machine, its facets and bound, routes and
labelled programs with the terms of the decomposition, the full-checking and sampled plays, sampled rounds under
per-stage coins, the receipt rounds that split the luck premise into its operand and receipt halves, a round's unfolded
share and the whole-unit luck, the counting objects of the tensor slot count, the output count, the fan-in count and
LIN-STEER, a fold of words by binding steps, and J10-R's noise, one seed a call at call scale. -/

@[expose] public section
