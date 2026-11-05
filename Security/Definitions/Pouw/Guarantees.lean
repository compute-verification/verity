module

public import Definitions.Pouw.Guarantees.Barrier
public import Definitions.Pouw.Guarantees.Deadline
public import Definitions.Pouw.Guarantees.Dither
public import Definitions.Pouw.Guarantees.DitherInstance
public import Definitions.Pouw.Guarantees.Eps1
public import Definitions.Pouw.Guarantees.Game
public import Definitions.Pouw.Guarantees.KeyedLeaf
public import Definitions.Pouw.Guarantees.NCP
public import Definitions.Pouw.Guarantees.OnsiteFreed
public import Definitions.Pouw.Guarantees.PearlC
public import Definitions.Pouw.Guarantees.PearlCTileCheck
public import Definitions.Pouw.Guarantees.Receipt
public import Definitions.Pouw.Guarantees.RelativeSecurity
public import Definitions.Pouw.Guarantees.ShaRealize
public import Definitions.Pouw.Guarantees.Window
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
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

/-!
# PoUW's guarantees (trusted)

Every statement a document outside the package cites as proved, as a `def G : Prop` over `Definitions.Pouw` and
`Definitions.Pouw.Assumptions`, by topic. Its proof is the theorem `Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`; a pin cited only
inside the package stays a lemma under its own name.
-/

@[expose] public section
