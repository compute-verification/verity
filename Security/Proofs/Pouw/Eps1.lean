module

public import Proofs.Pouw.Eps1.Basics
public import Proofs.Pouw.Eps1.WordLocal
public import Proofs.Pouw.Eps1.ClassP
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

/-!
# Dither's CUDA-core lower bounds in M3: proofs

Theorem B's two instances of `Definitions.Pouw.Guarantees.Eps1`, each as `Pouw.SecurityProofs.Eps1.G`, proved in
`Eps1/WordLocal.lean` from the lemmas in `Eps1/Basics.lean`, and Theorem C's count over issue slots (`ClassPSlots`),
proved in `Eps1/ClassP.lean`. Theorem C's packed form is retired: its paper proof caps a writer's creations at 4, and
a writer can create 8.
-/

@[expose] public section

namespace Pouw.SecurityProofs.Eps1

/-- Theorem B, zero instance. -/
theorem WordLocalZero : Pouw.Guarantees.Eps1.WordLocalZero := WordLocal.wordLocalZero

/-- Theorem B, general instance. -/
theorem WordLocalGeneral : Pouw.Guarantees.Eps1.WordLocalGeneral := WordLocal.wordLocalGeneral

/-- Theorem C's count over issue slots. -/
theorem ClassPSlots : Pouw.Guarantees.Eps1.ClassPSlots := ClassP.classPSlots

end Pouw.SecurityProofs.Eps1
