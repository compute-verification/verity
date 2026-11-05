module

public import Proofs.Core.Circuit
public import Proofs.Core.Annotation
public import Proofs.Core.Advice
public import Proofs.Core.PrivatePartition
public import Proofs.Core.Boolean
public import Proofs.Core.Game
public import Proofs.Core.Game.Basic
public import Proofs.Core.Game.Interleave
public import Proofs.Core.Game.Union
public import Proofs.Core.Game.Prob
public import Proofs.Core.Game.Poly
public import Proofs.Core.Game.Expect
public import Proofs.Core.Game.Runs
public import Proofs.Core.Game.Seq
public import Proofs.Core.Game.Lock
public import Proofs.Core.Game.Batch
public import Proofs.Core.Game.BatchInduced
public import Proofs.Core.Game.Rewinding
public import Proofs.Core.Game.Unambiguous
public import Proofs.Core.Challenger
public import Proofs.Core.ValueSalts
public import Proofs.Core.UnionBound
public import Proofs.Core.Scalar.Vectors
public import Proofs.Core.Prims.Vectors
public import Proofs.Core.Round
public import Proofs.Core.Prims.Round
public import Proofs.Core.TC
public import Proofs.Core.Fp32
public import Proofs.Core.Formalism
public import Proofs.Core.Expand
public import Proofs.Core.Padding
public import Proofs.Core.AndSize
public import Proofs.Core.Accel.Model
public import Proofs.Core.Accel.Sheet
public import Proofs.Core.Accel.Witness
public import Proofs.Core.NaiveStrassen
public import Proofs.Core.Bilinear
public import Proofs.Core.HongKung
public import Proofs.Core.Blocks
public import Proofs.Core.DeviceFigures
public import Proofs.Core.Site
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
# Proofs of core's guarantees

One theorem `Verity.SecurityProofs.G : Verity.Guarantees.G` per proved guarantee, and every lemma behind them.
-/

@[expose] public section
