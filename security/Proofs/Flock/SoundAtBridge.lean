module

public import Proofs.Flock.EndToEndRegisteredRows
public import Proofs.Flock.SoundAtBridge.NonVacuous
public import Proofs.Flock.SoundAtBridge.Reduce
public import Proofs.Flock.SoundAtBridge.Served
public import Proofs.Flock.SoundAtBridge.ServedProof
public import Proofs.Flock.SoundAtBridge.Statement
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

/-! `Flock.SecurityProofs.SoundAtBridge : Flock.Guarantees.SoundAtBridge`: the reduction
`Flock.SoundAtBridge.soundAtBridge_of_rows` at Flock's rows link, `Flock.SecurityProofs.EndToEndRegisteredRows`.
`Flock.SecurityProofs.ServedRowsLinked : Flock.Guarantees.ServedRowsLinked`: `Flock.SoundAtBridge.servedRowsLinked`.
Its witness is `Flock.SoundAtBridge.soundAtBridge_nonvacuous`. -/

@[expose] public section

/-- **PoUW's proof premise from Flock's rows link** (`Flock.Guarantees.SoundAtBridge`). -/
theorem Flock.SecurityProofs.SoundAtBridge : Flock.Guarantees.SoundAtBridge :=
  Flock.SoundAtBridge.soundAtBridge_of_rows Flock.SecurityProofs.EndToEndRegisteredRows

/-- **Flock's rows link at served roots** (`Flock.Guarantees.ServedRowsLinked`). -/
theorem Flock.SecurityProofs.ServedRowsLinked : Flock.Guarantees.ServedRowsLinked :=
  Flock.SoundAtBridge.servedRowsLinked
