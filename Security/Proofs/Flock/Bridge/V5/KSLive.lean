module

public import Proofs.Flock.Bridge.V5.Statement
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
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.V5.Assembly
import Proofs.Flock.Bridge.V5.EndToEndHidden
import Proofs.Flock.Bridge.V5.OpeningsCanon
import Proofs.Flock.Bridge.V5.Transports

/-!
# `SampledKSLiveV5`, proved

`Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5`: `AssemblyTargetV5` (`assemblyTargetV5`) with
canonical transport at public inputs (`transportsLiveV5`), M1's core (`faultyRightTargetC`) and `EndToEndHiddenV5`
(`endToEndHiddenV5_of`, from `openingsCanonV5`) plugged in.
-/

@[expose] public section

/-- **The sampled audit's knowledge soundness, live, at the verifier's public inputs, with or without shared rows**
(`Flock.Guarantees.SampledKSLiveV5`). -/
theorem Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5 :=
  Flock.Bridge.assemblyTargetV5 Flock.Bridge.transportsLiveV5 Flock.Bridge.faultyRightTargetC
    (Flock.Bridge.endToEndHiddenV5_of Flock.Bridge.openingsCanonV5)
