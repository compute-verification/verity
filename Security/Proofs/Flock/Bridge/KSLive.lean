module

public import Proofs.Flock.Bridge.FaultyRightC
public import Proofs.Flock.Bridge.Refused
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
import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.OpeningsCanon
import Proofs.Flock.Bridge.Transports

/-!
# `SampledKSLiveV2` and `SampledKSLiveV3`, proved

`Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`: `AssemblyTargetC` (`assemblyTargetC`) with
canonical transport (`transportsLiveC`), M1's core (`faultyRightTargetC`), the slot count (`partBlocks`), the count lemma
(`count`) and `EndToEndHiddenC` (`endToEndHiddenC_of`, from `openingsCanon`) plugged in.
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`: V2 with its file preconditions read off the
verifiers' acceptance (`sampledKSLiveV3_of`).
-/

@[expose] public section

/-- **The sampled audit's knowledge soundness, live, at the circuit files the verifier of record reads** (`Flock.Guarantees.SampledKSLiveV2`). -/
theorem Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2 :=
  Flock.Bridge.assemblyTargetC Flock.Bridge.transportsLiveC Flock.Bridge.faultyRightTargetC Flock.Bridge.partBlocks
    Flock.Bridge.count (Flock.Bridge.endToEndHiddenC_of Flock.Bridge.openingsCanon)

/-- **The sampled audit's knowledge soundness, live, on the event that the verifiers accept the files**
(`Flock.Guarantees.SampledKSLiveV3`): no hypothesis on the files' rows, names, roots or population order. -/
theorem Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3 :=
  Flock.Guarantees.sampledKSLiveV3_of Flock.SecurityProofs.SampledKSLiveV2
