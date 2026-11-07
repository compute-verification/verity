module

public import Proofs.Flock.Bridge.KSLive
public import Proofs.Flock.Bridge.V5.Assembly
public import Proofs.Flock.Bridge.V5.EndToEndHidden
public import Proofs.Flock.Bridge.V5.OpeningsCanon
public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.V5.Transports
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

/-!
# `SampledKSLiveV5`, proved

`Flock.Bridge.sampledKSLiveV5R : Flock.Bridge.SampledKSLiveV5R`: `AssemblyTargetV5` (`assemblyTargetV5`) with
canonical transport at public inputs (`transportsLiveV5`), M1's core (`faultyRightTargetC`) and `EndToEndHiddenV5`
(`endToEndHiddenV5_of`, from `openingsCanonV5`) plugged in. `Flock.SecurityProofs.SampledKSLiveV5 :
Flock.Guarantees.SampledKSLiveV5`: that bound under the roots guard, with `ecr/sha-512`'s term at the roots finder in place
of the guard (`Bridge.le_add_ecrBoundL`).
-/

@[expose] public section

/-- **V5's bound under the roots guard** (`Flock.Bridge.SampledKSLiveV5R`). -/
theorem Flock.Bridge.sampledKSLiveV5R : Flock.Bridge.SampledKSLiveV5R :=
  Flock.Bridge.assemblyTargetV5 Flock.Bridge.transportsLiveV5 Flock.Bridge.faultyRightTargetC
    (Flock.Bridge.endToEndHiddenV5_of Flock.Bridge.openingsCanonV5)

/-- **The sampled audit's knowledge soundness, live, at the verifier's public inputs, with or without shared rows**
(`Flock.Guarantees.SampledKSLiveV5`). -/
theorem Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5 := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup desc qr hpop lj tb law kd hlaw hb hdiv dj S₀ d₀ y₀ wr dg
    N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q L P vb σ Xr Cc Lm rows roots Cd κ r ns δ
    hTab hLink
  exact Flock.Bridge.le_add_ecrBoundL (FlockSoundness.Game.prob_le_one _ _ τ) fun hroots =>
    Flock.Bridge.sampledKSLiveV5R n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots desc qr hpop lj tb law
      kd hlaw hb hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q hTab hLink
