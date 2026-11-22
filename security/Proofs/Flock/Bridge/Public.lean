module

public import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1
public import Proofs.Flock.Bridge.Canon
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
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.OpeningsCanon
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.Transports

/-!
# The sampled audit's public error

`Bridge.deltaPub` is the sampled audit's error read off the public parameters alone:
$$\binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \frac{\mathrm{ksClosedM1}(k,q_F,q_S)}{1-\rho}
  + \frac{W\,\big(2(t'+v)(1+k)/2^{256} + 1/(eM) + k/(eR_w)\big)}{1-\rho},$$
the sample's miss, the knowledge term at Flock ZK's parameters (`ksClosedM1`, at rate at most `ρ`) and the binding
error at `W = Σ_u |io u|` commit strings, the committed wires each unit of the circuit's partition computes or reads,
with the link finders charged `t' + v` per run.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The sampled audit's public error**: the sample's miss, the knowledge term at Flock ZK's parameters
(`ksClosedM1`), and the binding error at `W` commit strings with the link finders charged `t'` per run, the last two
over `1 − ρ`. -/
noncomputable def _root_.Flock.Bridge.deltaPub (n kd K₀ k : ℕ) (qF qS t' : ℝ) (M Rw W : ℕ) (ρ : ℝ) : ℝ :=
  ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + ksClosedM1 k qF qS / (1 - ρ) +
    (W : ℝ) * (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)

end Flock.Guarantees

