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
# `SampledKSLiveV2` with a public error

`SampledKSLiveV2` bounds `Pr[verity-verify-proof accepts ∧ ¬ RelS]` by three terms, two of which read the prover's run:
`ksAvgStrictZ (planZHJ dj y₀) L k q_F q_S (reg σ)`, at the prover's registration, and
`Q_s = Σ_c |Ω_c|/|Ω|` (`ΩcZ`), at its binding. Neither depends on the prover in value, and each is at most a number read
off the public parameters:

* `ε_ks ≤ ksClosedM1 k qF qS / (1 − ρ)`: every table of `planZHJ` is at Flock ZK's parameters (`planZHJ_M1W`, from
  what `verify --zk` checks at setup, at every registration: `planZHJ` reads none), so `ksAvgStrictZ_planZHJ_le_closed`
  applies, and its rate `2^23/(ek)` is at most `ρ`;
* `Q_s ≤ W`, `W = Σ_u |io u|` (`Bridge.ioSum`) the committed wires each unit of the circuit's partition computes or
  reads: a sample reads a commit string only through such a wire of a sampled proof block (`qs_le_io`). It is `Σ_u`, not
  `k_d · max_u`, because the live subset sample on finite bytes samples every unit when it runs out (`SamplingDesign.execOS`).

So `SampledKSLivePub` is `SampledKSLiveV2` with its error `Bridge.deltaPub`:
$$\binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \frac{\mathrm{ksClosedM1}(k,q_F,q_S)}{1-\rho}
  + \frac{W\,\big(2(t'+v)(1+k)/2^{256} + 1/(eM) + k/(eR_w)\big)}{1-\rho},$$
at the knowledge finders' budgets `q_F = 2(t' + v)` and `q_S = t' + v` and the link finders' per-run cost `t' + v`
(`SampledKSLiveV2`'s: the prover's per-run time `t'` and one opening's verification `v`), a function of `n`, `k_d`,
`K₀`, the analysis' `k`, `R_w`, `M`, `t'`, `v`, `ρ` and the circuit's partition. Its rate hypothesis reads only `k`
(`rateM1 k ≤ ρ`) where `SampledKSLiveV2`'s reads the plan, and every other hypothesis, named assumption included, is
`SampledKSLiveV2`'s.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The sampled audit's public error**: the sample's miss, the knowledge term at Flock ZK's parameters
(`ksClosedM1`), and the binding error at `W` commit strings with the link finders charged `t'` per run, each over `1 − ρ`
where `SampledKSLiveV2` has it. -/
noncomputable def _root_.Flock.Bridge.deltaPub (n kd K₀ k : ℕ) (qF qS t' : ℝ) (M Rw W : ℕ) (ρ : ℝ) : ℝ :=
  ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + ksClosedM1 k qF qS / (1 - ρ) +
    (W : ℝ) * (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)

end Flock.Guarantees

