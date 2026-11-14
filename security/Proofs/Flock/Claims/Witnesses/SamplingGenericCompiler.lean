module

public import Proofs.Flock.Claims.Witnesses.SamplingGenericCost
public import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
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
# The generic sampled compiler's statements say something

Each closed statement of the generic sampled compiler (`CompilerKS`, `CompilerKSReg`, `CompilerKSRegX`,
`CompilerKSCount`, `CompilerBits`, `CompilerEfficient`, `CompilerZK`) is a guarantee that the lock's `claims.generic` names, held to two
witnesses (the lock's `witnesses`): honest ones, whose conclusions are its premises holding together at one instance,
and a nontrivial one, at which its conclusion fails. Each builds its instance: coins `Bool` and the hash SHA-512
(`FlockDeploy.sha512`), at the toys of `Witnesses.SamplingGeneric`, `SamplingGenericCost` and `SamplingGenericZK`.

* Honest: `twoL`, two sessions over `twoA` that each accept, under a binding commitment (`compilerKS_honest`), whose
  opening checks make no call (`compilerBits_honest`); `echoL`, one masked session over the zero-knowledge `echoArg`,
  with the toy witness in its sampling relation (`compilerZK_honest`).
* Nontrivial: `sksL` (accepts with an empty relation) fails `KS`, `KSReg`, `KSRegX`, `KSCount` and `Bits`; `effL` (its opening
  check calls more than it counts) fails `Efficient`; and `talkL` (a talking prover, a silent simulator) fails
  `ZkCompiled.ZK`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Witness (silentP)
open Claims.Witness.Sampling (toyWit)
open Claims.Witness.SamplingGeneric

/-- **`CompilerKS`'s premises hold together at two sessions**: `twoA` is knowledge sound, `twoL`'s commitment binds,
`twoL` meets the contract, the empty statement is in scope, and each run checks both proof blocks and is accepted. Also
`CompilerKSReg`'s, `CompilerKSRegX`'s and `CompilerKSCount`'s, and with `compilerBits_honest` `CompilerBits`' and `CompilerEfficient`'s. -/
theorem compilerKS_honest :
    (twoA Bool).KS FlockDeploy.sha512 ∧ (twoL Bool).commit.Binds FlockDeploy.sha512 ∧
      (twoL Bool).Sessions FlockDeploy.sha512 ∧ (twoL Bool).Scope [] ∧
      ∀ r, (twoL Bool).checked [] r = [0, 1] ∧ (run FlockDeploy.sha512 silentP (twoL Bool).V [] () r).out = .acc :=
  twoSessions_meets _

/-- **The opening checks' count holds at `twoL`**: `CompilerBits`' and `CompilerEfficient`'s premise beyond the
contract. -/
theorem compilerBits_honest : (twoL Bool).OpenCallsLe FlockDeploy.sha512 := twoOpenCalls_meets _

/-- **`CompilerZK`'s premises hold together at a nonempty relation**: `echoArg` is zero knowledge, `echoL` meets the
zero-knowledge contract, and the toy witness is in `echoL`'s sampling relation. -/
theorem compilerZK_honest :
    (echoArg Bool).ZK FlockDeploy.sha512 ∧ (echoL Bool).ZkSessions FlockDeploy.sha512 ∧
      Claims.Sampling.Rhon FlockDeploy.sha512 (echoL Bool).commit (echoL Bool).parse (echoL Bool).partition [] toyWit :=
  echo_meets _

/-- **`CompilerKS`'s conclusion fails** at `sksL`, which accepts with an empty relation. -/
theorem compilerKS_nontrivial : ¬ (sksL Bool).KS FlockDeploy.sha512 := compiledKS_fails _

/-- **`CompilerKSReg`'s conclusion fails** at `sksL`. -/
theorem compilerKSReg_nontrivial : ¬ (sksL Bool).KSReg FlockDeploy.sha512 := compiledKSReg_fails _

/-- **`CompilerKSRegX`'s conclusion fails** at `sksL`. -/
theorem compilerKSRegX_nontrivial : ¬ (sksL Bool).KSRegX FlockDeploy.sha512 := compiledKSRegX_fails _

/-- **`CompilerKSCount`'s conclusion fails** at `sksL`. -/
theorem compilerKSCount_nontrivial : ¬ (sksL Bool).KSCount FlockDeploy.sha512 := compiledKSCount_fails _

/-- **`CompilerBits`' conclusion fails** at `sksL`. -/
theorem compilerBits_nontrivial : ¬ (sksL Bool).Bits FlockDeploy.sha512 := compiledBits_fails _

/-- **`CompilerEfficient`'s conclusion fails** at `effL`, whose opening check makes a call it counts as none. -/
theorem compilerEfficient_nontrivial : ¬ (effL Bool).Efficient FlockDeploy.sha512 := compiledEfficient_fails _

/-- **`CompilerZK`'s conclusion fails** at `talkL`, whose prover talks and whose composed simulator is silent. -/
theorem compilerZK_nontrivial : ¬ (talkL Bool).ZK FlockDeploy.sha512 := zkCompiled_talk_fails _

end Claims.SamplingGeneric
