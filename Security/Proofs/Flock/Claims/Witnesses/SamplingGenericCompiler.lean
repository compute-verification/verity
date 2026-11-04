import Proofs.Flock.Claims.Witnesses.SamplingGenericCost
import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
import Proofs.Flock.Claims.Deploy

/-!
# The generic sampled compiler's statements say something

Each closed statement of the generic sampled compiler (`CompilerKS`, `CompilerKSReg`, `CompilerKSCount`,
`CompilerBits`, `CompilerEfficient`, `CompilerZK`) and the zero-knowledge leak form (`ZKOutLeak`) is a guarantee, held
to two witnesses (the lock's `witnesses`): honest ones, whose conclusions are its premises holding together at one
instance, and a nontrivial one, at which its conclusion fails. Each builds its instance: coins `Bool` and the hash
SHA-512 (`FlockDeploy.sha512`), at the toys of `Witnesses.SamplingGeneric`, `SamplingGenericCost` and
`SamplingGenericZK`.

* Honest: `twoL`, two sessions over `twoA` that each accept, under a binding commitment (`compilerKS_honest`), whose
  opening checks make no call (`compilerBits_honest`); `echoL`, one masked session over the zero-knowledge `echoArg`,
  with the toy witness in its sampling relation (`compilerZK_honest`); `echoArg`'s sessions are `ZKOut` at every
  witness, all in its relation (`zkOutLeak_honest`).
* Nontrivial: `sksL` (accepts with an empty relation) fails `KS`, `KSReg`, `KSCount` and `Bits`; `effL` (its opening
  check calls more than it counts) fails `Efficient`; `talkL` (a talking prover, a silent simulator) fails
  `ZkCompiled.ZK`; and the talking prover's leak beats the leak form's bound (`talk_leaks`).
-/

namespace Claims.SamplingGeneric

open Claims.Witness (silentP)
open Claims.Witness.Sampling (toyWit)
open Claims.Witness.SamplingGeneric

/-- **`CompilerKS`'s premises hold together at two sessions**: `twoA` is knowledge sound, `twoL`'s commitment binds,
`twoL` meets the contract, the empty statement is in scope, and each run checks both proof blocks and is accepted. Also
`CompilerKSReg`'s and `CompilerKSCount`'s, and with `compilerBits_honest` `CompilerBits`' and `CompilerEfficient`'s. -/
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

/-- **`CompilerKSCount`'s conclusion fails** at `sksL`. -/
theorem compilerKSCount_nontrivial : ¬ (sksL Bool).KSCount FlockDeploy.sha512 := compiledKSCount_fails _

/-- **`CompilerBits`' conclusion fails** at `sksL`. -/
theorem compilerBits_nontrivial : ¬ (sksL Bool).Bits FlockDeploy.sha512 := compiledBits_fails _

/-- **`CompilerEfficient`'s conclusion fails** at `effL`, whose opening check makes a call it counts as none. -/
theorem compilerEfficient_nontrivial : ¬ (effL Bool).Efficient FlockDeploy.sha512 := compiledEfficient_fails _

/-- **`CompilerZK`'s conclusion fails** at `talkL`, whose prover talks and whose composed simulator is silent. -/
theorem compilerZK_nontrivial : ¬ (talkL Bool).ZK FlockDeploy.sha512 := zkCompiled_talk_fails _

end Claims.SamplingGeneric

namespace Claims

open Claims.Witness.SamplingGeneric

/-- **`ZKOutLeak`'s premise holds at a nonempty relation**: the echoing simulator's transcript is the talking prover's,
so its sessions are `ZKOut` at error `0`, at every witness, all in the relation. -/
theorem zkOutLeak_honest :
    ZKOut (fun _ => Unit) (fun _ => Unit) FlockDeploy.sha512 (fun _ => 1) (fun _ _ => talkP) (fun _ (_ : Unit) => True)
      (fun _ _ => ()) echoSimOut (fun _ _ _ _ _ _ => 0) :=
  echoArg_zk (Coin := Bool) _

/-- **`ZKOutLeak`'s conclusion fails** at the talking prover against the verifier that accepts everything, with the
silent simulator and error `0`: its transcript is never empty, and the simulator's always is. -/
theorem zkOutLeak_nontrivial :
    ¬ Pr (fun ω : Unit × Unit => (runStar FlockDeploy.sha512 talkP acceptAllVs [] ω ()).msgs ≠ []) ≤
        (Fintype.card Unit : ℝ) * Pr (fun p : Unit × silentSimOut.C [] =>
          (silentSimOut.given p.1).out FlockDeploy.sha512 acceptAllVs () [] p.2 ≠ []) + ∑ _o : Unit, (0 : ℝ) :=
  talk_leaks _

end Claims
