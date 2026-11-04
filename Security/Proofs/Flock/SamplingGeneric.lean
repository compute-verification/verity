import Definitions.Claims.SamplingGeneric
import Proofs.Flock.SamplingGeneric.Fails
import Proofs.Flock.SamplingGeneric.Split
import Proofs.Flock.SamplingGeneric.Assemble
import Proofs.Flock.SamplingGeneric.Coins
import Proofs.Flock.SamplingGeneric.Session
import Proofs.Flock.SamplingGeneric.Count
import Proofs.Flock.SamplingGeneric.Bits
import Proofs.Flock.SamplingGeneric.Reg
import Proofs.Flock.SamplingGeneric.Efficient
import Proofs.Flock.SamplingGeneric.Counter

/-!
# The sampled compiler in the claims core, at Flock

The compiled extractor's failure event, its probability and `Compiled.KS` from the split are in
`SamplingGeneric/Fails.lean`.

`Arg.ofFlock` is Flock's `E^rep` at the deployed decoder with Flock's errors, so its knowledge soundness is `Flock.KS`'s
at that decoder (`flockArgKS`, by unfolding: its coins, extractor, `κ` and `γ` are `Flock.E D D.decode`'s, `Flock.κ`'s
and `Flock.γ`'s). So the compiler's knowledge soundness gives Flock's (`flockKS_of_compilerKS`);
it is two steps (`SamplingGeneric/Split.lean`: `KSOf`, `SessionKSOf`, composed by `compilerKS_of`), the second proved
from the session transfer and the session's fork (`SamplingGeneric/Session.lean`: `sessionKSOf`), the first from
`KSOf`'s deterministic half, a failed extraction giving the failure event or a `Bext` collision (`split`,
`SamplingGeneric/Assemble.lean`: `ksOf`). So **`CompilerKS` is proved** (`compilerKS`), and with it Flock's instance
(`flockKS`) and the count form (`compilerKSCount`, `SamplingGeneric/Count.lean`). `CompilerBits` follows from it and
the extractor's expected reruns and own calls (`SamplingGeneric/Bits.lean`, `ExpOwnOf`), so **`CompilerEfficient`,
`ExpOwnOf` and `CompilerBits` are proved** (`compilerEfficient`, `expOwnOf`, `SamplingGeneric/Efficient.lean`;
`compilerBits`). Their own-call bounds read the opening checks' count, `Compiled.OpenCallsLe`, without which they fail
(`Counter.not_ownLe_openCalls`, `SamplingGeneric/Counter.lean`); at Flock it is the frozen `Sampling.OpenCallsLe`
(`openCallsLe_of_deploy`). A session's coins in a compiled run, `coinsOf` and `place`, invert each other
(`SamplingGeneric/Coins.lean`). The committed form,
against the incorrect set of the plurality trace the prover's tape fixes, follows from the sessions' bound and the
rewinding lemma (`SamplingGeneric/Reg.lean`: `compilerKSReg_of`), so **`CompilerKSReg` is proved** (`compilerKSReg`),
and with it Flock's (`flockKSReg`).
-/

namespace Claims.SamplingGeneric

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

theorem flockArgKS : FlockArgKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ hKS hdec
  exact hKS D.decode hdec

theorem flockKS_of_compilerKS (h : CompilerKS) : FlockKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D) L H (flockArgKS H D hKS hdec) hB hS

/-- **The sampled compiler is knowledge sound**: `KSOf` and `SessionKSOf` (`compilerKS_of`). -/
theorem compilerKS : CompilerKS := compilerKS_of ksOf sessionKSOf

/-- **Flock is an instance**: the compiler at `Arg.ofFlock`, from `Flock.KS` and its decoder's list bound. -/
theorem flockKS : FlockKS (Coin := Coin) (Col := Col) (T := T) := flockKS_of_compilerKS compilerKS

/-- The count form, at the miss of a count. -/
theorem compilerKSCount : CompilerKSCount := compilerKSCount_of compilerKS

theorem flockKSReg_of_compilerKSReg (h : CompilerKSReg) : FlockKSReg (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D) L H (flockArgKS H D hKS hdec) hB hS

/-- **The committed form**: `CompilerKSReg` from the sessions' bound (`compilerKSReg_of`). -/
theorem compilerKSReg : CompilerKSReg := compilerKSReg_of sessionKSOf

/-- **Flock's committed form**: the compiler's at `Arg.ofFlock`. -/
theorem flockKSReg : FlockKSReg (Coin := Coin) (Col := Col) (T := T) := flockKSReg_of_compilerKSReg compilerKSReg

/-- **Bits**: `CompilerKS` and `ExpOwnOf` (`compilerBits_of`). -/
theorem compilerBits : CompilerBits := compilerBits_of compilerKS expOwnOf

omit [Inhabited Coin] [DecidableEq Col] in
/-- **The opening checks' count at Flock is the frozen one**: a compiled verifier over Flock at a deployed sampled
verifier `D`'s Flock record, with `D`'s parser, commitment, reveal and opening count, meets `Compiled.OpenCallsLe`
where `D` meets `Sampling.OpenCallsLe`, as the deployed verifier does (`SamplingDeploy.openCallsLe_deploy`). -/
theorem openCallsLe_of_deploy [Zero Col] {H : Msg → Msg} (D : Claims.Sampling.Deploy Coin Col T)
    (L : Compiled Coin (Arg.ofFlock D.flock)) (hparse : L.parse = D.parse) (hcommit : L.commit = D.commit)
    (hreveal : L.reveal = D.reveal) (hcalls : L.openCalls = D.openCalls) (h : Claims.Sampling.OpenCallsLe H D) :
    L.OpenCallsLe H := by
  intro x s hs u v w
  rw [hparse] at hs
  rw [hcommit, hreveal, hcalls]
  exact h x s hs u v w

end Claims.SamplingGeneric
