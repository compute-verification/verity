module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Fails
public import Proofs.Flock.SamplingGeneric.Split
public import Proofs.Flock.SamplingGeneric.Assemble
public import Proofs.Flock.SamplingGeneric.Coins
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.Flock.SamplingGeneric.Count
public import Proofs.Flock.SamplingGeneric.Bits

/-!
# The sampled compiler in the claims core, at Flock

The compiled extractor's failure event, its probability and `Compiled.KS` from the split are in
`SamplingGeneric/Fails.lean`.

`Arg.ofFlock` is Flock's `E^rep` at a decoder with Flock's errors, so its knowledge soundness is `Flock.KS`'s
(`flockArgKS`, by unfolding: its coins, extractor, `κ` and `γ` are `Flock.E`'s, `Flock.κ`'s and `Flock.γ`'s), and its
rounds are its shape's (`arg_ofFlock_wf`). So the compiler's knowledge soundness gives Flock's (`flockKS_of_compilerKS`);
it is two steps (`SamplingGeneric/Split.lean`: `KSOf`, `SessionKSOf`, composed by `compilerKS_of`), the second proved
from the session transfer and the session's fork (`SamplingGeneric/Session.lean`: `sessionKSOf`), the first from
`KSOf`'s deterministic half, a failed extraction giving the failure event or a `Bext` collision (`split`,
`SamplingGeneric/Assemble.lean`: `ksOf`). So **`CompilerKS` is proved** (`compilerKS`), and with it Flock's instance
(`flockKS`) and the count form (`compilerKSCount`, `SamplingGeneric/Count.lean`). `CompilerBits` follows from it and
the extractor's expected reruns and own calls (`SamplingGeneric/Bits.lean`, `ExpOwnOf`). A session's coins in a
compiled run, `coinsOf` and `place`, invert each other (`SamplingGeneric/Coins.lean`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Inhabited Coin] [DecidableEq Col] in
theorem arg_ofFlock_wf (D : Claims.Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    (Arg.ofFlock D decode).WF := fun _ => rfl

theorem flockArgKS : FlockArgKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ decode hKS hdec
  exact hKS decode hdec

theorem flockKS_of_compilerKS (h : CompilerKS) : FlockKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ decode L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D decode) L H (arg_ofFlock_wf D decode) (flockArgKS H D decode hKS hdec) hB hS

/-- **The sampled compiler is knowledge sound**: `KSOf` and `SessionKSOf` (`compilerKS_of`). -/
theorem compilerKS : CompilerKS := compilerKS_of ksOf sessionKSOf

/-- **Flock is an instance**: the compiler at `Arg.ofFlock`, from `Flock.KS` and its decoder's list bound. -/
theorem flockKS : FlockKS (Coin := Coin) (Col := Col) (T := T) := flockKS_of_compilerKS compilerKS

/-- The count form, at the miss of a count. -/
theorem compilerKSCount : CompilerKSCount := compilerKSCount_of compilerKS

end Claims.SamplingGeneric
