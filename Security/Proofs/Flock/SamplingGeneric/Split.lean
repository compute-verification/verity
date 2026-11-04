module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Fails

/-!
# The sampled compiler's knowledge soundness, in two steps

`CompilerKS` in the two steps `Claims.Sampling`'s proofs take at Flock, each a statement a proof can target alone, over
`A`'s knowledge soundness at the sampled proof blocks' sessions (`Compiled.SessionKS`, at the found event
`Compiled.Found`, `SamplingGeneric/Fails.lean`):

* `KSOf` (`Sampling.KS_of`, generic): `SessionKS`, the commitment's binding and the contract give `Compiled.KS`. A
  failed extraction at an accepted run is a failed session or a collision among the extraction's SHA-512 calls; failed
  sessions are rare where the sample is likely to hit an incorrect proof block (`Sessions.draws`). From
  `Compiled.KS_of_sessionKS`, it needs only that split.
* `SessionKSOf` (`Sampling.SessionKS_of`, generic): `A`'s `KS` and the contract give `SessionKS`: at each session
  `A`'s knowledge soundness at the prover `Sessions.runs` shows, a session's fork being the compiled verifier's at one
  of its `k(x)` rounds.

`compilerKS_of` composes them.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims

/-- **KS from the sessions' KS** (`Sampling.KS_of`, generic): a compiled sampled verifier that meets the contract, with
a binding commitment, whose sessions are knowledge sound (`SessionKS`), meets `KS`. -/
def KSOf : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), L.commit.Binds H → L.Sessions H → L.SessionKS H → L.KS H

/-- **The sessions' KS from the proof system's** (`Sampling.SessionKS_of`, generic): over a knowledge-sound proof
system, a compiled sampled verifier that meets the contract has knowledge-sound sessions. -/
def SessionKSOf : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.KS H → L.Sessions H → L.SessionKS H

theorem compilerKS_of (hks : KSOf) (hsess : SessionKSOf) : CompilerKS := by
  intro Coin _ _ _ A L H hA hB hS
  exact hks Coin A L H hB hS (hsess Coin A L H hA hS)

end Claims.SamplingGeneric
