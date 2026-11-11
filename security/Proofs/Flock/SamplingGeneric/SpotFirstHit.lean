module

public import Proofs.Flock.SamplingGeneric.Guarantees
public import Proofs.Flock.SamplingGeneric.TwoFirstHitKS

/-!
# SPOT[Flock]'s two-stage verifier at the first-hit error

`Flock.SecurityProofs.SpotFirstHitTwo : Flock.Guarantees.SpotFirstHitTwo` is `CompiledTwo.ksFirstHitTwoX` at
SPOT[Flock]'s code (`SpotFlock.code`), at `SpotKSTwo`'s registration, interior roots, design, replay sample and replay
cap: the two-stage committed form at the first-hit error (`CompiledTwo.KSFirstHitTwoX`), at SHA-512, under the two
facts of the frame `SpotKSTwo` takes (`SpotFlock.Built`, `SpotFlock.Addresses`).

The form's error is a function of two costs, a run's and a first-hit finder's, and the form takes the costs with the
three bounds that make them costs. The guarantee's statement writes the form out at the code, so that each premise it
takes is a hypothesis of the statement itself and the lock's record names it: the two facts of the frame, the scope
(`CompiledTwo.Scope`) and the three bounds (`CompiledTwo.BconfXLe`, `OffRegXLe`, `OffIregLe`). Flock's knowledge
soundness, the commitment's binding and the two-stage contract are proved at the code as `SpotKSTwo` has them
(`flockArgKS`, `SamplingDeploy.commit_binds`, `SpotFlock.sessionsTwo_spot`). Its nontrivial witness is
`Claims.SamplingGeneric.compilerFirstHitTwoXAt_nontrivial`; its honest instance waits on `Addresses` at a two-stage
registration, as `SpotKSTwo`'s does.
-/

@[expose] public section

namespace Flock.Guarantees

open Claims

/-- **SPOT[Flock]'s two-stage verifier meets the committed form at the first-hit error, at SHA-512, at its own frame,
where each value has one address**: at `L`, the compiled two-stage verifier at `SpotKSTwo`'s registration, code,
interior roots, design, replay sample and replay cap, `CompiledTwo.KSFirstHitTwoX` holds, written out so that every
premise it takes is a hypothesis here. At a statement in scope (`CompiledTwo.Scope`), every prover and every two costs
`c_R` and `c_F` in SHA-512 calls with `Pr[B_conf,X] ≤ 2·c_R/2^256` (`CompiledTwo.BconfXLe`) and each of the two
off-trace masses at most `2·c_F/2^256` (`CompiledTwo.OffRegXLe`, `CompiledTwo.OffIregLe`: an accepted run's extractions
open a gate away from the boundary trace read at the statement, or away from its replay block's committed interior):
an accepted run samples a proof block incorrect at that trace with probability at most
`κ_s + √(k·γ_s·Pr[B]) + (2·c_R + 4·c_F)/2^256`, with the threshold form and the headline at that error
(`CompiledTwo.FirstHitTwoXAt`). -/
def SpotFirstHitTwo : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.Built h → SpotFlock.Addresses h →
    ∀ L, L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
        SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h) (SpotFlock.cap h) →
      ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (cR cF : ℝ),
          L.BconfXLe FlockDeploy.sha512 x P cR → L.OffRegXLe FlockDeploy.sha512 x P cF →
            L.OffIregLe FlockDeploy.sha512 x P d.part cF → L.FirstHitTwoXAt FlockDeploy.sha512 x s Pt d P cR cF

end Flock.Guarantees

theorem Flock.SecurityProofs.SpotFirstHitTwo : Flock.Guarantees.SpotFirstHitTwo := by
  intro h hB hS L hL
  subst hL
  exact Claims.SamplingGeneric.CompiledTwo.ksFirstHitTwoX (H := FlockDeploy.sha512)
    (L := SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
      SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h) (SpotFlock.cap h))
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _))
    (SamplingDeploy.commit_binds (SpotFlock.code h) _)
    (SpotFlock.sessionsTwo_spot h
      (SpotFlock.sessionOpensTwo_of_frameIs h _ (SpotFlock.embeds_layoutFor h) (SpotFlock.frameIs_built h hB hS)))
