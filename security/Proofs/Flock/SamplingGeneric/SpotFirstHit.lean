module

public import Proofs.Flock.SamplingGeneric.Guarantees
public import Proofs.Flock.SamplingGeneric.TwoFirstHitKS
public import Proofs.Flock.SpotFlock.ConnectedClassAt

/-!
# SPOT[Flock]'s two-stage verifier at the first-hit error

`Flock.SecurityProofs.SpotFirstHitTwo : Flock.Guarantees.SpotFirstHitTwo` is `CompiledTwo.ksFirstHitTwoX` at
SPOT[Flock]'s code at the connected program (`SpotFlock.codeConnected`), at `SpotKSTwo`'s registration, interior
roots, design, replay sample and replay limit there: what the two-stage committed form at the first-hit error
concludes (`CompiledTwo.FirstHitTwoXAt`), at SHA-512, under the two facts of the frame `SpotKSTwo` takes
(`SpotFlock.BuiltWired` at the code's own wiring, `SpotFlock.AddressesConnected`), at two costs that are functions of
the prover.

The form's error is a function of two costs in SHA-512 calls, a run's and a first-hit finder's, and the form takes
them with three bounds on masses of the prover's runs and their extractions (`CompiledTwo.BconfXLe`, `OffRegXLe`,
`OffIregLe`). The guarantee takes none of the three. Each is proved at the code from the linear expected-time bound on
SHA-512 (`Claims.ExpectedCR`) at the one machine that would find the collision (`SpotFlock.connected_bconfX_le_at`,
`connected_offRegX_le_cls_at`, `connected_offIreg_le_cls_at`): `B_conf,X`, and the two by-class first-hit finders,
under the statement's roots and under the interior roots, at every number of tries a finder stops at. So the
guarantee's only premise on SHA-512 is that bound at those three machines, and its two costs are closed expressions:

* `c_R = T·(1 + e_s·N_s) + n_PU·openCalls` (`Compiled.runCost`) and
* `c_F = T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls` (`Compiled.firstHitCost`),

where `T` is the most SHA-512 calls one run of the prover makes, both parties' calls counted (`Compiled.runCalls`),
and `N_s`, `K_s` and `K_m` are three expectations over the run's coins: the sessions of a run's sample, the classes of
the gates the sample needs, and the classes' representatives its sessions need, session by session. A proof block
needs the gates its reveal opens at the connected program (`SpotFlock.needsConnected`) and a gate's class is its
key's there (`SpotFlock.clsConnected`, `SpotFlock.gateKeyConnected`).
The statement fixes each of the four by an equation, as it fixes the verifier, so none is a hypothesis on the prover:
no budget and no count is assumed.

The guarantee's statement writes all of this out at the code, so that each premise it takes is a hypothesis of the
statement itself and the lock's record names it: the two facts of the frame, the scope (`CompiledTwo.Scope`) and
`Claims.ExpectedCR`. Flock's knowledge soundness is proved at the code as `SpotKSTwo` has it (`flockArgKS`), and the
commitment's binding and the two-stage contract are `SpotFlock.connected_contract`'s. Its nontrivial witness is
`Claims.SamplingGeneric.compilerFirstHitTwoXAt_nontrivial`; its honest instance waits on `AddressesConnected` at a
two-stage registration, as `SpotKSTwo`'s does.
-/

@[expose] public section

namespace Flock.Guarantees

open Claims

/-- **SPOT[Flock]'s two-stage verifier meets the committed form at the first-hit error, at SHA-512, at the connected
program, with both costs functions of the prover**: at a held code `h` whose frame is SPOT's own at the connected
program (`SpotFlock.BuiltWired`) and at which each gate's value has one address there
(`SpotFlock.AddressesConnected`), at `L`, the compiled two-stage verifier at `SpotKSTwo`'s registration, code at the
connected program (`SpotFlock.codeConnected`), interior roots, design, replay sample and replay limit there, at a
statement in scope (`CompiledTwo.Scope`) and every prover, with `needs` the gates each proof block's reveal opens there
and `cls` the classes of the gates' keys there. Where the linear expected-time bound on SHA-512 (`Claims.ExpectedCR`)
holds at `B_conf,X` and at the two by-class first-hit finders at every number `K` of tries a finder stops at (the
finder under the statement's roots, `Compiled.finderXCls`, and the one under the interior roots,
`CompiledTwo.finderICls`), an accepted run samples a proof block incorrect at the boundary trace read at the statement
with probability at most `κ_s + √(k·γ_s·Pr[B]) + (2·c_R + 4·c_F)/2^256`, with the threshold form and the headline at that error
(`CompiledTwo.FirstHitTwoXAt`), at

* `c_R = T·(1 + e_s·N_s) + n_PU·openCalls` (`Compiled.runCost`) and
* `c_F = T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls` (`Compiled.firstHitCost`).

`T` is the most SHA-512 calls one run of the prover makes, both parties' calls counted (`Compiled.runCalls`); `N_s`
is the expected number of sessions in a run's sample, `K_s` the expected number of classes of the gates the sample
needs, and `K_m` the expected number of the classes' representatives its sessions need, session by session, each over
the run's coins. The four are fixed by equations and restrict no prover. The statement takes no bound on a mass of
the prover's runs: the three the form takes are proved at the code from the expected-time bound. -/
def SpotFirstHitTwo : Prop :=
  ∀ h : SpotFlock.Held,
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) → SpotFlock.AddressesConnected h →
    ∀ L, L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
        (SpotFlock.codeConnected h) SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h)
        (SpotFlock.cap h) →
      ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
        ∀ needs : ℕ → Finset ℕ, needs = SpotFlock.needsConnected h s →
        ∀ cls : Fin (L.nG x) → Fin (L.nG x), cls = SpotFlock.clsConnected h s Pt d (L.nG x) →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
          ExpectedCR FlockDeploy.sha512
              (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x) →
            (∀ K : ℕ, ExpectedCR FlockDeploy.sha512
              (L.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K)) →
            (∀ K : ℕ, ExpectedCR FlockDeploy.sha512
              (L.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) →
          ∀ T : ℕ, T = L.toCompiled.runCalls FlockDeploy.sha512 x P →
          ∀ Ns : ℝ, Ns = Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
              ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) →
          ∀ Ks : ℝ, Ks = Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
              (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) →
          ∀ Km : ℝ, Km = Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
              (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
                (L.checked x r) : ℝ)) →
            L.FirstHitTwoXAt FlockDeploy.sha512 x s Pt d P (L.toCompiled.runCost x T Ns)
              (L.toCompiled.firstHitCost x T Ns Ks Km)

end Flock.Guarantees

theorem Flock.SecurityProofs.SpotFirstHitTwo : Flock.Guarantees.SpotFirstHitTwo := by
  intro h hB hA L hL x s Pt d hx hs hPt hd needs hneeds cls hcls Ω _ _ P hBc hX hI T hT Ns hNs Ks hKs Km hKm
  obtain ⟨hbind, hSess, -⟩ := SpotFlock.connected_contract h hB hA hL
  have hin : Claims.Sampling.InScopeTwo Pt d s := L.inScopeTwo_of hx hs hPt hd
  have hrun := fun (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) =>
    (Claims.SamplingGeneric.Compiled.runCalls_le FlockDeploy.sha512 L.toCompiled x P ω r).trans hT.ge
  exact Claims.SamplingGeneric.CompiledTwo.ksFirstHitTwoX (H := FlockDeploy.sha512) (L := L)
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _))
    hbind hSess x s Pt d hx hs hPt hd Ω P (L.toCompiled.runCost x T Ns) (L.toCompiled.firstHitCost x T Ns Ks Km)
    (SpotFlock.connected_bconfX_le_at h hB hA hL x P T hrun hNs.ge hBc)
    (SpotFlock.connected_offRegX_le_cls_at h hB hA hL hs hPt hd hin P hneeds hcls T hrun hNs.ge hKs.ge hKm.ge hX)
    (SpotFlock.connected_offIreg_le_cls_at h hB hA hL hs hPt hd hin P hneeds hcls T hrun hNs.ge hKs.ge hKm.ge hI)
