module

public import Proofs.Flock.SpotFlock.ConnectedClassAt
public import Proofs.Flock.SamplingGeneric.TwoOneWaitKS
public import Proofs.Flock.SamplingGeneric.TwoCap
import Proofs.Flock.Claims.Prob

/-! SPOT[Flock]'s two-stage verifier at the boundary trace an extraction of one wait a session fixes, at the connected
program (`SpotOneWaitTwo`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The error at the one-wait extraction's boundary trace, at the connected program**:
`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(2·runCost + 4·firstHitCost)/2^256` (`CompiledTwo.εclsOneWait`), at `T` the most
SHA-512 calls a run of the prover makes (`Compiled.runCalls`), `N_s` the expected number of sessions in a run's
sample, `K_s` the expected number of classes of the gates the sample needs and `K_m` the expected number of the
classes' representatives its sessions need, session by session. -/
noncomputable def εoneWaitAt
    (L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)))
    (x : Msg) (s : AuditStatement) (Pt : Partition) (d : Design.Two) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : ℝ :=
  L.εclsOneWait FlockDeploy.sha512 x P (L.toCompiled.runCalls FlockDeploy.sha512 x P)
    (Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)))
    (Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (((L.toCompiled.sampleNeeds x (needsConnected h s) (L.checked x r)).image
        (clsConnected h s Pt d (L.nG x))).card : ℝ)))
    (Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (L.toCompiled.sessCountIn x (needsConnected h s)
        (fun l => (L.toCompiled.sampleNeeds x (needsConnected h s) l).image (clsConnected h s Pt d (L.nG x)))
        (L.checked x r) : ℝ)))

theorem εoneWaitAt_le_513
    (L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)))
    (x : Msg) (s : AuditStatement) (Pt : Partition) (d : Design.Two) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (hT : L.toCompiled.runCalls FlockDeploy.sha512 x P ≤ 513 * 2 ^ 80)
    (hk : ∀ u < L.toCompiled.nPU x, (Claims.SamplingGeneric.Arg.ofFlock
      (FlockDeploy.deployZk _root_.Flock.Readable.relation)).k (L.toCompiled.sh x u) ≤ 12343986)
    (hNs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ 2 ^ 26)
    (hKs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (((L.toCompiled.sampleNeeds x (needsConnected h s) (L.checked x r)).image
        (clsConnected h s Pt d (L.nG x))).card : ℝ)) ≤ 2 ^ 38)
    (hKm : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (L.toCompiled.sessCountIn x (needsConnected h s)
        (fun l => (L.toCompiled.sampleNeeds x (needsConnected h s) l).image (clsConnected h s Pt d (L.nG x)))
        (L.checked x r) : ℝ)) ≤ 2 ^ 38)
    (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    εoneWaitAt h L x s Pt d P ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB FlockDeploy.sha512 L.V P x) + 2 ^ (-(102 : ℝ)) := by
  unfold εoneWaitAt
  refine Claims.SamplingGeneric.CompiledTwo.εclsOneWait_le_513 FlockDeploy.sha512 L x P ?_ ?_ ?_ hT hk hNs hKs hKm
    hown
  all_goals exact Ex_nonneg (by intro r; exact Nat.cast_nonneg _)

end SpotFlock

namespace Flock.Guarantees

open Claims

/-- **SPOT[Flock]'s two-stage verifier meets the committed form at the boundary trace an extraction of one wait a
session fixes, at SHA-512, at the connected program.** At a held code `h` whose frame is SPOT's own at the connected
program (`SpotFlock.BuiltWired`) and at which each gate's value has one address there (`SpotFlock.AddressesConnected`),
at `L`, the compiled two-stage verifier at `SpotKSTwo`'s registration, code at the connected program
(`SpotFlock.codeConnected`), interior roots, design and replay sample there, at a statement in scope
(`CompiledTwo.Scope`) and every prover: where SHA-512 meets the linear expected-time bound (`Claims.ExpectedCR`) at
the one-wait compiled verifier's `B_conf,X` and at its two by-class finders, under the statement's roots and under
the interior roots, at every number `K` of tries they stop at, an accepted run samples a proof block incorrect at the
one-wait boundary trace read at the statement with probability at most

`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(2·runCost + 4·firstHitCost)/2^256`

(`SpotFlock.εoneWaitAt`: the two costs at one wait a session, at the most SHA-512 calls a run of the prover makes
and the sample's three expected counts), with the threshold form and the headline at that error
(`CompiledTwo.OneWaitTwoXAt`). -/
def SpotOneWaitTwo : Prop :=
  ∀ h : SpotFlock.Held,
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) → SpotFlock.AddressesConnected h →
    ∀ L, L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
        (SpotFlock.codeConnected h) SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h)
        (SpotFlock.cap h) →
      ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
          ExpectedCR FlockDeploy.sha512
              (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.oneWait.toCompiled Ω P x) →
            (∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.oneWait.toCompiled.finderXCls FlockDeploy.sha512 P x s
              (SpotFlock.needsConnected h s) (SpotFlock.clsConnected h s Pt d (L.nG x)) K)) →
            (∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.oneWait.finderICls FlockDeploy.sha512 P x s d.part
              (SpotFlock.needsConnected h s) (SpotFlock.clsConnected h s Pt d (L.nG x)) K)) →
            L.OneWaitTwoXAt FlockDeploy.sha512 x s Pt d P (SpotFlock.εoneWaitAt h L x s Pt d P)

end Flock.Guarantees

theorem Flock.SecurityProofs.SpotOneWaitTwo : Flock.Guarantees.SpotOneWaitTwo := by
  intro h hB hA L hL x s Pt d hx hs hPt hd Ω _ _ P hBc hX hI
  have hin : Claims.Sampling.InScopeTwo Pt d s := L.inScopeTwo_of hx hs hPt hd
  have hSess : L.Sessions FlockDeploy.sha512 := (SpotFlock.connected_contract h hB hA hL).2.1
  exact Claims.SamplingGeneric.CompiledTwo.oneWaitTwoXAt_of_event hSess hs hPt hd hin P
    (SpotFlock.connected_eventTwoX_oneWait_le_cls_at_budget h hB hA hL hs hPt hd hin P
      (needs := SpotFlock.needsConnected h s) rfl (cls := SpotFlock.clsConnected h s Pt d (L.nG x)) rfl
      (L.toCompiled.runCalls FlockDeploy.sha512 x P)
      (fun ζ r => L.toCompiled.runCalls_le FlockDeploy.sha512 x P ζ r) le_rfl le_rfl le_rfl hBc hX hI)
