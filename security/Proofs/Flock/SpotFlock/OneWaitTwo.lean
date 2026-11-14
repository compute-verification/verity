module

public import Proofs.Flock.SpotFlock.OneWaitClassAt
public import Proofs.Flock.SpotFlock.OffClassAt
public import Proofs.Flock.SamplingGeneric.TwoOneWaitKS
public import Proofs.Flock.SamplingGeneric.TwoCap
import Proofs.Flock.Claims.Prob

/-!
# SPOT[Flock]'s two-stage verifier at the one-wait extraction's boundary trace

`Flock.SecurityProofs.SpotOneWaitTwo : Flock.Guarantees.SpotOneWaitTwo` is the two-stage committed form at the
boundary trace that an extraction of one wait a session fixes (`CompiledTwo.OneWaitTwoXAt`), at SPOT[Flock]'s code
(`SpotFlock.code`), at `SpotKSTwo`'s registration, interior roots, design and replay sample, at SHA-512,
under the two facts of the frame `SpotKSTwo` takes (`SpotFlock.Built`, `SpotFlock.Addresses`).

Its error is `SpotFlock.εoneWaitAt`,

`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(2·runCost + 4·firstHitCost)/2^256`,

with `runCost = T·(1 + e_s·N_s) + n_PU·openCalls` and
`firstHitCost = T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls` at the one-wait extraction's
`e_s = 1 + k`, and every quantity read off the prover and the code:

* `T` is the most SHA-512 calls a run of the prover makes, both parties' counted (`Compiled.runCalls`);
* `N_s` is the expected number of sessions in a run's sample, `K_s` the expected number of classes of the gates the
  sample needs, and `K_m` the expected number of the classes' representatives its sessions need, session by session;
  a proof block needs the gates its reveal opens (`needsAt`) and a gate's class is its key's (`clsAt`, `gateKey`).

So the statement takes no bound on a count or on a mass. Beside the two facts of the frame and the scope, what it
takes is the linear expected-time bound on SHA-512 (`Claims.ExpectedCR`) at three finders built from the prover: the
one-wait compiled verifier's `B_conf,X`, and its two by-class finders at every number `K` of tries they stop at.

What the code supplies, each proved: that what Flock's extractor finds its one wait finds at half the rate, at the
masked record (`oneWaitFinds_zk`); Flock's knowledge soundness there; the commitment's binding, the two-stage
contract and the opening checks' count (`code_contract`); and that a one-wait extraction opens a class of the key
whole or not at all (`code_hwit_oneWait`). `εoneWaitAt_le_513` is the error at numbers. Its nontrivial witness is
`Claims.SamplingGeneric.compilerOneWaitTwoXAt_nontrivial`; its honest instance waits on `Addresses` at a two-stage
registration, as `SpotKSTwo`'s does.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The gates a proof block needs**, at the code: those its reveal opens (`Reveal.needs`). -/
def needsAt (s : AuditStatement) (u : ℕ) : Finset ℕ := (Reveal.needs (partition h) s u).toFinset

/-- **A gate's class**, at the code, among `n` gates: the representative of the gates with its key (`gateKey`: the
proof blocks that need the gate opened, the replay block that holds it, and whether it is on that block's
boundary). -/
noncomputable def clsAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) (n : ℕ) : Fin n → Fin n :=
  Verity.FirstHit.clsOf fun g : Fin n => gateKey h s Pt d g.val

/-- **The error at the one-wait extraction's boundary trace, at the code**:
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
      (((L.toCompiled.sampleNeeds x (needsAt h s) (L.checked x r)).image (clsAt h s Pt d (L.nG x))).card : ℝ)))
    (Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (L.toCompiled.sessCountIn x (needsAt h s)
        (fun l => (L.toCompiled.sampleNeeds x (needsAt h s) l).image (clsAt h s Pt d (L.nG x)))
        (L.checked x r) : ℝ)))

/-- **The error at the one-wait trace, at the code, at one wait a session with every call charged as `513`**:
where a run of the prover calls SHA-512 at most `513·2^80` times, each proof block's wait collects
`k ≤ 12343986` accepted reruns, the sample's three expected counts are at most `2^26`, `2^38` and `2^38`, and the
opening checks of the statement's proof blocks call SHA-512 `n_PU·openCalls ≤ 513·2^80` times, `εoneWaitAt` is at
most `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2^-102` (`CompiledTwo.εclsOneWait_le_513`). -/
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
      (((L.toCompiled.sampleNeeds x (needsAt h s) (L.checked x r)).image (clsAt h s Pt d (L.nG x))).card : ℝ)) ≤
        2 ^ 38)
    (hKm : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (L.toCompiled.sessCountIn x (needsAt h s)
        (fun l => (L.toCompiled.sampleNeeds x (needsAt h s) l).image (clsAt h s Pt d (L.nG x)))
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
session fixes, at SHA-512, at its own frame, where each value has one address.** At `L`, the compiled two-stage
verifier at `SpotKSTwo`'s registration, code, interior roots, design and replay sample, at a statement
in scope (`CompiledTwo.Scope`) and every prover: where SHA-512 meets the linear expected-time bound
(`Claims.ExpectedCR`) at the one-wait compiled verifier's `B_conf,X` and at its two by-class finders, under the
statement's roots and under the interior roots, at every number `K` of tries they stop at, an accepted run samples
a proof block incorrect at the one-wait boundary trace read at the statement with probability at most

`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(2·runCost + 4·firstHitCost)/2^256`

(`SpotFlock.εoneWaitAt`: the two costs at one wait a session, at the most SHA-512 calls a run of the prover makes
and the sample's three expected counts), with the threshold form and the headline at that error
(`CompiledTwo.OneWaitTwoXAt`). -/
def SpotOneWaitTwo : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.Built h → SpotFlock.Addresses h →
    ∀ L, L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
        SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h) (SpotFlock.cap h) →
      ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
          ExpectedCR FlockDeploy.sha512
              (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.oneWait.toCompiled Ω P x) →
            (∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.oneWait.toCompiled.finderXCls FlockDeploy.sha512 P x s
              (SpotFlock.needsAt h s) (SpotFlock.clsAt h s Pt d (L.nG x)) K)) →
            (∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.oneWait.finderICls FlockDeploy.sha512 P x s d.part
              (SpotFlock.needsAt h s) (SpotFlock.clsAt h s Pt d (L.nG x)) K)) →
            L.OneWaitTwoXAt FlockDeploy.sha512 x s Pt d P (SpotFlock.εoneWaitAt h L x s Pt d P)

end Flock.Guarantees

theorem Flock.SecurityProofs.SpotOneWaitTwo : Flock.Guarantees.SpotOneWaitTwo := by
  intro h hB hS L hL x s Pt d hx hs hPt hd Ω _ _ P hBc hX hI
  have hin : Claims.Sampling.InScopeTwo Pt d s := L.inScopeTwo_of hx hs hPt hd
  have hSess : L.Sessions FlockDeploy.sha512 := (SpotFlock.code_contract h hB hS hL).2.1
  have hx' : SpotFlock.parse h x = some s := by
    subst hL
    exact hs
  have hP' : SpotFlock.partition h s = some Pt := by
    subst hL
    exact hPt
  have hd' : SpotFlock.two h s = some d := by
    subst hL
    exact hd
  exact Claims.SamplingGeneric.CompiledTwo.oneWaitTwoXAt_of_event hSess hs hPt hd hin P
    (SpotFlock.code_eventTwoX_oneWait_le_cls_at_budget h hB hS hL hx' hP' hd' hin P
      (needs := SpotFlock.needsAt h s) rfl (cls := SpotFlock.clsAt h s Pt d (L.nG x)) rfl
      (L.toCompiled.runCalls FlockDeploy.sha512 x P)
      (fun ω r => L.toCompiled.runCalls_le FlockDeploy.sha512 x P ω r) le_rfl le_rfl le_rfl hBc hX hI)
