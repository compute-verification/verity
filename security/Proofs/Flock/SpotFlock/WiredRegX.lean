module

public import Proofs.Flock.SpotFlock.WiredKS
public import Proofs.Flock.SamplingGeneric.TwoCountX
import Proofs.Flock.Masked
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Sampling.Deployed

/-! SPOT[Flock]'s two-stage audit at the connected program at the committed trace, fixed before the sample (`ksRegTwoX_connected`, `exfilRegTwoX_connected`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot

variable (h : Held) {ω : ℕ → ℕ} {keep : ℕ → Bool}

theorem sessionsTwo_wired (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (hs : StmtOpensWired h ω keep) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeWired h ω keep) iroots
      (twoWired h ω) (rsWired h ω keep) (cap h)).Sessions FlockDeploy.sha512 :=
  SamplingDeploy.sessionsTwo_stage (baseWired h ω keep) (reg h) (twoWired h ω) (fun _ => L)
    (fun _ => h.stage * L) (fun _ => coinByte) (fun _ => h.stage) (stageOrderWired h ω keep)
    (fun _ => stageByte) iroots (cap h) callsUniform (stageUniform h.stage)
    (regMatchesTwo_wired h hW) (refusesTwo_wired h hW _) (replayCap_wired h hW _) (interiorFixed h _)
    (sessionRunsTo_wired h ω keep _root_.Flock.Readable.relation FlockDeploy.sha512) (sessionsIn_wired h ω keep)
    (sessionOpensTwo_wired h hW hs)

theorem ksRegTwoX_wired (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (hs : StmtOpensWired h ω keep) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeWired h ω keep) iroots
      (twoWired h ω) (rsWired h ω keep) (cap h)).KSRegTwoX FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSRegTwoX FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _))
    (SamplingDeploy.commit_binds _ _) (sessionsTwo_wired h hW hs)

theorem ksRegTwoX_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).KSRegTwoX FlockDeploy.sha512 :=
  ksRegTwoX_wired h (fun _ _ hd => rewires_of_two hd (wiresFirst_of_two h hd)) (stmtOpens_connected h hB hA)

theorem exfilRegTwoX_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).ExfilRegTwoX FlockDeploy.sha512 :=
  Claims.SamplingGeneric.CompiledTwo.exfilRegTwoX_of_ks (ksRegTwoX_connected h hB hA)

end SpotFlock
