module

public import Proofs.Flock.SpotFlock.WiredKS
public import Proofs.Flock.SamplingGeneric.TwoCountX
import Proofs.Flock.Masked
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# SPOT[Flock]'s two-stage audit at the connected program, at the committed trace

`ksTwo_connected` (`WiredKS`) is the extractor's form, whose witness may read the sample. This file gives the same
at the boundary trace read at the statement (`Compiled.regValX`), fixed before the sample, from the same facts of
the wired code:

* `sessionsTwo_wired`: the two-stage contract at the wired code (`sessionsTwo_stage` at `ksTwo_wired`'s facts).
* `ksRegTwoX_wired`: `CompiledTwo.KSRegTwoX` there (`compilerKSRegTwoX` at Flock's masked record and the
  commitment's binding).
* `ksRegTwoX_connected`, `exfilRegTwoX_connected`: at the code's own wiring (`codeConnected h`), under the three
  premises on that wiring (`WiresFirst`, `WiresMove`, `SourcesNamed`), from SPOT's own frame there and the addresses
  (`BuiltWired`, `AddressesConnected`); what the committed trace serves is counted by
  `CompiledTwo.exfilRegTwoX_of_ks`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot

variable (h : Held) {ω : ℕ → ℕ} {keep : ℕ → Bool}

/-- **The two-stage contract at the wired code**: `sessionsTwo_stage` at `ksTwo_wired`'s facts. -/
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

/-- **The committed form at the wired code**: `compilerKSRegTwoX` at Flock's masked record, the commitment's binding
and the contract (`sessionsTwo_wired`). -/
theorem ksRegTwoX_wired (hW : ∀ s d, two h s = some d → (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (hs : StmtOpensWired h ω keep) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeWired h ω keep) iroots
      (twoWired h ω) (rsWired h ω keep) (cap h)).KSRegTwoX FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSRegTwoX FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _))
    (SamplingDeploy.commit_binds _ _) (sessionsTwo_wired h hW hs)

/-- **SPOT[Flock]'s two-stage committed form at the connected program**: `CompiledTwo.KSRegTwoX` at the code's own
wiring, under `ksTwo_connected`'s premises and from its two facts: the wiring map's rewiring the members' programs
where the design evaluates, and the sessions' openings there. -/
theorem ksRegTwoX_connected (hF : WiresFirst h) (_ : WiresMove h) (hS : SourcesNamed h)
    (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).KSRegTwoX FlockDeploy.sha512 :=
  ksRegTwoX_wired h (fun _ _ hd => rewires_of_two hd hF) (stmtOpens_connected h hF hS hB hA)

/-- **What the committed trace serves at the connected program**: `CompiledTwo.ExfilRegTwoX` from
`ksRegTwoX_connected` (`CompiledTwo.exfilRegTwoX_of_ks`). -/
theorem exfilRegTwoX_connected (hF : WiresFirst h) (hM : WiresMove h) (hS : SourcesNamed h)
    (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).ExfilRegTwoX FlockDeploy.sha512 :=
  Claims.SamplingGeneric.CompiledTwo.exfilRegTwoX_of_ks (ksRegTwoX_connected h hF hM hS hB hA)

end SpotFlock
