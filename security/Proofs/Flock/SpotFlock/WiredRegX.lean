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

* `sessionsTwo_wired`: the two-stage contract at the wired code where the check passes (`sessionsTwo_stage` at
  `ksTwo_wired`'s facts).
* `ksRegTwoX_wired`, `ksRegTwoX_refused`: `CompiledTwo.KSRegTwoX` there (`compilerKSRegTwoX` at Flock's masked
  record and the commitment's binding), and where the check fails, with no premise.
* `ksRegTwoX_connected`, `exfilRegTwoX_connected`: at the code's own wiring (`codeConnected h`), from SPOT's own
  frame there and the addresses (`BuiltWired`, `AddressesWired`); what the committed trace serves is counted by
  `CompiledTwo.exfilRegTwoX_of_ks`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot

variable (h : Held) {ω : ℕ → ℕ} {keep : ℕ → Bool}

/-- **The two-stage contract at the wired code, where the check passes**: `sessionsTwo_stage` at `ksTwo_wired`'s
facts. -/
theorem sessionsTwo_wired (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) (hs : StmtOpensWired h ω keep) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeWired h ω keep true) iroots
      (twoGated h ω true) (rsWired h ω keep true) (cap h)).Sessions FlockDeploy.sha512 :=
  SamplingDeploy.sessionsTwo_stage (baseWired h ω keep true) (reg h) (twoGated h ω true) (fun _ => L)
    (fun _ => h.stage * L) (fun _ => coinByte) (fun _ => h.stage) (stageOrderWired h ω keep true)
    (fun _ => stageByte) iroots (cap h) callsUniform (stageUniform h.stage)
    (regMatchesTwo_wired h hW) (refusesTwo_wired h hW _) (replayCap_wired h hW _) (interiorFixed h _)
    (sessionRunsTo_wired h ω keep _root_.Flock.Readable.relation FlockDeploy.sha512) (sessionsIn_wired h ω keep)
    (sessionOpensTwo_wired h hW hs)

/-- **The committed form at the wired code, where the check passes**: `compilerKSRegTwoX` at Flock's masked record,
the commitment's binding and the contract (`sessionsTwo_wired`). -/
theorem ksRegTwoX_wired (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) (hs : StmtOpensWired h ω keep) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeWired h ω keep true) iroots
      (twoGated h ω true) (rsWired h ω keep true) (cap h)).KSRegTwoX FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSRegTwoX FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _))
    (SamplingDeploy.commit_binds _ _) (sessionsTwo_wired h hW hs)

variable (ω keep)

/-- **The committed form at the wired code, where the check fails**: the design does not evaluate, so no statement
is in scope. -/
theorem ksRegTwoX_refused :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeWired h ω keep false) iroots
      (twoGated h ω false) (rsWired h ω keep false) (cap h)).KSRegTwoX FlockDeploy.sha512 := by
  intro x hx
  obtain ⟨s, Pt, d, -, -, h2, -⟩ := hx
  have h2' : twoGated h ω false s = some d := h2
  rw [twoGated_false] at h2'
  cases h2'

/-- **SPOT[Flock]'s two-stage committed form at the connected program**: `CompiledTwo.KSRegTwoX` at the code's own
wiring, behind its check, as `ksTwo_connected`: where the check passes from the wiring map's rewiring the members'
programs and the sessions' openings there, and with no premise where it fails. -/
theorem ksRegTwoX_connected (hB : BuiltWired h (wireMap h) (returned h))
    (hA : AddressesWired h (wireMap h) (returned h)) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).KSRegTwoX FlockDeploy.sha512 := by
  have key : ∀ ok : Bool, wiredOk h = ok →
      (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
        (codeWired h (wireMap h) (returned h) ok) iroots (twoGated h (wireMap h) ok)
        (rsWired h (wireMap h) (returned h) ok) (cap h)).KSRegTwoX FlockDeploy.sha512 := by
    intro ok hok
    cases ok with
    | false => exact ksRegTwoX_refused h (wireMap h) (returned h)
    | true =>
      have hW := rewires_of_wiredOk h hok
      exact ksRegTwoX_wired h hW (stmtOpensWired h (wireMap h) (returned h) hB hA hW)
  exact key (wiredOk h) rfl

/-- **What the committed trace serves at the connected program**: `CompiledTwo.ExfilRegTwoX` from
`ksRegTwoX_connected` (`CompiledTwo.exfilRegTwoX_of_ks`). -/
theorem exfilRegTwoX_connected (hB : BuiltWired h (wireMap h) (returned h))
    (hA : AddressesWired h (wireMap h) (returned h)) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) iroots
      (twoConnected h) (rsConnected h) (cap h)).ExfilRegTwoX FlockDeploy.sha512 :=
  Claims.SamplingGeneric.CompiledTwo.exfilRegTwoX_of_ks (ksRegTwoX_connected h hB hA)

end SpotFlock
