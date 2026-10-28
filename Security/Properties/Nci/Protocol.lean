import NciCheck.Protocol

/-!
# What NCI's core protocol's acceptance means

`NciCheck.Protocol.check` (`verity/core/protocols/network/nci/lean/NciCheck/Protocol.lean`, package `nci_check`) is NCI
as a core protocol, `verity/inference-only/v0`, a component of both composites (`Core.Onsite.components`,
`Core.Remote.components`): it runs the check of record, `NciCheck.Baseline.evaluate`, on the instance its params and
evidence give (`instanceOf`) and the trusted input its params and commitment give (`trustedOf`).

* `ProtocolAccepts`: on every case it accepts, the check of record accepts that instance and trusted input with a
  verdict whose `complies` is `true`, the policy check (`NciCheck.Desc.evaluate`) gives that same verdict on the
  instance, and the profile is the verdict's report (`reportOf`).

With `Properties.Composition`'s `AnswerAccepts` and `AcceptMeans` it takes an accepted composite's answer to an
accepted check of record; `BaselineProvenance` (`Properties.Nci.Baseline`) and `DescWitnesses`
(`Properties.NciDesc.Verdict`) say what that verdict means: θ holds what the deployment's first record registers, and
the declaration complies.

Proved as `NciCheck.Properties.SecurityProofs.ProtocolAccepts : NciCheck.Properties.ProtocolAccepts` in
`Proofs.Properties.Nci.Protocol`.
-/

namespace NciCheck.Properties

open Lean (Json)
open NciCheck.Protocol (Params Commitment Evidence instanceOf trustedOf reportOf)

/-- **NCI's protocol accepts only a complying verdict of the check of record.** On every case `NciCheck.Protocol.check`
accepts, `NciCheck.Baseline.evaluate` accepts the instance and trusted input the case gives with a verdict `out` that
the policy check gives on the instance too, whose `complies` is `true` and whose report is the profile. -/
def ProtocolAccepts : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Unit) (prof : Json),
    NciCheck.Protocol.check p c e = .accept pub prof →
      ∃ out, NciCheck.Baseline.evaluate (instanceOf p e) (trustedOf p c) = .ok out ∧
        NciCheck.Desc.evaluate (instanceOf p e) = .ok out ∧ out.getObjVal? "complies" = .ok (.bool true) ∧
        prof = reportOf out

end NciCheck.Properties
