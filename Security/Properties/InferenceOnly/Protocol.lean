import InferenceOnlyCheck.Protocol

/-!
# What the inference-only policy's core protocol's acceptance means

`InferenceOnlyCheck.Protocol.check` (`verity/core/protocols/network/inference_only/lean/InferenceOnlyCheck/Protocol.lean`, package `inference_only_check`) is the
inference-only policy as a core protocol, `verity/inference-only/v0`, a subprotocol of both composites
(`Core.Onsite.protocol`, `Core.Remote.protocol`): once the run's record names the declared circuit (`names`), it runs the
check of record, `InferenceOnlyCheck.Baseline.evaluate`, on the instance its params and evidence give (`instanceOf`) and the
trusted input its params and commitment give (`trustedOf`).

* `ProtocolAccepts`: on every case it accepts, the check of record accepts that instance and trusted input with a
  verdict whose `complies` is `true`, the policy check (`InferenceOnlyCheck.Desc.evaluate`) gives that same verdict on the
  instance, and the profile is the verdict's report (`reportOf`).

With `Properties.Composition`'s `AnswerAccepts` and `AcceptMeans` it takes an accepted onsite or remote answer to an
accepted check of record; `BaselineProvenance` (`Properties.InferenceOnly.Baseline`) and `DescWitnesses` (`Properties.InferenceOnlyDesc.Verdict`)
say what that verdict means: θ holds what the deployment's first record registers, and the declaration complies.

Proved as `InferenceOnlyCheck.Properties.SecurityProofs.ProtocolAccepts : InferenceOnlyCheck.Properties.ProtocolAccepts` in
`Proofs.Properties.InferenceOnly.Protocol`.
-/

namespace InferenceOnlyCheck.Properties

open Lean (Json)
open InferenceOnlyCheck.Protocol (Params Commitment Evidence instanceOf trustedOf reportOf)

/-- **The inference-only policy's protocol accepts only a complying verdict of the check of record.** On every case `InferenceOnlyCheck.Protocol.check`
accepts, `InferenceOnlyCheck.Baseline.evaluate` accepts the instance and trusted input the case gives with a verdict `out` that
the policy check gives on the instance too, whose `complies` is `true` and whose report is the profile. -/
def ProtocolAccepts : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Unit) (prof : Json),
    InferenceOnlyCheck.Protocol.check p c e = .accept pub prof →
      ∃ out, InferenceOnlyCheck.Baseline.evaluate (instanceOf p e) (trustedOf p c) = .ok out ∧
        InferenceOnlyCheck.Desc.evaluate (instanceOf p e) = .ok out ∧ out.getObjVal? "complies" = .ok (.bool true) ∧
        prof = reportOf out

end InferenceOnlyCheck.Properties
