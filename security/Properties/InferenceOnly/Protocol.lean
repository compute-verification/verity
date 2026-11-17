module

public import InferenceOnlyCheck.Protocol

/-! What the inference-only policy's core protocol's acceptance means. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties

open Lean (Json)
open InferenceOnlyCheck.Protocol (Params Commitment Evidence instanceOf trustedOf reportOf)

/-- **The inference-only policy's protocol accepts only a complying verdict of the check of record.** On every case `InferenceOnlyCheck.Protocol.check`
accepts, `InferenceOnlyCheck.Theta.evaluate` accepts the instance and trusted input the case gives with a verdict `out` that
the policy check gives on the instance too, whose `complies` is `true` and whose report is the profile. -/
def ProtocolAccepts : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Unit) (prof : Json),
    InferenceOnlyCheck.Protocol.check p c e = .accept pub prof →
      ∃ out, InferenceOnlyCheck.Theta.evaluate (instanceOf p e) (trustedOf p c) = .ok out ∧
        InferenceOnlyCheck.Desc.evaluate (instanceOf p e) = .ok out ∧ out.getObjVal? "complies" = .ok (.bool true) ∧
        prof = reportOf out

end InferenceOnlyCheck.Properties
