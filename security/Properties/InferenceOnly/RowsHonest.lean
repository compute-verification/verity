module

public import Properties.InferenceOnly.RowsHonest.Case
public import InferenceOnlyVerifier.RowsProtocol

/-!
# The compiled rows check accepts an honest case (`RowsHonestAccepted`)

`RowsHonest.Case` holds an honest case of the rows check as the bytes the composite hands it: the parameters (two
templates, each naming its circuit file by its SHA-512, with `Γ` and `α` at a block's rows and fresh bits), the run's
own record and the evidence. `RowsHonestAccepted` says that they decode under the rows check's decoders and that its
compiled check accepts them on some blobs, which the parameters' digests name. The honest witness of `RowsComplies`
takes it, and nothing in Lean proves it yet: the kernel doesn't evaluate the circuit files' parse (`Flock.ProvedScope.circuitOf`
on files of tens of megabytes) or the SHA-512s the check computes.

It is what `test_inference_only_rows_protocol.py` checks: `inference-only-rows check` accepts that case on the two
circuit files its digests name, as the reference does, and the module is that test's rendering of the case.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record

/-- **The compiled rows check accepts the honest case** (witnessed outside Lean: `inference-only-rows check` accepts
it). -/
def RowsHonestAccepted : Prop :=
  ∃ (p : InferenceOnly.RowsProtocol.Params) (c : InferenceOnlyVerifier.Protocol.Commitment)
    (e : InferenceOnly.RowsProtocol.Evidence) (bs : Core.Blobs) (pub : Unit) (prof : Lean.Json),
    (Core.canonicalJson RowsHonest.params.toUTF8 >>= InferenceOnly.RowsProtocol.paramsOf) = .ok p ∧
      InferenceOnlyVerifier.Protocol.commitmentOf RowsHonest.commitment.toUTF8 = .ok c ∧
      InferenceOnly.RowsProtocol.evidenceOf RowsHonest.evidence.toUTF8 = .ok e ∧
      InferenceOnly.RowsProtocol.checkBlobs p c e bs = .accept pub prof

end InferenceOnlyVerifier.Properties.Record
