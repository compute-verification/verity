import CoreOnsite
import Properties.SampledProofs.OneStageZk

/-!
# Onsite Verity's sampled proofs subprotocol: what onsite's acceptance says of the proofs

The property of `Core.Onsite.protocol.answer` on the caller's blobs, the whole of what onsite's caller runs, stated
against it as it runs. It takes no premise.

`SampledProofsAccepts`: when onsite accepts, its parameters and evidence decode to the components' (`decodeParams`,
`decodeEvidence`), one of whose cases is one-stage v2's (`verity/sampled-proofs/one-stage/v2`), under the parameters
onsite names for it; v2's decoders read that case, and what they decode to is `Accepted` on the same blobs: every
session flock-verify `--zk` accepts (`AcceptsZK`) on the circuit and public files the parameters and the evidence commit.
It is `Properties.Composition`'s reduction of a composition's acceptance to its components' at onsite's components,
composed with `Properties.SampledProofs.OneStageZk.AcceptMeans`.

That the case's commitment is among those onsite's commitment binds is `Properties.Composition.AcceptMeans`'s.
-/

open SampledProofs.OneStageZk
open Core Core.Composition

namespace Properties.Onsite

/-- **Onsite's accept gives v2's**: onsite accepts only parameters and evidence naming a case of one-stage v2, under
the parameters onsite names for it, that v2 decodes to what it accepts on the caller's blobs (`Accepted`). -/
def SampledProofsAccepts : Prop :=
  ∀ (a b e : ByteArray) (blobs : Array ByteArray),
    (Core.Onsite.protocol.answer a b e blobs).verdict = .accept →
      ∃ (ps : List Named) (ev : List Case) (n : Named) (x : Case),
        (canonicalJson a >>= decodeParams) = .ok ps ∧ decodeEvidence e = .ok ev ∧
        (n, x) ∈ ps.zip ev ∧ n.id = PROTOCOL ∧ x.id = PROTOCOL ∧ x.params = n.params ∧
        ∃ (p : Params) (c : SampledProofs.OneStage.Commitment) (e' : Evidence) (pub : SampledProofs.OneStage.Public)
          (prof : Core.Integrity),
          (canonicalJson n.params.toUTF8 >>= paramsOfJson) = .ok p ∧
          SampledProofs.OneStage.commitmentOf x.commitment = .ok c ∧
          evidenceOf x.evidence = .ok e' ∧ Properties.SampledProofs.OneStageZk.Accepted p c e' blobs pub prof

end Properties.Onsite
