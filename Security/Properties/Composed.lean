import CoreComposition.Product
import Properties.SampledProofs.OneStageZk

/-!
# One-stage v2 inside a composition: what a composition's acceptance gives of its sampled proofs case

The conclusions `Properties.Onsite` and `Properties.Remote` state for the two compiled compositions that take one-stage
v2 (`verity/sampled-proofs/one-stage/v2`) as their sampled proofs subprotocol, in the composition's own decoders' terms
(`Core.Composition.decodeParams`, `decodeEvidence`) and v2's (`paramsOfJson`, `commitmentOf`, `evidenceOf`).

* `SampledProofsAccepted`: the composition's parameters and evidence name a case of v2, under the parameters the
  composition names for it, that v2's decoders read to what is `Accepted` on the caller's blobs.
* `SampledProofsRegistered`: the registration check (R6): the parameters name v2 with parameters whose committed
  circuit digest names one of the caller's blobs, the circuit file every session of `Accepted` is verified on.
-/

open SampledProofs.OneStageZk
open Core Core.Composition

namespace Properties.Composed

/-- **The composition's case of v2 is `Accepted`**: the parameters `a` and evidence `e` decode to the composition's
(`decodeParams`, `decodeEvidence`); one of whose cases is v2's, under the parameters the composition names for it; and
v2's decoders read it to what is `Accepted` on `blobs`. -/
def SampledProofsAccepted (a e : ByteArray) (blobs : Array ByteArray) : Prop :=
  ∃ (ps : List Named) (ev : List Case) (n : Named) (x : Case),
    (canonicalJson a >>= decodeParams) = .ok ps ∧ decodeEvidence e = .ok ev ∧
    (n, x) ∈ ps.zip ev ∧ n.id = PROTOCOL ∧ x.id = PROTOCOL ∧ x.params = n.params ∧
    ∃ (p : Params) (c : SampledProofs.OneStage.Commitment) (e' : Evidence) (pub : SampledProofs.OneStage.Public)
      (prof : Core.Integrity),
      (canonicalJson n.params.toUTF8 >>= paramsOfJson) = .ok p ∧
      SampledProofs.OneStage.commitmentOf x.commitment = .ok c ∧
      evidenceOf x.evidence = .ok e' ∧ Properties.SampledProofs.OneStageZk.Accepted p c e' blobs pub prof

/-- **The registration check (R6)**: the parameters `a` decode to the composition's, which name v2 with parameters `p`,
and `p`'s committed circuit digest (`proofs.circuit`) is the SHA-512 of a circuit file `F`, one of the caller's
`blobs`. -/
def SampledProofsRegistered (a : ByteArray) (blobs : Array ByteArray) : Prop :=
  ∃ (ps : List Named) (n : Named) (p : Params) (F : ByteArray),
    (canonicalJson a >>= decodeParams) = .ok ps ∧ n ∈ ps ∧ n.id = PROTOCOL ∧
    (canonicalJson n.params.toUTF8 >>= paramsOfJson) = .ok p ∧
    Properties.SampledProofs.OneStageZk.Blob blobs p.own.circuit F

end Properties.Composed
