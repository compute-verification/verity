import CoreComposition.Product
import CoreComposition.Registered
import Properties.SampledProofs.OneStageZk

/-!
# One-stage v2 inside a composition: what a composition's acceptance gives of its sampled proofs case

The conclusions `Properties.Onsite` and `Properties.Remote` state for the two compiled compositions that take one-stage
v2 (`verity/sampled-proofs/one-stage/v2`) as their sampled proofs subprotocol, in v2's decoders' terms (`paramsOfJson`,
`commitmentOf`, `evidenceOf`) and the composition's own: `product`'s (`Core.Composition.decodeParams`,
`decodeEvidence`) at remote, `registered`'s (`decodeRegisteredParams`, `decodeRegisteredEvidence`) at onsite.

* `CaseAccepted`: the components' parameters and cases name a case of v2, under the parameters the composition names
  for it, that v2's decoders read to what is `Accepted` on the caller's blobs. `SampledProofsAccepted` reads them with
  `product`'s decoders, `RegisteredSampledProofsAccepted` with `registered`'s.
* `CaseRegistered`: the registration check (R6): the components' parameters name v2 with parameters whose committed
  circuit digest names one of the caller's blobs, the circuit file every session of `Accepted` is verified on.
  `SampledProofsRegistered` and `RegisteredSampledProofsRegistered` read them as above.
-/

open SampledProofs.OneStageZk
open Core Core.Composition

namespace Properties.Composed

/-- **The components' case of v2 is `Accepted`**: one of the cases `ev` is v2's, under the parameters `ps` names for it,
and v2's decoders read it to what is `Accepted` on `blobs`. -/
def CaseAccepted (ps : List Named) (ev : List Case) (blobs : Array ByteArray) : Prop :=
  ∃ (n : Named) (x : Case), (n, x) ∈ ps.zip ev ∧ n.id = PROTOCOL ∧ x.id = PROTOCOL ∧ x.params = n.params ∧
    ∃ (p : Params) (c : SampledProofs.OneStage.Commitment) (e' : Evidence) (pub : SampledProofs.OneStage.Public)
      (prof : Core.Integrity),
      (canonicalJson n.params.toUTF8 >>= paramsOfJson) = .ok p ∧
      SampledProofs.OneStage.commitmentOf x.commitment = .ok c ∧
      evidenceOf x.evidence = .ok e' ∧ Properties.SampledProofs.OneStageZk.Accepted p c e' blobs pub prof

/-- **The registration check (R6)**: the components' parameters `ps` name v2 with parameters `p`, and `p`'s committed
circuit digest (`proofs.circuit`) is the SHA-512 of a circuit file `F`, one of the caller's `blobs`. -/
def CaseRegistered (ps : List Named) (blobs : Array ByteArray) : Prop :=
  ∃ (n : Named) (p : Params) (F : ByteArray), n ∈ ps ∧ n.id = PROTOCOL ∧
    (canonicalJson n.params.toUTF8 >>= paramsOfJson) = .ok p ∧
    Properties.SampledProofs.OneStageZk.Blob blobs p.own.circuit F

/-- `CaseAccepted` of the parameters `a` and evidence `e` as `product` decodes them (`decodeParams`,
`decodeEvidence`). -/
def SampledProofsAccepted (a e : ByteArray) (blobs : Array ByteArray) : Prop :=
  ∃ (ps : List Named) (ev : List Case),
    (canonicalJson a >>= decodeParams) = .ok ps ∧ decodeEvidence e = .ok ev ∧ CaseAccepted ps ev blobs

/-- `CaseRegistered` of the parameters `a` as `product` decodes them. -/
def SampledProofsRegistered (a : ByteArray) (blobs : Array ByteArray) : Prop :=
  ∃ ps : List Named, (canonicalJson a >>= decodeParams) = .ok ps ∧ CaseRegistered ps blobs

/-- `CaseAccepted` of the components' parameters and cases as `registered` decodes `a` and `e`
(`decodeRegisteredParams`, `decodeRegisteredEvidence`), beside the registration's own entry `r` and case `k`. -/
def RegisteredSampledProofsAccepted (a e : ByteArray) (blobs : Array ByteArray) : Prop :=
  ∃ (r : Named) (ps : List Named) (k : Case) (ev : List Case),
    (canonicalJson a >>= decodeRegisteredParams) = .ok (r, ps) ∧ decodeRegisteredEvidence e = .ok (k, ev) ∧
    CaseAccepted ps ev blobs

/-- `CaseRegistered` of the components' parameters as `registered` decodes `a`. -/
def RegisteredSampledProofsRegistered (a : ByteArray) (blobs : Array ByteArray) : Prop :=
  ∃ (r : Named) (ps : List Named), (canonicalJson a >>= decodeRegisteredParams) = .ok (r, ps) ∧
    CaseRegistered ps blobs

end Properties.Composed
