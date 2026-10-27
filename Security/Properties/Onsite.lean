import CoreOnsite
import Properties.Composed

/-!
# Onsite Verity's sampled proofs subprotocol: what onsite's acceptance says of the proofs

The properties of `Core.Onsite.protocol.answer` on the caller's blobs, the whole of what onsite's caller runs, stated
against it as it runs. They take no premise.

* `SampledProofsAccepts`: when onsite accepts, its parameters and evidence decode as a registered composition's
  (`decodeRegisteredParams`, `decodeRegisteredEvidence`: the registration's entry and case, NCI's, beside the
  components'), one of the components' cases is one-stage v2's (`verity/sampled-proofs/one-stage/v2`), under the parameters
  onsite names for it; v2's decoders read that case, and what they decode to is `Accepted` on the same blobs: every
  session flock-verify `--zk` accepts (`AcceptsZK`) on the circuit and public files the parameters and the evidence
  commit (`Properties.Composed.RegisteredSampledProofsAccepted`). A registered composition accepts only what `product`
  of its components accepts, and `Properties.Composition`'s reduction of that to its components' at onsite's
  components is composed with
  `Properties.SampledProofs.OneStageZk.AcceptMeans`.
* `SampledProofsRegistered`: when onsite accepts, the circuit file v2 read is one of the caller's blobs and its SHA-512
  is the digest v2's parameters commit (`Properties.Composed.RegisteredSampledProofsRegistered`, R6).

That the case's commitment is among those onsite's commitment binds is `Properties.Composition.AcceptMeans`'s.
-/

namespace Properties.Onsite

/-- **Onsite's accept gives v2's**: onsite accepts only parameters and evidence naming a case of one-stage v2, under
the parameters onsite names for it, that v2 decodes to what it accepts on the caller's blobs (`Accepted`). -/
def SampledProofsAccepts : Prop :=
  ∀ (a b e : ByteArray) (blobs : Array ByteArray),
    (Core.Onsite.protocol.answer a b e blobs).verdict = .accept →
      Properties.Composed.RegisteredSampledProofsAccepted a e blobs

/-- **Onsite's accept checks the registration**: the circuit file v2 verified on is one of the caller's blobs, and its
SHA-512 is the digest v2's parameters commit. -/
def SampledProofsRegistered : Prop :=
  ∀ (a b e : ByteArray) (blobs : Array ByteArray),
    (Core.Onsite.protocol.answer a b e blobs).verdict = .accept →
      Properties.Composed.RegisteredSampledProofsRegistered a blobs

end Properties.Onsite
