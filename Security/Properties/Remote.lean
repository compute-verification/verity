import CoreRemote
import Properties.Composed

/-!
# Remote Verity's sampled proofs subprotocol: what remote's acceptance says of the proofs

The properties of `Core.Remote.protocol.answer` on the caller's blobs, the whole of what remote's caller runs, stated
against it as it runs. They take no premise, and are onsite's (`Properties.Onsite`) at remote's components, PoUW and
one-stage v2.

* `SampledProofsAccepts`: when remote accepts, its parameters and evidence name a case of one-stage v2
  (`verity/sampled-proofs/one-stage/v2`), under the parameters remote names for it, that v2's decoders read to what is
  `Accepted` on the same blobs (`Properties.Composed.SampledProofsAccepted`).
* `SampledProofsRegistered`: when remote accepts, the circuit file v2 read is one of the caller's blobs and its SHA-512
  is the digest v2's parameters commit (`Properties.Composed.SampledProofsRegistered`, R6).
-/

namespace Properties.Remote

/-- **Remote's accept gives v2's**: remote accepts only parameters and evidence naming a case of one-stage v2, under
the parameters remote names for it, that v2 decodes to what it accepts on the caller's blobs (`Accepted`). -/
def SampledProofsAccepts : Prop :=
  ∀ (a b e : ByteArray) (blobs : Array ByteArray),
    (Core.Remote.protocol.answer a b e blobs).verdict = .accept → Properties.Composed.SampledProofsAccepted a e blobs

/-- **Remote's accept checks the registration**: the circuit file v2 verified on is one of the caller's blobs, and its
SHA-512 is the digest v2's parameters commit. -/
def SampledProofsRegistered : Prop :=
  ∀ (a b e : ByteArray) (blobs : Array ByteArray),
    (Core.Remote.protocol.answer a b e blobs).verdict = .accept → Properties.Composed.SampledProofsRegistered a blobs

end Properties.Remote
