/-
A composition whose components may read blobs (`Core.BlobProtocol`, the content-addressed files a check reads beside its
three byte strings). `productB` hands the caller's blobs to every component, and a component that reads none
(`Core.Protocol.withoutBlobs`, a coercion) ignores them. Its parameters, commitment, evidence and verdict rule are
`product`'s: on the caller's blobs it is `product` of its components on those blobs (`productB_on`), and without blobs
`product` of its components without them. So what `product` accepts, a `productB` accepts on blobs, with each component
answering its own case on the same blobs. `registeredB` is `registered` over blobs in the same way (`registeredB_on`):
the registration check reads none, and the components read the caller's.
-/
import CoreComposition.Registered
import CoreProtocol.Blobs

namespace Core

/-- A protocol that reads no blobs, as a `BlobProtocol` whose check ignores them. -/
def Protocol.withoutBlobs (P : Protocol) : BlobProtocol := { P with checkBlobs := fun p c e _ => P.check p c e }

instance instCoeProtocolBlobProtocol_coreComposition : Coe Protocol BlobProtocol := ⟨Protocol.withoutBlobs⟩

/-- The protocol on the blobs `bs`: its check is `checkBlobs` on them. -/
def BlobProtocol.onBlobs (P : BlobProtocol) (bs : Blobs) : Protocol :=
  { P.toProtocol with check := fun p c e => P.checkBlobs p c e bs }

theorem BlobProtocol.on_eq (P : BlobProtocol) (blobs : Array ByteArray) : P.on blobs = P.onBlobs (Blobs.ofArray blobs) :=
  rfl

theorem Protocol.withoutBlobs_onBlobs (P : Protocol) (bs : Blobs) : P.withoutBlobs.onBlobs bs = P := rfl

end Core

namespace Core.Composition

/-- The composition `id` of the components `cs`, which read the caller's blobs: `product id` of the components without
blobs, whose check on blobs `bs` is `product`'s check of the components on `bs`. `cs` must be listed in the order of
their ids, each once. -/
def productB (id : String) (cs : List BlobProtocol) : BlobProtocol :=
  { product id (cs.map (·.toProtocol)) with checkBlobs := fun ps c ev bs => check id (cs.map (·.onBlobs bs)) ps c ev }

/-- On the caller's blobs a `productB` is `product` of its components on them. -/
theorem productB_on (id : String) (cs : List BlobProtocol) (blobs : Array ByteArray) :
    (productB id cs).on blobs = product id (cs.map (·.onBlobs (Blobs.ofArray blobs))) := rfl

/-- The composition `id` of the components `cs`, which read the caller's blobs, after the registration check `R` on the
registration the component `s` commits to: `registered` of the components without blobs, whose check on blobs `bs` is
`registered`'s check of the components on `bs`. -/
def registeredB (id : String) (R : Protocol) (s : String) (cs : List BlobProtocol) : BlobProtocol :=
  { registered id R s (cs.map (·.toProtocol)) with
    checkBlobs := fun ps c ev bs => checkRegistered id R s (cs.map (·.onBlobs bs)) ps c ev }

/-- On the caller's blobs a `registeredB` is `registered` of its components on them. -/
theorem registeredB_on (id : String) (R : Protocol) (s : String) (cs : List BlobProtocol) (blobs : Array ByteArray) :
    (registeredB id R s cs).on blobs = registered id R s (cs.map (·.onBlobs (Blobs.ofArray blobs))) := rfl

/-- `ordered` of the components' ids. -/
def orderedB (cs : List BlobProtocol) : Bool := ordered (cs.map (·.toProtocol))

end Core.Composition
