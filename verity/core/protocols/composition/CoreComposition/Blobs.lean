/-
A composition whose components may read blobs (`Core.BlobProtocol`, the content-addressed files a check reads beside its
three byte strings). `productB` hands the caller's blobs to every component, and a component that reads none
(`Core.Protocol.withoutBlobs`, a coercion) ignores them. Its parameters, commitment, evidence and verdict rule are
`product`'s: on the caller's blobs it is `product` of its components on those blobs (`productB_on`), and without blobs
`product` of its components without them. So what `product` accepts, a `productB` accepts on blobs, with each component
answering its own case on the same blobs. `sameB` is `same` over blobs in the same way (`sameB_on`).
-/
import CoreComposition.Same
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

/-- The composition `id` of the components `cs`, which read the caller's blobs, whose components `ss` must all commit
to one record: `same` of the components without blobs, whose check on blobs `bs` is `same`'s check of the
components on `bs`. -/
def sameB (id : String) (ss : List String) (cs : List BlobProtocol) : BlobProtocol :=
  { same id ss (cs.map (·.toProtocol)) with checkBlobs := fun ps c ev bs => checkSame id ss (cs.map (·.onBlobs bs)) ps c ev }

/-- On the caller's blobs a `sameB` is `same` of its components on them. -/
theorem sameB_on (id : String) (ss : List String) (cs : List BlobProtocol) (blobs : Array ByteArray) :
    (sameB id ss cs).on blobs = same id ss (cs.map (·.onBlobs (Blobs.ofArray blobs))) := rfl

/-- `ordered` of the components' ids. -/
def orderedB (cs : List BlobProtocol) : Bool := ordered (cs.map (·.toProtocol))

end Core.Composition
