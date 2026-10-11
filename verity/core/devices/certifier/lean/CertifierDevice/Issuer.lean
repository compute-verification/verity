import CertifierDevice.Driver
import Lms.Sign

/-!
# The issuer: one network certificate per closed link-window

The executable form of `certificate.py`'s `Issuer`, `Certificate` and `CertifiedRecord`: a record's head (`recordHead`,
under the grid's `parameters_digest`) and digest, the message that binds it to the key, the certifier session and the
link-window (`certificateMessage`), and its LMS signature at a reserved leaf (`Lms.signAt`). The encodings are
`NetworkCheck`'s and the signature `Lms`'s.

Which leaf signs, and which session ids the key has started, is the signer's state (`lms.LeafState.reserve` and
`start`), which stays with the host: `Issuer.certify` is pure given the reservation's result, and `certifyWith` takes it
from `Core.World`, starting the issuer's session there (`Core.World.start`) before its first reservation, as
`certificate.Issuer` does. As in `Issuer.certify`, the head, its digest and the message come first, so a record with no
encoding is refused before a session is started or a leaf is spent, and a start or a reservation the host refuses
leaves the link-window without a certificate (fail-closed) and is listed in `refused`.
-/

namespace CertifierDevice

open NetworkCheck (Direction RecordHead)

/-- `certificate.Certificate`. -/
structure Certificate where
  keyId : ByteArray
  link : String
  direction : Direction
  window : Nat
  digest : ByteArray
  signature : ByteArray

/-- `certificate.CertifiedRecord`: the certificate and the record head it signs. -/
structure CertifiedRecord where
  head : RecordHead
  certificate : Certificate

/-- Σ_sync as `parameters_digest` reads it: its deltas, and each ρ as (numerator, denominator) in lowest terms, in the
set's order. -/
def checkSyncSet (ss : NetworkCertifier.SyncSet) : NetworkCheck.SyncSet :=
  { deltas := (ss.deltas.map (Int.ofNat ·)).toArray, rhos := (ss.rhos.map fun ρ => (ρ.num, ρ.den)).toArray }

/-- `parameters_digest`: every grid parameter, and Σ_sync where a clock applies (egress). -/
def parameters (p : Params) (ss : Option NetworkCertifier.SyncSet) : Except String ByteArray :=
  NetworkCheck.parametersDigest p (ss.map checkSyncSet)

/-- `certificate.Issuer`: the certifier's key, its session's id, the link-windows whose start or leaf the host refused,
and whether its session has started. -/
structure Issuer where
  key : Lms.PrivateKey
  session : ByteArray
  refused : Array (String × Direction × Nat) := #[]
  started : Bool := false

namespace Issuer

/-- What `Issuer.certify` computes before it signs: the record's head, its digest and the certificate message. -/
def prepare (i : Issuer) (record : Record) (direction : Direction) (params : Params) (parameters : ByteArray)
    (deadline : Bool) : Except String (RecordHead × ByteArray × ByteArray) := do
  let head ← NetworkCheck.recordHead record.link direction record.window record.grid
    ((Status.reason? record.status).map reasonValue) params parameters deadline
  let digest ← head.digest
  let message ← NetworkCheck.certificateMessage i.key.I i.session record.link direction record.window digest
  pure (head, digest, message)

/-- The prepared record signed at the reserved leaf, or listed in `refused` when the host refused the reservation (or the
session's start before it). -/
def finish (i : Issuer) (leaf : Except String Nat) (record : Record) (direction : Direction)
    (prepared : RecordHead × ByteArray × ByteArray) : Except String (Issuer × Option CertifiedRecord) := do
  let (head, digest, message) := prepared
  match leaf with
  | .error _ => pure ({ i with refused := i.refused.push (record.link, direction, record.window) }, none)
  | .ok q =>
    let signature ← Lms.signAt i.key q message
    pure (i, some ⟨head, ⟨i.key.I, record.link, direction, record.window, digest, signature.encode⟩⟩)

/-- **`Issuer.certify`**, its signer's reservation given (a refused start of the session is a refused reservation):
refused (an error) where the record has no head, as Python raises `ValueError`. -/
def certify (i : Issuer) (leaf : Except String Nat) (record : Record) (direction : Direction) (params : Params)
    (parameters : ByteArray) (deadline : Bool := true) : Except String (Issuer × Option CertifiedRecord) := do
  i.finish leaf record direction (← i.prepare record direction params parameters deadline)

end Issuer

/-- `Issuer.certify` against the world's signer state: the head and the message first, then the session's start unless
it has started, then the reservation, then the signature. -/
def certifyWith (w : Core.World Out) (i : Issuer) (record : Record) (direction : Direction) (params : Params)
    (parameters : ByteArray) (deadline : Bool := true) : IO (Issuer × Option CertifiedRecord) := do
  let prepared ← IO.ofExcept (i.prepare record direction params parameters deadline)
  let started ← if i.started then pure (.ok ()) else w.start i.session
  let leaf ← match started with
    | .ok () => w.reserve
    | .error e => pure (.error e)
  let (i', cr?) ← IO.ofExcept (i.finish leaf record direction prepared)
  pure ({ i' with started := i.started || started matches .ok () }, cr?)

/-- `CertifiedRecord.verify` under an `LmsKey` in certifier session `session`: the certificate is signed by `publicKey`
over its message in that session, names the head's link-window and signs the head's digest. -/
def CertifiedRecord.verify (c : CertifiedRecord) (params : Lms.Params) (publicKey session : ByteArray) : Bool :=
  let x := c.certificate
  let h := c.head
  match h.digest, NetworkCheck.certificateMessage x.keyId session x.link x.direction x.window x.digest,
      Lms.Signature.decode params x.signature with
  | .ok d, .ok m, .ok sig =>
    x.keyId == publicKey.extract 0 16 && x.link == h.link && x.direction == h.direction && (x.window : Int) == h.window
      && x.digest == d && (Lms.verify params publicKey m sig).toOption == some true
  | _, _, _ => false

end CertifierDevice
