import CertifierDevice.Driver
import Lms.Sign

/-!
# The issuer: one network certificate per closed link-window

The executable form of `certificate.py`'s `Issuer`, `Certificate` and `CertifiedRecord`: a record's head (`recordHead`,
under the grid's `parameters_digest`) and digest, that digest hidden under `salt_c`, 64 bytes of the certifier's own
coins drawn for the record (`certificateDigest`), the message that binds the hidden digest to the key, the certifier
session and the link-window (`certificateMessage`), and its LMS signature at a reserved leaf (`Lms.signAt`). The
encodings are `NetworkCheck`'s and the signature `Lms`'s.

Which leaf signs, and which session ids the key has started, is the signer's state (`lms.LeafState.reserve` and
`start`), which stays with the host: `Issuer.certify` is pure given the salt and the reservation's result, and
`certifyWith` takes both from `Core.World`, the salt from its coins (`Core.World.coins`) and the leaf from its signer
state, starting the issuer's session there (`Core.World.start`) before its first reservation, as `certificate.Issuer`
does. As in `Issuer.certify`, the salt, the head, its digests and the message come first, so a record with no encoding
is refused before a session is started or a leaf is spent, and a start or a reservation the host refuses leaves the
link-window without a certificate (fail-closed) and is listed in `refused`.

**The clock certificate** (`ClockCertificate`, `Issuer.clock`, `clockWith`): beside each egress window's record, the
sync the certifier's shaper ran the window under, signed as `certificate.Issuer.clock` signs it: the clock digest
(`clockDigest`, `tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖ u64be q)`, ρ = p/q in lowest
terms) under the record certificate's message for the same key, session and link-window, `egress`. The digest is first
and the message next, so a sync or a link-window with no encoding is refused before a session is started or a leaf is
spent, and a refused start or reservation is listed in `refused` as a record's is. `ClockCertificate.verify` is
`certificate.ClockCertificate.verify`: an egress certificate of the sync's clock digest, signed under the key in the
session. The record certificate signs `SHA-512("verity/net-cert/v1" ‖ salt_c ‖ R)`, so neither passes as the other
short of a SHA-512 preimage.
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

/-- `certificate.CertifiedRecord`: the record head, the salt the certifier drew for it (`salt_c`) and the certificate,
which signs the head's digest hidden under that salt. -/
structure CertifiedRecord where
  head : RecordHead
  salt : ByteArray
  certificate : Certificate

/-- Σ_sync as `parameters_digest` reads it: its deltas, and each ρ as (numerator, denominator) in lowest terms, in the
set's order. -/
def checkSyncSet (ss : NetworkCertifier.SyncSet) : NetworkCheck.SyncSet :=
  { deltas := (ss.deltas.map (Int.ofNat ·)).toArray, rhos := (ss.rhos.map fun ρ => (ρ.num, ρ.den)).toArray }

/-- `parameters_digest`: every grid parameter, and Σ_sync where a clock applies (egress). -/
def parameters (p : Params) (ss : Option NetworkCertifier.SyncSet) : Except String ByteArray :=
  NetworkCheck.parametersDigest p (ss.map checkSyncSet)

/-- `certificate.CLOCK_TAG`. -/
def clockTag : String := "verity-network-certifier/clock/v1"

/-- What `clock_digest` hashes under `clockTag`: `u64be δ ‖ u64be p ‖ u64be q` of the sync (δ, ρ = p/q, which `Rat`
keeps in lowest terms as `Fraction` does), refused unless each is a u64. -/
def clockInput (sync : NetworkCertifier.ClockSync) : Except String ByteArray := do
  pure ((← NetworkCheck.uBE 8 sync.delta "δ") ++ (← NetworkCheck.uBE 8 sync.rho.num "p")
    ++ (← NetworkCheck.uBE 8 sync.rho.den "q"))

/-- `certificate.clock_digest`: the digest a clock certificate signs. -/
def clockDigest (sync : NetworkCertifier.ClockSync) : Except String ByteArray := do
  Identity.taggedSha512 clockTag (← clockInput sync)

/-- `certificate.ClockCertificate`: the sync the certifier's shaper ran one egress link-window under, public, and the
certificate of its clock digest. -/
structure ClockCertificate where
  sync : NetworkCertifier.ClockSync
  certificate : Certificate

/-- `certificate.Issuer`: the certifier's key, its session's id, the link-windows whose start or leaf the host refused,
and whether its session has started. -/
structure Issuer where
  key : Lms.PrivateKey
  session : ByteArray
  refused : Array (String × Direction × Nat) := #[]
  started : Bool := false

namespace Issuer

/-- What `Issuer.certify` computes before it signs: the record's head, the digest the certificate signs (the head's
digest under `salt`, refused unless it is 64 bytes) and the certificate message. -/
def prepare (i : Issuer) (salt : ByteArray) (record : Record) (direction : Direction) (params : Params)
    (parameters : ByteArray) (deadline : Bool) : Except String (RecordHead × ByteArray × ByteArray) := do
  if salt.size ≠ NetworkCheck.saltBytes then
    throw s!"the certifier's coins gave {salt.size} bytes for a {NetworkCheck.saltBytes}-byte salt"
  let head ← NetworkCheck.recordHead record.link direction record.window record.grid
    ((Status.reason? record.status).map reasonValue) params parameters deadline
  let digest := NetworkCheck.certificateDigest salt (← head.digest)
  let message ← NetworkCheck.certificateMessage i.key.I i.session record.link direction record.window digest
  pure (head, digest, message)

/-- The prepared record signed at the reserved leaf, or listed in `refused` when the host refused the reservation (or the
session's start before it). -/
def finish (i : Issuer) (salt : ByteArray) (leaf : Except String Nat) (record : Record) (direction : Direction)
    (prepared : RecordHead × ByteArray × ByteArray) : Except String (Issuer × Option CertifiedRecord) := do
  let (head, digest, message) := prepared
  match leaf with
  | .error _ => pure ({ i with refused := i.refused.push (record.link, direction, record.window) }, none)
  | .ok q =>
    let signature ← Lms.signAt i.key q message
    pure (i, some ⟨head, salt, ⟨i.key.I, record.link, direction, record.window, digest, signature.encode⟩⟩)

/-- **`Issuer.certify`**, the record's salt and its signer's reservation given (a refused start of the session is a
refused reservation): refused (an error) where the salt is not 64 bytes or the record has no head, as Python raises
`ValueError`. -/
def certify (i : Issuer) (salt : ByteArray) (leaf : Except String Nat) (record : Record) (direction : Direction)
    (params : Params) (parameters : ByteArray) (deadline : Bool := true) :
    Except String (Issuer × Option CertifiedRecord) := do
  i.finish salt leaf record direction (← i.prepare salt record direction params parameters deadline)

/-- What `Issuer.clock` signs: the sync's clock digest and the certificate message over it for the egress link-window,
refused (an error) where `clock_digest` or `certificate_message` raises. -/
def prepareClock (i : Issuer) (link : String) (window : Nat) (sync : NetworkCertifier.ClockSync) :
    Except String (ByteArray × ByteArray) := do
  let digest ← clockDigest sync
  pure (digest, ← NetworkCheck.certificateMessage i.key.I i.session link .egress window digest)

/-- The prepared clock digest signed at the reserved leaf, or the link-window listed in `refused` when the host refused
the reservation (or the session's start before it). -/
def finishClock (i : Issuer) (leaf : Except String Nat) (link : String) (window : Nat)
    (sync : NetworkCertifier.ClockSync) (prepared : ByteArray × ByteArray) :
    Except String (Issuer × Option ClockCertificate) := do
  let (digest, message) := prepared
  match leaf with
  | .error _ => pure ({ i with refused := i.refused.push (link, .egress, window) }, none)
  | .ok q =>
    let signature ← Lms.signAt i.key q message
    pure (i, some ⟨sync, ⟨i.key.I, link, .egress, window, digest, signature.encode⟩⟩)

/-- **`Issuer.clock`**, its signer's reservation given: the clock certificate of the egress link-window, `sync` being
the one the certifier's shaper ran it under; refused (an error) where the sync isn't three u64s or the link-window has
no message, as Python raises `ValueError`. -/
def clock (i : Issuer) (leaf : Except String Nat) (link : String) (window : Nat) (sync : NetworkCertifier.ClockSync) :
    Except String (Issuer × Option ClockCertificate) := do
  i.finishClock leaf link window sync (← i.prepareClock link window sync)

end Issuer

/-- `Issuer.certify` against the world: the salt from its coins, the head and the message, then the session's start
unless it has started, then the reservation from its signer state, then the signature. -/
def certifyWith (w : Core.World Out) (i : Issuer) (record : Record) (direction : Direction) (params : Params)
    (parameters : ByteArray) (deadline : Bool := true) : IO (Issuer × Option CertifiedRecord) := do
  let salt ← w.coins NetworkCheck.saltBytes
  let prepared ← IO.ofExcept (i.prepare salt record direction params parameters deadline)
  let started ← if i.started then pure (.ok ()) else w.start i.session
  let leaf ← match started with
    | .ok () => w.reserve
    | .error e => pure (.error e)
  let (i', cr?) ← IO.ofExcept (i.finish salt leaf record direction prepared)
  pure ({ i' with started := i.started || started matches .ok () }, cr?)

/-- `Issuer.clock` against the world: the clock digest and the message, then the session's start unless it has started,
then the reservation from its signer state, then the signature. -/
def clockWith (w : Core.World Out) (i : Issuer) (link : String) (window : Nat) (sync : NetworkCertifier.ClockSync) :
    IO (Issuer × Option ClockCertificate) := do
  let prepared ← IO.ofExcept (i.prepareClock link window sync)
  let started ← if i.started then pure (.ok ()) else w.start i.session
  let leaf ← match started with
    | .ok () => w.reserve
    | .error e => pure (.error e)
  let (i', c?) ← IO.ofExcept (i.finishClock leaf link window sync prepared)
  pure ({ i' with started := i.started || started matches .ok () }, c?)

/-- `CertifiedRecord.verify` under an `LmsKey` in certifier session `session`: the certificate is signed by `publicKey`
over its message in that session, names the head's link-window and signs the head's digest under the record's salt. -/
def CertifiedRecord.verify (c : CertifiedRecord) (params : Lms.Params) (publicKey session : ByteArray) : Bool :=
  let x := c.certificate
  let h := c.head
  match h.digest, NetworkCheck.certificateMessage x.keyId session x.link x.direction x.window x.digest,
      Lms.Signature.decode params x.signature with
  | .ok d, .ok m, .ok sig =>
    x.keyId == publicKey.extract 0 16 && x.link == h.link && x.direction == h.direction && (x.window : Int) == h.window
      && x.digest == NetworkCheck.certificateDigest c.salt d && (Lms.verify params publicKey m sig).toOption == some true
  | _, _, _ => false

/-- `ClockCertificate.verify` under an `LmsKey` in certifier session `session`: the certificate is an egress one, signs
the sync's clock digest, and is signed by `publicKey` over its message in that session. -/
def ClockCertificate.verify (c : ClockCertificate) (params : Lms.Params) (publicKey session : ByteArray) : Bool :=
  let x := c.certificate
  match clockDigest c.sync, NetworkCheck.certificateMessage x.keyId session x.link x.direction x.window x.digest,
      Lms.Signature.decode params x.signature with
  | .ok d, .ok m, .ok sig =>
    x.direction == .egress && x.digest == d && x.keyId == publicKey.extract 0 16
      && (Lms.verify params publicKey m sig).toOption == some true
  | _, _, _ => false

end CertifierDevice
