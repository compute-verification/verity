import Lms.Verify

/-!
# HSS at two levels: a device's identity and its parties' keys (RFC 8554 §6 over SHA-512)

Executable forms of `verity/ml/crypto/lms.py`'s HSS verification, whose module docstring is the spec: the three role
digests a device's keys sign (`partyKeyMessage`, `periodRootMessage`, `livenessMessage`), the two-level signature's byte
format (`HssSignature.decode`: `lower ‖ certificate ‖ signature`, the lower key `16 + N` bytes and each signature
`signatureBytes`), and its verification (`verifyCertificate`, `hssVerify`, `verifyLiveness`). A role digest is
`SHA-512([tag] ‖ I ‖ [context] ‖ [record])`, each `[·]` a u64 big-endian length then the bytes. A Python `ValueError` is
an `Except.error` here. `MemoryChallenger.Difftest` holds these to the Python.
-/

namespace Lms

open Flock

/-- `lms._field`: a u64 big-endian length, then the bytes. -/
def lenField (b : ByteArray) : ByteArray := be64 b.size.toUInt64 ++ b

def PARTY_KEY_TAG : String := "verity/party-key/v1"
def PERIOD_ROOT_TAG : String := "verity/period-root/v1"
def LIVENESS_TAG : String := "verity/liveness/v1"

/-- `lms.DEPLOYMENT_BYTES`: a deployment configuration's hash, SHA-512 of the file. -/
def DEPLOYMENT_BYTES : Nat := 64

/-- `lms._digest(role, I, context, record)`, refused unless `I` is 16 bytes, and for a context or record of `2^64` bytes
or more, whose length is no u64 (Python's `int.to_bytes(8, "big")` raises there). -/
def roleDigest (tag : String) (I context record : ByteArray) : E ByteArray := do
  if I.size ≠ 16 then throw s!"an LMS identifier is 16 bytes, not {I.size}"
  if 2 ^ 64 ≤ context.size ∨ 2 ^ 64 ≤ record.size then throw "a field's length is a u64"
  pure (Sha512.hash (lenField tag.toUTF8 ++ I ++ lenField context ++ lenField record))

/-- `lms._u64`: eight bytes big-endian, refused at `2^64` and above. -/
def u64 (x : Nat) (what : String) : E ByteArray :=
  if x < 2 ^ 64 then pure (be64 x.toUInt64) else throw s!"{what} is a u64, not {x}"

/-- `lms._party_context`: the deployment's hash and the party's name, each length-prefixed. -/
def partyContext (deployment : ByteArray) (party : String) : E ByteArray := do
  if deployment.size ≠ DEPLOYMENT_BYTES then
    throw s!"a deployment's hash is {DEPLOYMENT_BYTES} bytes, not {deployment.size}"
  pure (lenField deployment ++ lenField party.toUTF8)

/-- `lms._period_context`: the party's context, the attachment id and the link's name, length-prefixed, and the
period's index (u64). -/
def periodContext (deployment : ByteArray) (party : String) (attachment : ByteArray) (link : String) (period : Nat) :
    E ByteArray := do
  pure ((← partyContext deployment party) ++ lenField attachment ++ lenField link.toUTF8 ++
    (← u64 period "a certification period's index"))

/-- `lms.party_key_message(I, deployment, party, lower)`: what a device's identity signs to certify a party's lower key. -/
def partyKeyMessage (I deployment : ByteArray) (party : String) (lower : ByteArray) : E ByteArray := do
  roleDigest PARTY_KEY_TAG I (← partyContext deployment party) lower

/-- A period root's context: the period's, then the record count (u64). -/
def periodRootContext (deployment : ByteArray) (party : String) (attachment : ByteArray) (link : String)
    (period count : Nat) : E ByteArray := do
  pure ((← periodContext deployment party attachment link period) ++ (← u64 count "a period's record count"))

/-- `lms.period_root_message(I, deployment, party, attachment, link, period, count, root)`: what a party's lower key
signs once per certification period. -/
def periodRootMessage (I deployment : ByteArray) (party : String) (attachment : ByteArray) (link : String)
    (period count : Nat) (root : ByteArray) : E ByteArray := do
  roleDigest PERIOD_ROOT_TAG I (← periodRootContext deployment party attachment link period count) root

/-- A liveness record's context: the period's, then the sampling block's index (u64). -/
def livenessContext (deployment : ByteArray) (party : String) (attachment : ByteArray) (link : String)
    (period block : Nat) : E ByteArray := do
  pure ((← periodContext deployment party attachment link period) ++ (← u64 block "a sampling block's index"))

/-- `lms.liveness_message(I, deployment, party, attachment, link, period, block)`: a party's public record that sampling
block `block` of the period was certified, with no content. -/
def livenessMessage (I deployment : ByteArray) (party : String) (attachment : ByteArray) (link : String)
    (period block : Nat) : E ByteArray := do
  roleDigest LIVENESS_TAG I (← livenessContext deployment party attachment link period block) ByteArray.empty

/-- A two-level HSS signature: the party's lower public key (`I ‖ T[1]`), the identity's signature of its
`partyKeyMessage` (its certificate), and the lower key's signature of the message. -/
structure HssSignature where
  lower : ByteArray
  certificate : Signature
  signature : Signature

/-- `HssSignature.decode`: exactly `16 + N + 2 · signatureBytes` bytes, `lower ‖ certificate ‖ signature`. -/
def HssSignature.decode (p : Params) (data : ByteArray) : E HssSignature := do
  let n := 16 + p.N
  let s := p.signatureBytes
  if data.size ≠ n + 2 * s then
    throw s!"an HSS signature is {n + 2 * s} bytes, not {data.size}"
  pure ⟨data.extract 0 n, ← Signature.decode p (data.extract n (n + s)), ← Signature.decode p (data.extract (n + s) data.size)⟩

/-- `lms.verify_certificate`: whether `certificate` is the identity's signature over `lower`'s `partyKeyMessage` for
`party` in the deployment; refused unless the identity is `16 + N` bytes, false for a lower key of another length. -/
def verifyCertificate (p : Params) (identity deployment : ByteArray) (party : String) (lower : ByteArray)
    (certificate : Signature) : E Bool := do
  if identity.size ≠ 16 + p.N then throw s!"an LMS public key is {16 + p.N} bytes, not {identity.size}"
  if lower.size ≠ 16 + p.N then return false
  verify p identity (← partyKeyMessage (identity.extract 0 16) deployment party lower) certificate

/-- `lms.hss_verify`: the lower key's certificate verifies under the identity, and the signature under the lower key. -/
def hssVerify (p : Params) (identity deployment : ByteArray) (party : String) (sig : HssSignature)
    (message : ByteArray) : E Bool := do
  if !(← verifyCertificate p identity deployment party sig.lower sig.certificate) then return false
  verify p sig.lower message sig.signature

/-- `lms.verify_liveness`: whether `sig` is the party's public liveness record of block `block` of the period, false for
a lower key of another length. -/
def verifyLiveness (p : Params) (identity deployment : ByteArray) (party : String) (attachment : ByteArray)
    (link : String) (period block : Nat) (sig : HssSignature) : E Bool := do
  if sig.lower.size ≠ 16 + p.N then return false
  hssVerify p identity deployment party sig
    (← livenessMessage (sig.lower.extract 0 16) deployment party attachment link period block)

end Lms
