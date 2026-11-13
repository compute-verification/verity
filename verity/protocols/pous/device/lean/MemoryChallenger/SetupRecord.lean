module

public import MemoryChallenger.Protocol
public import MemoryChallenger.StreamedWipe

/-!
# The memory challenger's setup record

What the memory challenger's device signs at setup, with the key and under the session id it signs that session's
record with (`memoryChallengerMessage`): the setup's published outputs it was given, the number of blocks, the
commitment to `W`, the salt and the root of the store's rows, against which the session's answers are checked. It says
the device was given these, never that setup's check passed: the device sees no store and runs no check at setup.
`memory_challenger.SetupRecord` and its functions mirror this file byte for byte.

The record (`SetupRecord.value`) is raw bytes, never a digest: `SETUP_RECORD_TAG` (NUL-ended), the blocks as a u64
big-endian, then the commitment, the salt and the root, each a u64 big-endian length then the bytes (`field`). It is at
least `SETUP_RECORD_TAG.size + 32 = 78` bytes long, so never 64 bytes, and PoUS's session record (`message`) and the wipe's (`StreamedWipe.message`) are SHA-512
digests, 64 bytes long: no setup record is either (`SetupRecord.value_ne_pous`, `SetupRecord.value_ne_wipe`). The role's
pre-image (`rolePreimage`) determines its key identifier, session and record (`rolePreimage_inj`), so a setup record's
pre-image is never a session's or a wipe's (`setupPreimage_ne_pous`, `setupPreimage_ne_wipe`), on the bytes hashed and
with no assumption on the hash, and a setup record's message equal to one of theirs is a SHA-512 collision
(`setupMessage_collision`).

`setupRecordOk` checks a signed setup record against a session's scheme, public key and id: the record is under the
session's scheme, which `schemes` accepts, and the session's key, and its signature verifies on its bytes under the
session's id.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace MemoryChallenger

open Lean (Json)
open Flock

/-- The setup record's tag, NUL-ended. -/
def SETUP_RECORD_TAG : ByteArray := "verity-pous/memory-challenger/setup-record/v1".toUTF8.push 0

/-- The setup's published outputs the device was given: its number of blocks, its commitment to `W`, its salt and its
rows' root. -/
structure SetupRecord where
  blocks : Nat
  commitment : ByteArray
  salt : ByteArray
  root : ByteArray
  deriving BEq, Inhabited

/-- `SetupRecord.value()`: `SETUP_RECORD_TAG ‖ u64be(blocks) ‖ [commitment] ‖ [salt] ‖ [root]`; blocks of `2^64` or
more are refused. -/
def SetupRecord.value (r : SetupRecord) : E ByteArray := do
  pure (SETUP_RECORD_TAG ++ (← beN 8 r.blocks "blocks") ++ field r.commitment ++ field r.salt ++ field r.root)

/-- `setup_record_message(I, session, record)`: the role's message of the session and the record's bytes. -/
def SetupRecord.message (I session : ByteArray) (r : SetupRecord) : E ByteArray := do
  memoryChallengerMessage I session (← r.value)

/-- A setup record with the scheme, public key and signature it is signed with. -/
structure SignedSetupRecord where
  record : SetupRecord
  scheme : String
  publicKey : ByteArray
  signature : ByteArray
  deriving BEq, Inhabited

private def hexJ (b : ByteArray) : Json := .str (hexOf b)

/-- `SignedSetupRecord.to_json()`: the record's fields beside the scheme, public key and signature, in one object. -/
def SignedSetupRecord.toJson (s : SignedSetupRecord) : Json :=
  Json.mkObj [("blocks", Lean.toJson s.record.blocks), ("commitment", hexJ s.record.commitment),
    ("public_key", hexJ s.publicKey), ("root", hexJ s.record.root), ("salt", hexJ s.record.salt), ("scheme", .str s.scheme),
    ("signature", hexJ s.signature)]

def SignedSetupRecord.ofJson (j : Json) : V SignedSetupRecord := do
  let ctx := "setup"
  pure { record := { blocks := ← J.nat j "blocks" ctx,
                     commitment := ← hexField j "commitment" ctx,
                     salt := ← hexField j "salt" ctx, root := ← hexField j "root" ctx },
         scheme := ← J.str j "scheme" ctx, publicKey := ← hexField j "public_key" ctx,
         signature := ← hexField j "signature" ctx }

/-- A signed setup record from exactly its canonical JSON. -/
def decodeSignedSetupRecord : Json → V SignedSetupRecord := exactly SignedSetupRecord.ofJson SignedSetupRecord.toJson

/-- `setup_record_ok`: `s` is under `scheme`, which `schemes` accepts, and `publicKey`, and its signature verifies on
its record's bytes under `session`. -/
def setupRecordOk (schemes : Schemes) (scheme : String) (publicKey session : ByteArray) (s : SignedSetupRecord) :
    Bool :=
  s.scheme == scheme && s.publicKey == publicKey &&
    match schemes.lookup s.scheme with
    | none => false
    | some p =>
      match s.record.value >>= fun r => lmsVerify p s.publicKey session r s.signature with
      | .ok b => b
      | .error _ => false

/-- The bytes `memoryChallengerMessage` hashes (`memoryChallengerMessage_eq`). -/
def rolePreimage (I session record : ByteArray) : ByteArray :=
  field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field session) ++ field record

/-! ## The bytes as lists -/

private theorem toList_append (a b : ByteArray) : (a ++ b).data.toList = a.data.toList ++ b.data.toList := by
  simp [ByteArray.data_append]

private theorem length_toList (b : ByteArray) : b.data.toList.length = b.size := by simp

private theorem size_append (a b : ByteArray) : (a ++ b).size = a.size + b.size := by
  rw [← length_toList, toList_append, List.length_append, length_toList, length_toList]

private theorem ext_toList {a b : ByteArray} (h : a.data.toList = b.data.toList) : a = b := by
  apply ByteArray.ext
  simpa using h

private theorem split_bytes {a a' : ByteArray} {r r' : List UInt8} (h : a.data.toList ++ r = a'.data.toList ++ r')
    (hl : a.size = a'.size) : a = a' ∧ r = r' := by
  obtain ⟨h1, h2⟩ := List.append_inj h (by rw [length_toList, length_toList, hl])
  exact ⟨ext_toList h1, h2⟩

private theorem be64_toList (x : UInt64) : (be64 x).data.toList =
    [(x >>> 56).toUInt8, (x >>> 48).toUInt8, (x >>> 40).toUInt8, (x >>> 32).toUInt8,
     (x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, (x >>> 0).toUInt8] := by
  simp [be64, List.range', ByteArray.push, ByteArray.emptyWithCapacity]
  rfl

private theorem be64_size (x : UInt64) : (be64 x).size = 8 := by
  rw [← length_toList, be64_toList]
  rfl

private theorem byte_toNat (x : UInt64) (k : Nat) (hk : k < 64) :
    ((x >>> (UInt64.ofNat k)).toUInt8).toNat = x.toNat / 2 ^ k % 256 := by
  simp [UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow, Nat.mod_eq_of_lt hk]

private theorem be64_inj {x y : UInt64} (h : be64 x = be64 y) : x = y := by
  have h' := congrArg (fun b => b.data.toList) h
  simp only [be64_toList, List.cons.injEq] at h'
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, -⟩ := h'
  have e : ∀ k, k < 64 → (x >>> (UInt64.ofNat k)).toUInt8 = (y >>> (UInt64.ofNat k)).toUInt8 →
      x.toNat / 2 ^ k % 256 = y.toNat / 2 ^ k % 256 := fun k hk he => by
    rw [← byte_toNat x k hk, ← byte_toNat y k hk, he]
  have a0 := e 56 (by omega) h0
  have a1 := e 48 (by omega) h1
  have a2 := e 40 (by omega) h2
  have a3 := e 32 (by omega) h3
  have a4 := e 24 (by omega) h4
  have a5 := e 16 (by omega) h5
  have a6 := e 8 (by omega) h6
  have a7 := e 0 (by omega) h7
  have hx := x.toNat_lt
  have hy := y.toNat_lt
  apply UInt64.toNat_inj.mp
  simp only [Nat.reducePow] at a0 a1 a2 a3 a4 a5 a6 a7 hx hy ⊢
  omega

private theorem be64_size_inj {n m : Nat} (hn : n < 2 ^ 64) (hm : m < 2 ^ 64)
    (h : be64 n.toUInt64 = be64 m.toUInt64) : n = m := by
  have := congrArg UInt64.toNat (be64_inj h)
  simp only [Nat.toUInt64, UInt64.toNat_ofNat'] at this
  rwa [Nat.mod_eq_of_lt hn, Nat.mod_eq_of_lt hm] at this

private theorem hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash
  exact Array.size_ofFn

private theorem field_size (b : ByteArray) : (field b).size = 8 + b.size := by
  rw [field, size_append, be64_size]

private theorem beN_size {n x : Nat} {what : String} {b : ByteArray} (h : beN n x what = .ok b) : b.size = n := by
  unfold beN at h
  split at h
  · cases h
    simp [ByteArray.size]
  · cases h

/-! ## Sizes -/

theorem SETUP_RECORD_TAG_size : SETUP_RECORD_TAG.size = 46 := by
  simp only [SETUP_RECORD_TAG, ByteArray.size_push, String.toUTF8_eq_toByteArray, String.size_toByteArray]
  decide

theorem SetupRecord.value_size {r : SetupRecord} {b : ByteArray} (h : r.value = .ok b) :
    b.size = SETUP_RECORD_TAG.size + 32 + r.commitment.size + r.salt.size + r.root.size := by
  unfold SetupRecord.value at h
  cases hb : beN 8 r.blocks "blocks" with
  | error e => simp [hb, bind, Except.bind] at h
  | ok n =>
    simp only [hb, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [size_append, field_size, beN_size hb]
    omega

/-- PoUS's session record is a digest. -/
theorem pous_record_size {c : SessionCommitment} {ended : String} {rounds : Array Round} {r : ByteArray}
    (h : message c ended rounds = .ok r) : r.size = 64 := by
  unfold message at h
  cases hr : recordsRoot c.session rounds <;> simp [hr, bind, Except.bind, pure, Except.pure] at h
  subst h
  exact hash_size _

/-- The wipe's session record is a digest. -/
theorem wipe_record_size {c : StreamedWipe.Commitment} {ended : String} {fillStartNs fillEndNs : Int}
    {batches : Array StreamedWipe.Batch} {egressBits : Int} {r : ByteArray}
    (h : StreamedWipe.message c ended fillStartNs fillEndNs batches egressBits = .ok r) : r.size = 64 := by
  unfold StreamedWipe.message at h
  cases hr : StreamedWipe.recordsRoot c.session batches <;> simp [hr, bind, Except.bind, pure, Except.pure] at h
  subst h
  exact hash_size _

/-! ## No setup record is another subprotocol's -/

/-- A setup record is never 64 bytes long, so never a digest. -/
theorem SetupRecord.value_ne_digest {r : SetupRecord} {b d : ByteArray} (h : r.value = .ok b) (hd : d.size = 64) :
    b ≠ d := by
  intro e
  have := SetupRecord.value_size h
  rw [e, hd, SETUP_RECORD_TAG_size] at this
  omega

theorem SetupRecord.value_ne_pous {r : SetupRecord} {b : ByteArray} {c : SessionCommitment} {ended : String}
    {rounds : Array Round} {d : ByteArray} (h : r.value = .ok b)
    (hd : MemoryChallenger.message c ended rounds = .ok d) : b ≠ d :=
  SetupRecord.value_ne_digest h (pous_record_size hd)

theorem SetupRecord.value_ne_wipe {r : SetupRecord} {b : ByteArray} {c : StreamedWipe.Commitment} {ended : String}
    {fillStartNs fillEndNs : Int} {batches : Array StreamedWipe.Batch} {egressBits : Int} {d : ByteArray}
    (h : r.value = .ok b) (hd : StreamedWipe.message c ended fillStartNs fillEndNs batches egressBits = .ok d) :
    b ≠ d :=
  SetupRecord.value_ne_digest h (wipe_record_size hd)

/-! ## The role's pre-image -/

/-- The role's message is SHA-512 of `rolePreimage`. -/
theorem memoryChallengerMessage_eq {I session record : ByteArray} (hI : I.size = 16) :
    memoryChallengerMessage I session record = .ok (Sha512.hash (rolePreimage I session record)) := by
  simp [memoryChallengerMessage, hI, rolePreimage, pure, Except.pure]

/-- The role's pre-image determines the key identifier, the session and the record, for 16-byte identifiers and
sessions under `2^64` bytes. -/
theorem rolePreimage_inj {I I' s s' r r' : ByteArray} (hI : I.size = 16) (hI' : I'.size = 16) (hs : s.size < 2 ^ 64)
    (hs' : s'.size < 2 ^ 64) (h : rolePreimage I s r = rolePreimage I' s' r') : I = I' ∧ s = s' ∧ r = r' := by
  have e := congrArg (fun b => b.data.toList) h
  simp only [rolePreimage, field, toList_append, List.append_assoc] at e
  have e := List.append_cancel_left e
  have e := List.append_cancel_left e
  obtain ⟨eI, e⟩ := split_bytes e (by rw [hI, hI'])
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  obtain ⟨el, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  have hl := be64_size_inj hs hs' el
  obtain ⟨es, e⟩ := split_bytes e hl
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact ⟨eI, es, ext_toList e⟩

/-- A setup record's pre-image is never the pre-image of a 64-byte record, whatever the key identifiers and sessions. -/
theorem setupPreimage_ne_digest {I I' s s' b d : ByteArray} {r : SetupRecord} (hI : I.size = 16)
    (hI' : I'.size = 16) (hs : s.size < 2 ^ 64) (hs' : s'.size < 2 ^ 64) (h : r.value = .ok b) (hd : d.size = 64) :
    rolePreimage I s b ≠ rolePreimage I' s' d :=
  fun e => SetupRecord.value_ne_digest h hd (rolePreimage_inj hI hI' hs hs' e).2.2

theorem setupPreimage_ne_pous {I I' s s' b d : ByteArray} {r : SetupRecord} {c : SessionCommitment}
    {ended : String} {rounds : Array Round} (hI : I.size = 16) (hI' : I'.size = 16) (hs : s.size < 2 ^ 64)
    (hs' : s'.size < 2 ^ 64) (h : r.value = .ok b) (hd : message c ended rounds = .ok d) :
    rolePreimage I s b ≠ rolePreimage I' s' d :=
  setupPreimage_ne_digest hI hI' hs hs' h (pous_record_size hd)

theorem setupPreimage_ne_wipe {I I' s s' b d : ByteArray} {r : SetupRecord} {c : StreamedWipe.Commitment}
    {ended : String} {fillStartNs fillEndNs : Int} {batches : Array StreamedWipe.Batch} {egressBits : Int}
    (hI : I.size = 16) (hI' : I'.size = 16) (hs : s.size < 2 ^ 64) (hs' : s'.size < 2 ^ 64) (h : r.value = .ok b)
    (hd : StreamedWipe.message c ended fillStartNs fillEndNs batches egressBits = .ok d) :
    rolePreimage I s b ≠ rolePreimage I' s' d :=
  setupPreimage_ne_digest hI hI' hs hs' h (wipe_record_size hd)

/-- A setup record's message equal to the role's message of a 64-byte record, under any key identifier and session,
is a SHA-512 collision: one signature never verifies as both. -/
theorem setupMessage_collision {I I' s s' d m : ByteArray} {r : SetupRecord} (hs : s.size < 2 ^ 64)
    (hs' : s'.size < 2 ^ 64) (hd : d.size = 64) (hm : r.message I s = .ok m)
    (hm' : memoryChallengerMessage I' s' d = .ok m) : ∃ x y, x ≠ y ∧ Sha512.hash x = Sha512.hash y := by
  unfold SetupRecord.message at hm
  cases hb : r.value with
  | error e => simp [hb, bind, Except.bind] at hm
  | ok b =>
    simp only [hb, bind, Except.bind] at hm
    by_cases hI : I.size = 16
    · by_cases hI' : I'.size = 16
      · rw [memoryChallengerMessage_eq hI] at hm
        rw [memoryChallengerMessage_eq hI'] at hm'
        exact ⟨_, _, setupPreimage_ne_digest hI hI' hs hs' hb hd, (Except.ok.inj hm).trans (Except.ok.inj hm').symm⟩
      · simp [memoryChallengerMessage, hI', bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at hm'
    · simp [memoryChallengerMessage, hI, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at hm

end MemoryChallenger
