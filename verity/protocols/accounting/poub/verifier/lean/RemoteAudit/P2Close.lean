module

public import MemoryChallenger.StreamedWipe

/-!
# P2's close: the bytes the memory challenger signs over a P2 session's log

The memory challenger signs P2's close with the key it signs PoUS's sessions and the wipe's with, as its role's message
(`MemoryChallenger.memoryChallengerMessage`): `SHA-512(rolePreimage I session record)`, the pre-image
`[verity/memory-challenger-record/v1] ‖ I ‖ [[session]] ‖ [record]`, each `[·]` a u64 big-endian length then the bytes
(`MemoryChallenger.field`). P2's record (`closeRecord`) is raw bytes, never a digest: `CLOSE_TAG` (NUL-ended), the log's
line count as a u64 big-endian, and the case's 64-byte chain (`chainOf` over the case's head, then the log's entries:
`P2Protocol.headOf`), so it is `CLOSE_TAG.size + 72` bytes long.
PoUS's session record (`MemoryChallenger.message`) and the wipe's (`MemoryChallenger.StreamedWipe.message`) are SHA-512
digests, 64 bytes long, so no close record is either (`closeRecord_ne_pous`, `closeRecord_ne_wipe`), and the role's
pre-image determines its key identifier, session and record (`rolePreimage_inj`): a close's pre-image is never a PoUS
session's or a wipe's (`closePreimage_ne_pous`, `closePreimage_ne_wipe`), on the bytes hashed, with no assumption on the
hash.
-/

@[expose] public section

namespace RemoteAudit.P2Close

open Flock

/-- P2's tag, NUL-ended: the case's chain starts from it (`chainOf`) and the close's record opens with it
(`closeRecord`). -/
def CLOSE_TAG : ByteArray := "verity/remote-bandwidth/p2/close/v0".toUTF8.push 0

/-- The chain over `entries`' bytes (a case's head, then its log's entries), SPOT's release-log chain (`Spot.ChallengerSession.logChain`) under
`CLOSE_TAG`: `c_0 = SHA-512([CLOSE_TAG])`, `c_(k+1) = SHA-512(c_k ‖ [entry_k])`. -/
def chainOf (entries : Array ByteArray) : ByteArray :=
  entries.foldl (fun h l => Sha512.hash (h ++ MemoryChallenger.field l))
    (Sha512.hash (MemoryChallenger.field CLOSE_TAG))

/-- The record the close signs: `CLOSE_TAG ‖ u64be(lines) ‖ chain`. -/
def closeRecord (lines : Nat) (chain : ByteArray) : ByteArray := CLOSE_TAG ++ be64 lines.toUInt64 ++ chain

/-- The message the close signs under the key whose identifier is `ident`: the role's message of the session and
`closeRecord`. -/
def closeMessage (ident session : ByteArray) (lines : Nat) (chain : ByteArray) : Except String ByteArray :=
  MemoryChallenger.memoryChallengerMessage ident session (closeRecord lines chain)

/-- The bytes the role's message hashes (`memoryChallengerMessage_eq`). -/
def rolePreimage (I session record : ByteArray) : ByteArray :=
  MemoryChallenger.field MemoryChallenger.MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++
    MemoryChallenger.field (MemoryChallenger.field session) ++ MemoryChallenger.field record

/-! ## The bytes as lists -/

private theorem toList_append (a b : ByteArray) : (a ++ b).data.toList = a.data.toList ++ b.data.toList := by
  simp [ByteArray.data_append]

private theorem length_toList (b : ByteArray) : b.data.toList.length = b.size := by simp

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

/-! ## Sizes -/

theorem chainOf_size (entries : Array ByteArray) : (chainOf entries).size = 64 := by
  unfold chainOf
  apply Array.foldl_induction (fun _ (b : ByteArray) => b.size = 64)
  · exact hash_size _
  · intro _ _ _
    exact hash_size _

theorem closeRecord_size (lines : Nat) (chain : ByteArray) :
    (closeRecord lines chain).size = CLOSE_TAG.size + 8 + chain.size := by
  rw [closeRecord, ← length_toList, toList_append, toList_append, List.length_append, List.length_append,
    length_toList, length_toList, length_toList, be64_size]

/-- PoUS's session record is a digest. -/
theorem pous_record_size {c : MemoryChallenger.SessionCommitment} {ended : String}
    {rounds : Array MemoryChallenger.Round} {r : ByteArray} (h : MemoryChallenger.message c ended rounds = .ok r) :
    r.size = 64 := by
  unfold MemoryChallenger.message at h
  cases hr : MemoryChallenger.recordsRoot c.session rounds <;> simp [hr, bind, Except.bind, pure, Except.pure] at h
  subst h
  exact hash_size _

/-- The wipe's session record is a digest. -/
theorem wipe_record_size {c : MemoryChallenger.StreamedWipe.Commitment} {ended : String} {startMark : ByteArray}
    {fillStartNs fillEndNs : Int} {batches : Array MemoryChallenger.StreamedWipe.Batch} {r : ByteArray}
    (h : MemoryChallenger.StreamedWipe.message c ended startMark fillStartNs fillEndNs batches = .ok r) :
    r.size = 64 := by
  unfold MemoryChallenger.StreamedWipe.message at h
  cases hr : MemoryChallenger.StreamedWipe.recordsRoot c.session batches <;>
    simp [hr, bind, Except.bind, pure, Except.pure] at h
  subst h
  exact hash_size _

/-! ## No close record is another subprotocol's -/

/-- A close record over a chain is never 64 bytes long, so never a digest. -/
theorem closeRecord_ne_digest {lines : Nat} {entries : Array ByteArray} {d : ByteArray} (hd : d.size = 64) :
    closeRecord lines (chainOf entries) ≠ d := by
  intro h
  have := congrArg ByteArray.size h
  rw [closeRecord_size, chainOf_size, hd] at this
  omega

theorem closeRecord_ne_pous {lines : Nat} {entries : Array ByteArray} {c : MemoryChallenger.SessionCommitment}
    {ended : String} {rounds : Array MemoryChallenger.Round} {r : ByteArray}
    (h : MemoryChallenger.message c ended rounds = .ok r) : closeRecord lines (chainOf entries) ≠ r :=
  closeRecord_ne_digest (pous_record_size h)

theorem closeRecord_ne_wipe {lines : Nat} {entries : Array ByteArray} {c : MemoryChallenger.StreamedWipe.Commitment}
    {ended : String} {startMark : ByteArray} {fillStartNs fillEndNs : Int}
    {batches : Array MemoryChallenger.StreamedWipe.Batch} {r : ByteArray}
    (h : MemoryChallenger.StreamedWipe.message c ended startMark fillStartNs fillEndNs batches = .ok r) :
    closeRecord lines (chainOf entries) ≠ r :=
  closeRecord_ne_digest (wipe_record_size h)

/-! ## The role's pre-image -/

/-- The role's message is SHA-512 of `rolePreimage`. -/
theorem memoryChallengerMessage_eq {I session record : ByteArray} (hI : I.size = 16) :
    MemoryChallenger.memoryChallengerMessage I session record = .ok (Sha512.hash (rolePreimage I session record)) := by
  simp [MemoryChallenger.memoryChallengerMessage, hI, rolePreimage, pure, Except.pure]

/-- The role's pre-image determines the key identifier, the session and the record, for 16-byte identifiers and
sessions under `2^64` bytes. -/
theorem rolePreimage_inj {I I' s s' r r' : ByteArray} (hI : I.size = 16) (hI' : I'.size = 16) (hs : s.size < 2 ^ 64)
    (hs' : s'.size < 2 ^ 64) (h : rolePreimage I s r = rolePreimage I' s' r') : I = I' ∧ s = s' ∧ r = r' := by
  have e := congrArg (fun b => b.data.toList) h
  simp only [rolePreimage, MemoryChallenger.field, toList_append, List.append_assoc] at e
  have e := List.append_cancel_left e
  have e := List.append_cancel_left e
  obtain ⟨eI, e⟩ := split_bytes e (by rw [hI, hI'])
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  obtain ⟨el, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  have hl := be64_size_inj hs hs' el
  obtain ⟨es, e⟩ := split_bytes e hl
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact ⟨eI, es, ext_toList e⟩

/-- A close's pre-image is never the pre-image of a 64-byte record, whatever the key identifiers and sessions. -/
theorem closePreimage_ne_digest {I I' s s' d : ByteArray} {lines : Nat} {entries : Array ByteArray}
    (hI : I.size = 16) (hI' : I'.size = 16) (hs : s.size < 2 ^ 64) (hs' : s'.size < 2 ^ 64) (hd : d.size = 64) :
    rolePreimage I s (closeRecord lines (chainOf entries)) ≠ rolePreimage I' s' d :=
  fun h => closeRecord_ne_digest hd (rolePreimage_inj hI hI' hs hs' h).2.2

theorem closePreimage_ne_pous {I I' s s' r : ByteArray} {lines : Nat} {entries : Array ByteArray}
    {c : MemoryChallenger.SessionCommitment} {ended : String} {rounds : Array MemoryChallenger.Round}
    (hI : I.size = 16) (hI' : I'.size = 16) (hs : s.size < 2 ^ 64) (hs' : s'.size < 2 ^ 64)
    (h : MemoryChallenger.message c ended rounds = .ok r) :
    rolePreimage I s (closeRecord lines (chainOf entries)) ≠ rolePreimage I' s' r :=
  closePreimage_ne_digest hI hI' hs hs' (pous_record_size h)

theorem closePreimage_ne_wipe {I I' s s' r : ByteArray} {lines : Nat} {entries : Array ByteArray}
    {c : MemoryChallenger.StreamedWipe.Commitment} {ended : String} {startMark : ByteArray}
    {fillStartNs fillEndNs : Int} {batches : Array MemoryChallenger.StreamedWipe.Batch}
    (hI : I.size = 16) (hI' : I'.size = 16) (hs : s.size < 2 ^ 64) (hs' : s'.size < 2 ^ 64)
    (h : MemoryChallenger.StreamedWipe.message c ended startMark fillStartNs fillEndNs batches = .ok r) :
    rolePreimage I s (closeRecord lines (chainOf entries)) ≠ rolePreimage I' s' r :=
  closePreimage_ne_digest hI hI' hs hs' (wipe_record_size h)

end RemoteAudit.P2Close
