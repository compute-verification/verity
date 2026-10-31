import Properties.Pouw.Lead.Commit
import Proofs.Properties.Pouw.Commit
import Proofs.Flock.Soundness.KeyedDraw

/-!
# row-bound: a SHA-512 frame-v3 leaf's first block binds its domain and position

`leafFirstBlock` proves `Properties.Pouw.Lead.Commit.LeafFirstBlock` by reading the leaf's message as bytes
(`Properties.Pouw.Commit.frameBytes_data`): 41 fixed bytes, then `be64 64 ‖ id`, then `be64 |rank| ‖ rank`, which end
by byte 128 when `rank` takes at most 7 bytes (`natLen_le_seven`), so the first block holds them whole and they parse
back (`be8_inj`, `natBytes_inj`). `rowBound` proves `RowBound`: equal messages give the first case, and different ones
with one digest are the collision.
-/

namespace Pouw.SecurityProofs.Lead.Commit

open PouwCommit Properties.Pouw.Lead.Commit
open FlockSoundness.Randomness (natLen natBytes_toList natBytes_inj)

/-- frame-v3's fixed prefix, `veritor/protocol/merkle/frame/v3\0`. -/
def frameMagic : ByteArray := "veritor/protocol/merkle/frame/v3".toUTF8.push 0

/-- `Properties.Pouw.Commit.frameBytes_data` with the prefix as `frameMagic`, the bytes PoUW's `FRAME` holds. -/
theorem frameBytes_data' (tag : String) (parts : List ByteArray) :
    (frameBytes tag parts).data.toList = frameMagic.data.toList ++ (Flock.be32 tag.toUTF8.size.toUInt32).data.toList ++
      tag.toUTF8.data.toList ++ Properties.Pouw.Commit.framed parts :=
  (Properties.Pouw.Commit.frameBytes_data tag parts).trans (by rfl)

theorem natBytes_size (n : Nat) : (natBytes n).size = natLen n := by
  simp [natBytes, natLen, ByteArray.size]

theorem natLen_le_seven {n : Nat} (h : n < 2 ^ 56) : natLen n ≤ 7 := by
  unfold natLen
  by_cases h0 : n = 0
  · simp [h0]
  · have := (Nat.log2_lt h0).2 h
    simp only [h0, ite_false]
    omega

theorem domain_sha512 {binding : ByteArray} {owner : Int} {count : Nat} {d : PouwCommit.Domain}
    (h : PouwCommit.Domain.mk? .sha512 binding owner count = .ok d) : d.id.size = 64 ∧ d.hash = .sha512 := by
  unfold PouwCommit.Domain.mk? at h
  split at h <;> split at h <;>
    simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind, pure, Except.pure] at h
  subst h
  exact ⟨Properties.Pouw.Commit.sha512_size _, rfl⟩

theorem firstBlock_data (m : ByteArray) : (firstBlock m).data.toList = m.data.toList.take 128 := by
  simp [firstBlock, ByteArray.data_extract, Array.toList_extract, List.extract]

theorem take_append_long {α : Type} {a b : List α} {n : Nat} (h : a.length ≤ n) :
    (a ++ b).take n = a ++ b.take (n - a.length) := by
  induction a generalizing n with
  | nil => simp
  | cons x xs ih =>
    cases n with
    | zero => simp at h
    | succ n => simp [ih (by simpa using h)]

/-- The leaf message's bytes: the 41 fixed bytes, the id's frame, the position's frame, then the rest. -/
theorem leafMessage_data (d : PouwCommit.Domain) (rank : Nat) (schema : String) (v : ByteArray) :
    (leafMessage d rank schema v).data.toList =
      (frameMagic.data.toList ++ (Flock.be32 "leaf".toUTF8.size.toUInt32).data.toList ++ "leaf".toUTF8.data.toList) ++
        ((be8 d.id.size).data.toList ++ (d.id.data.toList ++ ((be8 (natBytes rank).size).data.toList ++
          ((natBytes rank).data.toList ++ Properties.Pouw.Commit.framed [natBytes rank, schema.toUTF8, v])))) := by
  simp [leafMessage, frameBytes_data', Properties.Pouw.Commit.framed, List.append_assoc]

theorem prefix_length :
    (frameMagic.data.toList ++ (Flock.be32 "leaf".toUTF8.size.toUInt32).data.toList ++ "leaf".toUTF8.data.toList).length =
      41 := by
  decide +kernel

/-- A SHA-512 frame-v3 leaf's first block: the 41 fixed bytes, the id's frame, the position's frame, then part of the
rest, when the position takes at most 7 bytes. -/
theorem firstBlock_leaf (d : PouwCommit.Domain) (rank : Nat) (schema : String) (v : ByteArray) (hid : d.id.size = 64)
    (hr : natLen rank ≤ 7) : ∃ k, (firstBlock (leafMessage d rank schema v)).data.toList =
      (frameMagic.data.toList ++ (Flock.be32 "leaf".toUTF8.size.toUInt32).data.toList ++ "leaf".toUTF8.data.toList) ++
        ((be8 d.id.size).data.toList ++ (d.id.data.toList ++ ((be8 (natBytes rank).size).data.toList ++
          ((natBytes rank).data.toList ++
            (Properties.Pouw.Commit.framed [natBytes rank, schema.toUTF8, v]).take k)))) := by
  rw [firstBlock_data, leafMessage_data]
  have hI : d.id.data.toList.length = 64 := by simpa using hid
  have hN : (natBytes rank).data.toList.length = natLen rank := by simpa using natBytes_size rank
  have h8 := Properties.Pouw.Commit.be8_length
  rw [take_append_long (by rw [prefix_length]; omega), take_append_long (by rw [prefix_length, h8]; omega),
    take_append_long (by rw [prefix_length, h8, hI]; omega),
    take_append_long (by rw [prefix_length, h8, hI, h8]; omega),
    take_append_long (by rw [prefix_length, h8, hI, h8, hN]; omega)]
  exact ⟨_, rfl⟩

theorem leafFirstBlock : LeafFirstBlock := by
  intro binding owner count binding' owner' count' d d' rank rank' schema schema' v v' b hd hd' hb hb' hr hr'
  have h := congrArg (fun x : ByteArray => x.data.toList) (hb.trans hb'.symm)
  obtain ⟨hid, -⟩ := domain_sha512 hd
  obtain ⟨hid', -⟩ := domain_sha512 hd'
  obtain ⟨k, hk⟩ := firstBlock_leaf d rank schema v hid (natLen_le_seven hr)
  obtain ⟨k', hk'⟩ := firstBlock_leaf d' rank' schema' v' hid' (natLen_le_seven hr')
  have h8 := Properties.Pouw.Commit.be8_length
  rw [hk, hk'] at h
  have h := List.append_cancel_left h
  obtain ⟨-, h⟩ := List.append_inj h (by rw [h8, h8])
  obtain ⟨hI, h⟩ := List.append_inj h (by simp [hid, hid'])
  obtain ⟨hE, h⟩ := List.append_inj h (by rw [h8, h8])
  have hs : (natBytes rank).size = (natBytes rank').size :=
    Properties.Pouw.Commit.be8_inj _ _ (by rw [natBytes_size]; have := natLen_le_seven hr; omega)
      (by rw [natBytes_size]; have := natLen_le_seven hr'; omega) hE
  obtain ⟨hN, -⟩ := List.append_inj h (by simp [hs])
  exact ⟨ByteArray.ext (Array.toList_inj.1 hI), natBytes_inj hN⟩

/-- A SHA-512 domain's leaf is the SHA-512 digest of `leafMessage`. The first case reads `Domain.leaf` as a frame over
`natBytes`, the second as `Flock.FV3`'s leaf, whose `FV3.uint` is `natBytes`. -/
theorem leaf_eq_hash {d : PouwCommit.Domain} (hh : d.hash = .sha512) (rank : Nat) (schema : String) (v : ByteArray) :
    d.leaf rank schema v = Flock.Sha512.hash (leafMessage d rank schema v) := by
  first
  | unfold PouwCommit.Domain.leaf frameHash leafMessage; rw [hh]; rfl
  | have hu : ∀ n, Flock.FV3.uint n = natBytes n := by
      intro n
      have hl : (Nat.log2 n + 8) / 8 = max 1 (((if n = 0 then 0 else Nat.log2 n + 1) + 7) / 8) := by
        by_cases h0 : n = 0
        · subst h0; simp
        · rw [if_neg h0, Nat.max_eq_right (show 1 ≤ (Nat.log2 n + 1 + 7) / 8 by omega)] <;> omega
      simp only [Flock.FV3.uint, natBytes]
      rw [hl]
    obtain ⟨h, c, i⟩ := d
    cases hh
    show Flock.Sha512.hash (Flock.FV3.input "leaf" [i, Flock.FV3.uint rank, Flock.FV3.uint rank, schema.toUTF8, v]) = _
    rw [hu] <;> rfl

theorem rowBound : RowBound := by
  intro binding owner count binding' owner' count' d d' rank rank' schema schema' v v' x hd hd' hx hx' hr hr'
  obtain ⟨-, hh⟩ := domain_sha512 hd
  obtain ⟨-, hh'⟩ := domain_sha512 hd'
  by_cases hm : leafMessage d rank schema v = leafMessage d' rank' schema' v'
  · exact .inl (leafFirstBlock binding owner count binding' owner' count' d d' rank rank' schema schema' v v'
      (firstBlock (leafMessage d rank schema v)) hd hd' rfl (by rw [hm]) hr hr')
  · refine .inr ⟨hm, ?_⟩
    rw [← leaf_eq_hash hh rank schema v, ← leaf_eq_hash hh' rank' schema' v', hx, hx']

end Pouw.SecurityProofs.Lead.Commit
