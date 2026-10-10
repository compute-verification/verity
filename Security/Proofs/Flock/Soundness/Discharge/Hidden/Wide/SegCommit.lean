import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegChain
import Proofs.Flock.Soundness.Binding.RowSeg

/-!
# A segmented row's commit string from its chains

A `sha512/row-seg/v1` row of `l` bits commits in two levels. `seg_digest` is segment `j`'s chain: under its first block
`S(j, s, m)` and its bytes, its last compression's output is the digest of `RowSeg.segMsg l row j`, when the segment's
bytes reach its last block (`SegFits`). `seg_commit` is the outer chain: under `Q(l)`, its messages the segments' digests
(copies of the bits `D j` of each segment's last output) and then SHA-512's padding, the `hm96` slot outputs the commit
string of `RowSeg.enc l row`, the bytes `ExplicitLeaf.rowSeg l` binds. The outer message's digests always reach its last
block, so it needs no such condition.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock.HmNets hiding and
open Flock Hm Refine
open Binding (digBytes)
open Binding.RowSeg (count segBits segBytes segMsg digests enc)

theorem size_segmentPrefix (j s m : ℕ) : (HmRow.segmentPrefix j s m).size = 128 := by
  have h0 : ("verity/sha512-row-seg/v1/segment".toUTF8.push 0).size = 33 := by decide +kernel
  have h8 : ∀ x, (HmRow.be64 x).size = 8 := fun x => by simp [HmRow.be64, ByteArray.size]
  refine size_pad ?_
  rw [ByteArray.size_append, ByteArray.size_append, ByteArray.size_append, h0, h8, h8, h8]
  omega

/-- Segment `j`'s compressions after its first block: its bytes, then SHA-512's padding. -/
def segComps (m : ℕ) : ℕ := ((m + 7) / 8 + 17 + 127) / 128

/-- **Every segment's bytes reach its last block**: SHA-512's padding of no segment needs a block of its own. -/
def SegFits (l : ℕ) : Prop := ∀ j < count l, 128 * (segComps (segBits l j) - 1) ≤ (segBits l j + 7) / 8

theorem length_segBytes {l : ℕ} {row : List UInt8} (hrow : row.length = (l + 7) / 8) {j : ℕ} (hj : j < count l) :
    (segBytes l row j).length = (segBits l j + 7) / 8 := by
  unfold segBytes
  rw [List.length_take, List.length_drop, hrow]
  unfold count at hj
  unfold segBits HmRow.ROW_SEG_BITS at *
  omega

theorem digBytes_rowMsgP (pre : ByteArray) (data : List UInt8) :
    digBytes (H512 (pre.data.toList ++ data)) = (Sha512.hash (rowMsgP pre data)).data.toList := by
  simp only [Binding.digBytes, H512, enc512, rowMsgP]

/-- **A segment's chain hashes its message**: `n` compressions (`segComps m`), the first `cv` `S(j, s, m)`'s midstate,
each later one the previous output, the messages the segment's bytes and padding: the last output is the digest of
`segMsg l row j` in word order. -/
theorem seg_digest {l : ℕ} {row : List UInt8} (hrow : row.length = (l + 7) / 8) {j : ℕ} (hj : j < count l)
    (hfit : 128 * (segComps (segBits l j) - 1) ≤ (segBits l j + 7) / 8)
    (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < segComps (segBits l j), sub k < 3)
    (hrows : ∀ k < segComps (segBits l j), stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : ∀ q : ℕ, q < 512 → zs 0 (1536 * sub 0 + q) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.segmentPrefix j (count l) (segBits l j)) 0) q)
    (hchain : ∀ k, k + 1 < segComps (segBits l j) → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < segComps (segBits l j), ∀ o < 128, ∀ t < 8, zs k (1536 * sub k + 512 + wordBit (8 * o + t)) =
      bitL (segBytes l row j ++ (HmRow.padTail (128 + (segBits l j + 7) / 8)).data.toList) (8 * (128 * k + o) + t)) :
    ∀ t < 512, Hm.bit (zs (segComps (segBits l j) - 1))
        ((compression3.run St.empty).1.2.getD (sub (segComps (segBits l j) - 1)) []) (wordBit t) =
      bitL (digBytes (H512 (segMsg l row j))) t := by
  have hd := length_segBytes hrow hj
  have := chain_digestP (size_segmentPrefix j (count l) (segBits l j)) (n := segComps (segBits l j))
    (by unfold segComps; omega) (data := segBytes l row j) (by rw [hd]; exact hfit) (by rw [hd]; unfold segComps; omega)
    zs sub hsub hrows hcv0 hchain (by rw [hd]; exact hmsg)
  unfold segMsg
  rw [digBytes_rowMsgP]
  exact this

/-- Byte `a` of the outer message's data is byte `a % 64` of segment `a / 64`'s digest. -/
theorem bitL_digests {l : ℕ} {row : List UInt8} {a : ℕ} (ha : a < 64 * count l) (t : ℕ) :
    bitL (digests l row) (8 * a + t % 8) = bitL (digBytes (H512 (segMsg l row (a / 64)))) (8 * (a % 64) + t % 8) := by
  have hf : ∀ j, (digBytes (H512 (segMsg l row j))).length = 64 := fun j => hL512 _
  have e := Binding.RowSeg.take_drop_flatMap_range hf (n := count l) (j := a / 64) (by omega)
  unfold bitL
  rw [show (8 * a + t % 8) / 8 = a by omega, show (8 * (a % 64) + t % 8) / 8 = a % 64 by omega,
    show (8 * a + t % 8) % 8 = t % 8 by omega, show (8 * (a % 64) + t % 8) % 8 = t % 8 by omega, digests, ← e,
    List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt (by omega), List.getElem?_drop,
    show 64 * (a / 64) + a % 64 = a by omega]

theorem length_digests (l : ℕ) (row : List UInt8) : (digests l row).length = 64 * count l := by
  rw [digests, List.length_flatMap, List.map_congr_left (g := fun _ => 64) fun j _ => hL512 _, List.map_const',
    List.sum_replicate, List.length_range, smul_eq_mul, Nat.mul_comm]

/-- The outer message's compressions after `Q(l)`, its `hm96` slot's block not among them. -/
def outerComps (l : ℕ) : ℕ := (64 * count l + 17 + 127) / 128 - 1

/-- **The outer chain commits the row**: its `n` compressions (`outerComps l`, maybe none) and its `hm96` slot, the
first `cv` `Q(l)`'s midstate, each later one the previous output, the messages the segments' digests (bit
`wordBit (8 b + t)` of segment `a / 64`'s `D`, at the data's byte `a`, `b = a % 64`) and then SHA-512's padding: the slot
outputs the commit string of `RowSeg.enc l row` under the salt. -/
theorem seg_commit {l : ℕ} {row : List UInt8} (D : ℕ → ℕ → Bool)
    (hD : ∀ j < count l, ∀ t < 512, D j (wordBit t) = bitL (digBytes (H512 (segMsg l row j))) t)
    (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < outerComps l, sub k < 3)
    (hrows : ∀ k < outerComps l, stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : 0 < outerComps l → ∀ q : ℕ, q < 512 →
      zs 0 (1536 * sub 0 + q) = wordsBit (Sha512.compress Sha512.iv (HmRow.rowPrefixSeg l) 0) q)
    (hchain : ∀ k, k + 1 < outerComps l → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < outerComps l, ∀ o < 128, ∀ t < 8, zs k (1536 * sub k + 512 + wordBit (8 * o + t)) =
      D ((128 * k + o) / 64) (wordBit (8 * ((128 * k + o) % 64) + t)))
    (zh : Var → Bool) (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh : ∀ q : ℕ, q < 512 → zh q = if outerComps l = 0 then
      wordsBit (Sha512.compress Sha512.iv (HmRow.rowPrefixSeg l) 0) q
      else Hm.bit (zs (outerComps l - 1)) ((compression3.run St.empty).1.2.getD (sub (outerComps l - 1)) []) q)
    (hpadh : ∀ o < 128, ∀ t < 8, zh (512 + wordBit (8 * o + t)) =
      if 128 * outerComps l + o < 64 * count l then
        D ((128 * outerComps l + o) / 64) (wordBit (8 * ((128 * outerComps l + o) % 64) + t))
      else bitL (HmRow.padTail (128 + 64 * count l)).data.toList (8 * (128 * outerComps l + o - 64 * count l) + t))
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString (enc l row) s) t := by
  have hpre := size_rowPrefixSeg l
  have hlen := length_digests l row
  have h1 : 128 * outerComps l ≤ 64 * count l := by unfold outerComps; omega
  have h2 : 64 * count l + 17 ≤ 128 * (outerComps l + 1) := by unfold outerComps; omega
  have hdat : ∀ a < 64 * count l, ∀ t < 8,
      D (a / 64) (wordBit (8 * (a % 64) + t)) = bitL (digests l row) (8 * a + t) := by
    intro a ha t ht
    have e := bitL_digests (row := row) ha t
    rw [Nat.mod_eq_of_lt ht] at e
    rw [e, hD _ (by omega) _ (by omega)]
  have hpad : ∀ q : ℕ, q < 1024 → zh (512 + q) =
      wbit (blockV (padData (rowMsgP (HmRow.rowPrefixSeg l) (digests l row))) (outerComps l + 1)) q := by
    intro q hq
    have ho : 8 * (q / 64) + (7 - q % 64 / 8) < 128 := by omega
    have hq' : wordBit (8 * (8 * (q / 64) + (7 - q % 64 / 8)) + q % 8) = q := by unfold wordBit; omega
    have e := hpadh _ ho (q % 8) (Nat.mod_lt _ (by norm_num))
    rw [hq'] at e
    rw [e, wbit_blockV_rowP hpre _ _ hq, hlen]
    unfold bitL
    split
    · rename_i hlt
      rw [List.getD_append _ _ _ _ (by rw [hlen]; omega)]
      have := hdat _ hlt (q % 8) (Nat.mod_lt _ (by norm_num))
      unfold bitL at this
      rw [this]
    · rename_i hge
      rw [List.getD_append_right _ _ _ _ (by rw [hlen]; omega), hlen,
        show (8 * (128 * outerComps l + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 - 64 * count l =
          (8 * (128 * outerComps l + (8 * (q / 64) + (7 - q % 64 / 8)) - 64 * count l) + q % 8) / 8 by omega,
        show (8 * (128 * outerComps l + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 =
          (8 * (128 * outerComps l + (8 * (q / 64) + (7 - q % 64 / 8)) - 64 * count l) + q % 8) % 8 by omega]
  rcases Nat.eq_zero_or_pos (outerComps l) with h0 | hn
  · have hpad0 := hpad
    rw [h0] at hpad0
    exact row_commit0P hpre (by rw [hlen]; omega) zh hh
      (fun q hq => by rw [hcvh q hq, if_pos h0]) hpad0 hs hy
  · exact row_commitP hpre hn (by rw [hlen]; exact h1) (by rw [hlen]; exact h2) zs sub hsub hrows (hcv0 hn) hchain
      (fun k hk o ho t ht => by
        rw [hmsg k hk o ho t ht, hdat _ (by omega) t ht])
      zh hh (fun q hq => by rw [hcvh q hq, if_neg (by omega)]) hpad hs hy

end FlockSoundness.Discharge.Hidden
