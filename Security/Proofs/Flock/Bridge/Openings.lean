import Proofs.Flock.Bridge.Deployed
import Proofs.Flock.Bridge.Lowers
import Proofs.Flock.Bridge.PortRoots
import Proofs.Flock.Bridge.SampledSound
import Proofs.Flock.Soundness.Discharge.Hm.Mask
import Properties.Verity.Game

/-!
# C-Flock's row openings are the spec's hm96 row leaves (`Hash.RowLeaf.accepts`) at SHA-512

The openings bridge `SampledDeploy` reads: a row that C-Flock's registered check opens (`Flock.Registered.opensLeaf`,
under the hm96 leaf `commitDep` computes), in any of the port formats C-Flock hashes (`rowMsg`: `sha512/row-seg/v1`,
`sha512/row/v2` or `sha512/row/v1`), is an opening the spec's verifier accepts,
`Hash.RowLeaf.accepts Verity.sha512 Verity.sha512`, at the port's domain, key and format (`formatOf`). Each layer is the
code's bytes read as the spec's messages:

* encodings: `be64`/`be32` are `Hash.be 8`/`Hash.be 4` (`be64_eq`, `be32_eq`), and `FV3.uint` is `Hash.uint` below
  `2^64` (`uint_eq`);
* frames: `Registered.frame` is `Hash.Tree.frame` (`frame_eq`), so the code's node and leaf are the spec's `nodeMsg`
  and `leafMsg` digests (`node_eq`, `leaf_eq`), at a domain whose id is the port's domain;
* the tree: `climbL` is `Tree.climb` (`climbL_eq`), `log2ceil` is `Tree.depth` (`depth_eq`), and `opensLeaf` gives
  `Tree.accepts` (`opensLeaf_accepts`);
* the row: a row's `sha512/row/v2` message is `RowV2.msg` (`enc_eq`), its `sha512/row/v1` message `RowV1.msg`
  (`encV1_eq`), and its `sha512/row-seg/v1` outer message `RowSeg.outerMsg` at SHA-512, segment by segment
  (`encSeg_eq`), so a row's message in its port's format hashes to the spec's digest in that format (`rowMsg_sha512`);
* the leaf: `Hm96.Default512`'s prefixes and mask are `Hash.Hm96`'s at the default key (`saltPrefix_eq`,
  `leafPrefix_eq`, `mask_eq`, the mask through `bitL_mask`), so the hm96 leaf `hm96Leaf` computes at `sha512Msg` is
  `Hash.Hm96.leaf` over the row's digest (`hm96Leaf_spec`);
* `rowLeaf_accepts`: the composition, in every format.

The ports' rows are `16 · words` bits (`Layout.portRowW`), so a `sha512/row/v1` row is whole words, as the spec's `v1`
verifier requires (`Hash.RowLeaf.Format.fits`; `live_opens`).

The spec's domain id is the port's domain itself, the UTF-8 of the port's name in the circuit file (`domOf`).
-/

namespace Flock.Bridge.Openings

open Claims Claims.Sampling
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length be64_bs hash_size)

/-! ## Encodings -/

theorem bs_push (b : ByteArray) (x : UInt8) : bs (b.push x) = bs b ++ [x] := by
  rw [bs, bs, ByteArray.data_push, Array.toList_push]

theorem be32_bs (x : UInt32) : bs (be32 x) =
    [(x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, (x >>> 0).toUInt8] := by
  simp [be32, bs, List.range', ByteArray.push, ByteArray.emptyWithCapacity]
  rfl

theorem byte64 (x j : ℕ) (hj : j ≤ 7) :
    (x.toUInt64 >>> (UInt64.ofNat (8 * j))).toUInt8 = (x / 256 ^ j % 256).toUInt8 := by
  apply UInt8.toNat_inj.mp
  have h8 : 8 * j % 64 = 8 * j := Nat.mod_eq_of_lt (by omega)
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.toUInt64_eq, UInt64.toNat_ofNat',
    Nat.shiftRight_eq_div_pow]
  have e1 : 8 * j % 2 ^ 64 % 64 = 8 * j := by rw [Nat.mod_eq_of_lt (by omega : 8 * j < 2 ^ 64), h8]
  have e2 : (2 : ℕ) ^ 64 = 2 ^ (8 * j) * 2 ^ (64 - 8 * j) := by rw [← pow_add]; congr 1; omega
  have e3 : (256 : ℕ) ^ j = 2 ^ (8 * j) := by rw [pow_mul]; norm_num
  rw [e1, e2, Nat.mod_mul_right_div_self, e3, Nat.mod_mod_of_dvd _ (pow_dvd_pow 2 (by omega : 8 ≤ 64 - 8 * j))]
  simp

theorem byte32 (x j : ℕ) (hj : j ≤ 3) :
    (x.toUInt32 >>> (UInt32.ofNat (8 * j))).toUInt8 = (x / 256 ^ j % 256).toUInt8 := by
  apply UInt8.toNat_inj.mp
  have h8 : 8 * j % 32 = 8 * j := Nat.mod_eq_of_lt (by omega)
  simp only [UInt32.toNat_toUInt8, UInt32.toNat_shiftRight, Nat.toUInt32_eq, UInt32.toNat_ofNat',
    Nat.shiftRight_eq_div_pow]
  have e1 : 8 * j % 2 ^ 32 % 32 = 8 * j := by rw [Nat.mod_eq_of_lt (by omega : 8 * j < 2 ^ 32), h8]
  have e2 : (2 : ℕ) ^ 32 = 2 ^ (8 * j) * 2 ^ (32 - 8 * j) := by rw [← pow_add]; congr 1; omega
  have e3 : (256 : ℕ) ^ j = 2 ^ (8 * j) := by rw [pow_mul]; norm_num
  rw [e1, e2, Nat.mod_mul_right_div_self, e3, Nat.mod_mod_of_dvd _ (pow_dvd_pow 2 (by omega : 8 ≤ 32 - 8 * j))]
  simp

/-- **`be64` is `Hash.be 8`.** -/
theorem be64_eq (x : ℕ) : bs (be64 x.toUInt64) = Hash.be 8 x := by
  rw [be64_bs]
  have := fun j hj => byte64 x j hj
  rw [show (56 : UInt64) = UInt64.ofNat (8 * 7) from rfl, show (48 : UInt64) = UInt64.ofNat (8 * 6) from rfl,
    show (40 : UInt64) = UInt64.ofNat (8 * 5) from rfl, show (32 : UInt64) = UInt64.ofNat (8 * 4) from rfl,
    show (24 : UInt64) = UInt64.ofNat (8 * 3) from rfl, show (16 : UInt64) = UInt64.ofNat (8 * 2) from rfl,
    show (8 : UInt64) = UInt64.ofNat (8 * 1) from rfl, show (0 : UInt64) = UInt64.ofNat (8 * 0) from rfl,
    this 7 (by norm_num), this 6 (by norm_num), this 5 (by norm_num), this 4 (by norm_num), this 3 (by norm_num),
    this 2 (by norm_num), this 1 (by norm_num), this 0 (by norm_num)]
  rfl

/-- **`be32` is `Hash.be 4`.** -/
theorem be32_eq (x : ℕ) : bs (be32 x.toUInt32) = Hash.be 4 x := by
  rw [be32_bs]
  have := fun j hj => byte32 x j hj
  rw [show (24 : UInt32) = UInt32.ofNat (8 * 3) from rfl, show (16 : UInt32) = UInt32.ofNat (8 * 2) from rfl,
    show (8 : UInt32) = UInt32.ofNat (8 * 1) from rfl, show (0 : UInt32) = UInt32.ofNat (8 * 0) from rfl,
    this 3 (by norm_num), this 2 (by norm_num), this 1 (by norm_num), this 0 (by norm_num)]
  rfl

theorem be_getElem (k n i : ℕ) (hi : i < (Hash.be k n).length) :
    (Hash.be k n)[i] = (n / 256 ^ (k - 1 - i) % 256).toUInt8 := by
  simp [Hash.be]

theorem be_length (k n : ℕ) : (Hash.be k n).length = k := by simp [Hash.be]

theorem get!_eq (b : ByteArray) (i : ℕ) : b.get! i = b.data.toList.getD i default := by
  simp [ByteArray.get!, List.getD_eq_getElem?_getD, Array.getElem?_toList, getElem!_def]
  cases b.data[i]? <;> rfl

theorem be64_get (n i : ℕ) (hi : i < 8) : (be64 n.toUInt64).get! i = (n / 256 ^ (7 - i) % 256).toUInt8 := by
  rw [get!_eq, show (be64 n.toUInt64).data.toList = Hash.be 8 n from be64_eq n, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by rw [be_length]; exact hi), Option.getD_some, be_getElem]

/-- **`FV3.uint` is `Hash.uint`** below `2^64`: both are the big-endian bytes from the first nonzero one, one byte for
0. -/
theorem uint_eq (n : ℕ) (hn : n < 2 ^ 64) : bs (FV3.uint n) = Hash.uint n := by
  have hsz : Nat.size n ≤ 64 := Nat.size_le.mpr hn
  set m := max 1 ((Nat.size n + 7) / 8) with hm
  have hm1 : 1 ≤ m := le_max_left _ _
  have hm8 : m ≤ 8 := max_le (by norm_num) (by omega)
  have hlt : n < 256 ^ m := by
    have h : Nat.size n ≤ 8 * m := by omega
    have := Nat.size_le.mp h
    rwa [pow_mul, show (2 : ℕ) ^ 8 = 256 by norm_num] at this
  have hge : 2 ≤ m → 256 ^ (m - 1) ≤ n := by
    intro h2
    have hs : 8 * (m - 1) < Nat.size n := by omega
    have := Nat.lt_size.mp hs
    rwa [pow_mul, show (2 : ℕ) ^ 8 = 256 by norm_num] at this
  have hzero : ∀ i < 8 - m, (be64 n.toUInt64).get! i = 0 := by
    intro i hi
    rw [be64_get n i (by omega)]
    have : n / 256 ^ (7 - i) = 0 :=
      Nat.div_eq_of_lt (lt_of_lt_of_le hlt (Nat.pow_le_pow_right (by norm_num) (by omega)))
    rw [this]; rfl
  have hz : ((List.range 8).find? (fun i => (be64 n.toUInt64).get! i != 0)).getD 7 = 8 - m := by
    by_cases h0 : n = 0
    · subst h0
      have : m = 1 := by simp [hm]
      rw [this]
      have hnone : (List.range 8).find? (fun i => (be64 (0 : ℕ).toUInt64).get! i != 0) = none :=
        List.find?_range_eq_none.mpr fun i hi => by rw [be64_get 0 i hi]; simp
      rw [hnone]; rfl
    · have hnz : (be64 n.toUInt64).get! (8 - m) ≠ 0 := by
        rw [be64_get n _ (by omega), show 7 - (8 - m) = m - 1 by omega]
        intro h
        have h' := congrArg UInt8.toNat h
        rw [UInt8.toNat_ofNat'] at h'
        simp only [UInt8.toNat_zero, Nat.mod_mod] at h'
        by_cases h2 : 2 ≤ m
        · have hlo := hge h2
          have hhi : n / 256 ^ (m - 1) < 256 := by
            rw [Nat.div_lt_iff_lt_mul (by positivity), ← pow_succ', show m - 1 + 1 = m by omega]
            exact hlt
          have hpos : 0 < n / 256 ^ (m - 1) := Nat.div_pos hlo (by positivity)
          omega
        · have : m = 1 := by omega
          rw [this] at h' hlt
          simp at h' hlt
          omega
      have : (List.range 8).find? (fun i => (be64 n.toUInt64).get! i != 0) = some (8 - m) := by
        rw [List.find?_range_eq_some]
        refine ⟨by simpa using hnz, List.mem_range.mpr (by omega), fun j hj => ?_⟩
        simp [hzero j hj]
      rw [this]; rfl
  unfold FV3.uint bs
  simp only
  rw [hz, ByteArray.data_extract, Array.toList_extract, show (be64 n.toUInt64).data.toList = Hash.be 8 n from be64_eq n]
  apply List.ext_getElem
  · simp [Hash.uint, be_length, ← hm]; omega
  · intro i h1 h2
    simp only [List.extract, List.getElem_take, List.getElem_drop, be_getElem]
    simp only [Hash.uint, ← hm, be_getElem]
    have hi : i < m := by simpa [Hash.uint, be_length, ← hm] using h2
    rw [show 8 - 1 - (8 - m + i) = m - 1 - i by omega]

/-! ## Frames -/

theorem foldl_step (parts : List ByteArray) : ∀ acc : ByteArray,
    bs (parts.foldl Registered.step acc) = bs acc ++ (parts.map bs).flatMap fun p => Hash.be 8 p.length ++ p := by
  induction parts with
  | nil => intro acc; simp
  | cons p ps ih =>
    intro acc
    rw [List.foldl_cons, ih, Registered.step, bs_append, bs_append, be64_eq, List.map_cons, List.flatMap_cons,
      bs_length]
    simp

theorem FRAME_eq : bs FV3.FRAME = Hash.Tree.FRAME := by
  rw [FV3.FRAME, bs_push]; rfl

/-- **C-Flock's frame is the spec's.** -/
theorem frame_eq (tag : String) (parts : List ByteArray) :
    bs (Registered.frame tag parts) = Hash.Tree.frame tag (parts.map bs) := by
  rw [Registered.frame, foldl_step, bs_append, bs_append, FRAME_eq, be32_eq, Hash.Tree.frame]
  simp [Hash.utf8, bs]

theorem depth_eq (n : ℕ) : Hash.Tree.depth n = log2ceil n := rfl

/-! ## SHA-512 digests -/

/-- A 64-byte string as a spec digest, read as `Verity.sha512` reads SHA-512's output. -/
def digOf (b : ByteArray) : Hash.Digest := fun i => b.data.getD i.val 0

theorem digOf_bytes {b : ByteArray} (h : b.size = 64) : (digOf b).bytes = bs b := by
  apply List.ext_getElem
  · rw [Hash.Digest.bytes, List.length_ofFn, bs_length, h]
  · intro i h1 h2
    rw [Hash.Digest.bytes, List.length_ofFn] at h1
    simp only [Hash.Digest.bytes, List.getElem_ofFn]
    show b.data.getD i 0 = (bs b)[i]
    have hi : i < b.data.size := by rw [ByteArray.size_data, h]; exact h1
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
    simp [bs]

theorem sha512_bytes (m : Hash.Msg) : (Verity.sha512 m).bytes = bs (Sha512.hash (bytes m)) :=
  digOf_bytes (hash_size _)

theorem sha512Msg_eq (m : Claims.Msg) : sha512Msg m = (Verity.sha512 m).bytes := (sha512_bytes m).symm

theorem digest_inj {x y : Hash.Digest} (h : x.bytes = y.bytes) : x = y := List.ofFn_injective h

/-! ## The tree -/

/-- **C-Flock's node is the spec's node digest**, at a domain whose id is the port's. -/
theorem node_eq {dom l r : ByteArray} {D : Hash.Tree.Domain} {d i : ℕ} {x y : Hash.Digest}
    (hid : D.id = bs dom) (hd : d < 2 ^ 64) (hi : i < 2 ^ 64) (hl : bs l = x.bytes) (hr : bs r = y.bytes) :
    bs (Registered.node dom d i l r) = (Verity.sha512 (Hash.Tree.nodeMsg D d i x y)).bytes := by
  have e : bs (Registered.nodeIn dom d i l r) = Hash.Tree.nodeMsg D d i x y := by
    rw [Registered.nodeIn, frame_eq, Hash.Tree.nodeMsg]
    simp only [List.map, uint_eq d hd, uint_eq i hi, hid, hl, hr]
  rw [Registered.node_eq, sha512_bytes, ← e, bytes_bs]

/-- **C-Flock's frame leaf is the spec's leaf digest** at the rank's own position. -/
theorem leaf_eq {dom payload : ByteArray} {D : Hash.Tree.Domain} {pos : ℕ} (schema : String)
    (hid : D.id = bs dom) (hp : D.position pos = pos) (hpos : pos < 2 ^ 64) :
    bs (Registered.leafAs schema dom pos payload) =
      (Verity.sha512 (Hash.Tree.leafMsg D pos schema (bs payload))).bytes := by
  have e : bs (Registered.leafInAs schema dom pos payload) = Hash.Tree.leafMsg D pos schema (bs payload) := by
    rw [Registered.leafInAs, frame_eq, Hash.Tree.leafMsg, hp]
    simp only [List.map, uint_eq pos hpos, hid]
    rfl
  rw [Registered.leafAs_eq, sha512_bytes, ← e, bytes_bs]

/-- **C-Flock's climb is the spec's**, from a digest whose bytes are the code's, through 64-byte siblings. -/
theorem climbL_eq {dom : ByteArray} {D : Hash.Tree.Domain} (hid : D.id = bs dom) :
    ∀ (ss : List ByteArray) (cur : ByteArray) (x : Hash.Digest) (idx d : ℕ), (∀ s ∈ ss, s.size = 64) →
      bs cur = x.bytes → idx < 2 ^ 64 → d + ss.length < 2 ^ 64 →
        bs (Registered.climbL dom cur idx d ss) = (Hash.Tree.climb Verity.sha512 D d idx x (ss.map digOf)).2.bytes := by
  intro ss
  induction ss with
  | nil => intro cur x idx d _ hc _ _; exact hc
  | cons s rest ih =>
    intro cur x idx d hs hc hi hd
    have hs0 : bs s = (digOf s).bytes := (digOf_bytes (hs s (by simp))).symm
    have hd' : d < 2 ^ 64 := by simp at hd; omega
    have hi' : idx / 2 < 2 ^ 64 := lt_of_le_of_lt (Nat.div_le_self _ _) hi
    simp only [Registered.climbL, List.map_cons, Hash.Tree.climb]
    apply ih _ _ _ _ (fun t ht => hs t (by simp [ht])) _ hi' (by simp at hd ⊢; omega)
    split_ifs
    · exact node_eq hid hd' hi' hc hs0
    · exact node_eq hid hd' hi' hs0 hc

theorem log2ceil_le {n : ℕ} (hn : n < 2 ^ 64) : log2ceil n ≤ 64 := by
  unfold log2ceil
  split_ifs with h
  · omega
  · have : Nat.log2 (n - 1) < 64 := (Nat.log2_lt (by omega)).mpr (by omega)
    omega

/-- **C-Flock's registered opening is the spec's frame-v3 opening**, at a domain whose id is the port's, whose
positions are the ranks and whose count is the port's leaves. -/
theorem opensLeaf_accepts (reg : Registered.Port) (D : Hash.Tree.Domain) (hid : D.id = bs reg.domain)
    (hp : ∀ r, D.position r = r) (hcount : D.count = reg.leaves) (hlt : reg.leaves < 2 ^ 64) (root : Hash.Digest)
    (hroot : bs reg.root = root.bytes) (payload path : ByteArray) (pos : ℕ)
    (h : Registered.opensLeaf reg payload pos path = true) :
    Hash.Tree.accepts Verity.sha512 D root pos (Verity.sha512 (Hash.Tree.leafMsg D pos reg.schema (bs payload)))
      ((Registered.sibs path (log2ceil reg.leaves)).map digOf) := by
  simp only [Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hpos, hsize⟩, hclimb⟩ := h
  refine ⟨hcount ▸ hpos, ?_, ?_⟩
  · simp [Registered.sibs, hcount, depth_eq]
  · apply digest_inj
    rw [← hroot, ← hclimb, Registered.climb]
    refine (climbL_eq hid _ _ _ _ _ (Registered.sibs_size hsize) (leaf_eq _ hid (hp pos) (by omega)) (by omega)
      ?_).symm
    have := log2ceil_le hlt
    simp [Registered.sibs]
    omega

/-! ## The row's message and the hm96 leaf -/

theorem bitsVal_eq : ∀ b : List Bool, HmRow.bitsVal b = Hash.RowSeg.bitsVal b
  | [] => rfl
  | x :: b => by rw [HmRow.bitsVal, Hash.RowSeg.bitsVal, bitsVal_eq b]

theorem packBits_eq (b : List Bool) : HmRow.packBits b = Hash.RowSeg.packBits b := by
  simp only [HmRow.packBits, Hash.RowSeg.packBits, bitsVal_eq]

theorem pad_eq (b : ByteArray) (n : ℕ) : bs (Hm96.pad b n) = Hash.padTo n (bs b) := by
  rw [Hm96.pad, bs_append, Hash.padTo, bs_length]
  congr 1
  simp only [bs, Array.toList_replicate]
  congr 1
  omega

theorem rowPrefixV2_eq (l : ℕ) : bs (HmRow.rowPrefixV2 l) = Hash.RowV2.rowPrefix l := by
  rw [HmRow.rowPrefixV2, pad_eq, bs_append, bs_push, Hash.RowV2.rowPrefix]
  congr 2
  simp [bs, Hash.be, Array.toList_map, Array.toList_range]

/-- **`sha512/row/v2`'s bytes are `RowV2.msg`.** -/
theorem enc_eq (b : List Bool) : FlockLevel3.RowV2.enc b = Hash.RowV2.msg b := by
  rw [FlockLevel3.RowV2.enc, show (HmRow.rowPrefixV2 b.length).data.toList = bs (HmRow.rowPrefixV2 b.length) from rfl,
    rowPrefixV2_eq, packBits_eq, Hash.RowV2.msg]

theorem rowPrefixV1_eq (w : ℕ) : bs (HmRow.rowPrefix w) = Hash.RowV1.rowPrefix w := by
  rw [HmRow.rowPrefix, pad_eq, bs_append, bs_append, bs_push, be32_eq, Hash.RowV1.rowPrefix]
  rfl

/-- **`sha512/row/v1`'s bytes are `RowV1.msg`.** -/
theorem encV1_eq (b : List Bool) :
    (HmRow.rowPrefix (b.length / 16)).data.toList ++ HmRow.packBits b = Hash.RowV1.msg b := by
  rw [show (HmRow.rowPrefix (b.length / 16)).data.toList = bs (HmRow.rowPrefix (b.length / 16)) from rfl,
    rowPrefixV1_eq, packBits_eq, Hash.RowV1.msg]

theorem be64Nat_eq (x : ℕ) : bs (HmRow.be64 x) = Hash.be 8 x := by
  simp [HmRow.be64, bs, Hash.be, Array.toList_map, Array.toList_range]

theorem rowPrefixSeg_eq (l : ℕ) : bs (HmRow.rowPrefixSeg l) = Hash.RowSeg.outerPrefix l := by
  rw [HmRow.rowPrefixSeg, pad_eq, bs_append, bs_append, bs_append, bs_push, be64Nat_eq, be64Nat_eq,
    show (1024 : UInt32) = (1024 : ℕ).toUInt32 from rfl, be32_eq, Hash.RowSeg.outerPrefix]
  rfl

theorem segmentPrefix_eq (j s m : ℕ) : bs (HmRow.segmentPrefix j s m) = Hash.RowSeg.segmentPrefix j s m := by
  rw [HmRow.segmentPrefix, pad_eq, bs_append, bs_append, bs_append, bs_push, be64Nat_eq, be64Nat_eq, be64Nat_eq,
    Hash.RowSeg.segmentPrefix]
  rfl

theorem chunk_eq (b : List Bool) {a c i : ℕ} (hi : i < c) :
    ((((b.drop (8 * a)).take (8 * c)).drop (8 * i)).take 8) = (b.drop (8 * (a + i))).take 8 := by
  apply List.ext_getElem?
  intro k
  simp only [List.getElem?_take, List.getElem?_drop]
  split_ifs <;> first | rfl | omega | (congr 1; omega)

/-- A row's bytes from byte `a`, `c` of them, are its bits' bytes from bit `8 a`, `8 c` of them. -/
theorem packBits_drop_take (b : List Bool) (a c : ℕ) :
    Hash.RowSeg.packBits ((b.drop (8 * a)).take (8 * c)) = ((Hash.RowSeg.packBits b).drop a).take c := by
  apply List.ext_getElem
  · simp [Hash.RowSeg.packBits]; omega
  · intro i h1 h2
    simp only [Hash.RowSeg.packBits, List.length_map, List.length_range, List.length_take, List.length_drop] at h1 h2
    simp only [Hash.RowSeg.packBits, List.getElem_map, List.getElem_range, List.getElem_take, List.getElem_drop]
    rw [chunk_eq b (by omega : i < c)]

/-- **A segment's bytes are its bits' bytes**: C-Flock's `segBytes` of the row's bytes is the spec's segment, packed. -/
theorem segBytes_packBits (b : List Bool) (j : ℕ) :
    FlockSoundness.Binding.RowSeg.segBytes b.length (Hash.RowSeg.packBits b) j =
      Hash.RowSeg.packBits (Hash.RowSeg.segment b j) := by
  rw [Hash.RowSeg.segment, Hash.RowSeg.SEG_BITS, show 8192 * j = 8 * (1024 * j) by ring,
    show (8192 : ℕ) = 8 * 1024 from rfl, packBits_drop_take, FlockSoundness.Binding.RowSeg.segBytes, List.take_eq_take_iff]
  simp only [FlockSoundness.Binding.RowSeg.segBits, HmRow.ROW_SEG_BITS, Hash.RowSeg.packBits, List.length_drop,
    List.length_map, List.length_range]
  omega

theorem segMsg_eq (b : List Bool) (j : ℕ) :
    FlockSoundness.Binding.RowSeg.segMsg b.length (Hash.RowSeg.packBits b) j = Hash.RowSeg.segmentMsg b j := by
  have hl : FlockSoundness.Binding.RowSeg.segBits b.length j = (Hash.RowSeg.segment b j).length := by
    simp [FlockSoundness.Binding.RowSeg.segBits, Hash.RowSeg.segment, Hash.RowSeg.SEG_BITS, HmRow.ROW_SEG_BITS,
      Nat.min_comm]
  rw [FlockSoundness.Binding.RowSeg.segMsg, Hash.RowSeg.segmentMsg, ← hl, segBytes_packBits,
    show (HmRow.segmentPrefix j (FlockSoundness.Binding.RowSeg.count b.length)
        (FlockSoundness.Binding.RowSeg.segBits b.length j)).data.toList =
      bs (HmRow.segmentPrefix j (FlockSoundness.Binding.RowSeg.count b.length)
        (FlockSoundness.Binding.RowSeg.segBits b.length j)) from rfl, segmentPrefix_eq]
  rfl

/-- **`sha512/row-seg/v1`'s outer bytes are `RowSeg.outerMsg` at SHA-512.** -/
theorem encSeg_eq (b : List Bool) :
    FlockSoundness.Binding.RowSeg.enc b.length (HmRow.packBits b) = Hash.RowSeg.outerMsg Verity.sha512 b := by
  rw [packBits_eq, FlockSoundness.Binding.RowSeg.enc, FlockSoundness.Binding.RowSeg.digests, Hash.RowSeg.outerMsg,
    Hash.RowSeg.segmentMsgs, List.flatMap_map,
    show (HmRow.rowPrefixSeg b.length).data.toList = bs (HmRow.rowPrefixSeg b.length) from rfl, rowPrefixSeg_eq,
    show FlockSoundness.Binding.RowSeg.count b.length = Hash.RowSeg.segCount b.length from rfl]
  refine congrArg (Hash.RowSeg.outerPrefix b.length ++ ·) (List.flatMap_congr fun j _ => ?_)
  rw [segMsg_eq, sha512_bytes]
  simp only [FlockSoundness.Binding.digBytes, FlockSoundness.Refine.dig_eq, FlockSoundness.Refine.H512_val, bs, bytes]

/-- The default hm96 key's bytes. -/
def key512 : Hash.Msg := bs Hm96.Default512.key

theorem saltPrefix_eq : FlockSoundness.Refine.K512.saltPrefix = Hash.Hm96.saltPrefix := by
  show bs Hm96.Default512.saltPrefix = _
  rw [Hm96.Default512.saltPrefix, pad_eq, Hash.Hm96.saltPrefix]
  rfl

theorem leafPrefix_eq : FlockSoundness.Refine.K512.leafPrefix = Hash.Hm96.leafPrefix Verity.sha512 key512 := by
  show bs Hm96.Default512.leafPrefix = _
  rw [Hm96.Default512.leafPrefix, Hm96.leafPrefix, pad_eq, bs_append, Hash.Hm96.leafPrefix, sha512_bytes, key512,
    bytes_bs]
  rfl

theorem testBit_eq (n i : ℕ) : n.testBit i = (n / 2 ^ i % 2 == 1) := by
  rw [Nat.testBit_eq_decide_div_mod_eq]
  cases h : n / 2 ^ i % 2 == 1 <;> simp_all

theorem bitL_eq (l : List UInt8) (t : ℕ) : FlockSoundness.Discharge.Hm.bitL l t = Hash.Hm96.bit l t := by
  rw [FlockSoundness.Discharge.Hm.bitL, Hash.Hm96.bit, testBit_eq]

theorem bit_eq (b : ByteArray) (t : ℕ) : Hm96.bit b t = Hash.Hm96.bit (bs b) t := by
  rw [Hm96.bit, Hash.Hm96.bit, get!_eq, Nat.shiftRight_eq_div_pow]
  rfl

theorem foldl_filter (p f : ℕ → Bool) : ∀ (l : List ℕ) (a : Bool),
    (l.filter p).foldl (fun acc j => xor acc (f j)) a = l.foldl (fun acc j => acc ^^ (f j && p j)) a
  | [], _ => rfl
  | x :: l, a => by
    cases hp : p x
    · rw [List.filter_cons_of_neg (by simp [hp]), List.foldl_cons, foldl_filter p f l]
      simp [hp]
    · rw [List.filter_cons_of_pos hp, List.foldl_cons, List.foldl_cons, foldl_filter p f l]
      simp [hp]

theorem maskBit_eq (y : List UInt8) {t : ℕ} (ht : t < 512) :
    Hash.Hm96.maskBit key512 y t = FlockSoundness.Discharge.Hm.bitL (FlockSoundness.Refine.K512.mask y) t := by
  rw [FlockSoundness.Refine.K512_mask, FlockSoundness.Discharge.Hm.bitL_mask y ht, Hash.Hm96.maskBit,
    FlockSoundness.Discharge.Hm.maskV, foldl_filter]
  simp only [bit_eq, bitL_eq, Hash.Hm96.SALT_BYTES, key512]

set_option maxRecDepth 100000 in
theorem byte_bits : ∀ n < 256, (List.range 8).foldl (fun acc s => acc + if n.testBit s then 2 ^ s else 0) 0 = n := by
  decide

/-- **The default key's mask is `Hash.Hm96.mask`**, bit for bit through `bitL_mask`. -/
theorem mask_eq (y : List UInt8) : FlockSoundness.Refine.K512.mask y = Hash.Hm96.mask key512 y := by
  have hlen := FlockSoundness.Refine.K512.mask_len y
  apply List.ext_getElem
  · rw [hlen, Hash.Hm96.mask, List.length_map, List.length_range]; rfl
  · intro p h1 h2
    have h64 : p < 64 := by rw [hlen] at h1; exact h1
    simp only [Hash.Hm96.mask, List.getElem_map, List.getElem_range]
    set v := (FlockSoundness.Refine.K512.mask y)[p] with hv
    have hbit : ∀ s < 8, Hash.Hm96.maskBit key512 y (8 * p + s) = v.toNat.testBit s := by
      intro s hs
      rw [maskBit_eq y (by omega), FlockSoundness.Discharge.Hm.bitL, show (8 * p + s) / 8 = p by omega,
        show (8 * p + s) % 8 = s by omega, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1, Option.getD_some]
    have hfold : (List.range 8).foldl
        (fun acc s => acc + if Hash.Hm96.maskBit key512 y (8 * p + s) then 2 ^ s else 0) 0 =
        (List.range 8).foldl (fun acc s => acc + if v.toNat.testBit s then 2 ^ s else 0) 0 :=
      List.foldl_ext _ _ _ fun acc s hs => by rw [hbit s (List.mem_range.mp hs)]
    rw [hfold, byte_bits _ v.toNat_lt]
    simp

theorem xor_ofFn (x : Hash.Digest) (l : List UInt8) (hl : l.length = 64) :
    FlockSoundness.Merkle.xorBytes x.bytes l = List.ofFn fun i : Fin 64 => x i ^^^ l.getD i 0 := by
  apply List.ext_getElem
  · simp [FlockSoundness.Merkle.xorBytes, Hash.Digest.bytes, hl]
  · intro i h1 h2
    simp only [FlockSoundness.Merkle.xorBytes, List.getElem_zipWith, Hash.Digest.bytes, List.getElem_ofFn]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by simp at h2; omega), Option.getD_some]

/-- **The hm96 outer message at SHA-512 is the spec's**: `leafPrefix ‖ commitString` at the default key. -/
theorem hm96Outer_eq (row salt : Opening Bool → Claims.Msg) (o : Opening Bool) :
    hm96Outer sha512Msg row salt o =
      Hash.Hm96.leafPrefix Verity.sha512 key512 ++ Hash.Hm96.commitString Verity.sha512 key512 (Verity.sha512 (row o))
        (salt o) := by
  rw [hm96Outer, leafPrefix_eq, Hash.Hm96.commitString, sha512Msg_eq, sha512Msg_eq, saltPrefix_eq, mask_eq,
    xor_ofFn _ _ (by rw [← mask_eq]; exact FlockSoundness.Refine.K512.mask_len _)]

/-- **`hm96Leaf` at SHA-512, under a 192-byte salt, is the spec's hm96 leaf over the row message's SHA-512 digest.** -/
theorem hm96Leaf_spec (row salt : Opening Bool → Claims.Msg) (o : Opening Bool)
    (hs : (salt o).length = Hash.Hm96.SALT_BYTES) :
    ((hm96Leaf row salt o).eval sha512Msg).1 =
      some (Hash.Hm96.leaf Verity.sha512 key512 (Verity.sha512 (row o)) (salt o)).bytes := by
  have hlen : ∀ m, (sha512Msg m).length = 64 := fun m => by
    rw [sha512Msg_eq, Hash.Digest.bytes, List.length_ofFn]
  rw [hm96Leaf_fst, if_pos ⟨hlen _, hlen _, hs⟩, hm96Outer_eq, sha512Msg_eq, Hash.Hm96.leaf]

/-! ## The composition -/

/-- **A row C-Flock's registered check opens is one the spec's verifier accepts**: at a port whose schema is the
verifier's format's, a verifier of the default key whose domain has the port's id (`hid`), the ranks as positions and
the port's leaves as count, the row `row` (a length of the format's rows) under the salt `salt` (192 bytes), opened at
`pos` through `path` against the port's root under the hm96 leaf over the row's digest in that format, is
`Hash.RowLeaf.accepts` at SHA-512, with the path's siblings as digests. -/
theorem rowLeaf_accepts (reg : Registered.Port) (V : Hash.RowLeaf.Verifier)
    (hschema : reg.schema = V.format.schema) (hkey : V.key = key512)
    (hid : V.domain.id = bs reg.domain) (hp : ∀ r, V.domain.position r = r)
    (hcount : V.domain.count = reg.leaves) (hlt : reg.leaves < 2 ^ 64) (root : Hash.Digest)
    (hroot : bs reg.root = root.bytes) (row : List Bool) (hrow : row.length < 2 ^ 64)
    (hfits : V.format.fits row.length) (salt : Claims.Msg)
    (hsalt : salt.length = Hash.Hm96.SALT_BYTES) (pos : ℕ) (path : ByteArray)
    (h : Registered.opensLeaf reg
      (bytes (Hash.Hm96.leaf Verity.sha512 key512 (V.format.digest Verity.sha512 row) salt).bytes) pos path = true) :
    Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 V root pos
      ⟨row, salt, (Registered.sibs path (log2ceil reg.leaves)).map digOf⟩ := by
  refine ⟨hrow, hfits, hsalt, ?_⟩
  have ha := opensLeaf_accepts reg V.domain hid hp hcount hlt root hroot _ path pos h
  rw [bs_bytes, hschema] at ha
  unfold Hash.RowLeaf.leafMsg
  rw [hkey]
  exact ha

/-! ## An opening `commitDep` accepts -/

/-! ## The port's format -/

/-- **A port's format**: `sha512/row-seg/v1` at a segmented port, `sha512/row/v2` at a bit row, `sha512/row/v1`
otherwise, as C-Flock frames the port's rows (`Registered.portSchema`) and hashes them (`rowMsg`). -/
def formatOf (p : Flock.Port) : Hash.RowLeaf.Format :=
  if p.segs.isSome then .seg else if p.v2 then .v2 else .v1

theorem portSchema_eq (p : Flock.Port) : Registered.portSchema p = (formatOf p).schema := by
  unfold Registered.portSchema formatOf
  split_ifs <;> rfl

/-- **A row's message in its port's format hashes to the spec's digest in that format**, at SHA-512. -/
theorem rowMsg_sha512 (p : Flock.Port) (b : List Bool) :
    Verity.sha512 (rowMsg p b) = (formatOf p).digest Verity.sha512 b := by
  unfold rowMsg formatOf
  split_ifs
  · show _ = Hash.RowSeg.digest _ _
    rw [Hash.RowSeg.digest, encSeg_eq]
  · show _ = Hash.RowV2.digest _ _
    rw [Hash.RowV2.digest, enc_eq]
  · show _ = Hash.RowV1.digest _ _
    rw [Hash.RowV1.digest, encV1_eq]

theorem rowDep_port {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt} {c : Flock.Circuit}
    {o : Opening Bool} {r : RowAt} {p : Flock.Port} (hr : rowOfPos C rows o.pos = some r)
    (hp : c.ports[r.port]? = some p) (place : ℕ → (w : ℕ) × Fin w) (bits : Claims.Msg → ℕ → Bool) :
    rowDep C rows c place bits o = rowMsg p (SamplingDeploy.rowBits place bits o) := by
  unfold rowDep
  rw [show portOfPos C rows c o.pos = some p by simp [portOfPos, hr, hp]]

/-- Every length of whole u16 words is one of each format's row lengths. -/
theorem fits_of_words {F : Hash.RowLeaf.Format} {l : ℕ} (h : l % 16 = 0) : F.fits l := by
  cases F
  · trivial
  · trivial
  · exact h

/-- **What `commitDep` checked at SHA-512**: the opening's salt is `Hash.Hm96.SALT_BYTES` bytes, and the position's row and
root, the registered port whose root it is, and that port's check of the hm96 leaf over the row message's digest and the
opening's salt, through its path. -/
theorem commitDep_opens {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt} {c : Flock.Circuit}
    {nl : ℕ} {roots : List Claims.Msg} {o : Opening Bool}
    (h : (commitDep C rows c nl roots).Opens sha512Msg roots o = true) :
    (saltOf o).length = Hash.Hm96.SALT_BYTES ∧ ∃ r rt reg, rowOfPos C rows o.pos = some r ∧ roots[r.port]? = some rt ∧
      (portsOf c nl roots).find? (fun p => decide (bs p.root = rt)) = some reg ∧
      Registered.opensLeaf reg (bytes (Hash.Hm96.leaf Verity.sha512 key512
        (Verity.sha512 (rowDep C rows c (placeOf C rows) (bitsOf (log2ceil nl)) o)) (saltOf o)).bytes) r.leaf
        (bytes (pathOf (log2ceil nl) o)) = true := by
  unfold Commitment.Opens commitDep pathCommit at h
  dsimp only at h
  cases hat : atOf C rows roots o.pos with
  | none => rw [hat] at h; simp [OC.eval] at h
  | some ri =>
    rw [hat] at h
    dsimp only at h
    rw [ocBind_eval] at h
    by_cases hs : (saltOf o).length = Hash.Hm96.SALT_BYTES
    · rw [hm96Leaf_spec _ _ _ hs] at h
      dsimp only at h
      obtain ⟨reg, hf, hopen⟩ := (regPath_sha512 _ _ _ _ _).mp h
      obtain ⟨r, hr, hm⟩ := Option.bind_eq_some_iff.mp hat
      obtain ⟨rt, hrt, rfl⟩ := Option.map_eq_some_iff.mp hm
      exact ⟨hs, r, rt, reg, hr, hrt, hf, hopen⟩
    · rw [hm96Leaf_sha512_salt _ _ _ hs] at h
      simp [OC.eval] at h

theorem portsOf_mem {c : Flock.Circuit} {nl : ℕ} {roots : List Claims.Msg} {reg : Registered.Port}
    (h : reg ∈ portsOf c nl roots) :
    ∃ q < c.ports.size, reg.leaves = nl ∧ reg.domain = domOf c q ∧ reg.schema = schemaOf c q := by
  unfold portsOf at h
  obtain ⟨q, hq, rfl⟩ := Array.mem_map.mp h
  exact ⟨q, Array.mem_range.mp hq, rfl, rfl, rfl⟩

theorem climbL_size (dom : ByteArray) : ∀ (ss : List ByteArray) (cur : ByteArray) (idx d : ℕ), cur.size = 64 →
    (Registered.climbL dom cur idx d ss).size = 64
  | [], _, _, _, h => h
  | _ :: ss, _, _, _, _ => by
    simp only [Registered.climbL]
    apply climbL_size dom ss
    split_ifs <;> rw [Registered.node_eq] <;> exact hash_size _

/-- A registered root an opening reached is a 64-byte digest. -/
theorem root_size {reg : Registered.Port} {payload path : ByteArray} {pos : ℕ}
    (h : Registered.opensLeaf reg payload pos path = true) : reg.root.size = 64 := by
  simp only [Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at h
  rw [← h.2, Registered.climb]
  exact climbL_size _ _ _ _ _ (by rw [Registered.leafAs_eq]; exact hash_size _)

/-- The spec's verifier of a registered port in format `F`: the port's domain as id, the ranks as positions and the
port's leaves as count, the default key. -/
def verifierOf (reg : Registered.Port) (F : Hash.RowLeaf.Format) : Hash.RowLeaf.Verifier :=
  ⟨⟨bs reg.domain, reg.leaves, fun r => r⟩, key512, F⟩

theorem rowBits_place (place : ℕ → (w : ℕ) × Fin w) (bits : Claims.Msg → ℕ → Bool) (o : Opening Bool) :
    (SamplingDeploy.rowBits place bits o)[((place o.pos).2 : ℕ)]? = some o.val := by
  simp [SamplingDeploy.rowBits]

/-- **An opening `commitDep` accepts at SHA-512 is a spec row-leaf opening**: the position's row is opened at its leaf
index under the registered port whose root is the position's. When that port's schema is the format of the row's port
`p` (`formatOf`), and the row's width is a length of that format's rows, the spec's verifier of the port in that format
(`verifierOf`, the port's domain as id) accepts it at the root's digest. -/
theorem commitDep_accepts {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt} {c : Flock.Circuit}
    {nl : ℕ} {roots : List Claims.Msg} (hlt : nl < 2 ^ 64) {o : Opening Bool}
    (h : (commitDep C rows c nl roots).Opens sha512Msg roots o = true) :
    ∃ r rt reg, rowOfPos C rows o.pos = some r ∧ roots[r.port]? = some rt ∧
      (portsOf c nl roots).find? (fun p => decide (bs p.root = rt)) = some reg ∧
      ∀ p, c.ports[r.port]? = some p → reg.schema = (formatOf p).schema → (placeOf C rows o.pos).1 < 2 ^ 64 →
        (formatOf p).fits (placeOf C rows o.pos).1 →
        Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 (verifierOf reg (formatOf p)) (digOf reg.root) r.leaf
          ⟨SamplingDeploy.rowBits (placeOf C rows) (bitsOf (log2ceil nl)) o, saltOf o,
            (Registered.sibs (bytes (pathOf (log2ceil nl) o)) (log2ceil nl)).map digOf⟩ := by
  obtain ⟨hs, r, rt, reg, hr, hrt, hf, hopen⟩ := commitDep_opens h
  refine ⟨r, rt, reg, hr, hrt, hf, fun p hp hschema hw hfit => ?_⟩
  obtain ⟨q, hq, hleaves, -, -⟩ := portsOf_mem (Array.mem_of_find?_eq_some hf)
  rw [rowDep_port hr hp, rowMsg_sha512] at hopen
  have hlen : (SamplingDeploy.rowBits (placeOf C rows) (bitsOf (log2ceil nl)) o).length = (placeOf C rows o.pos).1 := by
    simp [SamplingDeploy.rowBits]
  have ha := rowLeaf_accepts reg (verifierOf reg (formatOf p)) hschema rfl rfl (fun _ => rfl) rfl (hleaves ▸ hlt)
    (digOf reg.root) (digOf_bytes (root_size hopen)).symm _ (by rw [hlen]; exact hw) (by rw [hlen]; exact hfit)
    (saltOf o) hs r.leaf _ hopen
  rwa [hleaves] at ha

theorem find?_map_range {α : Type} {m q : ℕ} {f : ℕ → α} {P : α → Bool} {a : α}
    (h : ((Array.range m).map f).find? P = some a) (hq : q < m) (hP : P (f q) = true)
    (hd : ∀ i < q, P (f i) = true → False) : a = f q := by
  obtain ⟨hPa, i, hi, rfl, hj⟩ := Array.find?_eq_some_iff_getElem.mp h
  simp only [Array.size_map, Array.size_range] at hi
  simp only [Array.getElem_map, Array.getElem_range] at hPa hj ⊢
  rcases Nat.lt_trichotomy i q with hlt | rfl | hgt
  · exact (hd i hlt hPa).elim
  · rfl
  · have := hj q hgt
    simp [hP] at this

/-- **The port a position's root finds is its row's own**, at a circuit that names each port once and whose roots
expose no collision (`PortRoots.rootsColl`): the registered port `find?` returns has the row's port's domain and root. -/
theorem find_port {c : Flock.Circuit} {pub : Flock.Public} {nl q : ℕ} {rt : Claims.Msg} {reg : Registered.Port}
    (hn : Registered.dupName c = none) (hc : PortRoots.rootsColl c pub nl = none)
    (hq : (rootsOf c pub nl)[q]? = some rt)
    (hf : (portsOf c nl (rootsOf c pub nl)).find? (fun p => decide (bs p.root = rt)) = some reg) :
    q < c.ports.size ∧ reg.leaves = nl ∧ reg.domain = domOf c q ∧ reg.root = bytes rt ∧ reg.schema = schemaOf c q := by
  have hqs : q < c.ports.size := by
    have := (List.getElem?_eq_some_iff.mp hq).1
    simpa [rootsOf] using this
  have hrt : (rootsOf c pub nl).getD q [] = rt := by
    rw [List.getD_eq_getElem?_getD, hq, Option.getD_some]
  have hd := PortRoots.rootsDistinct_of_fixes hn (PortRoots.fixes_of_none hc)
  unfold portsOf at hf
  have he := find?_map_range hf hqs (by simp only [bs_bytes, hrt, decide_eq_true_eq])
    (fun i hi hP => hd q i hi hqs (by simp only [bs_bytes, decide_eq_true_eq] at hP; rw [hP, hrt]))
  subst he
  exact ⟨hqs, rfl, rfl, show bytes ((rootsOf c pub nl).getD q []) = bytes rt by rw [hrt], rfl⟩

/-- **An opening `commitDep` accepts at a deployed registration is a spec row-leaf opening at its row's port**: at a
file that names each port once, whose roots expose no collision and with `nl < 2^64` leaves a port, the position's row
`r` is opened at rank `r.leaf` under port `r.port`'s root `rt`, by the spec's verifier of that port in its format
(`formatOf`, its name's bytes as the domain id), with the opened bit at the row's place, when the row's width is a length
of that format's rows. Every component is a function of the position. -/
theorem opens_at {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt} {c : Flock.Circuit}
    {pub : Flock.Public} {nl : ℕ} (hlt : nl < 2 ^ 64)
    (hn : Registered.dupName c = none) (hc : PortRoots.rootsColl c pub nl = none) {o : Opening Bool}
    (h : (commitDep C rows c nl (rootsOf c pub nl)).Opens sha512Msg (rootsOf c pub nl) o = true) :
    ∃ r rt p, rowOfPos C rows o.pos = some r ∧ (rootsOf c pub nl)[r.port]? = some rt ∧ c.ports[r.port]? = some p ∧
      ((placeOf C rows o.pos).1 < 2 ^ 64 → (formatOf p).fits (placeOf C rows o.pos).1 →
        ∃ ro : Hash.RowLeaf.Opening,
          Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 ⟨⟨bs (domOf c r.port), nl, fun r => r⟩, key512, formatOf p⟩
            (digOf (bytes rt)) r.leaf ro ∧ ro.row[((placeOf C rows o.pos).2 : ℕ)]? = some o.val) := by
  obtain ⟨r, rt, reg, hr, hrt, hf, ha⟩ := commitDep_accepts hlt h
  obtain ⟨hq, hleaves, hdom, hroot, hschema⟩ := find_port hn hc hrt hf
  have hp : c.ports[r.port]? = some c.ports[r.port] := Array.getElem?_eq_getElem hq
  refine ⟨r, rt, c.ports[r.port], hr, hrt, hp, fun hw hfit =>
    ⟨⟨SamplingDeploy.rowBits (placeOf C rows) (bitsOf (log2ceil nl)) o, saltOf o,
      (Registered.sibs (bytes (pathOf (log2ceil nl) o)) (log2ceil nl)).map digOf⟩, ?_, rowBits_place _ _ _⟩⟩
  have hs : reg.schema = (formatOf c.ports[r.port]).schema := by
    rw [hschema, ← portSchema_eq]
    simp [schemaOf, hp]
  have := ha _ hp hs hw hfit
  rwa [verifierOf, hleaves, hdom, hroot] at this

/-- **A value the witness opens has an opening that passes**: at its position, of its value. -/
theorem val_opens {H : Claims.Msg → Claims.Msg} {Cm : Commitment Bool} {s : Claims.Sampling.Stmt}
    {w : Claims.Sampling.Wit} {i : ℕ} {b : Bool} (h : w.val H Cm s i = some b) :
    ∃ o : Opening Bool, o.pos = i ∧ o.val = b ∧ Cm.Opens H s.roots o = true := by
  obtain ⟨o, hf, rfl⟩ := Option.map_eq_some_iff.mp h
  have := List.find?_some hf
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  exact ⟨o, this.1, rfl, this.2⟩

/-- **A live session's row is whole words**: the game's row at a position has its bit inside it, and its width is its
port's `16 · words` bits (`Layout.portRowW`). -/
theorem live_row {r : Flock.Bridge.Parsed} {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt r.n r.D r.hU kd K₀ desc) {i : ℕ} {ra : RowAt} (h : rowOfPos A.Cc A.rows i = some ra) :
    ra.place < ra.width ∧ ra.width % 16 = 0 := by
  obtain ⟨g, -, hg⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨x, -, rfl⟩ := Option.map_eq_some_iff.mp hg
  exact ⟨x.2.2.isLt, by simp [FlockSoundness.Discharge.Layout.portRowW]⟩

/-- **`hopen`'s premise at a live session**: a value a witness opens under the session's deployed commitment, at a
statement over its roots (`LiveAt.valE`'s, by `rfl`), was opened by a spec row-leaf opening at its row's port in that
port's format (`formatOf`: `sha512/row-seg/v1`, `sha512/row/v2` or `sha512/row/v1`), with `n < 2^64` units. -/
theorem live_opens {r : Flock.Bridge.Parsed} {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt r.n r.D r.hU kd K₀ desc) (hlt : r.n < 2 ^ 64) {wt : Claims.Sampling.Wit}
    {s : Claims.Sampling.Stmt} (hs : s.roots = A.roots) {i : ℕ} {b : Bool} (h : wt.val sha512Msg A.Cd s i = some b) :
    ∃ ra rt p, rowOfPos A.Cc A.rows i = some ra ∧ A.roots[ra.port]? = some rt ∧ r.D.unroll.ports[ra.port]? = some p ∧
      ((placeOf A.Cc A.rows i).1 < 2 ^ 64 →
        ∃ ro : Hash.RowLeaf.Opening,
          Hash.RowLeaf.accepts Verity.sha512 Verity.sha512
            ⟨⟨bs (domOf r.D.unroll ra.port), r.n, fun k => k⟩, key512, formatOf p⟩
            (digOf (bytes rt)) ra.leaf ro ∧ ro.row[((placeOf A.Cc A.rows i).2 : ℕ)]? = some b) := by
  obtain ⟨o, rfl, rfl, ho⟩ := val_opens h
  rw [hs] at ho
  obtain ⟨ra, rt, p, hr, hrt, hp, ha⟩ := opens_at hlt A.hdup A.hroots ho
  obtain ⟨hpl, hm⟩ := live_row A hr
  have hw1 : (placeOf A.Cc A.rows o.pos).1 = ra.width := by rw [placeOf_eq hr hpl]
  exact ⟨ra, rt, p, hr, hrt, hp, fun hw => ha hw (fits_of_words (by rw [hw1]; exact hm))⟩

/-! ## The constants -/

theorem constsOf_eq_parsed (r : Flock.Bridge.Parsed)
    (g : Fin (FlockSoundness.Discharge.Layout.keyN r.D.unroll r.hU r.n)) :
    constsOf r.D.unroll r.hU r.n g = Verity.parsedConsts (some r) g := by
  simp only [constsOf, Verity.parsedConsts, Fin.ext_iff, FlockSoundness.Discharge.Layout.oneG,
    FlockSoundness.Discharge.Layout.zeroG, Fin.coe_castLE]

/-- **`hconst` at the parse**: where the lowered program places a constant `b` at a wire, the parse's constants
(`FileRead.consts`' `parsedConsts`) hold `b` there, and so does the transcript at any values. -/
theorem hconst_parsed {r : Flock.Bridge.Parsed} {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt r.n r.D r.hU kd K₀ desc) (val : ℕ → Option Bool) (g : Fin A.Cc.N) (b : Bool)
    (h : (progOf A.Cc A.κ).gates[posOf A.Cc g]? = some (.const b)) :
    Verity.parsedConsts (some r) g = some b ∧
      ∀ i : Fin ((circuitOf r.D r.hU r.n).w g), (transcriptOf r.D r.hU r.n A.κ val g).getLsbD i.val = b := by
  obtain ⟨a, ha, hin, hk⟩ := const_pos h
  obtain rfl := posOf_injective ha
  refine ⟨(constsOf_eq_parsed r g).symm.trans hk, fun i => ?_⟩
  have hi : i.val = 0 := by have := i.2; simp [circuitOf_eq, Verity.Circuit.ofAudit] at this; omega
  show (BitVec.ofBool (openedOf A.Cc A.κ val g)).getLsbD i.val = b
  rw [openedOf_const val hin hk, hi]
  simp

end Flock.Bridge.Openings
