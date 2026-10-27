import Proofs.Flock.Bridge.Deployed
import Proofs.Flock.Soundness.Discharge.Hm.Mask
import Properties.Verity.Game

/-!
# C-Flock's row openings are the spec's hm96 row leaves (`Hash.RowLeaf.accepts`) at SHA-512

The openings bridge `SampledDeploy` reads: a `sha512/row/v2` row that C-Flock's registered check opens
(`Flock.Registered.opensLeaf`, under the hm96 leaf `commitDep` computes) is an opening the spec's verifier accepts,
`Hash.RowLeaf.accepts Verity.sha512 Verity.sha512`, at the port's domain, key and format. Each layer is the code's bytes
read as the spec's messages:

* encodings: `be64`/`be32` are `Hash.be 8`/`Hash.be 4` (`be64_eq`, `be32_eq`), and `FV3.uint` is `Hash.uint` below
  `2^64` (`uint_eq`);
* frames: `Registered.frame` is `Hash.Tree.frame` (`frame_eq`), so the code's node and leaf are the spec's `nodeMsg`
  and `leafMsg` digests (`node_eq`, `leaf_eq`), at a domain whose id's bytes are the port's domain (`hid`);
* the tree: `climbL` is `Tree.climb` (`climbL_eq`), `log2ceil` is `Tree.depth` (`depth_eq`), and `opensLeaf` gives
  `Tree.accepts` (`opensLeaf_accepts`);
* the leaf: `Hm96.Default512`'s prefixes and mask are `Hash.Hm96`'s at the default key (`saltPrefix_eq`,
  `leafPrefix_eq`, `mask_eq`, the mask through `bitL_mask`), and a row's `sha512/row/v2` message is `RowV2.msg`
  (`enc_eq`), so the hm96 leaf `hm96Leaf` computes at `sha512Msg` is `Hash.Hm96.leaf` over `RowV2.digest`
  (`hm96Leaf_spec`);
* `rowLeaf_accepts`: the composition.

`hid` reads the spec's domain id, a 64-byte `Digest`, as the port's domain, which C-Flock takes from the port's name in
the circuit file (`domOf`): the two agree only at a 64-byte name until `Hash.Tree.Domain.id` is a message.
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
    (hid : D.id.bytes = bs dom) (hd : d < 2 ^ 64) (hi : i < 2 ^ 64) (hl : bs l = x.bytes) (hr : bs r = y.bytes) :
    bs (Registered.node dom d i l r) = (Verity.sha512 (Hash.Tree.nodeMsg D d i x y)).bytes := by
  have e : bs (Registered.nodeIn dom d i l r) = Hash.Tree.nodeMsg D d i x y := by
    rw [Registered.nodeIn, frame_eq, Hash.Tree.nodeMsg]
    simp only [List.map, uint_eq d hd, uint_eq i hi, hid, hl, hr]
  rw [Registered.node_eq, sha512_bytes, ← e, bytes_bs]

/-- **C-Flock's frame leaf is the spec's leaf digest** at the rank's own position. -/
theorem leaf_eq {dom payload : ByteArray} {D : Hash.Tree.Domain} {pos : ℕ} (schema : String)
    (hid : D.id.bytes = bs dom) (hp : D.position pos = pos) (hpos : pos < 2 ^ 64) :
    bs (Registered.leafAs schema dom pos payload) =
      (Verity.sha512 (Hash.Tree.leafMsg D pos schema (bs payload))).bytes := by
  have e : bs (Registered.leafInAs schema dom pos payload) = Hash.Tree.leafMsg D pos schema (bs payload) := by
    rw [Registered.leafInAs, frame_eq, Hash.Tree.leafMsg, hp]
    simp only [List.map, uint_eq pos hpos, hid]
    rfl
  rw [Registered.leafAs_eq, sha512_bytes, ← e, bytes_bs]

/-- **C-Flock's climb is the spec's**, from a digest whose bytes are the code's, through 64-byte siblings. -/
theorem climbL_eq {dom : ByteArray} {D : Hash.Tree.Domain} (hid : D.id.bytes = bs dom) :
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
theorem opensLeaf_accepts (reg : Registered.Port) (D : Hash.Tree.Domain) (hid : D.id.bytes = bs reg.domain)
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

/-- **`hm96Leaf` at SHA-512, on a `sha512/row/v2` row, is the spec's hm96 leaf over `RowV2.digest`.** -/
theorem hm96Leaf_spec (bits : Opening Bool → List Bool) (salt : Opening Bool → Claims.Msg) (o : Opening Bool) :
    ((hm96Leaf (fun o => FlockLevel3.RowV2.enc (bits o)) salt o).eval sha512Msg).1 =
      some (Hash.Hm96.leaf Verity.sha512 key512 (Hash.RowV2.digest Verity.sha512 (bits o)) (salt o)).bytes := by
  have hlen : ∀ m, (sha512Msg m).length = 64 := fun m => by
    rw [sha512Msg_eq, Hash.Digest.bytes, List.length_ofFn]
  rw [hm96Leaf_fst, if_pos ⟨hlen _, hlen _⟩, hm96Outer_eq, sha512Msg_eq, Hash.Hm96.leaf, Hash.RowV2.digest, enc_eq]

/-! ## The composition -/

/-- **A row/v2 row C-Flock's registered check opens is one the spec's verifier accepts**: at a port of schema
`hm96-sha512/row/v2` and a verifier of that format and the default key, whose domain has the port's id (`hid`), the
ranks as positions and the port's leaves as count, the row `row` under the salt `salt` (192 bytes), opened at `pos`
through `path` against the port's root, is `Hash.RowLeaf.accepts` at SHA-512, with the path's siblings as digests. -/
theorem rowLeaf_accepts (reg : Registered.Port) (V : Hash.RowLeaf.Verifier) (hv2 : V.format = .v2)
    (hschema : reg.schema = Hash.RowLeaf.Format.v2.schema) (hkey : V.key = key512)
    (hid : V.domain.id.bytes = bs reg.domain) (hp : ∀ r, V.domain.position r = r)
    (hcount : V.domain.count = reg.leaves) (hlt : reg.leaves < 2 ^ 64) (root : Hash.Digest)
    (hroot : bs reg.root = root.bytes) (row : List Bool) (hrow : row.length < 2 ^ 64) (salt : Claims.Msg)
    (hsalt : salt.length = Hash.Hm96.SALT_BYTES) (pos : ℕ) (path : ByteArray)
    (h : Registered.opensLeaf reg
      (bytes (Hash.Hm96.leaf Verity.sha512 key512 (Hash.RowV2.digest Verity.sha512 row) salt).bytes) pos path = true) :
    Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 V root pos
      ⟨row, salt, (Registered.sibs path (log2ceil reg.leaves)).map digOf⟩ := by
  refine ⟨hrow, hsalt, ?_⟩
  have ha := opensLeaf_accepts reg V.domain hid hp hcount hlt root hroot _ path pos h
  rw [bs_bytes, hschema] at ha
  unfold Hash.RowLeaf.leafMsg
  rw [hv2, hkey]
  exact ha

/-! ## An opening `commitDep` accepts -/

/-- At a file whose ports are all row/v2 rows, an opening's row message is its row's `sha512/row/v2` bytes. -/
theorem rowDep_v2 {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt} {c : Flock.Circuit}
    (hv2 : RowsV2 c) (place : ℕ → (w : ℕ) × Fin w) (bits : Claims.Msg → ℕ → Bool) (o : Opening Bool) :
    rowDep C rows c place bits o = FlockLevel3.RowV2.enc (SamplingDeploy.rowBits place bits o) := by
  unfold rowDep
  cases hp : portOfPos C rows c o.pos with
  | none => rfl
  | some p =>
    obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.mp hp
    obtain ⟨hq, rfl⟩ := Array.getElem?_eq_some_iff.mp hr
    have h := (Array.all_eq_true.mp hv2) r.port hq
    simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at h
    simp only [rowMsg, h.1, h.2, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]

/-- **What `commitDep` checked at SHA-512**: the position's row and root, the registered port whose root it is, and that
port's check of the hm96 leaf over the row's `sha512/row/v2` digest and the opening's salt, through its path. -/
theorem commitDep_opens {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt} {c : Flock.Circuit}
    {nl : ℕ} {roots : List Claims.Msg} (hv2 : RowsV2 c) {o : Opening Bool}
    (h : (commitDep C rows c nl roots).Opens sha512Msg roots o = true) :
    ∃ r rt reg, rowOfPos C rows o.pos = some r ∧ roots[r.port]? = some rt ∧
      (portsOf c nl roots).find? (fun p => decide (bs p.root = rt)) = some reg ∧
      Registered.opensLeaf reg (bytes (Hash.Hm96.leaf Verity.sha512 key512 (Hash.RowV2.digest Verity.sha512
        (SamplingDeploy.rowBits (placeOf C rows) (bitsOf (log2ceil nl)) o)) (saltOf o)).bytes) r.leaf
        (bytes (pathOf (log2ceil nl) o)) = true := by
  have hrow : rowDep C rows c (placeOf C rows) (bitsOf (log2ceil nl)) =
      fun o => FlockLevel3.RowV2.enc (SamplingDeploy.rowBits (placeOf C rows) (bitsOf (log2ceil nl)) o) :=
    funext (rowDep_v2 hv2 _ _)
  unfold Commitment.Opens commitDep pathCommit at h
  rw [hrow] at h
  dsimp only at h
  cases hat : atOf C rows roots o.pos with
  | none => rw [hat] at h; simp [OC.eval] at h
  | some ri =>
    rw [hat] at h
    dsimp only at h
    rw [ocBind_eval, hm96Leaf_spec] at h
    dsimp only at h
    obtain ⟨reg, hf, hopen⟩ := (regPath_sha512 _ _ _ _ _).mp h
    obtain ⟨r, hr, hm⟩ := Option.bind_eq_some_iff.mp hat
    obtain ⟨rt, hrt, rfl⟩ := Option.map_eq_some_iff.mp hm
    exact ⟨r, rt, reg, hr, hrt, hf, hopen⟩

theorem portsOf_mem {c : Flock.Circuit} {nl : ℕ} {roots : List Claims.Msg} {reg : Registered.Port}
    (h : reg ∈ portsOf c nl roots) :
    ∃ q < c.ports.size, reg.leaves = nl ∧ reg.domain = domOf c q ∧ reg.schema = schemaOf c q := by
  unfold portsOf at h
  obtain ⟨q, hq, rfl⟩ := Array.mem_map.mp h
  exact ⟨q, Array.mem_range.mp hq, rfl, rfl, rfl⟩

theorem schemaOf_v2 {c : Flock.Circuit} (hv2 : RowsV2 c) {q : ℕ} (hq : q < c.ports.size) :
    schemaOf c q = Hash.RowLeaf.Format.v2.schema := by
  have h := (Array.all_eq_true.mp hv2) q hq
  simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at h
  simp [schemaOf, Array.getElem?_eq_getElem hq, Registered.portSchema, h.1, h.2, HmRow.ROW_SCHEMA_V2,
    Hash.RowLeaf.Format.schema]

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

/-- At a depth of at least 1, an accepted path's length fixes the salt's: 192 bytes. -/
theorem saltOf_length {d : ℕ} (hd : 1 ≤ d) {o : Opening Bool} (h : (pathOf d o).length = 64 * d) :
    (saltOf o).length = Hash.Hm96.SALT_BYTES := by
  unfold pathOf at h
  unfold saltOf saltLen Hash.Hm96.SALT_BYTES
  simp only [List.length_take, List.length_drop] at h ⊢
  omega

theorem log2ceil_pos {n : ℕ} (hn : 2 ≤ n) : 1 ≤ log2ceil n := by
  unfold log2ceil
  rw [if_neg (by omega)]
  omega

/-- The spec's verifier of a registered row/v2 port: the domain of id `id`, the ranks as positions and the port's
leaves as count, the default key. -/
def verifierOf (id : Hash.Digest) (reg : Registered.Port) : Hash.RowLeaf.Verifier :=
  ⟨⟨id, reg.leaves, fun r => r⟩, key512, .v2⟩

theorem rowBits_place (place : ℕ → (w : ℕ) × Fin w) (bits : Claims.Msg → ℕ → Bool) (o : Opening Bool) :
    (SamplingDeploy.rowBits place bits o)[((place o.pos).2 : ℕ)]? = some o.val := by
  simp [SamplingDeploy.rowBits]

/-- **An opening `commitDep` accepts at SHA-512 is a spec row-leaf opening**: at a file whose ports are all row/v2 rows
and at least two leaves a port, the position's row is opened at its leaf index under the registered port whose root is
the position's, which the spec's verifier of that port (`verifierOf`, at a domain id whose bytes are the port's domain)
accepts at the root's digest, with the opened bit at its place. -/
theorem commitDep_accepts {C : FlockSoundness.Audit.Circuit} {rows : Fin C.N → Option RowAt} {c : Flock.Circuit}
    {nl : ℕ} {roots : List Claims.Msg} (hv2 : RowsV2 c) (hnl : 2 ≤ nl) (hlt : nl < 2 ^ 64) {o : Opening Bool}
    (h : (commitDep C rows c nl roots).Opens sha512Msg roots o = true) :
    ∃ r rt reg, rowOfPos C rows o.pos = some r ∧ roots[r.port]? = some rt ∧
      (portsOf c nl roots).find? (fun p => decide (bs p.root = rt)) = some reg ∧
      ∀ id : Hash.Digest, id.bytes = bs reg.domain → (placeOf C rows o.pos).1 < 2 ^ 64 →
        Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 (verifierOf id reg) (digOf reg.root) r.leaf
          ⟨SamplingDeploy.rowBits (placeOf C rows) (bitsOf (log2ceil nl)) o, saltOf o,
            (Registered.sibs (bytes (pathOf (log2ceil nl) o)) (log2ceil nl)).map digOf⟩ := by
  obtain ⟨r, rt, reg, hr, hrt, hf, hopen⟩ := commitDep_opens hv2 h
  refine ⟨r, rt, reg, hr, hrt, hf, fun id hid hw => ?_⟩
  obtain ⟨q, hq, hleaves, -, hschema⟩ := portsOf_mem (Array.mem_of_find?_eq_some hf)
  have hpath : (pathOf (log2ceil nl) o).length = 64 * log2ceil nl := by
    have := hopen
    simp only [Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq, size_bytes, hleaves] at this
    exact this.1.2
  have ha := rowLeaf_accepts reg (verifierOf id reg) rfl (hschema.trans (schemaOf_v2 hv2 hq)) rfl hid
    (fun _ => rfl) rfl (hleaves ▸ hlt) (digOf reg.root) (digOf_bytes (root_size hopen)).symm _
    (by simpa [SamplingDeploy.rowBits] using hw) (saltOf o) (saltOf_length (log2ceil_pos hnl) hpath) r.leaf _ hopen
  rwa [hleaves] at ha

end Flock.Bridge.Openings
