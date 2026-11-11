module

public import Proofs.Flock.Verifier.Registered
public import Proofs.Flock.Bytes
public import Definitions.Hash.Tree
import Mathlib.Tactic.NormNum
import all Init.Data.Array.Basic

/-!
# The registered path check is `Hash.Tree`'s (frame-v3-sha512)

The verifier's path check (`Flock.Registered.opensLeaf`) as the trusted tree (`Hash.Tree`) states it, at a port whose
frame-v3 domain is 64 bytes, the width of `Hash.Tree.Domain.id`, and at a hash `H` that reads a byte string as the
code's SHA-512 does (`HashOf`; `Verity.sha512` is one):

* `bs_be64`, `bs_be32`, `bs_uint`, `bs_frame`: the code's frame (`Flock.Registered.frame`, what `FV3.hash` hashes) is
  `Hash.Tree.frame` over its parts' bytes, its lengths `Hash.be` and its integers `Hash.uint`.
* `dig`: a byte string as a digest. `leafInAs_bs`, `nodeIn_bs`: a leaf's and a node's frame are `Hash.Tree.leafMsg`'s
  and `Hash.Tree.nodeMsg`'s at a domain whose id is the code's domain, at identity positions.
* `climbL_spec`: the code's climb from a 64-byte digest through 64-byte siblings ends where `Hash.Tree.climb` does.
* `opensLeaf_accepts`: **what `opensLeaf` accepts, `Hash.Tree.accepts` accepts**, at the port's domain (`domainOf`:
  its id, its leaves, identity positions), its root, the digest of the leaf's message and the path's siblings.
-/

@[expose] public section

namespace Flock.Bridge.Spec

open Flock.Registered
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length be64_bs)

/-! ## Byte encodings -/

/-- `be k` reads its number modulo `256 ^ k`. -/
theorem be_mod (k n : ℕ) : Hash.be k (n % 256 ^ k) = Hash.be k n := by
  unfold Hash.be
  apply List.map_congr_left
  intro i hi
  rw [List.mem_range] at hi
  have hk : 256 ^ k = 256 ^ (k - 1 - i) * 256 ^ (i + 1) := by
    rw [← pow_add]; congr 1; omega
  rw [hk, Nat.mod_mul_right_div_self, Nat.mod_mod_of_dvd _ (dvd_pow_self 256 (Nat.succ_ne_zero i))]

theorem toUInt8_mod (n : ℕ) : (n % 256).toUInt8 = n.toUInt8 := by
  apply UInt8.toNat_inj.mp
  simp

theorem shift64_toUInt8 (x : UInt64) (j : ℕ) (hj : j < 8) :
    (x >>> (UInt64.ofNat (8 * j))).toUInt8 = (x.toNat / 256 ^ j % 256).toUInt8 := by
  apply UInt8.toNat_inj.mp
  rw [toUInt8_mod]
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.toUInt8_eq, UInt8.toNat_ofNat',
    UInt64.toNat_ofNat']
  rw [Nat.mod_eq_of_lt (a := 8 * j) (show 8 * j < 2 ^ 64 by omega), Nat.mod_eq_of_lt (a := 8 * j) (by omega),
    Nat.shiftRight_eq_div_pow, pow_mul]
  norm_num

theorem bs_be64 (x : UInt64) : bs (be64 x) = Hash.be 8 x.toNat := by
  rw [be64_bs]
  have e := fun j (hj : j < 8) => shift64_toUInt8 x j hj
  have e7 := e 7 (by omega); have e6 := e 6 (by omega); have e5 := e 5 (by omega); have e4 := e 4 (by omega)
  have e3 := e 3 (by omega); have e2 := e 2 (by omega); have e1 := e 1 (by omega); have e0 := e 0 (by omega)
  norm_num at e7 e6 e5 e4 e3 e2 e1 e0
  simp only [Hash.be, List.range_succ, List.range_zero, List.map_cons, List.map_nil, List.nil_append,
    List.cons_append]
  norm_num
  exact ⟨e7, e6, e5, e4, e3, e2, e1, e0⟩

/-- A part's length prefix is `Hash.be 8` of its size. -/
theorem bs_be64_size (p : ByteArray) : bs (be64 p.size.toUInt64) = Hash.be 8 p.size := by
  rw [bs_be64, Nat.toUInt64_eq, UInt64.toNat_ofNat', show (2 : ℕ) ^ 64 = 256 ^ 8 by norm_num, be_mod]

theorem shift32_toUInt8 (x : UInt32) (j : ℕ) (hj : j < 4) :
    (x >>> (UInt32.ofNat (8 * j))).toUInt8 = (x.toNat / 256 ^ j % 256).toUInt8 := by
  apply UInt8.toNat_inj.mp
  rw [toUInt8_mod]
  simp only [UInt32.toNat_toUInt8, UInt32.toNat_shiftRight, Nat.toUInt8_eq, UInt8.toNat_ofNat',
    UInt32.toNat_ofNat']
  rw [Nat.mod_eq_of_lt (a := 8 * j) (show 8 * j < 2 ^ 32 by omega), Nat.mod_eq_of_lt (a := 8 * j) (by omega),
    Nat.shiftRight_eq_div_pow, pow_mul]
  norm_num

theorem be32_bs (x : UInt32) : bs (be32 x) =
    [(x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, (x >>> 0).toUInt8] := by
  simp [be32, bs, List.range', ByteArray.push, ByteArray.emptyWithCapacity]
  rfl

theorem bs_be32 (x : UInt32) : bs (be32 x) = Hash.be 4 x.toNat := by
  rw [be32_bs]
  have e := fun j (hj : j < 4) => shift32_toUInt8 x j hj
  have e3 := e 3 (by omega); have e2 := e 2 (by omega); have e1 := e 1 (by omega); have e0 := e 0 (by omega)
  norm_num at e3 e2 e1 e0
  simp only [Hash.be, List.range_succ, List.range_zero, List.map_cons, List.map_nil, List.nil_append,
    List.cons_append]
  norm_num
  exact ⟨e3, e2, e1, e0⟩

/-- A tag's length prefix is `Hash.be 4` of its size. -/
theorem bs_be32_size (b : ByteArray) : bs (be32 b.size.toUInt32) = Hash.be 4 b.size := by
  rw [bs_be32, Nat.toUInt32_eq, UInt32.toNat_ofNat', show (2 : ℕ) ^ 32 = 256 ^ 4 by norm_num, be_mod]

theorem size_eq_log2 {x : ℕ} (hx : x ≠ 0) : Nat.size x = Nat.log2 x + 1 := by
  apply le_antisymm
  · exact Nat.size_le.mpr Nat.lt_log2_self
  · by_contra h
    have h' : Nat.size x ≤ Nat.log2 x := by omega
    have := Nat.size_le.mp h'
    have := Nat.log2_self_le hx
    omega

theorem uint_len (x : ℕ) : (Nat.log2 x + 8) / 8 = max 1 ((Nat.size x + 7) / 8) := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · simp
  · rw [size_eq_log2 (by omega)]
    omega

/-- **The code's frame integer is `Hash.uint`.** -/
theorem bs_uint (x : ℕ) : bs (FV3.uint x) = Hash.uint x := by
  simp only [FV3.uint, bs, Array.toList_map, Array.toList_range, Hash.uint, Hash.be, ← uint_len]
  apply List.map_congr_left
  intro i _
  rw [← toUInt8_mod, Nat.shiftRight_eq_div_pow, pow_mul]
  norm_num

theorem bs_FRAME : bs FV3.FRAME = Hash.Tree.FRAME := by
  simp [FV3.FRAME, Hash.Tree.FRAME, Hash.cstr, Hash.utf8, bs, ByteArray.push]

theorem bs_foldl_step : ∀ (parts : List ByteArray) (acc : ByteArray),
    bs (parts.foldl step acc) = bs acc ++ (parts.map bs).flatMap fun p => Hash.be 8 p.length ++ p
  | [], acc => by simp
  | p :: ps, acc => by
    rw [List.foldl_cons, bs_foldl_step ps, step, bs_append, bs_append, bs_be64_size]
    simp [List.flatMap_cons, bs_length, List.append_assoc]

/-- **A frame is `Hash.Tree.frame`** over its parts' bytes. -/
theorem bs_frame (tag : String) (parts : List ByteArray) :
    bs (frame tag parts) = Hash.Tree.frame tag (parts.map bs) := by
  rw [frame, bs_foldl_step, bs_append, bs_append, bs_FRAME, bs_be32_size]
  simp [Hash.Tree.frame, Hash.utf8, bs]

/-! ## Digests and the hash -/

/-- A byte string as a digest: its first 64 bytes, zero past its end. -/
def dig (b : ByteArray) : Hash.Digest := fun i => b.data.getD i.val 0

theorem dig_bytes {b : ByteArray} (h : b.size = 64) : (dig b).bytes = bs b := by
  apply List.ext_getElem
  · simp [Verity.Digest.bytes, bs, ← h]
  · intro i h1 h2
    have hi : i < b.data.size := by simpa [bs] using h2
    simp only [Verity.Digest.bytes, List.getElem_ofFn, dig, bs, Array.getElem_toList, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem hi, Option.getD_some]

theorem dig_inj {a b : ByteArray} (ha : a.size = 64) (hb : b.size = 64) (h : dig a = dig b) : a = b := by
  have := congrArg Verity.Digest.bytes h
  rw [dig_bytes ha, dig_bytes hb] at this
  exact Properties.NetworkCertifier.Bytes.bs_inj this

/-- **`H` reads a byte string as the code's SHA-512 does**: its digest of the string's bytes is SHA-512's output. -/
def HashOf (H : Hash.Msg → Hash.Digest) : Prop := ∀ x : ByteArray, H (bs x) = dig (Flock.Sha512.hash x)

/-! ## Leaves, nodes and the climb -/

/-- **A registered port's frame-v3 domain as the spec holds it**: its domain as the id, its leaves as the count, and
rank `r` at position `r`, as the code frames a leaf (`FV3.leaf`'s `uint i, uint i`). -/
def domainOf (reg : Registered.Port) : Hash.Tree.Domain := ⟨dig reg.domain, reg.leaves, id⟩

theorem leafInAs_bs {dom : ByteArray} (hd : dom.size = 64) (D : Hash.Tree.Domain) (hD : D.id = dig dom)
    (hp : ∀ r, D.position r = r) (schema : String) (pos : ℕ) (payload : ByteArray) :
    bs (leafInAs schema dom pos payload) = Hash.Tree.leafMsg D pos schema (bs payload) := by
  rw [leafInAs, bs_frame, Hash.Tree.leafMsg, hD, dig_bytes hd, hp]
  simp only [List.map_cons, List.map_nil, bs_uint]
  rfl

theorem nodeIn_bs {dom : ByteArray} (hd : dom.size = 64) (D : Hash.Tree.Domain) (hD : D.id = dig dom)
    (d idx : ℕ) {l r : ByteArray} (hl : l.size = 64) (hr : r.size = 64) :
    bs (nodeIn dom d idx l r) = Hash.Tree.nodeMsg D d idx (dig l) (dig r) := by
  rw [nodeIn, bs_frame, Hash.Tree.nodeMsg, hD, dig_bytes hd, dig_bytes hl, dig_bytes hr]
  simp only [List.map_cons, List.map_nil, bs_uint]

/-- **The code's climb is the spec's**, at a 64-byte domain, from a 64-byte digest through 64-byte siblings. -/
theorem climbL_spec {H : Hash.Msg → Hash.Digest} (hH : HashOf H) {dom : ByteArray} (hd : dom.size = 64)
    (D : Hash.Tree.Domain) (hD : D.id = dig dom) :
    ∀ (sibs : List ByteArray) (cur : ByteArray) (idx d : ℕ), cur.size = 64 → (∀ s ∈ sibs, s.size = 64) →
      dig (climbL dom cur idx d sibs) = (Hash.Tree.climb H D d idx (dig cur) (sibs.map dig)).2 := by
  intro sibs
  induction sibs with
  | nil => intro cur idx d _ _; rfl
  | cons s rest ih =>
    intro cur idx d hc hs
    have hs0 : s.size = 64 := hs s (by simp)
    have hr : ∀ x ∈ rest, x.size = 64 := fun x hx => hs x (by simp [hx])
    simp only [climbL, List.map_cons, Hash.Tree.climb]
    by_cases hp : idx % 2 = 0
    · simp only [hp, ↓reduceIte]
      rw [ih _ _ _ (by rw [node_eq]; exact Sha512.hash_size _) hr, node_eq, ← nodeIn_bs hd D hD d (idx / 2) hc hs0,
        hH]
    · simp only [hp, ↓reduceIte]
      rw [ih _ _ _ (by rw [node_eq]; exact Sha512.hash_size _) hr, node_eq, ← nodeIn_bs hd D hD d (idx / 2) hs0 hc,
        hH]

theorem log2ceil_eq_depth (n : ℕ) : log2ceil n = Hash.Tree.depth n := rfl

/-- **The registered path check is `Hash.Tree`'s**: a leaf payload `opensLeaf` accepts at a port whose domain is 64
bytes, the spec accepts at the port's domain (`domainOf`) and root, with the leaf the digest of the spec's leaf message
over the payload, and the siblings the path's. -/
theorem opensLeaf_accepts {H : Hash.Msg → Hash.Digest} (hH : HashOf H) {reg : Registered.Port}
    (hd : reg.domain.size = 64)
    {payload path : ByteArray} {pos : ℕ} (h : opensLeaf reg payload pos path = true) :
    Hash.Tree.accepts H (domainOf reg) (dig reg.root) pos
      (H (Hash.Tree.leafMsg (domainOf reg) pos reg.schema (bs payload)))
      ((sibs path (log2ceil reg.leaves)).map dig) := by
  simp only [opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hpos, hlen⟩, hc⟩ := h
  refine ⟨hpos, by simp [sibs_length, domainOf, log2ceil_eq_depth], ?_⟩
  have hl := climbL_spec hH hd (domainOf reg) rfl (sibs path (log2ceil reg.leaves))
    (leafAs reg.schema reg.domain pos payload) pos 0 (by rw [leafAs_eq]; exact Sha512.hash_size _) (sibs_size hlen)
  rw [← leafInAs_bs hd (domainOf reg) rfl (fun _ => rfl), hH, ← leafAs_eq, ← hl]
  exact congrArg dig hc

end Flock.Bridge.Spec
