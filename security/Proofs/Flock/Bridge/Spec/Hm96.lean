module

public import Proofs.Flock.Bridge.Spec.Row
public import Proofs.Flock.Soundness.Discharge.Hm.Mask
public import Proofs.Flock.Claims.Sampling.Leaf
import all Init.Data.Array.Basic

/-!
# hm96's mask and leaf are the spec's

The executable's hm96 leaf at its key (`Hm96.Default512`, through `FlockSoundness.Refine.K512` as
`Claims.Sampling.hm96Outer` reads it) as `Hash.Hm96` states it, at a hash `H` that reads bytes as SHA-512 does
(`HashOf`):

* `bit_eq_bitL`, `bit_hm96`, `maskV_eq`: a bit of a byte string, a key bit and a bit of `M·y` are `Hash.Hm96`'s
  (`bit`, `maskBit`), from the proofs' `bitL` and `maskV`.
* `bs_mask`: **the executable's mask `y ↦ M·y` is `Hash.Hm96.mask`** at the key's bytes, from `bitL_mask` and a
  byte's being the sum of its bits (`byte_bits`).
* `hm96Outer_eq`, `sha512Msg_hm96Outer`: the outer message the Bridge's leaf hashes is `Hash.Hm96`'s
  `leafPrefix ‖ commitString`, and its digest's bytes are `Hash.Hm96.leaf`'s.
-/

@[expose] public section

namespace Flock.Bridge.Spec

open Properties.NetworkCertifier.Bytes (bs bs_append bs_length)
open FlockSoundness.Discharge.Hm (bitL maskV bitL_mask)

/-! ## Bits -/

theorem bits_fold (n : ℕ) : ∀ k : ℕ,
    (List.range k).foldl (fun acc s => acc + if n.testBit s then 2 ^ s else 0) 0 = n % 2 ^ k
  | 0 => by simp [Nat.mod_one]
  | k + 1 => by
    rw [List.range_succ, List.foldl_append, bits_fold n k, List.foldl_cons, List.foldl_nil, Nat.mod_pow_succ,
      Nat.testBit_eq_decide_div_mod_eq]
    rcases Nat.mod_two_eq_zero_or_one (n / 2 ^ k) with h | h <;> simp [h]

/-- A byte is the sum of its bits. -/
theorem byte_bits (b : UInt8) :
    ((List.range 8).foldl (fun acc s => acc + if b.toNat.testBit s then 2 ^ s else 0) 0).toUInt8 = b := by
  rw [bits_fold, Nat.mod_eq_of_lt b.toNat_lt, Nat.toUInt8_eq, UInt8.ofNat_toNat]

/-- `Hash.Hm96.bit` is the proofs' `bitL`. -/
theorem bit_eq_bitL (s : Hash.Msg) (t : ℕ) : Hash.Hm96.bit s t = bitL s t := by
  simp only [Hash.Hm96.bit, bitL, Nat.testBit_eq_decide_div_mod_eq]
  generalize (s.getD (t / 8) 0).toNat / 2 ^ (t % 8) % 2 = r
  by_cases h : r = 1 <;> simp [h]

/-- The code's key bit is `Hash.Hm96.bit` of the key's bytes. -/
theorem bit_hm96 (key : ByteArray) (t : ℕ) : Flock.Hm96.bit key t = Hash.Hm96.bit (bs key) t := by
  rw [Flock.Hm96.bit, Hash.Hm96.bit, FlockSoundness.Discharge.Hm.get!_eq_getD, Nat.shiftRight_eq_div_pow, bs]

theorem foldl_filter_xor (p f : ℕ → Bool) : ∀ (l : List ℕ) (acc : Bool),
    (l.filter p).foldl (fun acc j => xor acc (f j)) acc = l.foldl (fun acc j => xor acc (p j && f j)) acc
  | [], _ => rfl
  | a :: l, acc => by
    rw [List.filter_cons, List.foldl_cons]
    cases hp : p a
    · rw [Bool.false_and, Bool.xor_false]; exact foldl_filter_xor p f l acc
    · simp only [Bool.true_and, ↓reduceIte, List.foldl_cons]; exact foldl_filter_xor p f l _

/-- Bit `i` of `M·y`, the proofs' (`maskV`), is `Hash.Hm96.maskBit`'s. -/
theorem maskV_eq (key : ByteArray) (y : Hash.Msg) (i : ℕ) : maskV key (bitL y) i = Hash.Hm96.maskBit (bs key) y i := by
  unfold maskV Hash.Hm96.maskBit
  rw [show 8 * Hash.Hm96.SALT_BYTES = 1536 from rfl, foldl_filter_xor]
  refine List.foldl_ext _ _ _ fun acc j _ => ?_
  rw [bit_hm96, bit_eq_bitL y, Bool.and_comm]

theorem mask_length (key y : Hash.Msg) : (Hash.Hm96.mask key y).length = 64 := by simp [Hash.Hm96.mask]

/-- **The executable's mask is `Hash.Hm96.mask`** at its key's bytes, at every salt (neither reads past byte 192). -/
theorem bs_mask (y : Hash.Msg) :
    bs (Flock.Hm96.Default512.mask ⟨y.toArray⟩) = Hash.Hm96.mask (bs Flock.Hm96.Default512.key) y := by
  apply List.ext_getElem
  · rw [mask_length, bs_length, FlockSoundness.Refine.mask_size]
  · intro p h1 h2
    have hp : p < 64 := by simpa [mask_length] using h2
    simp only [Hash.Hm96.mask, List.getElem_map, List.getElem_range]
    have hb : ∀ s ∈ List.range 8, Hash.Hm96.maskBit (bs Flock.Hm96.Default512.key) y (8 * p + s) =
        (bs (Flock.Hm96.Default512.mask ⟨y.toArray⟩))[p].toNat.testBit s := by
      intro s hs
      rw [List.mem_range] at hs
      rw [← maskV_eq, ← bitL_mask y (t := 8 * p + s) (by omega), bitL, show (8 * p + s) / 8 = p by omega,
        show (8 * p + s) % 8 = s by omega]
      exact congrArg (fun b : UInt8 => b.toNat.testBit s) (List.getD_eq_getElem _ 0 h1)
    rw [List.foldl_ext _ _ _ fun acc s hs => by rw [hb s hs], byte_bits]

/-! ## The leaf -/

theorem xorBytes_ofFn (d : Hash.Digest) (m : List UInt8) (hm : m.length = 64) :
    FlockSoundness.Merkle.xorBytes d.bytes m = List.ofFn fun i : Fin 64 => d i ^^^ m.getD i 0 := by
  apply List.ext_getElem
  · simp [FlockSoundness.Merkle.xorBytes, Verity.Digest.bytes, hm]
  · intro i h1 h2
    have hi : i < 64 := by simpa using h2
    simp only [FlockSoundness.Merkle.xorBytes, Verity.Digest.bytes, List.getElem_zipWith, List.getElem_ofFn,
      List.getD_eq_getElem _ _ (show i < m.length by omega)]

/-- `Claims.Sampling.sha512Msg` is the bytes of `H`'s digest. -/
theorem sha512Msg_eq {H : Hash.Msg → Hash.Digest} (hH : HashOf H) (m : Claims.Msg) :
    Claims.Sampling.sha512Msg m = (H m).bytes := by
  have h := hH (Claims.Sampling.bytes m)
  rw [Claims.Sampling.bs_bytes] at h
  rw [Claims.Sampling.sha512Msg, h, dig_bytes (Sha512.hash_size _)]

/-- **The outer message the Bridge's hm96 leaf hashes is `Hash.Hm96`'s**: `leafPrefix ‖ commitString` at the
executable's key, the row message's digest and the salt. -/
theorem hm96Outer_eq {H : Hash.Msg → Hash.Digest} (hH : HashOf H) (row salt : Claims.Sampling.Opening Bool → Claims.Msg)
    (o : Claims.Sampling.Opening Bool) :
    Claims.Sampling.hm96Outer Claims.Sampling.sha512Msg row salt o =
      Hash.Hm96.leafPrefix H (bs Flock.Hm96.Default512.key) ++
        Hash.Hm96.commitString H (bs Flock.Hm96.Default512.key) (H (row o)) (salt o) := by
  have em : ∀ y, FlockSoundness.Refine.K512.mask y = Hash.Hm96.mask (bs Flock.Hm96.Default512.key) y :=
    fun y => (FlockSoundness.Refine.K512_mask y).trans (bs_mask y)
  have el : FlockSoundness.Refine.K512.leafPrefix = Hash.Hm96.leafPrefix H (bs Flock.Hm96.Default512.key) :=
    leafPrefix_eq hH
  have es : FlockSoundness.Refine.K512.saltPrefix = Hash.Hm96.saltPrefix := saltPrefix_eq
  rw [Claims.Sampling.hm96Outer, Hash.Hm96.commitString, sha512Msg_eq hH, sha512Msg_eq hH, em, el, es,
    xorBytes_ofFn _ _ (mask_length _ _)]

/-- **The Bridge's hm96 leaf is `Hash.Hm96.leaf`**: the bytes SHA-512 gives its outer message are the spec leaf's
over the row message's digest and the salt, at the executable's key. -/
theorem sha512Msg_hm96Outer {H : Hash.Msg → Hash.Digest} (hH : HashOf H)
    (row salt : Claims.Sampling.Opening Bool → Claims.Msg) (o : Claims.Sampling.Opening Bool) :
    Claims.Sampling.sha512Msg (Claims.Sampling.hm96Outer Claims.Sampling.sha512Msg row salt o) =
      (Hash.Hm96.leaf H (bs Flock.Hm96.Default512.key) (H (row o)) (salt o)).bytes := by
  rw [sha512Msg_eq hH, hm96Outer_eq hH, Hash.Hm96.leaf]

end Flock.Bridge.Spec
