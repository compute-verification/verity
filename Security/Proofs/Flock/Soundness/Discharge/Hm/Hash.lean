/-
The verifier's SHA-512 (`Flock.Sha512.hash`) as the chain of its compressions: the padded message (`padData`), the
chaining value after each block (`chain`), and the digest's bytes, big-endian per word (`digestOf`), whose bit `t` is bit
`wordBit t` of the chaining value in word order.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Exec

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-- The 16 bytes of a message length `len` (in bytes) as SHA-512's 128-bit big-endian bit count. -/
def lenBytes (len : ℕ) : List UInt8 := (List.range' 0 16).map fun a => ((len * 8) >>> (8 * (15 - a)) % 256).toUInt8

/-- The number of zero bytes SHA-512's padding puts after `msg ‖ 0x80`. -/
def zeroPad (len : ℕ) : ℕ := (240 - (len + 1) % 128) % 128

/-- SHA-512's padded message (FIPS 180-4 §5.1.2) as `Sha512.hash` builds it: `msg ‖ 0x80 ‖ 0…0 ‖ len`. -/
def padData (msg : ByteArray) : ByteArray :=
  (List.range' 0 16).foldl (fun b a => b.push ((msg.size * 8) >>> (8 * (15 - a)) % 256).toUInt8)
    ((List.range' 0 ((240 - (msg.push 128).size % 128) % 128)).foldl (fun b _ => b.push 0) (msg.push 128))

/-- The chaining value after the first `n` blocks of `data`. -/
def chain (data : ByteArray) (n : ℕ) : Array UInt64 :=
  (List.range' 0 n).foldl (fun h blk => Sha512.compress h data (128 * blk)) Sha512.iv

/-- The digest of a chaining value: its eight words, big-endian. -/
def digestOf (h : Array UInt64) : ByteArray :=
  ⟨Array.ofFn (n := 64) fun i => (h[i.val / 8]! >>> (8 * (7 - i.val % 8)).toUInt64).toUInt8⟩

/-- **`Sha512.hash` is the chain of its compressions over the padded message.** -/
theorem hash_eq (msg : ByteArray) : Sha512.hash msg = digestOf (chain (padData msg) ((padData msg).size / 128)) := by
  simp only [Sha512.hash, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero,
    Nat.add_sub_cancel, Nat.div_one]
  rfl

theorem foldl_push_data (f : ℕ → UInt8) : ∀ (l : List ℕ) (b : ByteArray),
    (l.foldl (fun b a => b.push (f a)) b).data.toList = b.data.toList ++ l.map f
  | [], b => by simp
  | a :: l, b => by
      rw [List.foldl_cons, foldl_push_data f l, ByteArray.data_push, Array.toList_push, List.map_cons,
        List.append_assoc]
      rfl

/-- The padded message's bytes. -/
theorem padData_data (msg : ByteArray) :
    (padData msg).data.toList = msg.data.toList ++ [128] ++ List.replicate (zeroPad msg.size) 0 ++ lenBytes msg.size := by
  rw [padData, foldl_push_data (fun a => ((msg.size * 8) >>> (8 * (15 - a)) % 256).toUInt8),
    foldl_push_data (fun _ => 0), ByteArray.data_push, Array.toList_push, List.map_const', List.length_range',
    ByteArray.size_push]
  rfl

theorem size_padData (msg : ByteArray) : (padData msg).size = msg.size + 1 + zeroPad msg.size + 16 := by
  rw [← ByteArray.size_data, ← Array.length_toList, padData_data]
  simp only [List.length_append, Array.length_toList, ByteArray.size_data, List.length_cons, List.length_nil,
    List.length_replicate, lenBytes, List.length_map, List.length_range']

theorem size_padData_mod (msg : ByteArray) : (padData msg).size % 128 = 0 := by
  have := size_padData msg
  unfold zeroPad at this
  omega

/-- The chain over `n + 1` blocks is one more compression of the chain over `n`. -/
theorem chain_succ (data : ByteArray) (n : ℕ) :
    chain data (n + 1) = Sha512.compress (chain data n) data (128 * n) := by
  rw [chain, chain, List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add,
    Nat.one_mul]

/-- Bit `t` of a byte string: bit `t % 8` of byte `t / 8` (`Hm96.bit`, `Binding.byteOf`). -/
def bitL (l : List UInt8) (t : ℕ) : Bool := (l.getD (t / 8) 0).toNat.testBit (t % 8)

theorem getD_digestOf (h : Array UInt64) {i : ℕ} (hi : i < 64) :
    (digestOf h).data.toList.getD i 0 = (h[i / 8]! >>> (8 * (7 - i % 8)).toUInt64).toUInt8 := by
  simp only [digestOf, Array.toList_ofFn, List.getD_eq_getElem?_getD, List.getElem?_ofFn, hi, dite_true,
    Option.getD_some]

/-- **The digest's bit order**: bit `t < 512` of `digestOf h` (byte-string order) is bit `wordBit t` of `h`'s words in
word order. -/
theorem bitL_digestOf (h : Array UInt64) {t : ℕ} (ht : t < 512) :
    bitL (digestOf h).data.toList t = wbit (toBV h) (wordBit t) := by
  rw [bitL, getD_digestOf h (by omega), wbit, getD_toBV]
  have e1 : wordBit t / 64 = t / 8 / 8 := by unfold wordBit; omega
  have e2 : wordBit t % 64 = 8 * (7 - t / 8 % 8) + t % 8 := by unfold wordBit; omega
  rw [e1, e2, BitVec.getLsbD, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.testBit_mod_two_pow,
    Nat.testBit_shiftRight]
  have e3 : (8 * (7 - t / 8 % 8)).toUInt64.toNat % 64 = 8 * (7 - t / 8 % 8) := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [e3]
  simp [show t % 8 < 8 by omega]

theorem blocks_pos (msg : ByteArray) : 0 < (padData msg).size / 128 := by
  have := size_padData msg
  have := size_padData_mod msg
  omega

/-- The chaining value before the last block of `msg`'s padded message. -/
def lastCv (msg : ByteArray) : List (BitVec 64) :=
  (List.range 8).map fun i => (chain (padData msg) ((padData msg).size / 128 - 1))[i]!.toBitVec

/-- The last block of `msg`'s padded message: its sixteen big-endian words. -/
def lastBlock (msg : ByteArray) : List (BitVec 64) :=
  (List.range 16).map fun i => (wordBE (padData msg) (128 * ((padData msg).size / 128 - 1)) i).toBitVec

/-- **SHA-512's digest bits are the last compression's**: bit `t < 512` of `Sha512.hash msg` (byte-string order) is bit
`wordBit t` of `compressV` of the chaining value before the last block and the last block. -/
theorem bitL_hash (msg : ByteArray) {t : ℕ} (ht : t < 512) :
    bitL (Sha512.hash msg).data.toList t = wbit (compressV (lastCv msg) (lastBlock msg)) (wordBit t) := by
  rw [hash_eq, bitL_digestOf _ ht, lastCv, lastBlock, ← compress_eq, ← chain_succ,
    Nat.sub_add_cancel (blocks_pos msg)]

/-! ## Message words -/

theorem getLsbD_step (x : UInt64) (b : UInt8) (p : ℕ) :
    (x <<< 8 ||| b.toUInt64).toBitVec.getLsbD p =
      (if p < 8 then b.toNat.testBit p else decide (p < 64) && x.toBitVec.getLsbD (p - 8)) := by
  rw [UInt64.toBitVec_or, BitVec.getLsbD_or, UInt64.toBitVec_shiftLeft, UInt8.toBitVec_toUInt64]
  simp only [BitVec.getLsbD_setWidth]
  rw [show UInt64.toBitVec 8 % 64 = 8#64 from rfl, BitVec.shiftLeft_eq', BitVec.getLsbD_shiftLeft]
  rw [show (8#64).toNat = 8 from rfl]
  have hb : b.toBitVec.getLsbD p = b.toNat.testBit p := rfl
  rw [hb]
  by_cases hp : p < 8
  · simp [hp, show p < 64 by omega]
  · have : b.toNat.testBit p = false :=
      Nat.testBit_lt_two_pow (lt_of_lt_of_le (UInt8.toNat_lt b) (Nat.pow_le_pow_right (by norm_num) (by omega)))
    simp [hp, this]

/-- `k ≤ 8` bytes `g 0, …, g (k-1)` read big-endian: bit `p` is bit `p % 8` of byte `k - 1 - p / 8`. -/
theorem getLsbD_foldBE (g : ℕ → UInt8) : ∀ k ≤ 8, ∀ p : ℕ,
    ((List.range' 0 k).foldl (fun (x : UInt64) j => x <<< 8 ||| (g j).toUInt64) 0).toBitVec.getLsbD p =
      (decide (p < 8 * k) && (g (k - 1 - p / 8)).toNat.testBit (p % 8))
  | 0, _, p => by simp
  | k + 1, hk, p => by
      rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul,
        getLsbD_step]
      by_cases hp : p < 8
      · simp only [hp, ↓reduceIte, show p < 8 * (k + 1) by omega, decide_true, Bool.true_and,
          show k + 1 - 1 - p / 8 = k by omega, Nat.mod_eq_of_lt hp]
      · simp only [hp, ↓reduceIte]
        rw [getLsbD_foldBE g k (by omega) (p - 8), show (p - 8) % 8 = p % 8 by omega,
          show k - 1 - (p - 8) / 8 = k + 1 - 1 - p / 8 by omega]
        by_cases h2 : p < 8 * (k + 1)
        · simp [h2, show p < 64 by omega, show p - 8 < 8 * k by omega]
        · simp [h2, show ¬ (p - 8 < 8 * k) by omega]

/-- Word `i` of the block at `off` of a byte list, big-endian. -/
def wordL (l : List UInt8) (off i : ℕ) : UInt64 :=
  (List.range' 0 8).foldl (fun x j => x <<< 8 ||| (l.getD (off + 8 * i + j) 0).toUInt64) 0

theorem get!_eq_getD (b : ByteArray) (k : ℕ) : b.get! k = b.data.toList.getD k 0 := by
  obtain ⟨bs⟩ := b
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, ByteArray.get!, getElem!_def]
  cases bs[k]? <;> rfl

theorem wordBE_eq_wordL (b : ByteArray) (off i : ℕ) : wordBE b off i = wordL b.data.toList off i := by
  simp only [wordBE, wordL, get!_eq_getD]

theorem getLsbD_wordL (l : List UInt8) (off i : ℕ) {p : ℕ} (hp : p < 64) :
    (wordL l off i).toBitVec.getLsbD p = (l.getD (off + 8 * i + (7 - p / 8)) 0).toNat.testBit (p % 8) := by
  rw [wordL, getLsbD_foldBE _ 8 le_rfl p]
  simp [hp]

theorem wordL_congr {l l' : List UInt8} {off off' i i' : ℕ}
    (h : ∀ j < 8, l.getD (off + 8 * i + j) 0 = l'.getD (off' + 8 * i' + j) 0) : wordL l off i = wordL l' off' i' := by
  unfold wordL
  refine List.foldl_ext _ _ _ fun x j hj => ?_
  rw [h j (by simp at hj; omega)]

/-! ## The compression reads only its block -/

theorem compress_congr (h : Array UInt64) {d d' : ByteArray} {off off' : ℕ}
    (hd : ∀ k < 128, d.get! (off + k) = d'.get! (off' + k)) : Sha512.compress h d off = Sha512.compress h d' off' := by
  have hw : (List.range' 0 16).foldl (fun w i => w.push (wordBE d off i)) (Array.mkEmpty 80) =
      (List.range' 0 16).foldl (fun w i => w.push (wordBE d' off' i)) (Array.mkEmpty 80) := by
    refine List.foldl_ext _ _ _ fun w i hi => ?_
    simp only [List.mem_range'_1] at hi
    rw [wordBE_eq_wordL, wordBE_eq_wordL, wordL_congr fun j hj => ?_]
    rw [← get!_eq_getD, ← get!_eq_getD, Nat.add_assoc, Nat.add_assoc, hd _ (by omega)]
  rw [compress_unfold, compress_unfold, hw]

theorem size_compress (h : Array UInt64) (data : ByteArray) (off : ℕ) : (Sha512.compress h data off).size = 8 := by
  rw [compress_unfold]
  rfl

theorem map_range_eq_toBV (w : Array UInt64) (hw : w.size = 8) :
    (List.range 8).map (fun i => w[i]!.toBitVec) = toBV w := by
  apply List.ext_getElem
  · simp [length_toBV, hw]
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_range, toBV, Array.getElem_toList]
    have : i < w.size := by simp at h1; omega
    simp [this]

end FlockSoundness.Discharge.Hm
