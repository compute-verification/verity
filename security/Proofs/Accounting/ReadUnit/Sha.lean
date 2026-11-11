/-
SHA-512 over message words as the read's generators build it (`sound_shaW`); the verifier's SHA-512 of a message after a
prefix of whole blocks as the same chain (`hash_chainW`: the prefix's chaining value, then one compression a block of the
rest and its padding, each block's words big-endian); and the words' bytes as the read's ports hold them in leaf order
(`wordsBytes_eq`, `wordsDigest_bytes`).
-/
module

public import Flock.ReadNet.Sha
public import Proofs.Accounting.ReadUnit.Net
public import Proofs.Flock.Soundness.Discharge.Hm.Slot
public import Proofs.Hash.Bytes
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- `n` compressions from the chaining value `cv` over the words `ws`, sixteen a block. -/
def chainW : ℕ → List (BitVec 64) → List (BitVec 64) → List (BitVec 64)
  | 0, cv, _ => cv
  | n + 1, cv, ws => chainW n (compressV cv (ws.take 16)) (ws.drop 16)

/-- Words' bytes, each word big-endian: a digest's 64 bytes from its eight words. -/
def bytesBE (ws : List (BitVec 64)) : Hash.Msg := ws.flatMap fun w => Hash.be 8 w.toNat

/-- A byte string's 64-bit words, each big-endian. -/
def wordsBE (l : List UInt8) : List (BitVec 64) := (List.range (l.length / 8)).map fun i => (wordL l 0 i).toBitVec

/-- `G` is the verifier's SHA-512 (`Flock.Sha512.hash`) on every message. -/
def IsSha512 (G : Hash.Msg → Verity.Digest) : Prop := ∀ m, (G m).bytes = (Flock.Sha512.hash ⟨m.toArray⟩).data.toList

/-- The blocks of a message of `total` bytes once padded. -/
def blocks (total : ℕ) : ℕ := (total + 17 + zeroPad total) / 128

/-! ## The block generator -/

/-- **The block generator computes the chain**: whenever its rows hold, `shaW`'s output words are `chainW` of its
chaining value's and message's. -/
theorem sound_shaW : ∀ (n : ℕ) (cv msg : List Word), Sound (Flock.ReadNet.shaW n cv msg) fun z out =>
    cv.length = 8 → Words64 cv → msg.length = 16 * n → Words64 msg →
      out.length = 8 ∧ Words64 out ∧ out.map (bv z) = chainW n (cv.map (bv z)) (msg.map (bv z))
  | 0, _, _ => Sound.pure' fun _ hl hw _ _ => ⟨hl, hw, rfl⟩
  | n + 1, cv, msg => by
    unfold Flock.ReadNet.shaW
    refine Sound.bind' (sound_compress cv (msg.take 16)) fun st =>
      Sound.mono (sound_shaW n st (msg.drop 16)) fun z out h hst hcl hcw hml hmw => ?_
    obtain ⟨s1, s2⟩ := hst hcl hcw (by rw [List.length_take, hml]; omega) (hmw.take 16)
    have hsl : st.length = 8 := by
      rw [length_of_map_bv s2, length_compressV (cv := cv.map (bv z)) (by rw [List.length_map, hcl])]
    obtain ⟨o1, o2, o3⟩ := h hsl s1 (by rw [List.length_drop, hml]; omega) (hmw.drop 16)
    refine ⟨o1, o2, ?_⟩
    rw [o3, s2]
    simp only [chainW, List.map_take, List.map_drop]

/-! ## The padding and the chain -/

theorem padTail_bytes (total : ℕ) :
    Flock.ReadNet.padTail total = ([128] : List UInt8) ++ List.replicate (zeroPad total) (0 : UInt8) ++ lenBytes total :=
  rfl

theorem length_padTail (total : ℕ) : (Flock.ReadNet.padTail total).length = zeroPad total + 17 := by
  simp only [padTail_bytes, lenBytes, List.length_append, List.length_cons, List.length_nil, List.length_replicate,
    List.length_map, List.length_range']
  omega

theorem blocks_mul (total : ℕ) : 128 * blocks total = total + (Flock.ReadNet.padTail total).length := by
  rw [length_padTail, blocks]
  unfold zeroPad
  omega

/-- The chain over the first `n` blocks reads only those blocks. -/
theorem chain_prefix {d d' : ByteArray} (n : ℕ) (h : ∀ i < 128 * n, d.get! i = d'.get! i) :
    chain d n = chain d' n := by
  induction n with
  | zero => simp only [chain, List.range'_zero, List.foldl_nil]
  | succ n ih =>
      rw [chain_succ, chain_succ, ih fun i hi => h i (by omega)]
      exact compress_congr _ fun k hk => h _ (by omega)

theorem size_chain (d : ByteArray) {k : ℕ} (hk : 0 < k) : (chain d k).size = 8 := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_lt hk
  rw [Nat.zero_add, chain_succ]
  exact size_compress _ _ _

theorem chainV_toBV (d : ByteArray) {k : ℕ} (hk : 0 < k) : chainV d k = toBV (chain d k) := by
  rw [chainV, map_range_eq_toBV _ (size_chain d hk)]

/-- **The words of a padded message's blocks are its chain's**: if every block from `b` on is `W`'s next sixteen words,
`chainW` from the chaining value after `b` blocks is the chaining value after `b + k`. -/
theorem chainW_eq_chainV (P : ByteArray) : ∀ (k b : ℕ) (W : List (BitVec 64)),
    (∀ j < k, (W.drop (16 * j)).take 16 = blockV P (b + j)) → chainW k (chainV P b) W = chainV P (b + k)
  | 0, b, _, _ => rfl
  | k + 1, b, W, h => by
    have h0 : W.take 16 = blockV P b := by simpa using h 0 (by omega)
    have ih := chainW_eq_chainV P k (b + 1) (W.drop 16) fun j hj => by
      have e : (W.drop 16).drop (16 * j) = W.drop (16 * (j + 1)) := by
        rw [List.drop_drop]
        congr 1
        omega
      rw [e, h (j + 1) (by omega)]
      congr 1
      omega
    simp only [chainW]
    rw [h0, ← chainV_succ, ih]
    congr 1
    omega

theorem take_drop_map_range {α : Type} (g : ℕ → α) {n a k : ℕ} (h : a + k ≤ n) :
    (((List.range n).map g).drop a).take k = (List.range k).map fun i => g (a + i) := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_drop, List.length_map, List.length_range]
    omega
  · intro i h1 h2
    simp only [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range]

/-- Block `np + j` of `pre ‖ rest`, for `pre` of `np` blocks, is the sixteen words of `rest` from word `16 j`. -/
theorem blockV_eq {P : ByteArray} {pre rest : List UInt8} (hP : P.data.toList = pre ++ rest) {np : ℕ}
    (hpre : pre.length = 128 * np) {j : ℕ} (hj : 16 * j + 16 ≤ rest.length / 8) :
    ((wordsBE rest).drop (16 * j)).take 16 = blockV P (np + j) := by
  rw [wordsBE, take_drop_map_range _ hj, blockV]
  refine List.map_congr_left fun i hi => ?_
  rw [List.mem_range] at hi
  try dsimp only
  rw [wordBE_eq_wordL, hP]
  congr 1
  refine wordL_congr fun t ht => ?_
  rw [List.getD_append_right _ _ _ _ (by omega)]
  congr 1
  omega

/-! ## Bytes of words -/

theorem getD_flatMap8 {α : Type} (f : α → List UInt8) (hf : ∀ a, (f a).length = 8) (d : α) :
    ∀ (l : List α) (i : ℕ), i < 8 * l.length → (l.flatMap f).getD i 0 = (f (l.getD (i / 8) d)).getD (i % 8) 0
  | [], i, h => by simp at h
  | a :: l, i, h => by
    rw [List.flatMap_cons]
    by_cases hi : i < 8
    · rw [List.getD_append _ _ _ _ (by rw [hf]; exact hi), Nat.div_eq_of_lt hi, Nat.mod_eq_of_lt hi,
        List.getD_cons_zero]
    · rw [List.getD_append_right _ _ _ _ (by rw [hf]; omega), hf,
        getD_flatMap8 f hf d l (i - 8) (by rw [List.length_cons] at h; omega),
        show i / 8 = (i - 8) / 8 + 1 by omega, show i % 8 = (i - 8) % 8 by omega, List.getD_cons_succ]

theorem getD_be8 (n : ℕ) {m : ℕ} (hm : m < 8) : (Hash.be 8 n).getD m 0 = (n / 256 ^ (7 - m) % 256).toUInt8 := by
  simp [Hash.be, List.getD_eq_getElem?_getD, hm]

theorem length_bytesBE : ∀ ws : List (BitVec 64), (bytesBE ws).length = 8 * ws.length
  | [] => rfl
  | w :: ws => by
    have := length_bytesBE ws
    simp only [bytesBE, List.flatMap_cons, List.length_append, Hash.BindingProofs.be_length, List.length_cons] at this ⊢
    omega

theorem getD_bytesBE {ws : List (BitVec 64)} {i : ℕ} (hi : i < 8 * ws.length) :
    (bytesBE ws).getD i 0 = ((ws.getD (i / 8) 0).toNat / 256 ^ (7 - i % 8) % 256).toUInt8 := by
  rw [bytesBE, getD_flatMap8 (fun w : BitVec 64 => Hash.be 8 w.toNat) (fun w => Hash.BindingProofs.be_length 8 _) 0
    ws i hi, getD_be8 _ (Nat.mod_lt _ (by norm_num))]

theorem toUInt8_shift (w : UInt64) {k : ℕ} (hk : k < 8) :
    (w >>> (8 * (7 - k)).toUInt64).toUInt8 = (w.toBitVec.toNat / 256 ^ (7 - k) % 256).toUInt8 := by
  have e3 : (8 * (7 - k)).toUInt64.toNat % 64 = 8 * (7 - k) := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  have e4 : (2 : ℕ) ^ (8 * (7 - k)) = 256 ^ (7 - k) := by rw [pow_mul]; norm_num
  apply UInt8.toNat_inj.1
  rw [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, e3, Nat.shiftRight_eq_div_pow, e4, Nat.toUInt8_eq,
    UInt8.toNat_ofNat']
  simp only [Nat.reducePow, Nat.mod_mod]
  try rfl

/-- **A chaining value's digest is its words' big-endian bytes.** -/
theorem digestOf_eq_bytesBE (h : Array UInt64) (hs : h.size = 8) :
    (FlockSoundness.Discharge.Hm.digestOf h).data.toList = bytesBE (toBV h) := by
  have hl : (toBV h).length = 8 := by simp [toBV, hs]
  apply List.ext_getElem
  · rw [length_bytesBE, hl]
    simp [FlockSoundness.Discharge.Hm.digestOf]
  · intro i h1 h2
    have hi : i < 64 := by rw [length_bytesBE, hl] at h2; omega
    rw [← List.getD_eq_getElem _ 0 h1, ← List.getD_eq_getElem _ 0 h2, getD_digestOf h hi,
      getD_bytesBE (ws := toBV h) (by rw [hl]; omega), getD_toBV]
    exact toUInt8_shift _ (Nat.mod_lt _ (by norm_num))

/-! ## SHA-512 after a prefix of whole blocks -/

/-- **The verifier's SHA-512 after a prefix of whole blocks is the chain from the prefix's chaining value**: the digest
of `pre ‖ data`, for `pre` of `np` blocks, is the chain from `pre`'s chaining value over the words of `data` and its
padding, as big-endian bytes. -/
theorem hash_chainW {G : Hash.Msg → Verity.Digest} (hG : IsSha512 G) {pre data : List UInt8} {np : ℕ}
    (hpre : pre.length = 128 * np) :
    (G (pre ++ data)).bytes =
      bytesBE (chainW (blocks (pre.length + data.length) - np) (chainV ⟨pre.toArray⟩ np)
        (wordsBE (data ++ Flock.ReadNet.padTail (pre.length + data.length)))) := by
  have hd : (⟨(pre ++ data).toArray⟩ : ByteArray).data.toList = pre ++ data := by simp
  have hs : (⟨(pre ++ data).toArray⟩ : ByteArray).size = pre.length + data.length := by simp [ByteArray.size]
  have hP := padData_data (⟨(pre ++ data).toArray⟩ : ByteArray)
  have hPs := size_padData (⟨(pre ++ data).toArray⟩ : ByteArray)
  have hpos := blocks_pos (⟨(pre ++ data).toArray⟩ : ByteArray)
  rw [hd, hs] at hP
  rw [hs] at hPs
  rw [hG (pre ++ data), hash_eq]
  generalize padData (⟨(pre ++ data).toArray⟩ : ByteArray) = P at hP hPs hpos ⊢
  have hP' : P.data.toList = pre ++ (data ++ Flock.ReadNet.padTail (pre.length + data.length)) := by
    rw [hP, padTail_bytes]
    simp only [List.append_assoc]
  have hb : P.size / 128 = blocks (pre.length + data.length) := by
    rw [hPs, blocks]
    omega
  rw [hb] at hpos ⊢
  have hm := blocks_mul (pre.length + data.length)
  have hnp : np ≤ blocks (pre.length + data.length) := by omega
  have hrest : (data ++ Flock.ReadNet.padTail (pre.length + data.length)).length / 8 =
      16 * (blocks (pre.length + data.length) - np) := by
    rw [List.length_append]
    omega
  have hc : chain P np = chain ⟨pre.toArray⟩ np := chain_prefix np fun i hi => by
    rw [get!_eq_getD, get!_eq_getD, hP', List.getD_append _ _ _ _ (by omega)]
  have hpreP : chainV P np = chainV ⟨pre.toArray⟩ np := by
    unfold chainV
    rw [hc]
  rw [← hpreP, chainW_eq_chainV P (blocks (pre.length + data.length) - np) np
      (wordsBE (data ++ Flock.ReadNet.padTail (pre.length + data.length)))
      fun j hj => blockV_eq hP' hpre (by rw [hrest]; omega),
    Nat.add_sub_cancel' hnp, chainV_toBV P hpos, digestOf_eq_bytesBE _ (size_chain P hpos)]

/-- The verifier's SHA-512 is a hash `IsSha512` describes. -/
theorem isSha512_sha512 : IsSha512 Verity.sha512 := by
  intro m
  have hs : (Flock.Sha512.hash ⟨m.toArray⟩).data.size = 64 := by
    rw [hash_eq]
    simp [FlockSoundness.Discharge.Hm.digestOf]
  apply List.ext_getElem
  · simp [Verity.Digest.bytes, hs]
  · intro i h1 h2
    have hi : i < (Flock.Sha512.hash ⟨m.toArray⟩).data.size := by
      rw [hs]
      simpa [Verity.Digest.bytes] using h1
    simp only [Verity.Digest.bytes, List.getElem_ofFn, Array.getElem_toList]
    show (Flock.Sha512.hash ⟨m.toArray⟩).data.getD i 0 = _
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]

/-! ## Words in leaf order -/

/-- Bits `[a, a + k)` of a number's bits are its `k` bits from bit `a`. -/
theorem num_chunk (l : List Bool) {a k : ℕ} (h : a + k ≤ l.length) :
    num ((l.drop a).take k) = num l / 2 ^ a % 2 ^ k := by
  have e := num_append (l.take a) (l.drop a)
  rw [List.take_append_drop, List.length_take, Nat.min_eq_left (by omega)] at e
  have e2 := num_append ((l.drop a).take k) ((l.drop a).drop k)
  rw [List.take_append_drop, List.length_take, List.length_drop, Nat.min_eq_left (by omega)] at e2
  have h1 := num_lt (l.take a)
  rw [List.length_take, Nat.min_eq_left (by omega)] at h1
  have h2 := num_lt ((l.drop a).take k)
  rw [List.length_take, List.length_drop, Nat.min_eq_left (by omega)] at h2
  rw [e, e2, Nat.add_mul_div_left _ _ (Nat.two_pow_pos a), Nat.div_eq_of_lt h1, Nat.zero_add,
    Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h2]

theorem num_evs_valN (z : Var → Bool) : ∀ w : Word, num (evs z w) = valN z w
  | [] => rfl
  | f :: w => by
    show (ev z f).toNat + 2 * num (evs z w) = (ev z f).toNat + 2 * valN z w
    rw [num_evs_valN z w]

theorem valN_lt (z : Var → Bool) : ∀ w : Word, valN z w < 2 ^ w.length
  | [] => by simp [valN]
  | f :: w => by
    have := valN_lt z w
    have : (ev z f).toNat ≤ 1 := Bool.toNat_le _
    simp only [valN, List.length_cons, Nat.pow_succ]
    omega

theorem num_evs_word (z : Var → Bool) {w : Word} (hw : w.length = 64) : num (evs z w) = (bv z w).toNat := by
  rw [num_evs_valN, bv, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by rw [← hw]; exact valN_lt z w)]

theorem bytesOf_append64 {A : List Bool} (B : List Bool) (hA : A.length = 64) :
    ReadUnit.bytesOf (A ++ B) =
      (List.range 8).map (fun m => UInt8.ofNat (num ((A.drop (8 * (7 - m))).take 8))) ++ ReadUnit.bytesOf B := by
  have hn : (A ++ B).length / 64 = B.length / 64 + 1 := by rw [List.length_append, hA]; omega
  rw [ReadUnit.bytesOf, ReadUnit.bytesOf, hn, List.range_succ_eq_map (n := B.length / 64), List.flatMap_cons,
    List.flatMap_map]
  congr 1
  · refine List.map_congr_left fun m hm => ?_
    rw [List.mem_range] at hm
    have e : ((A ++ B).drop (64 * 0 + 8 * (7 - m))).take 8 = (A.drop (8 * (7 - m))).take 8 := by
      rw [Nat.mul_zero, Nat.zero_add, List.drop_append_of_le_length (by omega),
        List.take_append_of_le_length (by rw [List.length_drop]; omega)]
    simp only [e]
  · congr 1
    funext w
    refine List.map_congr_left fun m hm => ?_
    rw [List.mem_range] at hm
    have e : (A ++ B).drop (64 * (w + 1) + 8 * (7 - m)) = B.drop (64 * w + 8 * (7 - m)) := by
      rw [List.drop_append, List.drop_eq_nil_of_le (by omega), List.nil_append, hA]
      congr 1
      omega
    simp only [Nat.succ_eq_add_one, e]

theorem be8_word (z : Var → Bool) {w : Word} (hw : w.length = 64) :
    (List.range 8).map (fun m => UInt8.ofNat (num (((evs z w).drop (8 * (7 - m))).take 8))) =
      Hash.be 8 (bv z w).toNat := by
  rw [Hash.be]
  refine List.map_congr_left fun m hm => ?_
  rw [List.mem_range] at hm
  have hl : 8 * (7 - m) + 8 ≤ (evs z w).length := by simp only [evs, List.length_map, hw]; omega
  have e1 : (2 : ℕ) ^ (8 * (7 - m)) = 256 ^ (7 - m) := by rw [pow_mul]; norm_num
  simp only [num_chunk _ hl, num_evs_word z hw, e1, Nat.toUInt8_eq, show 8 - 1 - m = 7 - m by omega]
  try rfl

/-- The bytes words hold in leaf order are their values' big-endian bytes. -/
theorem wordsBytes_eq (z : Var → Bool) {n : ℕ} {ws : List Word} (h : Shaped n ws) :
    wordsBytes z ws = bytesBE (ws.map (bv z)) := by
  induction ws generalizing n with
  | nil => simp [wordsBytes, ReadUnit.bytesOf, evs, bytesBE]
  | cons w ws ih =>
    have hw : w.length = 64 := h.2 w (by simp)
    have ih' := ih (n := n - 1) ⟨by rw [← h.1, List.length_cons]; omega, fun x hx => h.2 x (by simp [hx])⟩
    have e : wordsBytes z (w :: ws) = ReadUnit.bytesOf (evs z w ++ evs z ws.flatten) := by
      simp only [wordsBytes, evs, List.flatten_cons, List.map_append]
    rw [e, bytesOf_append64 _ (by simp only [evs, List.length_map, hw]),
      show ReadUnit.bytesOf (evs z ws.flatten) = wordsBytes z ws from rfl, ih', be8_word z hw]
    simp only [bytesBE, List.map_cons, List.flatMap_cons]

/-- The digest 8 words hold in leaf order has their values' big-endian bytes. -/
theorem wordsDigest_bytes (z : Var → Bool) {ws : List Word} (h : Shaped 8 ws) :
    (wordsDigest z ws).bytes = bytesBE (ws.map (bv z)) := by
  rw [← wordsBytes_eq z h]
  have hl := length_wordsBytes z h
  apply List.ext_getElem
  · simp [Verity.Digest.bytes, hl]
  · intro i h1 h2
    simp only [Verity.Digest.bytes, List.getElem_ofFn, wordsDigest, ReadUnit.digestOf]
    exact List.getD_eq_getElem _ _ h2

/-! ## Bits of big-endian bytes -/

theorem testBit_byte (v : BitVec 64) {k b : ℕ} (hb : b < 8) :
    ((v.toNat / 256 ^ (7 - k) % 256).toUInt8).toNat.testBit b = v.getLsbD (8 * (7 - k) + b) := by
  have e : (256 : ℕ) ^ (7 - k) = 2 ^ (8 * (7 - k)) := by rw [pow_mul]; norm_num
  rw [Nat.toUInt8_eq, UInt8.toNat_ofNat', e, ← Nat.shiftRight_eq_div_pow, show (256 : ℕ) = 2 ^ 8 from rfl, Nat.mod_mod,
    Nat.testBit_mod_two_pow, Nat.testBit_shiftRight]
  simp [hb]
  exact BitVec.testBit_toNat _

theorem bitL_bytesBE {vs : List (BitVec 64)} {t : ℕ} (ht : t < 64 * vs.length) :
    bitL (bytesBE vs) t = (vs.getD (t / 64) 0).getLsbD (8 * (7 - t / 8 % 8) + t % 8) := by
  rw [bitL, getD_bytesBE (by omega), testBit_byte _ (Nat.mod_lt _ (by norm_num)), show t / 8 / 8 = t / 64 by omega]

theorem wordL_bytesBE {vs : List (BitVec 64)} (pad : List UInt8) {i : ℕ} (hi : i < vs.length) :
    (wordL (bytesBE vs ++ pad) 0 i).toBitVec = vs[i] := by
  apply BitVec.eq_of_getLsbD_eq
  intro p hp
  have hb : 0 + 8 * i + (7 - p / 8) < (bytesBE vs).length := by rw [length_bytesBE]; omega
  rw [getLsbD_wordL _ _ _ hp, List.getD_append _ _ _ _ hb, getD_bytesBE (by rw [length_bytesBE] at hb; exact hb),
    testBit_byte _ (Nat.mod_lt _ (by norm_num)), show (0 + 8 * i + (7 - p / 8)) / 8 = i by omega,
    show 8 * (7 - (0 + 8 * i + (7 - p / 8)) % 8) + p % 8 = p by omega, List.getD_eq_getElem _ _ hi]

theorem length_wordsBE (l : List UInt8) : (wordsBE l).length = l.length / 8 := by simp [wordsBE]

/-- The words of words' bytes and a tail are the words and the tail's words. -/
theorem wordsBE_append (vs : List (BitVec 64)) (pad : List UInt8) :
    wordsBE (bytesBE vs ++ pad) = vs ++ wordsBE pad := by
  apply List.ext_getElem
  · simp only [length_wordsBE, List.length_append, length_bytesBE]
    omega
  · intro i h1 h2
    simp only [wordsBE, List.getElem_map, List.getElem_range]
    by_cases hi : i < vs.length
    · rw [List.getElem_append_left hi]
      exact wordL_bytesBE pad hi
    · rw [List.getElem_append_right (by omega)]
      simp only [List.getElem_map, List.getElem_range]
      congr 1
      refine wordL_congr fun t ht => ?_
      rw [List.getD_append_right _ _ _ _ (by rw [length_bytesBE]; omega), length_bytesBE]
      congr 1
      omega

/-- Bit `t` of the bytes words hold is the form at `wordBit t` of their bits. -/
theorem ev_getD_wordBit (z : Var → Bool) {ws : List Word} (hw : Words64 ws) {t : ℕ} (ht : t < 64 * ws.length) :
    ev z (ws.flatten.getD (wordBit t) Lin.zero) = bitL (bytesBE (ws.map (bv z))) t := by
  have h1 : ev z (ws.flatten.getD (wordBit t) Lin.zero) = FlockSoundness.Discharge.Hm.bit z ws.flatten (wordBit t) := by
    rw [FlockSoundness.Discharge.Hm.bit, List.getD_eq_getElem?_getD]
  rw [h1, bit_flatten z hw, bitL_bytesBE (by rw [List.length_map]; exact ht),
    show wordBit t / 64 = t / 64 by unfold wordBit; omega,
    show wordBit t % 64 = 8 * (7 - t / 8 % 8) + t % 8 by unfold wordBit; omega]
  congr 1
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases ws[t / 64]? <;> rfl

/-! ## The constants a part's message carries -/

theorem chain_one (pre : ByteArray) : chain pre 1 = Flock.Sha512.compress Flock.Sha512.iv pre 0 := by
  rw [chain_succ, Nat.mul_zero]
  rfl

theorem midstate_eq (pre : ByteArray) : Flock.ReadNet.midstate pre = (chainV pre 1).map constW := by
  rw [Flock.ReadNet.midstate, chainV, List.map_map, chain_one]
  rfl

theorem map_bv_midstate (z : Var → Bool) (pre : ByteArray) :
    (Flock.ReadNet.midstate pre).map (bv z) = chainV pre 1 := by
  rw [midstate_eq, map_bv_constW]

theorem words64_midstate (pre : ByteArray) : Words64 (Flock.ReadNet.midstate pre) := by
  rw [midstate_eq]
  exact words64_constW _

theorem length_midstate (pre : ByteArray) : (Flock.ReadNet.midstate pre).length = 8 := by
  simp [Flock.ReadNet.midstate]

theorem constWords_eq (l : List UInt8) : Flock.ReadNet.constWords l = (wordsBE l).map constW := by
  rw [Flock.ReadNet.constWords, wordsBE, List.map_map]
  rfl

theorem map_bv_constWords (z : Var → Bool) (l : List UInt8) : (Flock.ReadNet.constWords l).map (bv z) = wordsBE l := by
  rw [constWords_eq, map_bv_constW]

theorem words64_constWords (l : List UInt8) : Words64 (Flock.ReadNet.constWords l) := by
  rw [constWords_eq]
  exact words64_constW _

theorem length_constWords (l : List UInt8) : (Flock.ReadNet.constWords l).length = l.length / 8 := by
  simp [Flock.ReadNet.constWords]

/-- A chaining value after one block of `pre` and its bytes' chain is the hash of `pre ‖ data`. -/
theorem hash_mid {G : Hash.Msg → Verity.Digest} (hG : IsSha512 G) {pre : ByteArray} (hpre : pre.size = 128)
    (data : List UInt8) :
    (G (pre.data.toList ++ data)).bytes =
      bytesBE (chainW (blocks (128 + data.length) - 1) (chainV pre 1)
        (wordsBE (data ++ Flock.ReadNet.padTail (128 + data.length)))) := by
  have hl : pre.data.toList.length = 128 := by rw [Array.length_toList, ByteArray.size_data, hpre]
  have hb : (⟨pre.data.toList.toArray⟩ : ByteArray) = pre := ByteArray.ext (by simp)
  rw [hash_chainW hG (np := 1) (by omega), hb, hl]

/-! ## Byte strings of the code -/

theorem pad_data (b : ByteArray) (n : ℕ) : (Flock.Hm96.pad b n).data.toList = Hash.padTo n b.data.toList := by
  rw [Flock.Hm96.pad, ByteArray.toList_data_append, Hash.padTo]
  simp only [Array.toList_replicate, Array.length_toList, ByteArray.size_data]
  congr 2
  omega

theorem push_zero_data (s : String) : (s.toUTF8.push 0).data.toList = Hash.cstr s := by
  simp [Hash.cstr, Hash.utf8, ByteArray.data_push]

theorem be64_data (x : ℕ) : (Flock.HmRow.be64 x).data.toList = Hash.be 8 x := by
  simp [Flock.HmRow.be64, Hash.be]

theorem length_padTo {n : ℕ} {m : List UInt8} (h : m.length ≤ n) : (Hash.padTo n m).length = n := by
  simp only [Hash.padTo, List.length_append, List.length_replicate]
  omega

theorem size_of_data {b : ByteArray} {l : List UInt8} (h : b.data.toList = l) : b.size = l.length := by
  rw [← h, Array.length_toList, ByteArray.size_data]

end Accounting.ReadUnit.Gates
