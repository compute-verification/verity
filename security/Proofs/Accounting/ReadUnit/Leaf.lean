/-
Part B's leaf, hm96-sha512's leaf generator against `Hash.Hm96`: on a digest `x` and a salt `y`, both in leaf order,
its first 16 words hold the commit string `b ‖ c` (`commitString_eq`: `b = x ⊕ M·y` bit by bit, `bytes_b`; `c` the
salt's SHA-512) and its last eight the leaf `SHA-512(leafPrefix(key) ‖ b ‖ c)`, under the pinned key (`leaf_spec`,
`Gates.Hm96LeafSpec`).
-/
module

public import Flock.ReadNet.Leaf
public import Proofs.Accounting.ReadUnit.Sha
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
import all Init.Data.String.Defs

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- The pinned hm96 key's bytes. -/
def pinnedKey : Hash.Msg := Flock.Hm96.Default512.key.data.toList

/-! ## Bits and the mask -/

theorem hm96Bit_eq (s : List UInt8) (t : ℕ) : Hash.Hm96.bit s t = bitL s t := by
  rw [Hash.Hm96.bit, bitL, Nat.testBit_eq_decide_div_mod_eq]
  rcases Nat.mod_two_eq_zero_or_one ((s.getD (t / 8) 0).toNat / 2 ^ (t % 8)) with h | h <;> rw [h] <;> rfl

theorem flockBit_eq (b : ByteArray) (t : ℕ) : Flock.Hm96.bit b t = bitL b.data.toList t := by
  rw [Flock.Hm96.bit, bitL, get!_eq_getD, Nat.shiftRight_eq_div_pow, Nat.testBit_eq_decide_div_mod_eq]
  rcases Nat.mod_two_eq_zero_or_one ((b.data.toList.getD (t / 8) 0).toNat / 2 ^ (t % 8)) with h | h <;> rw [h] <;> rfl

theorem foldl_filter_xor (p f : ℕ → Bool) : ∀ (l : List ℕ) (c : Bool),
    (l.filter p).foldl (fun acc j => xor acc (f j)) c = l.foldl (fun acc j => acc ^^ (p j && f j)) c
  | [], _ => rfl
  | j :: l, c => by
    cases hp : p j
    · rw [List.filter_cons_of_neg (by simp [hp])]
      simp only [List.foldl_cons, hp, Bool.false_and, Bool.xor_false]
      exact foldl_filter_xor p f l c
    · rw [List.filter_cons_of_pos hp]
      simp only [List.foldl_cons, hp, Bool.true_and]
      exact foldl_filter_xor p f l _

theorem foldl_congr_mem {f g : Bool → ℕ → Bool} :
    ∀ {l : List ℕ}, (∀ acc j, j ∈ l → f acc j = g acc j) → ∀ c, l.foldl f c = l.foldl g c
  | [], _, _ => rfl
  | j :: l, h, c => by
    rw [List.foldl_cons, List.foldl_cons, h c j (by simp)]
    exact foldl_congr_mem (fun acc i hi => h acc i (by simp [hi])) _

/-- `M·y`'s bit `i`, as a sum in order. -/
theorem maskBit_eq (key y : Hash.Msg) (i : ℕ) :
    Hash.Hm96.maskBit key y i = (List.range 1536).foldl (fun acc j => acc ^^ (bitL y j && bitL key (i + j))) false := by
  rw [Hash.Hm96.maskBit, show 8 * Hash.Hm96.SALT_BYTES = 1536 from rfl, foldl_filter_xor]
  simp only [hm96Bit_eq]

theorem foldl_pow_testBit (f : ℕ → Bool) : ∀ n : ℕ,
    (List.range n).foldl (fun acc s => acc + if f s then 2 ^ s else 0) 0 < 2 ^ n ∧
      ∀ i < n, ((List.range n).foldl (fun acc s => acc + if f s then 2 ^ s else 0) 0).testBit i = f i
  | 0 => ⟨by simp, fun i h => absurd h (by omega)⟩
  | n + 1 => by
    obtain ⟨hlt, hbit⟩ := foldl_pow_testBit f n
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    try dsimp only
    generalize (List.range n).foldl (fun acc s => acc + if f s then 2 ^ s else 0) 0 = F at hlt hbit ⊢
    refine ⟨by split <;> rw [Nat.pow_succ] <;> omega, fun i hi => ?_⟩
    by_cases hin : i < n
    · cases hf : f n
      · rw [ite_eq_right (by simp), Nat.add_zero]
        exact hbit i hin
      · rw [ite_eq_left rfl, Nat.add_comm, Nat.testBit_two_pow_add_gt hin]
        exact hbit i hin
    · have hni : i = n := by omega
      rw [hni]
      cases hf : f n
      · rw [ite_eq_right (by simp), Nat.add_zero]
        exact Nat.testBit_lt_two_pow hlt
      · rw [ite_eq_left rfl, Nat.add_comm, Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow hlt]
        rfl

/-- Bit `s` of `M·y`'s byte `p` is `M·y`'s bit `8 p + s`. -/
theorem testBit_mask (key y : Hash.Msg) {p s : ℕ} (hp : p < 64) (hs : s < 8) :
    ((Hash.Hm96.mask key y).getD p 0).toNat.testBit s = Hash.Hm96.maskBit key y (8 * p + s) := by
  obtain ⟨hlt, hbit⟩ := foldl_pow_testBit (fun s => Hash.Hm96.maskBit key y (8 * p + s)) 8
  rw [Hash.Hm96.mask, List.getD_eq_getElem _ _ (by simpa using hp)]
  simp only [List.getElem_map, List.getElem_range]
  rw [Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.mod_eq_of_lt hlt]
  exact hbit s hs

theorem bitL_bytes (d : Verity.Digest) {i s : ℕ} (hi : i < 64) (hs : s < 8) :
    bitL d.bytes (8 * i + s) = (d ⟨i, hi⟩).toNat.testBit s := by
  rw [bitL, show (8 * i + s) / 8 = i by omega, show (8 * i + s) % 8 = s by omega,
    List.getD_eq_getElem _ _ (by simpa [Verity.Digest.bytes] using hi)]
  simp only [Verity.Digest.bytes, List.getElem_ofFn]

/-! ## `b = x ⊕ M·y` -/

/-- Word `w` of `b` as the leaf generator builds it, before it commits it. -/
def bWord (x y : List Word) (w : ℕ) : Word :=
  (List.range 64).map fun p => maskBit Flock.Hm96.Default512.key
    ((List.range 1536).map fun j => y.flatten.getD (wordBit j) Lin.zero) (wordBit (64 * w + p))
    (x.flatten.getD (wordBit (wordBit (64 * w + p))) Lin.zero)

/-- Bit `i` of `b` as the leaf generator's net computes it: the digest's bit `i` and `M·y`'s. -/
theorem ev_bi (z : Var → Bool) {x y : List Word} (hx : Shaped 8 x) (hy : Shaped 24 y) {i : ℕ} (hi : i < 512) :
    ev z (maskBit Flock.Hm96.Default512.key ((List.range 1536).map fun j => y.flatten.getD (wordBit j) Lin.zero) i
      (x.flatten.getD (wordBit i) Lin.zero)) =
      (bitL (wordsDigest z x).bytes i ^^ Hash.Hm96.maskBit pinnedKey (wordsBytes z y) i) := by
  rw [maskBit_hom (ev z) (ev_xor z), maskBit_eq, ev_getD_wordBit z hx.2 (by rw [hx.1]; omega),
    wordsDigest_bytes z hx, wordsBytes_eq z hy]
  refine Bool.xor_right_inj.2 ?_
  refine foldl_congr_mem (fun acc j hj => ?_) false
  rw [List.mem_range] at hj
  try dsimp only
  rw [List.getD_eq_getElem _ _ (by simpa using hj)]
  simp only [List.getElem_map, List.getElem_range]
  rw [ev_getD_wordBit z hy.2 (by rw [hy.1]; omega), flockBit_eq, Bool.and_comm]
  rfl

theorem bytesBE_append (a b : List (BitVec 64)) : bytesBE (a ++ b) = bytesBE a ++ bytesBE b := List.flatMap_append

/-- **`b`'s bytes are `x ⊕ M·y`.** -/
theorem bytes_b (z : Var → Bool) {x y : List Word} (hx : Shaped 8 x) (hy : Shaped 24 y) {b : List Word}
    (hb : b.map (bv z) = (List.range 8).map fun w => bv z (bWord x y w)) :
    bytesBE (b.map (bv z)) =
      List.ofFn fun i : Fin 64 => wordsDigest z x i ^^^ (Hash.Hm96.mask pinnedKey (wordsBytes z y)).getD i 0 := by
  apply List.ext_getElem
  · simp [length_bytesBE, hb]
  · intro i h1 h2
    have hi : i < 64 := by simpa using h2
    apply UInt8.toNat_inj.1
    apply Nat.eq_of_testBit_eq
    intro s
    by_cases hs : s < 8
    · rw [← List.getD_eq_getElem _ 0 h1, getD_bytesBE (ws := b.map (bv z)) (by rw [hb]; simp; omega),
        testBit_byte _ hs, List.getElem_ofFn, UInt8.toNat_xor, Nat.testBit_xor, testBit_mask _ _ hi hs, hb,
        List.getD_eq_getElem _ _ (by simp; omega)]
      simp only [List.getElem_map, List.getElem_range]
      rw [getLsbD_bv z (by simp [bWord]), bWord]
      simp only [FlockSoundness.Discharge.Hm.bit, List.getElem?_map, List.getElem?_range (by omega : 8 * (7 - i % 8) + s < 64),
        Option.map_some, Option.getD_some]
      rw [show wordBit (64 * (i / 8) + (8 * (7 - i % 8) + s)) = 8 * i + s by unfold wordBit; omega,
        ev_bi z hx hy (by omega), bitL_bytes _ hi hs]
    · have h8 : ∀ u : UInt8, u.toNat.testBit s = false := fun u =>
        Nat.testBit_lt_two_pow (lt_of_lt_of_le (show u.toNat < 2 ^ 8 from u.toBitVec.isLt)
          (Nat.pow_le_pow_right (by norm_num) (by omega)))
      rw [h8, h8]

/-! ## The prefixes -/

theorem saltPrefix_data : Flock.Hm96.Default512.saltPrefix.data.toList = Hash.Hm96.saltPrefix := by
  rw [Flock.Hm96.Default512.saltPrefix, pad_data,
    show Flock.Hm96.sha512.saltTag = "verity/hm96-sha512/salt/v1".toUTF8.push 0 from rfl, push_zero_data]
  rfl

theorem leafPrefix_data :
    Flock.Hm96.Default512.leafPrefix.data.toList = Hash.Hm96.leafPrefix Verity.sha512 pinnedKey := by
  have hb : (⟨pinnedKey.toArray⟩ : ByteArray) = Flock.Hm96.Default512.key := ByteArray.ext (by simp [pinnedKey])
  have hk : (Verity.sha512 pinnedKey).bytes = (Flock.Sha512.hash Flock.Hm96.Default512.key).data.toList := by
    rw [isSha512_sha512 pinnedKey, hb]
  rw [Flock.Hm96.Default512.leafPrefix, Flock.Hm96.leafPrefix, pad_data, ByteArray.toList_data_append,
    show Flock.Hm96.sha512.leafTag = "verity/hm96-sha512/leaf/v1".toUTF8.push 0 from rfl, push_zero_data,
    Hash.Hm96.leafPrefix, hk]
  rfl

theorem size_leafPrefix : Flock.Hm96.Default512.leafPrefix.size = 128 := by
  rw [size_of_data leafPrefix_data, Hash.Hm96.leafPrefix, length_padTo]
  have : (Hash.cstr "verity/hm96-sha512/leaf/v1").length = 27 := by
    rw [← push_zero_data, Array.length_toList, ByteArray.size_data]
    decide +kernel
  simp only [List.length_append, this, Hash.BindingProofs.bytes_length]
  norm_num

/-! ## The leaf -/

/-- **The commit string is `b ‖ c`**: the leaf generator's `b` and `c` words hold hm96's commit string of the digest and
the salt, under the pinned key. -/
theorem commitString_eq (z : Var → Bool) {x y b c : List Word} (hx : Shaped 8 x) (hy : Shaped 24 y)
    (hb : b.map (bv z) = (List.range 8).map fun w => bv z (bWord x y w))
    (hc : c.map (bv z) = chainW 2 ((Flock.ReadNet.midstate Flock.Hm96.Default512.saltPrefix).map (bv z))
      ((y ++ Flock.ReadNet.constWords (Flock.ReadNet.padTail 320)).map (bv z))) :
    Hash.Hm96.commitString Verity.sha512 pinnedKey (wordsDigest z x) (wordsBytes z y) = bytesBE ((b ++ c).map (bv z)) := by
  rw [Hash.Hm96.commitString, List.map_append, bytesBE_append, bytes_b z hx hy hb]
  congr 1
  rw [hc, ← saltPrefix_data, wordsBytes_eq z hy, hash_mid isSha512_sha512 size_saltPrefix, map_bv_midstate,
    List.map_append, map_bv_constWords, length_bytesBE, List.length_map, hy.1, wordsBE_append]
  rfl

/-- **Part B, the leaf**: the leaf generator meets `Gates.Hm96LeafSpec` under the pinned key. -/
theorem leaf_spec : Hm96LeafSpec pinnedKey Flock.ReadNet.leaf := by
  intro x y
  simp only [Flock.ReadNet.leaf]
  refine Sound.bind' (sound_mapM_word (fun w => commitW (bWord x y w)) (fun _ => True)
    (fun z w => bv z (bWord x y w))
    (fun w => Sound.mono (sound_commitW (bWord x y w)) fun z r h _ => h (by simp [bWord])) (List.range 8)) fun b =>
    Sound.bind' (sound_shaW 2 _ _) fun c => Sound.bind' (sound_shaW 2 _ _) fun lf =>
      Sound.pure' fun z hlf hc hb => ?_
  intro hx hy
  obtain ⟨hb1, hb2⟩ := hb fun _ _ => trivial
  have hbl : b.length = 8 := by rw [length_of_map_bv hb2]; simp
  obtain ⟨c1, c2, c3⟩ := hc (length_midstate _) (words64_midstate _)
    (by simp [hy.1, length_constWords, length_padTail, zeroPad]) (Words64.append hy.2 (words64_constWords _))
  obtain ⟨l1, l2, l3⟩ := hlf (length_midstate _) (words64_midstate _)
    (by simp [hbl, c1, length_constWords, length_padTail, zeroPad]) ((hb1.append c2).append (words64_constWords _))
  have hdrop : (b ++ c ++ lf).drop 16 = lf := List.drop_left' (by simp [hbl, c1])
  refine ⟨⟨by simp [hbl, c1, l1], (hb1.append c2).append l2⟩, ?_⟩
  have hcs := commitString_eq z hx hy hb2 c3
  have hcl : (Hash.Hm96.commitString Verity.sha512 pinnedKey (wordsDigest z x) (wordsBytes z y)).length = 128 := by
    rw [hcs, length_bytesBE, List.length_map, List.length_append, hbl, c1]
  rw [hdrop, ← Hash.BindingProofs.bytes_inj, wordsDigest_bytes z ⟨l1, l2⟩, l3, map_bv_midstate, List.map_append,
    map_bv_constWords, Hash.Hm96.leaf, ← leafPrefix_data, hash_mid isSha512_sha512 size_leafPrefix, hcl, hcs,
    wordsBE_append]
  rfl

end Accounting.ReadUnit.Gates
