module

public import Proofs.Properties.Verity.Sampled.OpenEmpty

/-!
# hm96-sha512's mask at any key is the Hankel product

`Flock.Hm96.sha512.mask key y` sets bit `t < 512` of its output to `⊕_{j < 1536} key_{t+j} · y_j` (`bitL_sha512_mask`)
at every key, where `Hm.bitL_mask` has it at the pinned key only: the executable's loop is a fold over the output's bits
(`sha512_mask_eq`), each step setting its own bit and keeping the others (`MaskInv`, `maskInv_step`).

* **A coin tree's mask** (`treeMy_eq_hankel`): `treeMy key y`, the mask of a sample's salt under the key `key` as bits, is
  `CROnly.hankel` of the key's bits applied to the salt's bits over `ZMod 2`.
* **At a sampled key** (`treeMy_treeKey`): at the key `treeKey κ` of the device's 2047 bits `κ`, it is the Hankel
  product of `κ` itself, the mask whose hiding gap `CROnly.hm96_sha512_sum` averages over the key.
-/

@[expose] public section

namespace Verity.Game

open FlockSoundness.Discharge.Hm (bitL get!_eq_getD maskV foldl_range'_induct)
open FlockSoundness.Discharge.ZkView (testBit_foldr_bit foldr_bit_lt)

/-- `Hm96.bit` is `Hm.bitL` of the bytes. -/
theorem hm96Bit_eq_bitL (b : ByteArray) (t : ℕ) : Flock.Hm96.bit b t = bitL b.data.toList t := by
  rw [bitL, ← get!_eq_getD]
  simp only [Flock.Hm96.bit, Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
  generalize (b.get! (t / 8)).toNat / 2 ^ (t % 8) % 2 = r
  by_cases h : r = 1 <;> simp [h]

/-- **The bytes of `8·n` bits read them back**: bit `t < 8·n` of `bytesOfBits n f` is `f t`. -/
theorem bit_bytesOfBits (n : ℕ) (f : ℕ → Bool) {t : ℕ} (ht : t < 8 * n) :
    Flock.Hm96.bit (bytesOfBits n f) t = f t := by
  obtain ⟨q, r, hr, rfl⟩ : ∃ q r, r < 8 ∧ t = 8 * q + r := ⟨t / 8, t % 8, Nat.mod_lt _ (by omega), by omega⟩
  have hq : q < n := by omega
  have hfold : (List.range 8).foldr (fun b acc => 2 * acc + (f (8 * q + b)).toNat) 0 =
      ((List.range 8).map fun b => f (8 * q + b)).foldr Nat.bit 0 := by
    rw [List.foldr_map]
    congr 1
    funext b acc
    rw [Nat.bit_val]
  have hlt := foldr_bit_lt ((List.range 8).map fun b => f (8 * q + b))
  rw [List.length_map, List.length_range] at hlt
  rw [hm96Bit_eq_bitL, bitL, show (8 * q + r) / 8 = q by omega, show (8 * q + r) % 8 = r by omega]
  simp only [bytesOfBits, Array.toList_ofFn, List.getD_eq_getElem?_getD, List.getElem?_ofFn, hq, dite_true,
    Option.getD_some]
  rw [hfold, Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.mod_eq_of_lt hlt, testBit_foldr_bit, List.getD_eq_getElem?_getD,
    List.getElem?_map, List.getElem?_range hr, Option.map_some, Option.getD_some]

/-- A loop in `Id` whose every step yields `g` of its state is a fold. -/
theorem forIn_id_eq_foldl {α β : Type} (l : List α) (g : β → α → β) (f : α → β → Id (ForInStep β))
    (hf : ∀ a b, f a b = ForInStep.yield (g b a)) (init : β) : forIn l init f = l.foldl g init := by
  induction l generalizing init with
  | nil => rfl
  | cons a l ih =>
    rw [List.forIn_cons, hf a init]
    exact ih (g init a)

/-- Whether hm96-sha512's mask sets bit `i`: the XOR of the key's bits `i + j` over the salt's set bits `j`. -/
def maskCond (key y : ByteArray) (i : ℕ) : Bool :=
  ((List.range 1536).filter (Flock.Hm96.bit y ·)).foldl (fun acc j => acc ^^ Flock.Hm96.bit key (i + j)) false

/-- hm96-sha512's mask loop's step at bit `i`. -/
def maskStep (key y : ByteArray) (out : ByteArray) (i : ℕ) : ByteArray :=
  if maskCond key y i then out.set! (i / 8) (out.get! (i / 8) ||| (1 <<< (i % 8).toUInt8)) else out

/-- **hm96-sha512's mask is a fold of its steps** over the 512 output bits, from 64 zero bytes. -/
theorem sha512_mask_eq (key y : ByteArray) :
    Flock.Hm96.sha512.mask key y =
      (List.range' 0 512).foldl (maskStep key y) (ByteArray.mk (Array.replicate 64 0)) := by
  have h1 : 8 * Flock.Hm96.sha512.saltBytes = 1536 := rfl
  have h2 : 8 * Flock.Hm96.sha512.digestBytes = 512 := rfl
  have h3 : Flock.Hm96.sha512.digestBytes = 64 := rfl
  have h4 : ([:512] : Std.Legacy.Range).size = 512 := rfl
  unfold Flock.Hm96.mask
  rw [h1, h2, h3]
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range', h4]
  simp only [bind_pure, Id.run]
  refine forIn_id_eq_foldl _ (maskStep key y) _ (fun a b => ?_) _
  unfold maskStep maskCond
  split <;> rfl

theorem data_getD_set! (out : ByteArray) (k j : ℕ) (v : UInt8) (hk : k < out.size) :
    (out.set! k v).data.toList.getD j 0 = if k = j then v else out.data.toList.getD j 0 := by
  obtain ⟨bs⟩ := out
  change k < bs.size at hk
  show (bs.setIfInBounds k v).toList.getD j 0 = _
  rw [Array.toList_setIfInBounds, List.getD_eq_getElem?_getD, List.getElem?_set, Array.length_toList]
  split_ifs with h
  · rfl
  · rw [List.getD_eq_getElem?_getD]

theorem testBit_or_shl (x : UInt8) {s b : ℕ} (hs : s < 8) (hb : b < 8) :
    (x ||| (1 : UInt8) <<< s.toUInt8).toNat.testBit b = (x.toNat.testBit b || decide (s = b)) := by
  rw [UInt8.toNat_or, Nat.testBit_or, UInt8.toNat_shiftLeft]
  have h1 : s.toUInt8.toNat % 8 = s := by
    rw [Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hs (by decide)), Nat.mod_eq_of_lt hs]
  rw [h1, Nat.testBit_mod_two_pow, show (1 : UInt8).toNat = 1 from rfl, Nat.one_shiftLeft, Nat.testBit_two_pow]
  simp [hb]

/-- **The mask loop's invariant before bit `i`**: 64 bytes, whose bit `t < 512` is `maskCond`'s at `t` where `t < i`
and clear elsewhere. -/
def MaskInv (key y : ByteArray) (i : ℕ) (out : ByteArray) : Prop :=
  out.size = 64 ∧ ∀ t < 512, bitL out.data.toList t = (decide (t < i) && maskCond key y t)

theorem maskInv_step (key y : ByteArray) {i : ℕ} (hi : i < 512) {out : ByteArray} (h : MaskInv key y i out) :
    MaskInv key y (i + 1) (maskStep key y out i) := by
  obtain ⟨hs, hb⟩ := h
  unfold maskStep
  split
  · next hc =>
    refine ⟨by rw [ByteArray.size_set!, hs], fun t ht => ?_⟩
    rw [bitL, data_getD_set! _ _ _ _ (by rw [hs]; omega)]
    split
    · next he =>
      rw [testBit_or_shl _ (Nat.mod_lt _ (by omega)) (Nat.mod_lt _ (by omega)), get!_eq_getD, he]
      have hbt := hb t ht
      rw [bitL] at hbt
      rw [hbt]
      by_cases hti : t = i
      · subst hti
        simp [hc]
      · have h1 : decide (i % 8 = t % 8) = false := decide_eq_false (by omega)
        have h2 : decide (t < i + 1) = decide (t < i) := decide_eq_decide.2 (by omega)
        rw [h1, h2, Bool.or_false]
    · next he =>
      have hbt := hb t ht
      rw [bitL] at hbt
      rw [hbt]
      have h2 : decide (t < i + 1) = decide (t < i) := decide_eq_decide.2 (by omega)
      rw [h2]
  · next hc =>
    refine ⟨hs, fun t ht => ?_⟩
    rw [hb t ht]
    by_cases hti : t = i
    · subst hti
      simp [hc]
    · have h2 : decide (t < i + 1) = decide (t < i) := decide_eq_decide.2 (by omega)
      rw [h2]

theorem maskInv_final (key y : ByteArray) : MaskInv key y 512 (Flock.Hm96.sha512.mask key y) := by
  rw [sha512_mask_eq]
  have h0 : MaskInv key y 0 (ByteArray.mk (Array.replicate 64 0)) := by
    refine ⟨Array.size_replicate .., fun t _ => ?_⟩
    rw [bitL, Array.toList_replicate, List.getD_eq_getElem?_getD, List.getElem?_replicate,
      decide_eq_false (Nat.not_lt_zero t), Bool.false_and]
    split <;> exact Nat.zero_testBit _
  have := foldl_range'_induct (fun i out => MaskInv key y i out) (maskStep key y) 0 512 _ h0
    fun i out _ hi h => maskInv_step key y (by omega) h
  simpa using this

theorem foldl_filter_xor (p f : ℕ → Bool) : ∀ (l : List ℕ) (acc : Bool),
    (l.filter p).foldl (fun acc j => acc ^^ f j) acc = l.foldl (fun acc j => acc ^^ (f j && p j)) acc
  | [], _ => rfl
  | a :: l, acc => by
    rw [List.filter_cons, List.foldl_cons]
    cases hp : p a
    · rw [ite_eq_right (by simp), Bool.and_false, Bool.xor_false]
      exact foldl_filter_xor p f l acc
    · rw [ite_eq_left rfl, List.foldl_cons, Bool.and_true]
      exact foldl_filter_xor p f l _

/-- **hm96-sha512's mask at any key**: bit `t < 512` of `mask key y` is `⊕_{j < 1536} key_{t+j} · y_j` (`Hm.maskV`). -/
theorem bitL_sha512_mask (key y : ByteArray) {t : ℕ} (ht : t < 512) :
    bitL (Flock.Hm96.sha512.mask key y).data.toList t = maskV key (Flock.Hm96.bit y) t := by
  rw [(maskInv_final key y).2 t ht, decide_eq_true ht, Bool.true_and, maskCond, maskV, foldl_filter_xor]

/-- **A coin tree's mask is the Hankel product** of the key's bits and the salt's, over `ZMod 2`. -/
theorem treeMy_eq_hankel (key : ByteArray) (y : Fin 1536 → Bool) :
    treeMy key y = FlockSoundness.CROnly.hankel (n := 512) (l := 1536)
      (fun s => if Flock.Hm96.bit key s.val then (1 : ZMod 2) else 0) (fun j => if y j then 1 else 0) := by
  funext t
  have hy : ∀ j : Fin 1536, (if y j then (1 : ZMod 2) else 0) = if Flock.Hm96.bit (treeSalt y) j then 1 else 0 :=
    fun j => by rw [treeSalt, bit_bytesOfBits 192 _ (by omega), dite_eq_left j.isLt]
  rw [FlockSoundness.Discharge.ZkView.hankel_ind (Flock.Hm96.bit key) (Flock.Hm96.bit (treeSalt y)) _ hy t,
    ← FlockSoundness.Discharge.ZkView.maskV_eq, ← bitL_sha512_mask key (treeSalt y) t.isLt]
  rfl

/-- **A coin tree's mask at a sampled key** `treeKey κ` is the Hankel product of the key's 2047 bits `κ`. -/
theorem treeMy_treeKey (κ : Fin 2047 → Bool) (y : Fin 1536 → Bool) :
    treeMy (treeKey κ) y = FlockSoundness.CROnly.hankel (n := 512) (l := 1536)
      (fun s => if κ s then (1 : ZMod 2) else 0) (fun j => if y j then 1 else 0) := by
  rw [treeMy_eq_hankel]
  congr 1
  funext s
  rw [treeKey, bit_bytesOfBits 256 _ (by omega), dite_eq_left s.isLt]

end Verity.Game
