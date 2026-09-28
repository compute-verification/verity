import Proofs.Flock.Soundness.Discharge.ZkView.Statements
import Proofs.Flock.Soundness.Discharge.Hm.Commit

/-!
# The executable's hm96-sha512 leaf splits, and the honest prover's leaves are the executable's

`leafSplit`: bit `t < 512` of `Default512.commitString x y` is bit `t` of `x` xor bit `t` of `Default512.mask y`, which
`Hm.bitL_mask` gives as `⊕_{j < 1536} key_{t+j} · y_j`, the Hankel product `My y` at `t` over `ZMod 2`. The other half is
`c(y)` itself.
-/

namespace FlockSoundness.Discharge.ZkView

open Flock Hm

theorem testBit_foldr_bit (l : List Bool) (b : ℕ) : (l.foldr Nat.bit 0).testBit b = l.getD b false := by
  induction l generalizing b with
  | nil => simp
  | cons a l ih =>
    cases b with
    | zero => rw [List.foldr_cons, Nat.testBit_bit_zero]; rfl
    | succ b => rw [List.foldr_cons, Nat.testBit_bit_succ, ih]; rfl

theorem foldr_bit_lt (l : List Bool) : l.foldr Nat.bit 0 < 2 ^ l.length := by
  induction l with
  | nil => simp
  | cons a l ih => rw [List.foldr_cons, List.length_cons, Nat.bit_lt_two_pow_succ_iff]; exact ih

theorem testBit_byteOf (f : Fin 8 → ZMod 2) (b : Fin 8) : (byteOf f).toNat.testBit b = decide (f b = 1) := by
  have hlt := foldr_bit_lt (List.ofFn fun i => decide (f i = 1))
  rw [List.length_ofFn] at hlt
  rw [byteOf, Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.mod_eq_of_lt hlt, testBit_foldr_bit, List.getD_eq_getElem?_getD,
    List.getElem?_ofFn]
  simp [b.isLt]

theorem bitL_bytesOf {k : ℕ} (v : Fin (8 * k) → ZMod 2) {t : ℕ} (ht : t < 8 * k) :
    bitL (bytesOf v).data.toList t = decide (v ⟨t, ht⟩ = 1) := by
  obtain ⟨q, r, hr, rfl⟩ : ∃ q r, r < 8 ∧ t = 8 * q + r := ⟨t / 8, t % 8, Nat.mod_lt _ (by norm_num), by omega⟩
  have hq : q < k := by omega
  have e := testBit_byteOf (fun b : Fin 8 => v ⟨8 * q + b.val, by have := b.isLt; omega⟩) ⟨r, hr⟩
  rw [bitL, show (8 * q + r) / 8 = q by omega, show (8 * q + r) % 8 = r by omega]
  simp only [bytesOf, Array.toList_ofFn, List.getD_eq_getElem?_getD, List.getElem?_ofFn, hq, dite_true,
    Option.getD_some]
  exact e

theorem bitL_saltBytes (y : Salt) {j : ℕ} (hj : j < 1536) :
    bitL (saltBytes y).data.toList j = decide (y ⟨j, hj⟩ = 1) :=
  bitL_bytesOf (k := 192) y hj

theorem ind_xsum (f : ℕ → Bool) :
    ∀ N : ℕ, (if xsum (List.range N) f then (1 : ZMod 2) else 0) = ∑ j : Fin N, if f j then 1 else 0
  | 0 => by simp [xsum_nil]
  | N + 1 => by
    rw [List.range_succ, xsum_append, Fin.sum_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last]
    rw [← ind_xsum f N, xsum_cons, xsum_nil, Bool.xor_false]
    cases xsum (List.range N) f <;> by_cases h : f N = true <;> simp [h]
    decide

theorem xor_ind : ∀ a b : Bool,
    (a ^^ b) = decide (((if a then 1 else 0 : ZMod 2) + if b then 1 else 0) = 1) := by decide

theorem zmod2_ind : ∀ a : ZMod 2, a = if decide (a = 1) then 1 else 0 := by decide

/-- The Hankel product of a key and a salt given by their bits is the XOR sum `⊕_{j < 1536} κ_{t+j} · y_j`. -/
theorem hankel_ind (κb yb : ℕ → Bool) (y : Salt) (hy : ∀ j : Fin 1536, y j = if yb j then 1 else 0) (t : Fin 512) :
    CROnly.hankel (n := 512) (l := 1536) (fun s => if κb s.val then (1 : ZMod 2) else 0) y t =
      if xsum (List.range 1536) (fun j => κb (t + j) && yb j) then 1 else 0 := by
  rw [ind_xsum, CROnly.hankel]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [hy j]
  cases κb (t + j) <;> cases yb j <;> simp

theorem maskV_eq (key : ByteArray) (yb : ℕ → Bool) (i : ℕ) :
    maskV key yb i = xsum (List.range 1536) (fun j => Hm96.bit key (i + j) && yb j) := by
  unfold maskV xsum; rfl

/-- `My y` at `t` is `Hm.maskV`'s bit `t` of the salt's bytes. -/
theorem My_apply (y : Salt) (t : Fin 512) :
    My y t = if maskV Hm96.Default512.key (bitL (saltBytes y).data.toList) t then 1 else 0 := by
  rw [maskV_eq]
  exact hankel_ind (Hm96.bit Hm96.Default512.key) (bitL (saltBytes y).data.toList) y
    (fun j => by rw [bitL_saltBytes y j.isLt]; exact zmod2_ind _) t

theorem leafSplit : LeafSplit := by
  intro x y hx
  rw [Hm96.Default512.leaf, Hm96.Default512.commitString, leafH, saltHash]
  refine congrArg (fun b => Hm96.Default512.treeLeaf (b ++ Sha512.hash (Hm96.Default512.saltPrefix ++ saltBytes y))) ?_
  have hA : ∀ g : ℕ → UInt8, ((Array.range 64).map g).toList.length = 64 := fun g => by
    rw [Array.toList_map, List.length_map, Array.toList_range, List.length_range]
  have hB : ∀ v : Fin (8 * 64) → ZMod 2, (bytesOf (k := 64) v).data.toList.length = 64 := fun v => by
    rw [bytesOf, Array.toList_ofFn, List.length_ofFn]
  refine ByteArray.ext (Array.toList_inj.mp (eq_of_bitL (by rw [hA, hB]) fun t ht => ?_))
  rw [hA] at ht
  have ht' : t < 512 := ht
  rw [bitL_bytesOf (k := 64) _ ht', bitL]
  rw [Array.toList_map, Array.toList_range, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (show t / 8 < 64 by omega), Option.map_some, Option.getD_some]
  rw [UInt8.toNat_xor, Nat.testBit_xor, get!_eq_getD, get!_eq_getD]
  have hm : ∀ s : ByteArray,
      bitL (Hm96.Default512.mask s).data.toList t = maskV Hm96.Default512.key (bitL s.data.toList) t := by
    intro s
    have h := bitL_mask s.data.toList ht'
    rwa [Array.toArray_toList] at h
  have hm' := hm (saltBytes y)
  rw [bitL] at hm'
  rw [hm']
  dsimp only
  simp only [Pi.add_apply, bitsOf, My_apply y ⟨t, ht'⟩, bitL]
  exact xor_ind _ _

theorem Coined.content_size {K W Hi VL : Type} [Field K] {z : Sizes} (T : Coined K W Hi VL z) (w : W) (b : Base z Hi)
    (p : LeafPos z) : (T.content w b p).size = 64 := by
  rcases p with p | ⟨r, p⟩ | r <;> exact Sha512.hash_size _

theorem honest_msg : HonestMsg := by
  intro K W Hi VL _ J n Cn zs T out hc w x c
  dsimp only [honest, ZK.Strategy.comap, pairProver]
  refine congrArg (msgsAt T out w x.1 c) (funext fun l => ?_)
  rw [Coined.exeLeaf, leafSplit _ _ ((T c l.1).content_size _ _ _)]
  rfl

end FlockSoundness.Discharge.ZkView
