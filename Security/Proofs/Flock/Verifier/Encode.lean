/-
Level 3: the row encoding a Merkle leaf hashes is injective. `F128.toBytes` is `le64(lo) ‖ le64(hi)`, 16 bytes, and a row's
bytes are its elements' encodings in order, so two different rows are two different byte strings.
-/
import Flock.Merkle

namespace Flock

theorem UInt64.byte_toNat (x : UInt64) (k : Nat) (hk : k < 64) :
    (x >>> k.toUInt64).toUInt8.toNat = x.toNat / 2 ^ k % 256 := by
  rw [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow]
  congr 3
  simp [Nat.mod_eq_of_lt (show k < 2 ^ 64 by omega)]
  omega

/-- A `UInt64` is its eight little-endian bytes. -/
theorem UInt64.eq_of_bytes (x y : UInt64)
    (h : ∀ i, i < 8 → (x >>> (8 * i).toUInt64).toUInt8 = (y >>> (8 * i).toUInt64).toUInt8) : x = y := by
  have e := fun i (hi : i < 8) => congrArg UInt8.toNat (h i hi)
  have hx := x.toNat_lt
  have hy := y.toNat_lt
  have e0 := e 0 (by omega); have e1 := e 1 (by omega); have e2 := e 2 (by omega); have e3 := e 3 (by omega)
  have e4 := e 4 (by omega); have e5 := e 5 (by omega); have e6 := e 6 (by omega); have e7 := e 7 (by omega)
  simp only [UInt64.byte_toNat _ _ (by omega : (8 * 0) < 64), UInt64.byte_toNat _ _ (by omega : (8 * 1) < 64),
    UInt64.byte_toNat _ _ (by omega : (8 * 2) < 64), UInt64.byte_toNat _ _ (by omega : (8 * 3) < 64),
    UInt64.byte_toNat _ _ (by omega : (8 * 4) < 64), UInt64.byte_toNat _ _ (by omega : (8 * 5) < 64),
    UInt64.byte_toNat _ _ (by omega : (8 * 6) < 64), UInt64.byte_toNat _ _ (by omega : (8 * 7) < 64)]
    at e0 e1 e2 e3 e4 e5 e6 e7
  simp only [Nat.reduceMul, Nat.reducePow, Nat.pow_zero, Nat.div_one] at e0 e1 e2 e3 e4 e5 e6 e7
  apply UInt64.toNat_inj.mp
  omega

theorem F128.toBytes_size (a : F128) : a.toBytes.size = 16 := by
  rw [← ByteArray.size_data]
  simp [F128.toBytes]

theorem F128.toBytes_inj {a b : F128} (h : a.toBytes = b.toBytes) : a = b := by
  have hd := ByteArray.mk.inj h
  have byte := fun (i : Nat) (hi : i < 16) => by
    have := congrArg (fun arr : Array UInt8 => arr[i]!) hd
    rwa [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi), Array.getElem_ofFn,
      Array.getElem_ofFn] at this
  have lo := UInt64.eq_of_bytes a.lo b.lo fun i hi => by
    simpa [hi] using byte i (by omega)
  have hi := UInt64.eq_of_bytes a.hi b.hi fun i hi => by
    simpa [show ¬ (i + 8 < 8) by omega] using byte (i + 8) (by omega)
  cases a; cases b; simp_all

theorem rowBytes_foldl (L : List F128) (acc : ByteArray) :
    L.foldl F128.pushBytes acc = acc ++ L.foldl F128.pushBytes ByteArray.empty := by
  induction L generalizing acc with
  | nil => simp
  | cons a L ih => simp only [List.foldl_cons]; rw [ih, ih (F128.pushBytes _ a)]; simp [F128.pushBytes, ByteArray.append_assoc]

theorem rowBytes_cons (a : F128) (L : List F128) :
    (a :: L).foldl F128.pushBytes ByteArray.empty = a.toBytes ++ L.foldl F128.pushBytes ByteArray.empty := by
  rw [List.foldl_cons, rowBytes_foldl]; simp [F128.pushBytes]

theorem rowBytes_size (L : List F128) : (L.foldl F128.pushBytes ByteArray.empty).size = 16 * L.length := by
  induction L with
  | nil => simp
  | cons a L ih => rw [rowBytes_cons, ByteArray.size_append, ih, F128.toBytes_size]; simp [Nat.mul_succ]; omega

/-- **The row encoding is injective:** two rows with one byte string are one row. -/
theorem rowBytes_inj {r r' : Array F128} (h : MerkleScheme.rowBytes r = MerkleScheme.rowBytes r') : r = r' := by
  simp only [MerkleScheme.rowBytes, ← Array.foldl_toList] at h
  suffices ∀ L L' : List F128, L.foldl F128.pushBytes ByteArray.empty = L'.foldl F128.pushBytes ByteArray.empty → L = L' by
    exact Array.toList_inj.mp (this _ _ h)
  intro L
  induction L with
  | nil =>
    intro L' h
    cases L' with
    | nil => rfl
    | cons b L' => have := congrArg ByteArray.size h; rw [rowBytes_size, rowBytes_size] at this; simp at this
  | cons a L ih =>
    intro L' h
    cases L' with
    | nil => have := congrArg ByteArray.size h; rw [rowBytes_size, rowBytes_size] at this; simp at this
    | cons b L' =>
      rw [rowBytes_cons, rowBytes_cons] at h
      have hab := ByteArray.append_inj_left h (by rw [F128.toBytes_size, F128.toBytes_size])
      rw [hab] at h
      rw [F128.toBytes_inj hab, ih L' ((ByteArray.append_right_inj _).mp h)]

end Flock
