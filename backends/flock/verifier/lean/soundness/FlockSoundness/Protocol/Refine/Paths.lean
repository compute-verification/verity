import FlockSoundness.Protocol.Refine.Rep

/-!
# The Merkle checks are the model's path checks (PROTOCOL.md §14)

For the unsalted SHA-512 scheme (`MerkleScheme.sha512`), a passed `merkleCheck` is the model's `Merkle.Verifies` at
every query, with the model's hash `H512` (SHA-512 on byte lists, into 64-byte digests) and encodings `enc512` (a row is
its words' 16-byte encodings, a digest its bytes).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Model

/-! ## SHA-512 digests -/

/-! ## The word encoding is injective -/

theorem u64_of_bytes (x y : UInt64) (h : ∀ i < 8, (x >>> (8 * i).toUInt64).toUInt8 = (y >>> (8 * i).toUInt64).toUInt8) :
    x = y := by
  apply UInt64.toNat_inj.1
  apply Nat.eq_of_testBit_eq
  intro j
  by_cases hj : j < 64
  · have hi := congrArg UInt8.toNat (h (j / 8) (by omega))
    simp only [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, toNat_toUInt64 (8 * (j / 8)) (by omega),
      Nat.mod_eq_of_lt (by omega : 8 * (j / 8) < 64)] at hi
    have hb := congrArg (fun n => n.testBit (j % 8)) hi
    simp only [Nat.testBit_mod_two_pow, Nat.testBit_shiftRight] at hb
    rw [show 8 * (j / 8) + j % 8 = j by omega] at hb
    simpa [show j % 8 < 8 by omega] using hb
  · have h64 : 2 ^ 64 ≤ 2 ^ j := Nat.pow_le_pow_right (by norm_num) (by omega)
    rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le x.toNat_lt h64), Nat.testBit_lt_two_pow (lt_of_lt_of_le y.toNat_lt h64)]

theorem toBytes_get (a : F128) (i : ℕ) (hi : i < 16) :
    a.toBytes.data[i]! = if i < 8 then (a.lo >>> (8 * i).toUInt64).toUInt8 else (a.hi >>> (8 * (i - 8)).toUInt64).toUInt8 := by
  unfold F128.toBytes
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_ofFn]

theorem toBytes_inj {a b : F128} (h : a.toBytes = b.toBytes) : a = b := by
  have hd := congrArg ByteArray.data h
  have hlo : a.lo = b.lo := u64_of_bytes a.lo b.lo fun i hi => by
    have := congrArg (·[i]!) hd
    simp only [toBytes_get a i (by omega), toBytes_get b i (by omega), hi, ite_true] at this
    exact this
  have hhi : a.hi = b.hi := u64_of_bytes a.hi b.hi fun i hi => by
    have := congrArg (·[i + 8]!) hd
    simp only [toBytes_get a (i + 8) (by omega), toBytes_get b (i + 8) (by omega),
      show ¬ i + 8 < 8 by omega, ite_false, Nat.add_sub_cancel] at this
    exact this
  cases a; cases b
  simp_all

theorem toBytes_length (a : F128) : a.toBytes.data.toList.length = 16 := by
  simp [F128.toBytes]

/-- A row's bytes: its words' 16-byte encodings, in order. -/
def colBytes (col : List F128) : List UInt8 := col.flatMap fun a => a.toBytes.data.toList

theorem colBytes_inj : Function.Injective colBytes := by
  intro l₁
  induction l₁ with
  | nil =>
      intro l₂ h
      cases l₂ with
      | nil => rfl
      | cons b l₂ =>
          have := congrArg List.length h
          simp only [colBytes, List.flatMap_nil, List.flatMap_cons, List.length_append, toBytes_length,
            List.length_nil] at this
          omega
  | cons a l₁ ih =>
      intro l₂ h
      cases l₂ with
      | nil =>
          have := congrArg List.length h
          simp only [colBytes, List.flatMap_nil, List.flatMap_cons, List.length_append, toBytes_length,
            List.length_nil] at this
          omega
      | cons b l₂ =>
          simp only [colBytes, List.flatMap_cons] at h
          obtain ⟨h1, h2⟩ := List.append_inj h (by rw [toBytes_length, toBytes_length])
          have hab : a = b := toBytes_inj (ByteArray.ext (Array.ext' h1))
          rw [hab, ih h2]

/-! ## The model's hash and encodings -/

/-- A 64-byte digest. -/
abbrev Dig := {b : ByteArray // b.size = 64}

/-- SHA-512 on byte lists, into 64-byte digests. -/
def H512 (l : List UInt8) : Dig := ⟨Sha512.hash ⟨l.toArray⟩, sha512_size _⟩

/-- The tree's encodings: a row is its words' 16-byte encodings, a digest its 64 bytes. -/
def enc512 : Merkle.Enc (List 𝔽) Dig where
  col := colBytes
  dig := fun d => d.1.data.toList
  col_inj := colBytes_inj
  dig_inj := fun a b h => Subtype.ext (ByteArray.ext (Array.ext' h))
  dig_len := fun a b => by
    rw [Array.length_toList, Array.length_toList, ByteArray.size_data, ByteArray.size_data, a.2, b.2]

/-- A digest from bytes: the bytes when there are 64 of them, else a fixed digest. -/
def toDig (b : ByteArray) : Dig :=
  if h : b.size = 64 then ⟨b, h⟩ else ⟨⟨Array.replicate 64 0⟩, show (Array.replicate 64 (0 : UInt8)).size = 64 by simp⟩

/-! ## The proof's openings pass the model's path checks -/

/-- A cap of digests, entry by index. -/
def capDig (cap : Array ByteArray) : Cap Dig := fun i => toDig (cap.getD i ByteArray.empty)

/-- The proof's opening at level `ℓ`, query `j`: the opened row, and its siblings as digests. -/
def openRow (p : LigeritoProof) (sch : Flock.Schedule) (ℓ j : ℕ) : List 𝔽 × List Dig :=
  (((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf,
    (sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap)
      j).map toDig)

end FlockSoundness.Refine
