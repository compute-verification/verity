/-
Values of the frame-v3 read's forms at `z`: numbers from bits and their bits (`testBit_num`), a byte's value (`byteV`),
indexing a list of equal-length chunks (`getD_flatMap`), and words holding bytes (`HoldsBytes`): bit `p` of word `i` is
bit `p % 8` of byte `8 i + 7 - p / 8`, each word's bytes big-endian and each byte least significant bit first.
-/
module

public import Proofs.Accounting.ReadUnit.Net
public import Proofs.Flock.Soundness.Discharge.Hm.Compress
public import Flock.ReadNet.Frame

@[expose] public section

namespace Accounting.ReadUnit.FrameRead

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Flock.ReadNet Flock.ReadNet.Frame
open Accounting.ReadUnit.Gates

/-! ## Numbers from bits -/

theorem num_nil : ReadUnit.num [] = 0 := rfl

theorem num_cons (b : Bool) (bs : List Bool) : ReadUnit.num (b :: bs) = b.toNat + 2 * ReadUnit.num bs := rfl

/-- Bit `q` of the number bits hold is bit `q` of the list, 0 past its end. -/
theorem testBit_num : ∀ (l : List Bool) (q : ℕ), (ReadUnit.num l).testBit q = l.getD q false
  | [], q => by simp [num_nil]
  | b :: l, 0 => by
    rw [num_cons, Nat.testBit_zero, List.getD_cons_zero]
    cases b <;> simp
  | b :: l, q + 1 => by
    rw [num_cons, Nat.testBit_add_one, List.getD_cons_succ, ← testBit_num l q]
    congr 1
    cases b <;> simp <;> omega

theorem getD_evs (z : Var → Bool) (l : List Lin) (q : ℕ) : (evs z l).getD q false = ev z (l.getD q Lin.zero) := by
  rw [evs, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases l[q]? <;> rfl

theorem testBit_num_evs (z : Var → Bool) (l : List Lin) (q : ℕ) :
    (ReadUnit.num (evs z l)).testBit q = ev z (l.getD q Lin.zero) := by
  rw [testBit_num, getD_evs]

theorem num_evs_lt (z : Var → Bool) (l : List Lin) : ReadUnit.num (evs z l) < 2 ^ l.length := by
  simpa [evs] using num_lt (l.map (ev z))

/-- Two bytes with the same 8 low bits are equal. -/
theorem uint8_ext {a b : UInt8} (h : ∀ q < 8, a.toNat.testBit q = b.toNat.testBit q) : a = b := by
  apply UInt8.toNat_inj.1
  refine Nat.eq_of_testBit_eq fun q => ?_
  by_cases hq : q < 8
  · exact h q hq
  · have ha8 : a.toNat < 2 ^ 8 := a.toNat_lt_size
    have hb8 : b.toNat < 2 ^ 8 := b.toNat_lt_size
    have ha : a.toNat < 2 ^ q := lt_of_lt_of_le ha8 (Nat.pow_le_pow_right (by norm_num) (by omega))
    have hb : b.toNat < 2 ^ q := lt_of_lt_of_le hb8 (Nat.pow_le_pow_right (by norm_num) (by omega))
    rw [Nat.testBit_lt_two_pow ha, Nat.testBit_lt_two_pow hb]

/-! ## Bytes -/

/-- The byte a byte's forms hold at `z`. -/
def byteV (z : Var → Bool) (b : Byte) : UInt8 := UInt8.ofNat (ReadUnit.num (evs z b))

/-- The bytes byte forms hold at `z`. -/
def bytesV (z : Var → Bool) (bs : List Byte) : Hash.Msg := bs.map (byteV z)

theorem testBit_byteV (z : Var → Bool) {b : Byte} (hb : b.length = 8) {q : ℕ} (_hq : q < 8) :
    (byteV z b).toNat.testBit q = ev z (b.getD q Lin.zero) := by
  have h := num_evs_lt z b
  rw [hb] at h
  rw [byteV, UInt8.toNat_ofNat', Nat.mod_eq_of_lt h, testBit_num_evs]

theorem length_constByte (b : UInt8) : (constByte b).length = 8 := by simp [constByte]

theorem byteV_constByte (z : Var → Bool) (b : UInt8) : byteV z (constByte b) = b := by
  refine uint8_ext fun q hq => ?_
  rw [testBit_byteV z (length_constByte b) hq, constByte, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range hq]
  rfl

theorem bytesV_constBytes (z : Var → Bool) (m : List UInt8) : bytesV z (constBytes m) = m := by
  simp [bytesV, constBytes, Function.comp_def, byteV_constByte]

theorem bytesV_append (z : Var → Bool) (a b : List Byte) : bytesV z (a ++ b) = bytesV z a ++ bytesV z b :=
  List.map_append

theorem length_bytesV (z : Var → Bool) (a : List Byte) : (bytesV z a).length = a.length := List.length_map _

theorem length_constBytes (m : List UInt8) : (constBytes m).length = m.length := List.length_map _

theorem getD_bytesV (z : Var → Bool) (a : List Byte) (j : ℕ) : (bytesV z a).getD j 0 = byteV z (a.getD j []) := by
  rw [bytesV, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases a[j]? <;> rfl

/-- Every byte of `bs` is 8 forms. -/
def Bytes8 (bs : List Byte) : Prop := ∀ b ∈ bs, b.length = 8

theorem bytes8_constBytes (m : List UInt8) : Bytes8 (constBytes m) := by
  intro b hb
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 hb
  exact length_constByte u

theorem bytes8_append {a b : List Byte} (ha : Bytes8 a) (hb : Bytes8 b) : Bytes8 (a ++ b) := by
  intro x hx
  rcases List.mem_append.1 hx with h | h
  exacts [ha x h, hb x h]

theorem length_getD_of_bytes8 {bs : List Byte} (h : Bytes8 bs) {j : ℕ} (hj : j < bs.length) :
    (bs.getD j []).length = 8 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]
  exact h _ (List.getElem_mem hj)

/-! ## Lists of equal-length chunks -/

theorem getD_append_left' {α : Type} {l l' : List α} {d : α} {n : ℕ} (h : n < l.length) :
    (l ++ l').getD n d = l.getD n d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_append_left h]

theorem getD_append_right' {α : Type} {l l' : List α} {d : α} {n : ℕ} (h : l.length ≤ n) :
    (l ++ l').getD n d = l'.getD (n - l.length) d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_append_right h]

/-- Entry `k i + j` of chunks of length `k` is entry `j` of chunk `i`. -/
theorem getD_flatMap {α β : Type} (f : α → List β) (k : ℕ) (a₀ : α) (d : β) :
    ∀ l : List α, (∀ a ∈ l, (f a).length = k) → ∀ i j : ℕ, i < l.length → j < k →
      (l.flatMap f).getD (k * i + j) d = (f (l.getD i a₀)).getD j d
  | [], _, i, _, hi, _ => absurd hi (by simp)
  | a :: l, hl, i, j, hi, hj => by
    have ha : (f a).length = k := hl a List.mem_cons_self
    rw [List.flatMap_cons]
    cases i with
    | zero => rw [Nat.mul_zero, Nat.zero_add, getD_append_left' (by omega), List.getD_cons_zero]
    | succ i =>
      rw [getD_append_right' (by rw [ha, Nat.mul_succ]; omega), ha,
        show k * (i + 1) + j - k = k * i + j by rw [Nat.mul_succ]; omega, List.getD_cons_succ,
        getD_flatMap f k a₀ d l (fun b hb => hl b (List.mem_cons_of_mem _ hb)) i j (by simpa using hi) hj]

theorem length_flatMap_const {α β : Type} (f : α → List β) (k : ℕ) :
    ∀ l : List α, (∀ a ∈ l, (f a).length = k) → (l.flatMap f).length = k * l.length
  | [], _ => by simp
  | a :: l, hl => by
    rw [List.flatMap_cons, List.length_append, hl a List.mem_cons_self,
      length_flatMap_const f k l fun b hb => hl b (List.mem_cons_of_mem _ hb), List.length_cons]
    ring

/-! ## Words holding bytes -/

/-- **Words hold bytes** at `z`: `ws` are 64-bit words, eight bytes each, and bit `p` of word `i` is bit `p % 8` of byte
`8 i + 7 - p / 8` of `bs`. -/
def HoldsBytes (z : Var → Bool) (ws : List Word) (bs : List UInt8) : Prop :=
  Words64 ws ∧ 8 * ws.length = bs.length ∧
    ∀ i < ws.length, ∀ p < 64,
      ev z ((ws.getD i []).getD p Lin.zero) = (bs.getD (8 * i + 7 - p / 8) 0).toNat.testBit (p % 8)

/-- The words `wordsB` makes of 8-form bytes hold the bytes' values. -/
theorem holds_wordsB (z : Var → Bool) {data : List Byte} (h8 : Bytes8 data) (hm : data.length % 8 = 0) :
    HoldsBytes z (wordsB data) (bytesV z data) := by
  have hchunk : ∀ k < data.length / 8, ∀ b ∈ List.range 8, (data.getD (8 * k + 7 - b) []).length = 8 := by
    intro k hk b hb
    rw [List.mem_range] at hb
    exact length_getD_of_bytes8 h8 (by omega)
  refine ⟨?_, ?_, ?_⟩
  · intro w hw
    simp only [wordsB] at hw
    obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hw
    rw [List.mem_range] at hk
    rw [length_flatMap_const _ 8 _ (hchunk k hk), List.length_range]
  · rw [wordsB, List.length_map, List.length_range, length_bytesV]
    omega
  · intro i hi p hp
    rw [wordsB, List.length_map, List.length_range] at hi
    have e : (wordsB data).getD i [] = (List.range 8).flatMap fun b => data.getD (8 * i + 7 - b) [] := by
      rw [wordsB, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi]
      rfl
    have key := getD_flatMap (fun b => data.getD (8 * i + 7 - b) []) 8 0 Lin.zero (List.range 8) (hchunk i hi)
      (p / 8) (p % 8) (by simp; omega) (Nat.mod_lt _ (by norm_num))
    rw [show 8 * (p / 8) + p % 8 = p by omega, List.getD_eq_getElem?_getD (l := List.range 8),
      List.getElem?_range (by omega)] at key
    simp only [Option.getD_some] at key
    rw [e, key, getD_bytesV,
      testBit_byteV z (length_getD_of_bytes8 h8 (j := 8 * i + 7 - p / 8) (by omega)) (Nat.mod_lt _ (by norm_num))]

end Accounting.ReadUnit.FrameRead
