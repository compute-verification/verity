import NetworkCheck.RecordsGrid
import Proofs.Properties.NetworkCertifier.Bytes

/-!
# Bytes the committed digest is written in

A byte array's fold is its list's (`foldl_eq`); `hexOf` writes each byte as two characters, one byte per pair, so two
byte arrays with one hex string are one (`hexOf_inj`); lists cut into pieces of one length `k > 0` are one when their
concatenations are (`flatten_inj`); `k`-byte pieces of whole bits are `k` bytes (`bytesOfBits_size`).
-/

open NetworkCheck.Check (bytesOfBits bytesOfBitsList)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj)

set_option maxRecDepth 20000

namespace Properties.NetworkRecords.Bytes

theorem loop_eq {β : Type} (f : β → UInt8 → β) (as : ByteArray) (stop : Nat) (h : stop ≤ as.size) :
    ∀ (i j : Nat) (b : β), i + j = stop →
      ByteArray.foldlM.loop (m := Id) (fun b a => pure (f b a)) as stop h i j b =
        ((as.data.toList.drop j).take i).foldl f b := by
  intro i
  induction i with
  | zero => intro j b _; unfold ByteArray.foldlM.loop; split <;> rfl
  | succ i ih =>
    intro j b hij
    unfold ByteArray.foldlM.loop
    have hj : j < stop := by omega
    rw [dite_eq_left_of_eq_true (eq_true hj)]
    have hjs : j < as.data.toList.length := by simp; omega
    rw [List.drop_eq_getElem_cons hjs]
    simp only [List.take_succ_cons, List.foldl_cons]
    rw [← ih (j + 1) _ (by omega)]
    simp [Array.getElem_toList]
    rfl

theorem foldl_eq {β : Type} (f : β → UInt8 → β) (init : β) (as : ByteArray) :
    as.foldl f init = as.data.toList.foldl f init := by
  unfold ByteArray.foldl ByteArray.foldlM
  rw [dite_eq_left_of_eq_true (eq_true (Nat.le_refl _))]
  dsimp only
  rw [loop_eq f as as.size (Nat.le_refl _) (as.size - 0) 0 init (by omega)]
  simp only [List.drop_zero, Nat.sub_zero]
  rw [List.take_of_length_le (by simp)]
  rfl

theorem foldl_push (d₁ d₂ : UInt8 → Char) : ∀ (l : List UInt8) (s : String),
    (l.foldl (fun s x => (s.push (d₁ x)).push (d₂ x)) s).toList = s.toList ++ l.flatMap fun x => [d₁ x, d₂ x]
  | [], s => by simp
  | x :: l, s => by
    rw [List.foldl_cons, foldl_push d₁ d₂ l]
    simp [String.toList_push]

theorem hexOf_toList (b : ByteArray) :
    (Flock.hexOf b).toList = b.data.toList.flatMap fun x => (Flock.hexOf ⟨#[x]⟩).toList := by
  unfold Flock.hexOf
  rw [foldl_eq, foldl_push]
  simp only [String.toList_empty, List.nil_append]
  congr 1
  funext x
  rw [foldl_eq, foldl_push]
  simp

theorem hexOf_length (x : UInt8) : (Flock.hexOf ⟨#[x]⟩).toList.length = 2 := by
  unfold Flock.hexOf
  rw [foldl_eq, foldl_push]
  simp

/-- A hex digit's value, of `0-9` and `a-f`. -/
def nib (c : Char) : Nat := if c.toNat < 87 then c.toNat - 48 else c.toNat - 87

/-- A two-digit hex string's value. -/
def dec : List Char → Nat
  | [a, b] => nib a * 16 + nib b
  | _ => 0

theorem dec_hex_nat : ∀ n < 256, dec (Flock.hexOf ⟨#[n.toUInt8]⟩).toList = n := by decide +kernel

theorem dec_hex (x : UInt8) : dec (Flock.hexOf ⟨#[x]⟩).toList = x.toNat := by
  simpa using dec_hex_nat x.toNat x.toNat_lt

/-- Lists cut into pieces of one length `k > 0` are one when their concatenations are. -/
theorem flatten_inj {α : Type} {k : Nat} (hk : 0 < k) :
    ∀ {L L' : List (List α)}, (∀ x ∈ L, x.length = k) → (∀ x ∈ L', x.length = k) → L.flatten = L'.flatten →
      L = L'
  | [], [], _, _, _ => rfl
  | [], x :: _, _, h', e => by
    have := congrArg List.length e
    simp only [List.flatten_nil, List.length_nil, List.flatten_cons, List.length_append] at this
    have := h' x (List.mem_cons_self ..)
    omega
  | x :: _, [], h, _, e => by
    have := congrArg List.length e
    simp only [List.flatten_nil, List.length_nil, List.flatten_cons, List.length_append] at this
    have := h x (List.mem_cons_self ..)
    omega
  | x :: L, y :: L', h, h', e => by
    simp only [List.flatten_cons] at e
    have hx := h x (List.mem_cons_self ..)
    have hy := h' y (List.mem_cons_self ..)
    obtain ⟨rfl, e'⟩ := List.append_inj e (hx.trans hy.symm)
    rw [flatten_inj hk (fun z hz => h z (List.mem_cons_of_mem _ hz)) (fun z hz => h' z (List.mem_cons_of_mem _ hz)) e']

/-- Two byte arrays with one hex string are one. -/
theorem hexOf_inj {a b : ByteArray} (h : Flock.hexOf a = Flock.hexOf b) : a = b := by
  have e := congrArg String.toList h
  rw [hexOf_toList, hexOf_toList, List.flatMap_def, List.flatMap_def] at e
  have e' := flatten_inj (k := 2) (by decide) (by simp [hexOf_length]) (by simp [hexOf_length]) e
  have e'' := congrArg (List.map dec) e'
  simp only [List.map_map] at e''
  have hd : (dec ∘ fun x => (Flock.hexOf ⟨#[x]⟩).toList) = UInt8.toNat := funext dec_hex
  rw [hd] at e''
  have hi : a.data.toList = b.data.toList := List.map_injective_iff.mpr (fun x y h => UInt8.toNat_inj.mp h) e''
  exact bs_inj hi

theorem bytesOfBitsList_length : ∀ x : List Bool, (bytesOfBitsList x).length = x.length / 8
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest => by
    simp only [bytesOfBitsList, List.length_cons, bytesOfBitsList_length rest]
    omega
  | [] | [_] | [_, _] | [_, _, _] | [_, _, _, _] | [_, _, _, _, _] | [_, _, _, _, _, _] | [_, _, _, _, _, _, _] => by
    simp [bytesOfBitsList]

theorem bytesOfBits_size {x : List Bool} {k : Nat} (h : x.length = 8 * k) : (bytesOfBits x).size = k := by
  show (bytesOfBitsList x).toArray.size = k
  rw [List.size_toArray, bytesOfBitsList_length, h]
  omega

end Properties.NetworkRecords.Bytes
