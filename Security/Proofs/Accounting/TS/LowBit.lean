import Definitions.Accounting.Guarantees.TS.LowBit
import Mathlib.Data.FinEnum
import Mathlib.Data.Fintype.Card

/-!
# A one-register fold's low bit

`OneRegisterLowBit`: a step that binds its word is injective in it, so on the finite type of words it is onto, and its
bit-0 map `h` is onto in the word's bit at each state bit; binding its state makes `h` onto in the state's bit too. A
map on two bits that is onto in each argument is `a ⊕ e ⊕ h(0, 0)`, so each step adds its word's bit 0 and a constant,
and a run adds them all. `LiLowBit`: an odd multiplier keeps bit 0, an addition XORs its operands' bits 0, and so does
an XOR.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS

namespace LowBit

theorem getLsbD_ofNat_toNat (w : ℕ) (b : Bool) : (BitVec.ofNat (w + 1) b.toNat).getLsbD 0 = b := by
  cases b <;> simp

/-- A map on two bits that is onto in each argument is their XOR, plus its value at `(0, 0)`. -/
theorem latin (h : Bool → Bool → Bool) (hw : ∀ a t, ∃ e, h a e = t) (hs : ∀ e t, ∃ a, h a e = t) :
    ∀ a e, h a e = (a ^^ e ^^ h false false) := by
  have k1 : h false true = !h false false := by
    obtain ⟨e, he⟩ := hw false (!h false false)
    cases e
    · cases hb : h false false <;> simp [hb] at he
    · exact he
  have k2 : h true false = !h false false := by
    obtain ⟨a, ha⟩ := hs false (!h false false)
    cases a
    · cases hb : h false false <;> simp [hb] at ha
    · exact ha
  have k3 : h true true = h false false := by
    obtain ⟨e, he⟩ := hw true (!h true false)
    cases e
    · cases hb : h true false <;> simp [hb] at he
    · rw [he, k2, Bool.not_not]
  intro a e
  cases a <;> cases e <;> simp [k1, k2, k3]

/-- A step that binds its word is onto in it. -/
theorem word_surj {w : ℕ} (g : BindingStep 1 (BitVec (w + 1))) (s : Fin 1 → BitVec (w + 1)) :
    Function.Surjective fun d => g.1 s d 0 := by
  apply Finite.surjective_of_injective
  intro d d' hdd
  apply g.2.1 s
  funext j
  rw [Fin.fin_one_eq_zero j]
  exact hdd

/-- A step that binds its state is onto in it. -/
theorem state_surj {w : ℕ} (g : BindingStep 1 (BitVec (w + 1))) (d : BitVec (w + 1)) :
    Function.Surjective fun x : BitVec (w + 1) => g.1 (fun _ => x) d 0 := by
  apply Finite.surjective_of_injective
  intro x x' hxx
  have h1 : g.1 (fun _ => x) d = g.1 (fun _ => x') d := by
    funext j
    rw [Fin.fin_one_eq_zero j]
    exact hxx
  exact congrFun (g.2.2 d h1) 0

/-- A run of steps that each add their word's bit 0 and a constant `C i` adds them all. -/
theorem foldl_lowBit {w : ℕ} :
    ∀ (N : ℕ) (G : Fin N → (Fin 1 → BitVec (w + 1)) → BitVec (w + 1) → Fin 1 → BitVec (w + 1)) (C : Fin N → Bool), (∀ i s e, (G i s e 0).getLsbD 0 = ((s 0).getLsbD 0 ^^ e.getLsbD 0 ^^ C i)) →
    ∀ (s : Fin 1 → BitVec (w + 1)) (d : Fin N → BitVec (w + 1)),
      (Fin.foldl N (fun t i => G i t (d i)) s 0).getLsbD 0 =
        Fin.foldl N (fun x i => x ^^ (d i).getLsbD 0 ^^ C i) ((s 0).getLsbD 0)
  | 0, _, _, _, _, _ => by simp
  | N + 1, G, C, hG, s, d => by
    have ih := foldl_lowBit N (fun i => G i.castSucc) (fun i => C i.castSucc) (fun i => hG i.castSucc) s
      (fun i => d i.castSucc)
    try simp only at ih
    simp only [Fin.foldl_succ_last, hG]
    rw [ih]

theorem xor_shuffle (p q u v : Bool) : (p ^^ q ^^ u ^^ v) = ((p ^^ u) ^^ (q ^^ v)) := by
  cases p <;> cases q <;> cases u <;> cases v <;> rfl

theorem foldl_xor_split : ∀ (N : ℕ) (a b : Fin N → Bool) (x : Bool),
    Fin.foldl N (fun y i => y ^^ a i ^^ b i) x =
      (Fin.foldl N (fun y i => y ^^ a i) x ^^ Fin.foldl N (fun y i => y ^^ b i) false)
  | 0, _, _, _ => by simp
  | N + 1, a, b, x => by
    have ih := foldl_xor_split N (fun i => a i.castSucc) (fun i => b i.castSucc) x
    try simp only at ih
    simp only [Fin.foldl_succ_last]
    rw [ih]
    exact xor_shuffle _ _ _ _

theorem getLsbD_mul_odd {n : ℕ} (p a : BitVec (n + 1)) (ha : a.toNat % 2 = 1) :
    (p * a).getLsbD 0 = p.getLsbD 0 := by
  have h1 : a.setWidth 1 = 1#1 := by
    apply BitVec.eq_of_toNat_eq
    simp [BitVec.toNat_setWidth, ha]
  have h2 := congrArg (fun x => x.getLsbD 0) (BitVec.setWidth_mul p a (by omega : 1 ≤ n + 1))
  simp only [h1, BitVec.mul_one, BitVec.getLsbD_setWidth] at h2
  simpa using h2

theorem getLsbD_add_zero {n : ℕ} (p q : BitVec (n + 1)) : (p + q).getLsbD 0 = (p.getLsbD 0 ^^ q.getLsbD 0) := by
  rw [BitVec.getLsbD_add (by omega : 0 < n + 1), BitVec.carry_zero, Bool.xor_false]

theorem liHalf_lowBit {n : ℕ} (k : BitVec (n + 1)) {a : BitVec (n + 1)} (ha : a.toNat % 2 = 1) (y : BitVec (n + 1))
    (d : BitVec (n + 1) × BitVec (n + 1)) :
    (Pouw.Receipt.liHalf k a y d).getLsbD 0 = (y.getLsbD 0 ^^ d.1.getLsbD 0 ^^ k.getLsbD 0 ^^ d.2.getLsbD 0) := by
  simp only [Pouw.Receipt.liHalf]
  rw [getLsbD_add_zero, getLsbD_mul_odd _ _ ha, BitVec.getLsbD_xor, BitVec.getLsbD_xor]

end LowBit

open LowBit

theorem OneRegisterLowBit : Accounting.Guarantees.TS.OneRegisterLowBit := by
  intro w N g hg
  choose H hH using hg
  have lat : ∀ i a e, H i a e = (a ^^ e ^^ H i false false) := by
    intro i
    apply latin
    · intro a t
      obtain ⟨d, hd⟩ := word_surj (g i) (fun _ => BitVec.ofNat (w + 1) a.toNat) (BitVec.ofNat (w + 1) t.toNat)
      have hd' : (g i).1 (fun _ => BitVec.ofNat (w + 1) a.toNat) d 0 = BitVec.ofNat (w + 1) t.toNat := hd
      have he := hH i (fun _ => BitVec.ofNat (w + 1) a.toNat) d
      simp only [hd', getLsbD_ofNat_toNat] at he
      exact ⟨d.getLsbD 0, he.symm⟩
    · intro e t
      obtain ⟨x, hx⟩ := state_surj (g i) (BitVec.ofNat (w + 1) e.toNat) (BitVec.ofNat (w + 1) t.toNat)
      have hx' : (g i).1 (fun _ => x) (BitVec.ofNat (w + 1) e.toNat) 0 = BitVec.ofNat (w + 1) t.toNat := hx
      have he := hH i (fun _ => x) (BitVec.ofNat (w + 1) e.toNat)
      simp only [hx', getLsbD_ofNat_toNat] at he
      exact ⟨x.getLsbD 0, he.symm⟩
  refine ⟨Fin.foldl N (fun x i => x ^^ H i false false) false, fun s d => ?_⟩
  exact (foldl_lowBit N (fun i => (g i).1) (fun i => H i false false) (fun i s e => (hH i s e).trans (lat i _ _))
    s d).trans (foldl_xor_split N (fun i => (d i).getLsbD 0) (fun i => H i false false) ((s 0).getLsbD 0))

theorem LiLowBit : Accounting.Guarantees.TS.LiLowBit := by
  intro n hs
  induction hs with
  | nil => intro y; rfl
  | cons hd tl ih =>
    intro y
    simp only [List.map_cons, Pouw.Receipt.liChain, List.foldl_cons, Pouw.Receipt.LiOddHalfStep.val] at ih ⊢
    rw [ih, liHalf_lowBit _ hd.2.1.2]

end Accounting.SecurityProofs.TS
