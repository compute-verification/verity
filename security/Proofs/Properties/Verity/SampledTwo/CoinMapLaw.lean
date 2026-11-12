module

public import Proofs.Properties.Verity.SampledTwo.CoinMap

/-!
# The two-stage leg's coin map: the lemmas its law is proved from

`CoinMapLaw`'s first clause at the coin map (`Proofs.Properties.Verity.SampledTwo.CoinMap`) is proved along `relK`, one
coin node at a time. At a written node the node's value is a function of a block of the run's coins, and everything
the map writes after it reads only coins past that block; so the block is uniform and independent of the rest, and the
write costs only its shortfall below uniform. This module holds those pieces.

* **An element's bits** (`natOfBits_lt`, `testBit_natOfBits`, `bit_elemOfBits`, `eq_of_bit`, `bitsEquiv`): an element of
  GF(2^128) is its 128 bits, `F128.bit` one way and `elemOfBits` the other.
* **Averages over blocks** (`avg_split`, `avg_restrict`, `avg_indep`): uniform coins restricted along an injective map
  are uniform, and a function of a block of uniform coordinates is independent of what reads only the others.
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open scoped ENNReal

/-! ## An element's bits -/

theorem shiftBit (x : UInt64) {i : ℕ} (hi : i < 64) : ((x >>> i.toUInt64) &&& 1 == 1) = x.toNat.testBit i := by
  have h : ((x >>> i.toUInt64) &&& 1).toNat = x.toNat >>> i % 2 := by
    rw [UInt64.toNat_and, UInt64.toNat_shiftRight, UInt64.toNat_one, Nat.and_one_is_mod]
    simp [Nat.mod_eq_of_lt hi, Nat.toUInt64]
  rw [Bool.beq_eq_decide_eq]
  simp only [← UInt64.toNat_inj, h, UInt64.toNat_one, Nat.testBit]
  rcases Nat.mod_two_eq_zero_or_one (x.toNat >>> i) with h2 | h2 <;> simp [h2]

theorem natOfBits_lt : ∀ (n : ℕ) (g : ℕ → Bool), natOfBits n g < 2 ^ n
  | 0, _ => by simp [natOfBits]
  | n + 1, g => by
    have h := natOfBits_lt n (fun j => g (j + 1))
    have hb := Bool.toNat_le (g 0)
    rw [natOfBits, pow_succ]
    omega

theorem testBit_natOfBits : ∀ (n : ℕ) (g : ℕ → Bool) (j : ℕ), (natOfBits n g).testBit j = (decide (j < n) && g j)
  | 0, g, j => by simp [natOfBits]
  | n + 1, g, 0 => by
    rw [natOfBits, ← Nat.bit_val, Nat.testBit_bit_zero]
    simp
  | n + 1, g, j + 1 => by
    rw [natOfBits, ← Nat.bit_val, Nat.testBit_bit_succ, testBit_natOfBits n]
    simp

/-- `elemOfBits g`'s bit `b < 128` is `g b`. -/
theorem bit_elemOfBits (g : ℕ → Bool) {b : ℕ} (hb : b < 128) : (elemOfBits g).bit b = g b := by
  unfold Flock.F128.bit elemOfBits
  dsimp only
  by_cases h : b < 64
  · rw [if_pos h, shiftBit _ h, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (natOfBits_lt 64 g), testBit_natOfBits]
    simp [h]
  · rw [if_neg h, shiftBit _ (by omega : b - 64 < 64), UInt64.toNat_ofNat', Nat.mod_eq_of_lt (natOfBits_lt 64 _),
      testBit_natOfBits, show 64 + (b - 64) = b by omega]
    simp [show b - 64 < 64 by omega]

theorem testBit_toNat_ge (x : UInt64) {j : ℕ} (hj : 64 ≤ j) : x.toNat.testBit j = false :=
  Nat.testBit_lt_two_pow (lt_of_lt_of_le x.toNat_lt (Nat.pow_le_pow_right (by norm_num) hj))

/-- **An element is its 128 bits.** -/
theorem eq_of_bit {x y : Flock.F128} (h : ∀ b < 128, x.bit b = y.bit b) : x = y := by
  obtain ⟨xl, xh⟩ := x
  obtain ⟨yl, yh⟩ := y
  have hl : xl = yl := by
    apply UInt64.toNat_inj.mp
    apply Nat.eq_of_testBit_eq
    intro j
    by_cases hj : j < 64
    · have e := h j (by omega)
      unfold Flock.F128.bit at e
      dsimp only at e
      rwa [if_pos hj, if_pos hj, shiftBit _ hj, shiftBit _ hj] at e
    · rw [testBit_toNat_ge _ (by omega), testBit_toNat_ge _ (by omega)]
  have hh : xh = yh := by
    apply UInt64.toNat_inj.mp
    apply Nat.eq_of_testBit_eq
    intro j
    by_cases hj : j < 64
    · have e := h (j + 64) (by omega)
      unfold Flock.F128.bit at e
      dsimp only at e
      rwa [if_neg (by omega), if_neg (by omega), Nat.add_sub_cancel, shiftBit _ hj, shiftBit _ hj] at e
    · rw [testBit_toNat_ge _ (by omega), testBit_toNat_ge _ (by omega)]
  rw [hl, hh]

/-- **An element of GF(2^128) as its 128 bits** (`F128.bit`). -/
def bitsEquiv : Flock.F128 ≃ (Fin 128 → Bool) where
  toFun a := fun b => a.bit b.val
  invFun g := elemOfBits fun b => if h : b < 128 then g ⟨b, h⟩ else false
  left_inv a := eq_of_bit fun b hb => by
    rw [bit_elemOfBits _ hb]
    simp [hb]
  right_inv g := funext fun b => by
    show (elemOfBits _).bit b.val = g b
    rw [bit_elemOfBits _ b.isLt]
    simp

/-! ## Averages over blocks -/

/-- An average over a type split in two is the average over each part. -/
theorem avg_split {R P Q : Type} [Fintype R] [Fintype P] [Fintype Q] (e : R ≃ P × Q) (F : P → Q → ℝ≥0∞) :
    avg (fun r => F (e r).1 (e r).2) = avg fun x => avg fun y => F x y := by
  rw [Tape.avg_equiv e (fun p => F p.1 p.2)]
  exact Tape.avg_prod _

/-- **Uniform coordinates restricted along an injective map are uniform.** -/
theorem avg_restrict {I J T : Type} [Fintype I] [DecidableEq I] [Fintype J] [Fintype T] (κ : J → I)
    (hκ : Function.Injective κ) (H : (J → T) → ℝ≥0∞) :
    avg (fun f : I → T => H (f ∘ κ)) = avg H := by
  classical
  let e := Equiv.piEquivPiSubtypeProd (fun a : I => a ∈ Set.range κ) (fun _ => T)
  let k : J ≃ Set.range κ := Equiv.ofInjective κ hκ
  calc avg (fun f : I → T => H (f ∘ κ))
      = avg (fun f : I → T => H fun j => (e f).1 (k j)) := rfl
    _ = avg fun p : {a // a ∈ Set.range κ} → T => avg fun _ : {a // a ∉ Set.range κ} → T => H fun j => p (k j) :=
        avg_split e (fun p _ => H fun j => p (k j))
    _ = avg fun p : {a // a ∈ Set.range κ} → T => H fun j => p (k j) := by
        simp only [avg_const]
    _ = avg H := Tape.avg_equiv (Equiv.arrowCongr k.symm (Equiv.refl T)) H

/-- **A function of a block of uniform coordinates is independent of what reads only the others**: where `v` reads only
the coordinates `A` and is uniform, and `G x` reads only the others, averaging `G (v r) r` is averaging `G x r` over
independent `x` and `r`. -/
theorem avg_indep {ι T C : Type} [Fintype ι] [DecidableEq ι] [Fintype T] [Nonempty T] [Fintype C] (A : ι → Prop)
    [DecidablePred A] (v : (ι → T) → C) (hvA : ∀ r r' : ι → T, (∀ a, A a → r a = r' a) → v r = v r')
    (hv : ∀ H : C → ℝ≥0∞, avg (fun r => H (v r)) = avg H) (G : C → (ι → T) → ℝ≥0∞)
    (hG : ∀ x (r r' : ι → T), (∀ a, ¬ A a → r a = r' a) → G x r = G x r') :
    avg (fun r => G (v r) r) = avg fun x => avg fun r => G x r := by
  let e := Equiv.piEquivPiSubtypeProd A (fun _ => T)
  let y₀ : {a // ¬ A a} → T := fun _ => Classical.arbitrary T
  let x₀ : {a // A a} → T := fun _ => Classical.arbitrary T
  have hsymm : ∀ (p : {a // A a} → T) (q : {a // ¬ A a} → T) (a : ι),
      e.symm (p, q) a = if h : A a then p ⟨a, h⟩ else q ⟨a, h⟩ := fun _ _ _ => rfl
  let v₁ : ({a // A a} → T) → C := fun p => v (e.symm (p, y₀))
  let G₂ : C → ({a // ¬ A a} → T) → ℝ≥0∞ := fun x q => G x (e.symm (x₀, q))
  have hv₁ : ∀ r, v r = v₁ (e r).1 := fun r => hvA _ _ fun a ha => by
    show r a = e.symm ((e r).1, y₀) a
    rw [hsymm, dif_pos ha]
    rfl
  have hG₂ : ∀ x r, G x r = G₂ x (e r).2 := fun x r => hG x _ _ fun a ha => by
    show r a = e.symm (x₀, (e r).2) a
    rw [hsymm, dif_neg ha]
    rfl
  have hrest : ∀ x, avg (fun r => G x r) = avg fun q => G₂ x q := fun x => by
    simp only [hG₂ x]
    rw [avg_split e (fun _ q => G₂ x q)]
    simp only [avg_const]
  calc avg (fun r => G (v r) r)
      = avg (fun r => G₂ (v₁ (e r).1) (e r).2) := congrArg avg (funext fun r => by rw [hv₁ r, hG₂])
    _ = avg fun p => avg fun q => G₂ (v₁ p) q := avg_split e (fun p q => G₂ (v₁ p) q)
    _ = avg (fun r => avg fun q => G₂ (v₁ (e r).1) q) := by
        rw [avg_split e (fun p (_ : {a // ¬ A a} → T) => avg fun q => G₂ (v₁ p) q)]
        simp only [avg_const]
    _ = avg fun r => avg fun q => G₂ (v r) q := congrArg avg (funext fun r => by rw [hv₁ r])
    _ = avg fun x => avg fun q => G₂ x q := hv fun x => avg fun q => G₂ x q
    _ = avg fun x => avg fun r => G x r := congrArg avg (funext fun x => (hrest x).symm)

end Verity.SampledTwo.CoinMap
