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

theorem avg_swap {C D : Type} [Fintype C] [Fintype D] (f : C → D → ℝ≥0∞) :
    avg (fun c => avg fun d => f c d) = avg fun d => avg fun c => f c d := by
  unfold avg
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun d _ => Finset.sum_congr rfl fun c _ => ?_
  ring

/-- An average over independent coordinates of a function of one of them is its average over that one. -/
theorem avg_coord {C : Type} [Fintype C] [DecidableEq C] {T : C → Type} [∀ c, Fintype (T c)] [∀ c, Nonempty (T c)]
    (c : C) (F : T c → ℝ≥0∞) : avg (fun f : (c : C) → T c => F (f c)) = avg F := by
  calc avg (fun f : (c : C) → T c => F (f c))
      = avg (fun y : T c × ((j : {j // j ≠ c}) → T j) => F y.1) :=
        Tape.avg_equiv (Equiv.piSplitAt c T) (fun y => F y.1)
    _ = avg F := by
        rw [Tape.avg_prod]
        simp only [avg_const]

/-- `avg_le_write` at a function of the node's value alone. -/
theorem avg_le_write1 {P C : Type} [Fintype P] [Nonempty P] [Fintype C] [Nonempty C] (w : P → C → C)
    (A : C → ℝ≥0∞) (hA : ∀ c, A c ≤ 1) : avg A ≤ avg (fun p : P => avg fun c : C => A (w p c)) + shortfall w := by
  have h := avg_le_write (T := fun _ : C => Unit) w (fun z => A z.1) fun z => hA z.1
  have e : ∀ B : C → ℝ≥0∞, avg (fun z : C × (C → Unit) => B z.1) = avg B := fun B => by
    rw [Tape.avg_prod]
    simp only [avg_const]
  rw [e A] at h
  simpa only [e] using h

/-! ## The law at one node -/

variable {β : Type}

/-- A tape of a coin node, averaged over a function of its value and the tape after it. -/
theorem avg_tape_coin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) (Φ : (c : C) → Tape (k c) (s c) → ℝ≥0∞) :
    avg (fun t : C × ((c : C) → Tape (k c) (s c)) => Φ t.1 (t.2 t.1)) =
      avg fun c => avg fun τ : Tape (k c) (s c) => Φ c τ := by
  classical
  rw [Tape.avg_prod]
  exact congrArg avg (funext fun c => avg_coord (T := fun c => Tape (k c) (s c)) c (Φ c))

/-- A tape of a coin node, averaged over a function of the tape after a fixed value. -/
theorem avg_tape_at {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) (c₀ : C) (Φ : Tape (k c₀) (s c₀) → ℝ≥0∞) :
    avg (fun t : C × ((c : C) → Tape (k c) (s c)) => Φ (t.2 c₀)) = avg Φ := by
  classical
  rw [Tape.avg_prod]
  simp only [avg_const]
  exact avg_coord (T := fun c => Tape (k c) (s c)) c₀ Φ

/-- **The law at a node** with the map `P` of uniform coins `R` and the term `G`: every bounded function of the play's
output is on average at most its average at the mapped tape and uniform coins, plus `G`. -/
def LawHolds {R : Type} [Fintype R] (g : Game β) (s : Strategy g) (P : R → Tape g s → Tape g s) (G : ℝ≥0∞) : Prop :=
  ∀ H : β → ℝ≥0∞, (∀ b, H b ≤ 1) →
    avg (fun t : Tape g s => H (Tape.out g s t)) ≤
      avg (fun r : R => avg fun t : Tape g s => H (Tape.out g s (P r t))) + G

variable {R : Type} [Fintype R] [Nonempty R]

theorem law_mono {g : Game β} {s : Strategy g} {P : R → Tape g s → Tape g s} {G G' : ℝ≥0∞} (h : LawHolds g s P G)
    (hG : G ≤ G') : LawHolds g s P G' :=
  fun H hH => (h H hH).trans (add_le_add_left hG _)

theorem law_id (g : Game β) (s : Strategy g) : LawHolds g s (fun (_ : R) t => t) 0 := fun H _ => by
  rw [avg_const, add_zero]

theorem law_opt {γ : Type} (n : Game β) (y : γ → Game β) {fn : R → (s : Strategy n) → Tape n s → Tape n s}
    {fy : R → (a : γ) → (s : Strategy (y a)) → Tape (y a) s → Tape (y a) s} {gn : Strategy n → ℝ≥0∞}
    {gy : (a : γ) → Strategy (y a) → ℝ≥0∞} (hn : ∀ s, LawHolds n s (fun r => fn r s) (gn s))
    (hy : ∀ a s, LawHolds (y a) s (fun r => fy r a s) (gy a s)) :
    ∀ (o : Option γ) (s : Strategy (optGame o n y)),
      LawHolds (optGame o n y) s (fun r => placeOpt n y (fn r) (fy r) o s) (gapOpt n y gn gy o s)
  | none, s => hn s
  | some a, s => hy a s

theorem law_ite {c : Prop} (dc : Decidable c) (G₁ G₂ : Game β) {f₁ : R → (s : Strategy G₁) → Tape G₁ s → Tape G₁ s}
    {f₂ : R → (s : Strategy G₂) → Tape G₂ s → Tape G₂ s} {g₁ : Strategy G₁ → ℝ≥0∞} {g₂ : Strategy G₂ → ℝ≥0∞}
    (h₁ : ∀ s, LawHolds G₁ s (fun r => f₁ r s) (g₁ s)) (h₂ : ∀ s, LawHolds G₂ s (fun r => f₂ r s) (g₂ s)) :
    ∀ s : Strategy (@ite _ c dc G₁ G₂),
      LawHolds (@ite _ c dc G₁ G₂) s (fun r => placeIte dc G₁ G₂ s (f₁ r) (f₂ r)) (gapIte dc G₁ G₂ s g₁ g₂) := by
  cases dc with
  | isFalse _ => exact h₂
  | isTrue _ => exact h₁

/-- **At a coin node the map keeps**: the node's term is the average of the terms after it. -/
theorem law_keep {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β} (s : Strategy (@Game.coin β C iF iN k))
    {f : R → (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s} {G : C → ℝ≥0∞}
    (h : ∀ c, LawHolds (k c) (s c) (fun r => f r c (s c)) (G c)) :
    LawHolds (@Game.coin β C iF iN k) s (fun r => placeCoin k id (f r) s) (avg G) := by
  intro H hH
  have e : ∀ r, avg (fun t : Tape (@Game.coin β C iF iN k) s =>
      H (Tape.out _ s (placeCoin k id (f r) s t))) =
        avg fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) (f r c (s c) τ)) :=
    fun r => avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) (f r c (s c) τ))
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => H (Tape.out _ s t))
      = avg fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) τ) :=
        avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) τ)
    _ ≤ avg fun c => (avg fun r : R => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) (f r c (s c) τ))) +
          G c := avg_mono fun c => h c H hH
    _ = (avg fun c => avg fun r : R => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) (f r c (s c) τ))) +
          avg G := avg_add _ _
    _ = (avg fun r : R => avg fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) (f r c (s c) τ))) +
          avg G := by rw [avg_swap]
    _ = (avg fun r : R => avg fun t : Tape (@Game.coin β C iF iN k) s =>
          H (Tape.out _ s (placeCoin k id (f r) s t))) + avg G :=
        congrArg (· + avg G) (congrArg avg (funext fun r => (e r).symm))

/-- **The split of the coins at a write**: averaging a function of the written value `v r` and of the coins is averaging
it at independent values, for every function that `Rel`-related coins give the same value. -/
def Splits {C : Type} [Fintype C] (v : R → C) (Rel : R → R → Prop) : Prop :=
  ∀ F : C → R → ℝ≥0∞, (∀ c r r', Rel r r' → F c r = F c r') →
    avg (fun r => F (v r) r) = avg fun c => avg fun r => F c r

/-- **At a coin node the map writes `v r`**, a value independent of what the map reads after it: the node's term is the
average of the terms after it. -/
theorem law_set {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β} (s : Strategy (@Game.coin β C iF iN k))
    {f : R → (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s} {G : C → ℝ≥0∞} (v : R → C)
    (Rel : R → R → Prop) (hv : Splits v Rel) (hf : ∀ c r r' τ, Rel r r' → f r c (s c) τ = f r' c (s c) τ)
    (h : ∀ c, LawHolds (k c) (s c) (fun r => f r c (s c)) (G c)) :
    LawHolds (@Game.coin β C iF iN k) s (fun r => placeCoin k (fun _ => v r) (f r) s) (avg G) := by
  intro H hH
  let B : C → R → ℝ≥0∞ := fun c r => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) (f r c (s c) τ))
  have e : ∀ r, avg (fun t : Tape (@Game.coin β C iF iN k) s =>
      H (Tape.out _ s (placeCoin k (fun _ => v r) (f r) s t))) = B (v r) r :=
    fun r => avg_tape_at s (v r) fun τ => H (Tape.out (k (v r)) (s (v r)) (f r (v r) (s (v r)) τ))
  have hB : ∀ c r r', Rel r r' → B c r = B c r' := fun c r r' hr =>
    congrArg avg (funext fun τ => by rw [hf c r r' τ hr])
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => H (Tape.out _ s t))
      = avg fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) τ) :=
        avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) τ)
    _ ≤ avg fun c => (avg fun r : R => B c r) + G c := avg_mono fun c => h c H hH
    _ = (avg fun c => avg fun r : R => B c r) + avg G := avg_add _ _
    _ = (avg fun r : R => B (v r) r) + avg G := by rw [hv B hB]
    _ = (avg fun r : R => avg fun t : Tape (@Game.coin β C iF iN k) s =>
          H (Tape.out _ s (placeCoin k (fun _ => v r) (f r) s t))) + avg G :=
        congrArg (· + avg G) (congrArg avg (funext fun r => (e r).symm))

/-- **At a coin node the map writes `W (e r)` of the node's value**, with `e r` independent of what the map reads after
it: the node's term is `W`'s shortfall below uniform plus the average of the terms after the written value. -/
theorem law_write {C P : Type} {iF : Fintype C} {iN : Nonempty C} [Fintype P] [Nonempty P] {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) {f : R → (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s}
    {G : C → ℝ≥0∞} (W : P → C → C) (e : R → P) (Rel : R → R → Prop) (he : Splits e Rel)
    (hf : ∀ c r r' τ, Rel r r' → f r c (s c) τ = f r' c (s c) τ)
    (h : ∀ c, LawHolds (k c) (s c) (fun r => f r c (s c)) (G c)) :
    LawHolds (@Game.coin β C iF iN k) s (fun r => placeCoin k (W (e r)) (f r) s)
      (shortfall W + avg fun p => avg fun c => G (W p c)) := by
  intro H hH
  let A : C → ℝ≥0∞ := fun c => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) τ)
  let B : C → R → ℝ≥0∞ := fun c r => avg fun τ : Tape (k c) (s c) => H (Tape.out (k c) (s c) (f r c (s c) τ))
  have hA : ∀ c, A c ≤ 1 := fun c =>
    (avg_mono fun τ => hH (Tape.out (k c) (s c) τ)).trans_eq (avg_const 1)
  have e₁ : ∀ r, avg (fun t : Tape (@Game.coin β C iF iN k) s =>
      H (Tape.out _ s (placeCoin k (W (e r)) (f r) s t))) = avg fun c => B (W (e r) c) r := fun r => by
    show avg (fun t : C × ((c : C) → Tape (k c) (s c)) =>
      H (Tape.out (k (W (e r) t.1)) (s (W (e r) t.1)) (f r (W (e r) t.1) (s (W (e r) t.1)) (t.2 (W (e r) t.1))))) = _
    rw [Tape.avg_prod]
    exact congrArg avg (funext fun c => avg_coord (T := fun c => Tape (k c) (s c)) (W (e r) c)
      fun τ => H (Tape.out (k (W (e r) c)) (s (W (e r) c)) (f r (W (e r) c) (s (W (e r) c)) τ)))
  have hB : ∀ p r r', Rel r r' → (avg fun c => B (W p c) r) = avg fun c => B (W p c) r' := fun p r r' hr =>
    congrArg avg (funext fun c => congrArg avg (funext fun τ => by rw [hf (W p c) r r' τ hr]))
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => H (Tape.out _ s t))
      = avg A := avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) τ)
    _ ≤ avg (fun p : P => avg fun c : C => A (W p c)) + shortfall W := avg_le_write1 W A hA
    _ ≤ avg (fun p : P => avg fun c : C => (avg fun r : R => B (W p c) r) + G (W p c)) + shortfall W := by
        gcongr with p c
        exact h (W p c) H hH
    _ = (avg fun p : P => avg fun c : C => avg fun r : R => B (W p c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W := by
        simp only [avg_add]
    _ = (avg fun p : P => avg fun r : R => avg fun c : C => B (W p c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W := by
        simp only [avg_swap (fun (c : C) (r : R) => B _ r)]
    _ = (avg fun r : R => avg fun c : C => B (W (e r) c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W := by
        rw [he (fun p r => avg fun c => B (W p c) r) hB]
    _ = (avg fun r : R => avg fun t : Tape (@Game.coin β C iF iN k) s =>
          H (Tape.out _ s (placeCoin k (W (e r)) (f r) s t))) +
          (shortfall W + avg fun p => avg fun c => G (W p c)) := by
        rw [add_assoc, add_comm (avg fun p : P => avg fun c : C => G (W p c)) (shortfall W)]
        exact congrArg (· + _) (congrArg avg (funext fun r => (e₁ r).symm))

theorem law_congr {g : Game β} {s : Strategy g} {P P' : R → Tape g s → Tape g s} {G : ℝ≥0∞}
    (hP : ∀ r t, P r t = P' r t) (h : LawHolds g s P' G) : LawHolds g s P G := by
  have : P = P' := funext fun r => funext fun t => hP r t
  subst this
  exact h

theorem splits_map {C D : Type} [Fintype C] [Fintype D] {v : R → C} {Rel : R → R → Prop} (h : Splits v Rel)
    (φ : C ≃ D) : Splits (fun r => φ (v r)) Rel := fun F hF =>
  calc avg (fun r => F (φ (v r)) r) = avg fun c => avg fun r => F (φ c) r :=
        h (fun c r => F (φ c) r) fun c => hF (φ c)
    _ = avg fun d => avg fun r => F d r := Tape.avg_equiv φ fun d => avg fun r => F d r

theorem splits_congr {C : Type} [Fintype C] {v v' : R → C} {Rel : R → R → Prop} (hv : ∀ r, v' r = v r)
    (h : Splits v Rel) : Splits v' Rel := by
  have : v' = v := funext hv
  subst this
  exact h

/-! ## The chunks a write reads -/

/-- **Coins that agree off the chunk**: every element but round `i`'s `m` from `u`. -/
def OffChunk {K : ℕ} (i : Fin K) (u m : ℕ) (r r' : Fin K → FlockDeploy.Coin) : Prop :=
  ∀ (j : Fin K) (a : Fin 65536), ¬ (j = i ∧ u ≤ a.val ∧ a.val < u + m) → r j a = r' j a

/-- **A chunk is uniform and independent of the coins off it.** -/
theorem splits_chunk {K : ℕ} (i : Fin K) (u m : ℕ) (hm : u + m ≤ 65536) :
    Splits (fun r : Fin K → FlockDeploy.Coin => fun q : Fin m => (r i ⟨u + q.val, by omega⟩ : FlockLevel3.GF128))
      (OffChunk i u m) := by
  classical
  intro F hF
  let e : (Fin K × Fin 65536 → FlockLevel3.GF128) ≃ (Fin K → FlockDeploy.Coin) :=
    Equiv.curry (Fin K) (Fin 65536) FlockLevel3.GF128
  let κ : Fin m → Fin K × Fin 65536 := fun q => (i, ⟨u + q.val, by omega⟩)
  have hκ : Function.Injective κ := fun q q' h => by
    have := congrArg (fun a => a.2.val) h
    exact Fin.ext (by simpa [κ] using this)
  let A : Fin K × Fin 65536 → Prop := fun a => a.1 = i ∧ u ≤ a.2.val ∧ a.2.val < u + m
  let v' : (Fin K × Fin 65536 → FlockLevel3.GF128) → (Fin m → FlockLevel3.GF128) := fun ρ => ρ ∘ κ
  have hvA : ∀ ρ ρ' : Fin K × Fin 65536 → FlockLevel3.GF128, (∀ a, A a → ρ a = ρ' a) → v' ρ = v' ρ' :=
    fun ρ ρ' h => funext fun q => h (κ q) ⟨rfl, Nat.le_add_right _ _, by have := q.isLt; simp only; omega⟩
  have hv : ∀ H : (Fin m → FlockLevel3.GF128) → ℝ≥0∞, avg (fun ρ => H (v' ρ)) = avg H := fun H =>
    avg_restrict κ hκ H
  calc avg (fun r : Fin K → FlockDeploy.Coin => F (fun q : Fin m => (r i ⟨u + q.val, by omega⟩ : FlockLevel3.GF128)) r)
      = avg (fun ρ => F (v' ρ) (e ρ)) :=
        (Tape.avg_equiv e fun r => F (fun q : Fin m => (r i ⟨u + q.val, by omega⟩ : FlockLevel3.GF128)) r).symm
    _ = avg fun x => avg fun ρ => F x (e ρ) :=
        avg_indep A v' hvA hv (fun x ρ => F x (e ρ)) fun x ρ ρ' h => hF x _ _ fun j a hn => h (j, a) hn
    _ = avg fun x => avg fun r => F x r := congrArg avg (funext fun x => Tape.avg_equiv e (F x))

/-- A fresh coin's 512 bits from its 4 elements. -/
def freshEquiv : (Fin 4 → FlockLevel3.GF128) ≃ (Fin 512 → Bool) :=
  ((Equiv.arrowCongr (Equiv.refl (Fin 4)) bitsEquiv).trans (Equiv.curry (Fin 4) (Fin 128) Bool).symm).trans
    (Equiv.arrowCongr finProdFinEquiv (Equiv.refl Bool))

theorem freshVal_eq (c : FlockDeploy.Coin) (u : ℕ) (hu : u + 4 ≤ 65536) :
    freshVal c u = freshEquiv fun q : Fin 4 => (c ⟨u + q.val, by omega⟩ : FlockLevel3.GF128) := funext fun j => by
  have h1 : (128 * u + j.val) / 128 = u + j.val / 128 := by omega
  have h2 : (128 * u + j.val) % 128 = j.val % 128 := by omega
  have h3 : u + j.val / 128 < 65536 := by have := j.isLt; omega
  show (c.at ((128 * u + j.val) / 128)).bit ((128 * u + j.val) % 128) =
    Flock.F128.bit (c ⟨u + (finProdFinEquiv.symm j).1.val, _⟩) (finProdFinEquiv.symm j).2.val
  rw [h1, h2, finProdFinEquiv_symm_apply, Fin.coe_divNat, Fin.coe_modNat]
  simp only [FlockDeploy.Coin.at, dif_pos h3]

/-- **A fresh coin's write is uniform and independent of the coins off its chunk.** -/
theorem splits_fresh {K : ℕ} (i : Fin K) (u : ℕ) (hu : u + 4 ≤ 65536) :
    Splits (fun r : Fin K → FlockDeploy.Coin => freshVal (r i) u) (OffChunk i u 4) :=
  splits_congr (fun r => freshVal_eq (r i) u hu) (splits_map (splits_chunk i u 4 hu) freshEquiv)

/-- An opening's `n` elements, each as its 128 bits. -/
def openElems (c : FlockDeploy.Coin) (u n : ℕ) : Fin n → Fin 128 → Bool := fun q b => (c.at (u + q.val)).bit b.val

theorem openBits_eq (c : FlockDeploy.Coin) (u n : ℕ) : openBits c u n = bitsOf (openElems c u n) := by
  unfold openBits bitsOf openElems coinBit
  congr 1
  funext t
  have h1 : (128 * u + t.val) / 128 = u + t.val / 128 := by omega
  have h2 : (128 * u + t.val) % 128 = t.val % 128 := by omega
  rw [h1, h2]

/-- **An opening's elements are uniform and independent of the coins off its chunk.** -/
theorem splits_open {K : ℕ} (i : Fin K) (u n : ℕ) (hu : u + n ≤ 65536) :
    Splits (fun r : Fin K → FlockDeploy.Coin => openElems (r i) u n) (OffChunk i u n) := by
  refine splits_congr (fun r => funext fun q => ?_)
    (splits_map (splits_chunk i u n hu) (Equiv.arrowCongr (Equiv.refl (Fin n)) bitsEquiv))
  have h3 : u + q.val < 65536 := by have := q.isLt; omega
  funext b
  show (FlockDeploy.Coin.at (r i) (u + q.val)).bit b.val = Flock.F128.bit (r i ⟨u + q.val, h3⟩) b.val
  simp only [FlockDeploy.Coin.at, dif_pos h3]

end Verity.SampledTwo.CoinMap
