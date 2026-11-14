module

public import Proofs.Properties.Verity.SampledTwo.CoinMap

/-!
# The two-stage leg's coin map: its sampling design (`lawAt`)

`LawAt` at the coin map (`Proofs.Properties.Verity.SampledTwo.CoinMap`) is proved along `relK`, one coin node at a
time. At a written node the node's value is a function of a chunk of the run's coins, and everything the map writes
after it reads only coins past that chunk; so the chunk is uniform and independent of the rest, and the write costs only
its shortfall below uniform.

* **An element's bits** (`natOfBits_lt`, `testBit_natOfBits`, `bit_elemOfBits`, `eq_of_bit`, `bitsEquiv`): an element of
  GF(2^128) is its 128 bits, `F128.bit` one way and `elemOfBits` the other.
* **Averages over blocks** (`avg_split`, `avg_restrict`, `avg_indep`, `avg_coord`, `avg_le_write1`): uniform coins
  restricted along an injective map are uniform, and a function of a block of uniform coordinates is independent of
  what reads only the others.
* **The sampling design at a node** (`LawHolds`, `law_opt`, `law_ite`, `law_keep`, `law_set`, `law_write`, `law_send`): a bounded
  function of the play is on average at most its average at the mapped tape and uniform coins, plus a term; a kept coin
  node's term is the average of the terms after it, a node written with a value independent of what the map reads after
  it (`Splits`) costs nothing more, and a node written through `W` costs `W`'s shortfall.
* **The chunks** (`OffChunk`, `splits_chunk`, `splits_fresh`, `splits_open`): a round's chunk of elements, a fresh coin's
  512 bits and an opening's elements split from the coins off the chunk.
* **Along the releases** (`AgreeFrom`, `placeK_congr`, `law_fresh`, `law_hello`, `law_round`, `law_open`,
  `law_placeK`): the coin map reads the run's coins only from what the releases before it gave each round, and its sampling design
  holds along `relK` at `gapK`.
* **The first clause** (`law_place`, `lawAt`).
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open Claims
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
theorem avg_restrict {I J T : Type} [Fintype I] [DecidableEq I] [Fintype J] [DecidableEq J] [Fintype T] [Nonempty T]
    (κ : J → I)
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
  simp only [Tape.avg_prod, avg_const] at h
  exact h

/-! ## The sampling design at one node -/

variable {β : Type}

/-- A coin node's tape: its value. -/
abbrev tapeVal {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β} (s : Strategy (@Game.coin β C iF iN k))
    (t : Tape (@Game.coin β C iF iN k) s) : C :=
  (show C × ((c : C) → Tape (k c) (s c)) from t).1

/-- A coin node's tape: the tape after the value `c`. -/
abbrev tapeAt {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β} (s : Strategy (@Game.coin β C iF iN k))
    (t : Tape (@Game.coin β C iF iN k) s) (c : C) : Tape (k c) (s c) :=
  (show C × ((c : C) → Tape (k c) (s c)) from t).2 c

/-- A tape of a coin node, averaged over a function of its value and the tape after `w` of it. -/
theorem avg_tape_map {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) (w : C → C) (Φ : (c : C) → Tape (k (w c)) (s (w c)) → ℝ≥0∞) :
    avg (fun t : Tape (@Game.coin β C iF iN k) s => Φ (tapeVal s t) (tapeAt s t (w (tapeVal s t)))) =
      avg fun c => avg fun τ : Tape (k (w c)) (s (w c)) => Φ c τ := by
  classical
  letI : ∀ c, Fintype (Tape (k c) (s c)) := fun c => Tape.instFintype (k c) (s c)
  have : ∀ c, Nonempty (Tape (k c) (s c)) := fun c => Tape.instNonempty (k c) (s c)
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => Φ (tapeVal s t) (tapeAt s t (w (tapeVal s t))))
      = avg fun c => avg fun g : (c : C) → Tape (k c) (s c) => Φ c (g (w c)) :=
        Tape.avg_prod (C := C) (D := (c : C) → Tape (k c) (s c)) fun t => Φ t.1 (t.2 (w t.1))
    _ = avg fun c => avg fun τ : Tape (k (w c)) (s (w c)) => Φ c τ :=
        congrArg avg (funext fun c => avg_coord (T := fun c => Tape (k c) (s c)) (w c) (Φ c))

/-- A tape of a coin node, averaged over a function of its value and the tape after it. -/
theorem avg_tape_coin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) (Φ : (c : C) → Tape (k c) (s c) → ℝ≥0∞) :
    avg (fun t : Tape (@Game.coin β C iF iN k) s => Φ (tapeVal s t) (tapeAt s t (tapeVal s t))) =
      avg fun c => avg fun τ : Tape (k c) (s c) => Φ c τ :=
  avg_tape_map s (fun c => c) Φ

/-- A tape of a coin node, averaged over a function of the tape after a fixed value. -/
theorem avg_tape_at {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    (s : Strategy (@Game.coin β C iF iN k)) (c₀ : C) (Φ : Tape (k c₀) (s c₀) → ℝ≥0∞) :
    avg (fun t : Tape (@Game.coin β C iF iN k) s => Φ (tapeAt s t c₀)) = avg Φ := by
  have h := avg_tape_map s (fun _ => c₀) fun _ => Φ
  simp only [avg_const] at h
  exact h

/-- **The sampling design at a node** with the map `P` of uniform coins `R` and the term `G`: every bounded function of the play's
output is on average at most its average at the mapped tape and uniform coins, plus `G`. -/
def LawHolds {R : Type} [Fintype R] (g : Game β) (s : Strategy g) (P : R → Tape g s → Tape g s) (G : ℝ≥0∞) : Prop :=
  ∀ H : β → ℝ≥0∞, (∀ b, H b ≤ 1) →
    avg (fun t : Tape g s => H (Tape.out g s t)) ≤
      avg (fun r : R => avg fun t : Tape g s => H (Tape.out g s (P r t))) + G

variable {R : Type} [Fintype R] [Nonempty R]

theorem law_mono {g : Game β} {s : Strategy g} {P : R → Tape g s → Tape g s} {G G' : ℝ≥0∞} (h : LawHolds g s P G)
    (hG : G ≤ G') : LawHolds g s P G' :=
  fun H hH => (h H hH).trans (add_le_add le_rfl hG)

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
    _ = (avg fun r : R => B (v r) r) + avg G := congrArg (· + avg G) (hv B hB).symm
    _ = (avg fun r : R => avg fun t : Tape (@Game.coin β C iF iN k) s =>
          H (Tape.out _ s (placeCoin k (fun _ => v r) (f r) s t))) + avg G :=
        congrArg (· + avg G) (congrArg avg (funext fun r => (e r).symm))

open Classical in
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
      H (Tape.out _ s (placeCoin k (W (e r)) (f r) s t))) = avg fun c => B (W (e r) c) r := fun r =>
    avg_tape_map s (W (e r)) fun c τ =>
      H (Tape.out (k (W (e r) c)) (s (W (e r) c)) (f r (W (e r) c) (s (W (e r) c)) τ))
  have hB : ∀ p r r', Rel r r' → (avg fun c => B (W p c) r) = avg fun c => B (W p c) r' := fun p r r' hr =>
    congrArg avg (funext fun c => congrArg avg (funext fun τ => by rw [hf (W p c) r r' τ hr]))
  calc avg (fun t : Tape (@Game.coin β C iF iN k) s => H (Tape.out _ s t))
      = avg A := avg_tape_coin s fun c τ => H (Tape.out (k c) (s c) τ)
    _ ≤ avg (fun p : P => avg fun c : C => A (W p c)) + shortfall W := avg_le_write1 W A hA
    _ ≤ avg (fun p : P => avg fun c : C => (avg fun r : R => B (W p c) r) + G (W p c)) + shortfall W :=
        add_le_add (avg_mono fun p => avg_mono fun c => h (W p c) H hH) le_rfl
    _ = (avg fun p : P => avg fun c : C => avg fun r : R => B (W p c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W := by
        simp only [avg_add]
    _ = (avg fun p : P => avg fun r : R => avg fun c : C => B (W p c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W :=
        congrArg (fun z => z + (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W)
          (congrArg avg (funext fun p => avg_swap fun (c : C) (r : R) => B (W p c) r))
    _ = (avg fun r : R => avg fun c : C => B (W (e r) c) r) +
          (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W :=
        congrArg (fun z => z + (avg fun p : P => avg fun c : C => G (W p c)) + shortfall W)
          (he (fun p r => avg fun c => B (W p c) r) hB).symm
    _ = (avg fun r : R => avg fun t : Tape (@Game.coin β C iF iN k) s =>
          H (Tape.out _ s (placeCoin k (W (e r)) (f r) s t))) +
          (shortfall W + avg fun p => avg fun c => G (W p c)) := by
        have hX : (avg fun r : R => avg fun c : C => B (W (e r) c) r) =
            avg fun r : R => avg fun t : Tape (@Game.coin β C iF iN k) s =>
              H (Tape.out _ s (placeCoin k (W (e r)) (f r) s t)) :=
          congrArg avg (funext fun r => (e₁ r).symm)
        rw [hX]
        ring

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
    fun ρ ρ' h => funext fun q => h (κ q) ⟨rfl, Nat.le_add_right _ _, by
      have := q.isLt
      show u + q.val < u + m
      omega⟩
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
    Flock.F128.bit (c ⟨u + ((finProdFinEquiv (m := 4) (n := 128)).symm j).1.val, _⟩)
      ((finProdFinEquiv (m := 4) (n := 128)).symm j).2.val
  rw [h1, h2, finProdFinEquiv_symm_apply]
  simp only [Fin.coe_divNat, Fin.coe_modNat, FlockDeploy.Coin.at, dif_pos h3]

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

theorem law_send {M : Type} (kf : M → Game β) (m : M) (s' : Strategy (kf m)) {P : R → Tape (kf m) s' → Tape (kf m) s'}
    {G : ℝ≥0∞} (h : LawHolds (kf m) s' P G) : LawHolds (Game.send M kf) ⟨m, s'⟩ P G := h

/-! ## Along the releases -/

section Along

variable (L : Locate) {K : ℕ}

theorem usedOf_append (i : ℕ) : ∀ (q l₁ l₂ : List Verity.Release),
    usedOf L i q (l₁ ++ l₂) = usedOf L i q l₁ + usedOf L i (q ++ l₁) l₂
  | q, [], l₂ => by simp [usedOf]
  | q, rl :: l₁, l₂ => by
    simp only [List.cons_append, usedOf]
    rw [usedOf_append i (q ++ [rl]) l₁ l₂]
    simp only [List.append_assoc, List.singleton_append]
    omega

theorem usedOf_snoc (i : ℕ) (pre : List Verity.Release) (rl : Verity.Release) :
    usedOf L i [] (pre ++ [rl]) =
      usedOf L i [] pre + (if locAt L pre rl.frame rl.answers = some i then widthOf rl.frame else 0) := by
  rw [usedOf_append]
  simp [usedOf]

/-- **Coins that agree where the releases after `pre` read**: at each round's elements from what `pre` gave it. -/
def AgreeFrom (pre : List Verity.Release) (r r' : Fin K → FlockDeploy.Coin) : Prop :=
  ∀ (j : Fin K) (a : Fin 65536), usedOf L j.val [] pre ≤ a.val → r j a = r' j a

theorem AgreeFrom.snoc {pre : List Verity.Release} {r r' : Fin K → FlockDeploy.Coin} (h : AgreeFrom L pre r r')
    (rl : Verity.Release) : AgreeFrom L (pre ++ [rl]) r r' := fun j a ha =>
  h j a (by rw [usedOf_snoc] at ha; omega)

theorem slot_spec {pre : List Verity.Release} {fr : Verity.Frame} {b : ByteArray} {m : ℕ} {j : Fin K} {u : ℕ}
    (h : slot L K pre fr b m = some (j, u)) :
    locAt L pre fr b = some j.val ∧ u = usedOf L j.val [] pre ∧ u + m ≤ 65536 := by
  unfold slot at h
  split at h
  next i hi =>
    by_cases h₁ : i < K
    · rw [dif_pos h₁] at h
      by_cases h₂ : usedOf L i [] pre + m ≤ 65536
      · rw [if_pos h₂] at h
        cases h
        exact ⟨hi, rfl, h₂⟩
      · rw [if_neg h₂] at h
        cases h
    · rw [dif_neg h₁] at h
      cases h
  next => cases h

/-- Coins that agree off a write's chunk agree where the releases after it read. -/
theorem offChunk_agree {pre : List Verity.Release} {fr : Verity.Frame} {b : ByteArray} {m : ℕ} {j : Fin K} {u : ℕ}
    (h : slot L K pre fr b m = some (j, u)) (hw : widthOf fr = m) {r r' : Fin K → FlockDeploy.Coin}
    (hr : OffChunk j u m r r') (cs : List Bool) : AgreeFrom L (pre ++ [⟨fr, b, cs⟩]) r r' := by
  obtain ⟨hl, hu, -⟩ := slot_spec L h
  intro j' a ha
  refine hr j' a fun ⟨hj, h₁, h₂⟩ => ?_
  rw [hj] at ha
  rw [usedOf_snoc] at ha
  have hc : locAt L pre (⟨fr, b, cs⟩ : Verity.Release).frame (⟨fr, b, cs⟩ : Verity.Release).answers = some j.val := hl
  rw [if_pos hc] at ha
  have hw' : widthOf (⟨fr, b, cs⟩ : Verity.Release).frame = m := hw
  omega

/-! ### The coin map reads the run's coins only where the releases after it read -/

theorem placeOpt_congr {β γ : Type} (n : Game β) (y : γ → Game β) {fn fn' : (s : Strategy n) → Tape n s → Tape n s}
    {fy fy' : (a : γ) → (s : Strategy (y a)) → Tape (y a) s → Tape (y a) s} (hn : ∀ s t, fn s t = fn' s t)
    (hy : ∀ a s t, fy a s t = fy' a s t) :
    ∀ (o : Option γ) (s : Strategy (optGame o n y)) (t : Tape (optGame o n y) s),
      placeOpt n y fn fy o s t = placeOpt n y fn' fy' o s t
  | none, s, t => hn s t
  | some a, s, t => hy a s t

theorem placeIte_congr {β : Type} {c : Prop} (dc : Decidable c) (G₁ G₂ : Game β) (s : Strategy (@ite _ c dc G₁ G₂))
    {f₁ f₁' : (s : Strategy G₁) → Tape G₁ s → Tape G₁ s} {f₂ f₂' : (s : Strategy G₂) → Tape G₂ s → Tape G₂ s}
    (h₁ : ∀ s t, f₁ s t = f₁' s t) (h₂ : ∀ s t, f₂ s t = f₂' s t) (t : Tape (@ite _ c dc G₁ G₂) s) :
    placeIte dc G₁ G₂ s f₁ f₂ t = placeIte dc G₁ G₂ s f₁' f₂' t := by
  cases dc
  · exact h₂ s t
  · exact h₁ s t

theorem placeCoin_congr {β C : Type} {iF : Fintype C} {iN : Nonempty C} (k : C → Game β) {w w' : C → C}
    {f f' : (c : C) → (s : Strategy (k c)) → Tape (k c) s → Tape (k c) s} (hw : ∀ c, w c = w' c)
    (hf : ∀ c s t, f c s t = f' c s t) (s : Strategy (@Game.coin β C iF iN k))
    (t : Tape (@Game.coin β C iF iN k) s) : placeCoin k w f s t = placeCoin k w' f' s t := by
  have e₁ : w = w' := funext hw
  have e₂ : f = f' := funext fun c => funext fun s => funext fun t => hf c s t
  subst e₁ e₂
  rfl

theorem coinBit_congr {c c' : FlockDeploy.Coin} {u : ℕ} (h : ∀ a : Fin 65536, u ≤ a.val → c a = c' a) {t : ℕ}
    (ht : 128 * u ≤ t) : coinBit c t = coinBit c' t := by
  unfold coinBit FlockDeploy.Coin.at
  by_cases ha : t / 128 < 65536
  · rw [dif_pos ha, dif_pos ha, h ⟨t / 128, ha⟩ (by show u ≤ t / 128; omega)]
  · rw [dif_neg ha, dif_neg ha]

theorem freshVal_congr {c c' : FlockDeploy.Coin} {u : ℕ} (h : ∀ a : Fin 65536, u ≤ a.val → c a = c' a) :
    freshVal c u = freshVal c' u :=
  funext fun _ => coinBit_congr h (Nat.le_add_right _ _)

theorem openBits_congr {c c' : FlockDeploy.Coin} {u n : ℕ} (h : ∀ a : Fin 65536, u ≤ a.val → c a = c' a) :
    openBits c u n = openBits c' u n := by
  unfold openBits
  congr 1
  funext t
  exact coinBit_congr h (Nat.le_add_right _ _)

/-- A coin map along `relK k` reads the run's coins only where the releases after `pre` read. -/
def RecCongr {β : Type} {k : ℕ} (rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k) : Prop :=
  ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (s : Strategy (relK k fs ts f)) (t : Tape (relK k fs ts f) s) (r r' : Fin K → FlockDeploy.Coin),
    AgreeFrom L pre r r' → rc r fs ts pre f s t = rc r' fs ts pre f s t

theorem placeFresh_congr {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} (hrc : RecCongr L rc)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (l : String) (b : ByteArray) (s : Strategy (freshG k fs ts f l b)) (t : Tape (freshG k fs ts f l b) s)
    {r r' : Fin K → FlockDeploy.Coin} (h : AgreeFrom L pre r r') :
    placeFresh L r k (rc r) fs ts pre f l b s t = placeFresh L r' k (rc r') fs ts pre f l b s t := by
  unfold placeFresh
  refine placeOpt_congr (f []) (freshY k ts f l b) (fun _ _ => rfl) (fun fs' s' t' => ?_) _ s t
  refine placeCoin_congr _ (fun v => ?_) (fun x s₃ t₃ => ?_) s' t'
  rotate_left
  · exact hrc _ _ _ _ _ _ r r' (h.snoc L _)
  split
  next i u hs =>
    obtain ⟨-, hu, -⟩ := slot_spec L hs
    exact freshVal_congr fun a ha => h i a (by rw [← hu]; exact ha)
  next => rfl

theorem placeHello_congr {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} (hrc : RecCongr L rc)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (spec : Flock.CoinTree.Spec) (ν : ByteArray) (s : Strategy (helloG k fs ts f spec ν))
    (t : Tape (helloG k fs ts f spec ν) s) {r r' : Fin K → FlockDeploy.Coin} (h : AgreeFrom L pre r r') :
    placeHello k (rc r) fs ts pre f spec ν s t = placeHello k (rc r') fs ts pre f spec ν s t := by
  unfold placeHello
  refine placeIte_congr _ (helloX k fs ts f spec ν) (f []) s (fun s' t' => ?_) (fun _ _ => rfl) t
  refine placeCoin_congr _ (fun _ => rfl) (fun _ _ _ => ?_) s' t'
  exact hrc _ _ _ _ _ _ r r' (h.snoc L _)

theorem placeRound_congr {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} (hrc : RecCongr L rc)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (i sn n : ℕ) (b : ByteArray) (T : OpenTree) (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog))
    (s : Strategy (openP k fs ts f i sn n b T p)) (t : Tape (openP k fs ts f i sn n b T p) s)
    {r r' : Fin K → FlockDeploy.Coin} (h : AgreeFrom L pre r r') :
    placeRound L r k (rc r) fs ts pre f i sn n b T p s t = placeRound L r' k (rc r') fs ts pre f i sn n b T p s t := by
  unfold placeRound
  refine placeCoin_congr _ (fun dr => ?_) (fun _ _ _ => ?_) s t
  rotate_left
  · exact hrc _ _ _ _ _ _ r r' (h.snoc L _)
  split
  next j u hs =>
    obtain ⟨-, hu, -⟩ := slot_spec L hs
    rw [openBits_congr fun a ha => h j a (by rw [← hu]; exact ha)]
  next => rfl

theorem placeOpen_congr {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} (hrc : RecCongr L rc)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (i sn n : ℕ) (b : ByteArray) (s : Strategy (openG k fs ts f i sn n b)) (t : Tape (openG k fs ts f i sn n b) s)
    {r r' : Fin K → FlockDeploy.Coin} (h : AgreeFrom L pre r r') :
    placeOpen L r k (rc r) fs ts pre f i sn n b s t = placeOpen L r' k (rc r') fs ts pre f i sn n b s t := by
  unfold placeOpen
  exact placeIte_congr _ _ _ s (fun s' t' => placeOpt_congr _ _ (fun _ _ => rfl)
    (fun T s₃ t₃ => placeOpt_congr _ _ (fun _ _ => rfl)
      (fun p s₄ t₄ => placeRound_congr L hrc fs ts pre f i sn n b T p s₄ t₄ h) _ s₃ t₃) _ s' t')
    (fun _ _ => rfl) t

theorem placeK_congr {β : Type} : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (s : Strategy (relK k fs ts f))
    (t : Tape (relK k fs ts f) s) (r r' : Fin K → FlockDeploy.Coin),
    AgreeFrom L pre r r' → placeK L r k fs ts pre f s t = placeK L r' k fs ts pre f s t
  | 0, _, _, _, _, _, _, _, _, _ => rfl
  | _ + 1, _, _, _, _, ⟨none, _⟩, _, _, _, _ => rfl
  | k + 1, fs, ts, pre, f, ⟨some (.fresh l, b), s'⟩, t, _, _, h =>
    placeFresh_congr L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f s t r r' h => placeK_congr k fs ts pre f s t r r' h) fs ts pre f l b s' t h
  | k + 1, fs, ts, pre, f, ⟨some (.hello spec, ν), s'⟩, t, _, _, h =>
    placeHello_congr L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f s t r r' h => placeK_congr k fs ts pre f s t r r' h) fs ts pre f spec ν s' t h
  | k + 1, fs, ts, pre, f, ⟨some (.opening i sn n, b), s'⟩, t, _, _, h =>
    placeOpen_congr L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f s t r r' h => placeK_congr k fs ts pre f s t r r' h) fs ts pre f i sn n b s' t h

/-! ### The sampling design frame by frame -/

/-- The sampling design along `relK k` from every layout, trees, releases before and continuation. -/
def RecLaw {β : Type} {k : ℕ} (rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k) (gr : GapRec β k) : Prop :=
  ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β)
    (s : Strategy (relK k fs ts f)), LawHolds (relK k fs ts f) s (fun r => rc r fs ts pre f s) (gr fs ts pre f s)

theorem law_fresh {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} {gr : GapRec β k}
    (hlaw : RecLaw rc gr) (hrc : RecCongr L rc) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (l : String) (b : ByteArray)
    (s : Strategy (freshG k fs ts f l b)) :
    LawHolds (freshG k fs ts f l b) s (fun r => placeFresh L r k (rc r) fs ts pre f l b s)
      (gapFresh k gr fs ts pre f l b s) := by
  unfold placeFresh gapFresh
  refine law_opt _ _ (fun s => law_id _ s) (fun fs' s' => ?_) _ s
  rcases hs : slot L K pre (.fresh l) b 4 with _ | ⟨i, u⟩
  · refine law_keep s' fun x => ?_
    exact hlaw _ _ _ _ _
  · have hv := splits_fresh i u (slot_spec L hs).2.2
    refine law_set (R := Fin K → FlockDeploy.Coin) s' _ _ hv ?_ fun x => ?_
    · intro x r r' τ hr
      exact hrc _ _ _ _ _ _ r r' (offChunk_agree L hs rfl hr _)
    · exact hlaw _ _ _ _ _

theorem law_hello {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} {gr : GapRec β k}
    (hlaw : RecLaw rc gr) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (spec : Flock.CoinTree.Spec) (ν : ByteArray)
    (s : Strategy (helloG k fs ts f spec ν)) :
    LawHolds (helloG k fs ts f spec ν) s (fun r => placeHello k (rc r) fs ts pre f spec ν s)
      (gapHello k gr fs ts pre f spec ν s) := by
  unfold placeHello gapHello
  refine law_ite _ _ _ (fun s' => ?_) (fun s => law_id _ s) s
  refine law_keep s' fun tr => ?_
  exact hlaw _ _ _ _ _

theorem law_round {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} {gr : GapRec β k}
    (hlaw : RecLaw rc gr) (hrc : RecCongr L rc) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (i sn n : ℕ) (b : ByteArray) (T : OpenTree)
    (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)) (s : Strategy (openP k fs ts f i sn n b T p)) :
    LawHolds (openP k fs ts f i sn n b T p) s (fun r => placeRound L r k (rc r) fs ts pre f i sn n b T p s)
      (gapRound L K k gr fs ts pre f i sn n b T p s) := by
  unfold placeRound gapRound
  rcases hs : slot L K pre (.opening i sn n) b n with _ | ⟨j, u⟩
  · refine law_keep s fun dr => ?_
    exact hlaw _ _ _ _ _
  · obtain ⟨-, -, hu⟩ := slot_spec L hs
    have hv := splits_open j u n hu
    refine law_mono (law_congr (P' := fun r : Fin K → FlockDeploy.Coin =>
        placeCoin _ (fun dr => openW n (bitsOf (openElems (r j) u n)) dr)
        (fun dr s₅ t₅ => rc r fs (ts.set i (T.bump p.1)) (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩])
          (fun log => f (⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ :: log)) s₅ t₅) s) ?_
      (law_write (R := Fin K → FlockDeploy.Coin) s
        (G := fun dr => gr fs (ts.set i (T.bump p.1)) (pre ++ [⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩])
          (fun log => f (⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ :: log)) (s dr))
        (fun (e : Fin n → Fin 128 → Bool) dr => openW n (bitsOf e) dr) _ _ hv ?_ fun dr => ?_)) ?_
    · intro r t
      show placeCoin _ (fun dr => openW n (openBits (r j) u n) dr) _ s t = _
      rw [openBits_eq]
    · intro dr r r' τ hr
      exact hrc _ _ _ _ _ _ r r' (offChunk_agree L hs rfl hr _)
    · exact hlaw _ _ _ _ _
    · exact add_le_add le_self_add le_rfl

theorem law_open {β : Type} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k} {gr : GapRec β k}
    (hlaw : RecLaw rc gr) (hrc : RecCongr L rc) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (i sn n : ℕ) (b : ByteArray)
    (s : Strategy (openG k fs ts f i sn n b)) :
    LawHolds (openG k fs ts f i sn n b) s (fun r => placeOpen L r k (rc r) fs ts pre f i sn n b s)
      (gapOpen L K k gr fs ts pre f i sn n b s) := by
  unfold placeOpen gapOpen
  refine law_ite _ _ _ (fun s' => ?_) (fun s => law_id _ s) s
  refine law_opt _ _ (fun s => law_id _ s) (fun T s₃ => ?_) _ s'
  refine law_opt _ _ (fun s => law_id _ s) (fun p s₄ => ?_) _ s₃
  exact law_round L hlaw hrc fs ts pre f i sn n b T p s₄

/-- **The sampling design along the releases**: at every layout, trees, releases before, continuation and strategy, the coin map
`placeK` is uniform up to the hiding term `gapK`. -/
theorem law_placeK {β : Type} : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β) (s : Strategy (relK k fs ts f)),
    LawHolds (R := Fin K → FlockDeploy.Coin) (relK k fs ts f) s (fun r => placeK L r k fs ts pre f s)
      (gapK L K k fs ts pre f s)
  | 0, _, _, _, _, s => law_id _ s
  | _ + 1, _, _, _, _, ⟨none, _⟩ => law_id _ _
  | k + 1, fs, ts, pre, f, ⟨some (.fresh l, b), s'⟩ => by
    refine law_send _ _ _ ?_
    exact law_congr (fun r t => rfl) (law_fresh L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f s => law_placeK k fs ts pre f s)
      (fun fs ts pre f s t r r' h => placeK_congr L k fs ts pre f s t r r' h) fs ts pre f l b s')
  | k + 1, fs, ts, pre, f, ⟨some (.hello spec, ν), s'⟩ => by
    refine law_send _ _ _ ?_
    exact law_congr (fun r t => rfl) (law_hello (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f s => law_placeK k fs ts pre f s) fs ts pre f spec ν s')
  | k + 1, fs, ts, pre, f, ⟨some (.opening i sn n, b), s'⟩ => by
    refine law_send _ _ _ ?_
    exact law_congr (fun r t => rfl) (law_open L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f s => law_placeK k fs ts pre f s)
      (fun fs ts pre f s t r r' h => placeK_congr L k fs ts pre f s t r r' h) fs ts pre f i sn n b s')

end Along

theorem law_along {g g' : Game β} (h : g = g') {Pl : R → (s : Strategy g') → Tape g' s → Tape g' s}
    {G : Strategy g' → ℝ≥0∞} (hl : ∀ s, LawHolds g' s (fun r => Pl r s) (G s)) (s : Strategy g) :
    LawHolds g s (fun r => along h (Pl r) s) (alongVal h G s) := by
  subst h
  exact hl s

/-! ## The first clause -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The coin map is uniform up to the hiding term at the period's game.** -/
theorem law_place (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c))
    (L : Locate) (K : ℕ) :
    LawHolds (R := Fin K → FlockDeploy.Coin) (V.game θ F c) σ (fun r t => place V θ F c σ L r t)
      (gapOf V θ F c σ L K) :=
  law_along (game_eq V θ F c)
    (Pl := fun r s t => placeK L r (frames (V.families θ)) (V.families θ) [] [] (finish (V.signer θ) θ F c s.1) s.2 t)
    (G := fun s => gapK L K (frames (V.families θ)) (V.families θ) [] [] (finish (V.signer θ) θ F c s.1) s.2)
    (fun s => by
      rcases s with ⟨m, s'⟩
      refine law_send _ _ _ ?_
      exact law_congr (fun r t => rfl) (law_placeK L _ _ _ _ _ s')) σ

/-- **`LawAt`**: the period's play is at most as likely, at every event, as the frame's play at a uniform tape and
uniform run coins, plus the hiding term. -/
theorem lawAt (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (L : Locate) (P : Claims.Prover (TapeOf V θ F c σ)) : LawAt V A x θ F c σ L P := by
  classical
  intro E
  have h := law_place V θ F c σ L (A.L.V.k x) (fun b => if E b then 1 else 0) fun b => by
    by_cases hb : E b <;> simp [hb]
  calc Verity.Game.prob E (V.game θ F c) σ
      = prCoin fun t : Tape (V.game θ F c) σ => E (Tape.out _ σ t) := prob_eq_prCoin _ _ _
    _ ≤ (avg fun r : Coins A x => avg fun t : Tape (V.game θ F c) σ =>
          if E (Tape.out _ σ (place V θ F c σ L r t)) then 1 else 0) + gapOf V θ F c σ L (A.L.V.k x) := h
    _ = prCoin (fun s : TapeOf V θ F c σ × Coins A x => E ((frame V A x θ F c σ L P).out s.1 s.2)) +
          gapOf V θ F c σ L (A.L.V.k x) := by
        unfold prCoin
        rw [avg_prod, avg_swap]
        rfl
    _ = ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x => E ((frame V A x θ F c σ L P).out s.1 s.2)) +
          gapOf V θ F c σ L (A.L.V.k x) := by
        rw [prCoin_eq_ofReal_pr]

end Verity.SampledTwo.CoinMap
