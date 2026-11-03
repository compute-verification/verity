import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Rat.Cast.Order

/-!
# Served sparse noise in the bit model: the committed words and the served layouts (trusted definitions)

**Draft semantics, pending review**, for the definitions here. They are targets of `MH100w` programs
(`Pouw.Dimension.H100`), whose semantics `H100W.lean` reviews against the H200's measurements.

* **The grid** is every value of a call's coin bits, `Bits m C`: bit `b` of row `i`'s 32-bit coin word `c`. Targets and
  free values are affine in the bits (`aff`).
* **The free values** (`FreeB`) are the constants and the raw coin words `Σ_b 2^b·x(r, c, b)`.
* **A committed word** (`ssnWordB`) of row `i`, column `j`, after step `t` is its data case's constant `κ i j t` plus,
  at every bit, the weight of the used lane that reads it (`bitCoef`): step `s` reads lane `q` from bit `pos s q` of
  coin word `src s` and weighs it `W s q j` in column `j`. `ssnWords` is the set of a call's words over `n` columns,
  and `ShiftSeparatedOn` says no two differ by a constant and none is a free value plus one.
* **The 4-lane served layout** (`l4Src`, `l4Pos`): step 0's four lanes at bits 0, 8, 16 and 24 of word 0, every later
  step's lane at word 0's other 28 bits in order, then at words 1, 2, …. Its weights `w1l` put the column code
  `ssnCode` (the base-13 digits of `ssnC j`, less 6) on step 0 and weight 1 on every later step's lane 0; `l4Words`
  are its committed words.
* **The 5-lane served layout** (`l5Src`, `l5Pos`): the same with a fifth step-0 lane at bit 1 of word 0, so later steps
  start at word 0's other 27 bits. Its weights `w5` put the balanced base-13 code `code5` of
  `z_j = 185,646 ± (⌊j/2⌋ + 1)` on step 0; `l5Words` are its committed words.
-/

namespace Pouw.Dimension.Ssn

open Finset

/-- An affine function of `q`-ary digits: `c + Σ_k a k · x k`. -/
def aff {ι : Type} [Fintype ι] {q : ℕ} (c : ℚ) (a : ι → ℚ) : (ι → Fin q) → ℚ :=
  fun x => c + ∑ k, a k * ((x k : ℕ) : ℚ)

/-- Targets `T` shift-separated on the grid points `S`: no two differ on `S` by a constant, and none is a free value
plus a constant on `S`. -/
def ShiftSeparatedOn {G : Type} (S : Set G) (F : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  (∀ t ∈ T, ∀ t' ∈ T, (∃ q : ℚ, ∀ g ∈ S, t g = t' g + q) → t = t') ∧
    ∀ t ∈ T, ∀ f ∈ F, ¬ ∃ q : ℚ, ∀ g ∈ S, f g = t g + q

section Bits

variable {m S C L : ℕ}

/-- The call's coin bits: `x (i, c, b)` is bit `b` of row `i`'s coin word `c`. -/
abbrev Bits (m C : ℕ) : Type := Fin m × Fin C × Fin 32 → Fin 2

/-- Row `r`'s raw coin word `c`, as coefficients on the bits: `2^b` at `(r, c, b)`. -/
def coinCoef (r : Fin m) (c : Fin C) : Fin m × Fin C × Fin 32 → ℚ :=
  fun k => if k.1 = r ∧ k.2.1 = c then 2 ^ (k.2.2 : ℕ) else 0

/-- The free values in the bit model: the constants and the raw coin words. -/
def FreeB (m C : ℕ) : Set (Bits m C → ℚ) :=
  {f | (∃ q : ℚ, f = fun _ => q) ∨ ∃ r c, f = aff 0 (coinCoef (m := m) (C := C) r c)}

/-- The weight bit `k` carries in row `i`'s word of column `j` after step `t`: the weight of the used lane that reads it
(step `s` reads lane `q` from bit `pos s q` of coin word `src s`). -/
def bitCoef (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ) (t : Fin S) :
    Fin m × Fin C × Fin 32 → ℚ :=
  fun k => if k.1 = i then
    ∑ s : Fin S, ∑ q : Fin L, if s ≤ t ∧ used s q ∧ src s = k.2.1 ∧ pos s q = k.2.2 then (W s q j : ℚ) else 0
  else 0

/-- The committed word of row `i`, column `j`, after step `t`, in the bit model. -/
def ssnWordB (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ)
    (t : Fin S) : Bits m C → ℚ :=
  aff (κ i j t : ℚ) (bitCoef src used pos W i j t)

/-- A call's committed words over `n` columns: every row, column `j < n` and step. -/
def ssnWords (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (n : ℕ) :
    Finset (Bits m C → ℚ) :=
  (univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S => ssnWordB src used pos κ W x.1 x.2.1 x.2.2

end Bits

section Served

variable {S C : ℕ}

/-- The 4-lane layout's coin word for step `s`: word 0 through step 28, then words 1, 2, … 32 steps each. -/
def l4Word (s : ℕ) : ℕ := if s < 29 then 0 else 1 + (s - 29) / 32

/-- The 4-lane layout's bit for step `s`, lane `q`: step 0's lanes at bits `8q` of word 0; steps 1–28 at word 0's other
28 bits in order (1–7, 9–15, 17–23, 25–31); later steps at bits 0–31 of each next word. -/
def l4Pos (s q : ℕ) : ℕ := if s = 0 then 8 * q else if s < 29 then s + (s - 1) / 7 else (s - 29) % 32

theorem l4Word_lt (hC : S + 3 ≤ 32 * C) {s : ℕ} (hs : s < S) : l4Word s < C := by
  unfold l4Word
  split_ifs <;> omega

/-- The 4-lane layout's coin word for each step, with `C ≥ 1 + ⌈max(0, S − 29)/32⌉` words a row. -/
def l4Src (hC : S + 3 ≤ 32 * C) (s : Fin S) : Fin C := ⟨l4Word s, l4Word_lt hC s.isLt⟩

/-- The 5-lane layout's coin word for step `s`: word 0 through step 27, then words 1, 2, … 32 steps each. -/
def l5Word (s : ℕ) : ℕ := if s < 28 then 0 else 1 + (s - 28) / 32

/-- The 5-lane layout's bit for step `s`, lane `q`: step 0's lanes 0–3 at bits `8q` and lane 4 at bit 1 of word 0;
steps 1–27 at word 0's other 27 bits in order (2–7, 9–15, 17–23, 25–31); later steps at bits 0–31 of each next
word. -/
def l5Pos (s q : ℕ) : ℕ :=
  if s = 0 then (if q < 4 then 8 * q else 1) else if s < 28 then s + 1 + s / 7 else (s - 28) % 32

theorem l5Word_lt (hC : S + 4 ≤ 32 * C) {s : ℕ} (hs : s < S) : l5Word s < C := by
  unfold l5Word
  split_ifs <;> omega

/-- The 5-lane layout's coin word for each step, with `C ≥ 1 + ⌈max(0, S − 28)/32⌉` words a row. -/
def l5Src (hC : S + 4 ≤ 32 * C) (s : Fin S) : Fin C := ⟨l5Word s, l5Word_lt hC s.isLt⟩

/-- Column `j`'s code number: `j + 1`, skipping `14,280`, whose base-13 digits are all `6`. -/
def ssnC (j : ℕ) : ℕ := if j + 1 < 14280 then j + 1 else j + 2

/-- Digit `e` of `c` in base 13. -/
def ssnDigit (c : ℕ) (e : Fin 4) : ℕ := c / 13 ^ (e : ℕ) % 13

/-- Step 0's 4-lane column code: `C[e, j] = digit e of c_j − 6`, in `[−6, 6]⁴ ∖ {0}`. -/
def ssnCode (e : Fin 4) (j : ℕ) : ℤ := (ssnDigit (ssnC j) e : ℤ) - 6

/-- The 4-lane layout's weights: step 0 the code on four lanes, every later step one lane of weight 1 for every
column. -/
def w1l (s : Fin S) (q : Fin 4) (j : ℕ) : ℤ := if (s : ℕ) = 0 then ssnCode q j else if q = 0 then 1 else 0

/-- Column `j`'s balanced code number `z_j = (⌊j/2⌋ + 1)·(−1)^j`, shifted by `(13⁵ − 1)/2` into `[0, 13⁵)`. -/
def z5 (j : ℕ) : ℕ := if j % 2 = 0 then 185646 + (j / 2 + 1) else 185646 - (j / 2 + 1)

/-- The 5-lane balanced base-13 code: `C[e, j]` is `z_j`'s `e`-th balanced digit, in `[−6, 6]`. -/
def code5 (e : Fin 5) (j : ℕ) : ℤ := ((z5 j / 13 ^ (e : ℕ) % 13 : ℕ) : ℤ) - 6

/-- The 5-lane layout's weights: step 0 the code on five lanes, every later step one lane of weight 1 for every
column. -/
def w5 (s : Fin S) (q : Fin 5) (j : ℕ) : ℤ := if (s : ℕ) = 0 then code5 q j else if q = 0 then 1 else 0

/-- The 4-lane layout's committed words over `n` columns, for the data case `κ`. -/
def l4Words {m : ℕ} (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (n : ℕ) : Finset (Bits m C → ℚ) :=
  ssnWords (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l n

/-- The 5-lane layout's committed words over `n` columns, for the data case `κ`. -/
def l5Words {m : ℕ} (hC : S + 4 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (n : ℕ) : Finset (Bits m C → ℚ) :=
  ssnWords (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 n

end Served

end Pouw.Dimension.Ssn
