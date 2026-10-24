import Definitions.Pouw.PearlC.Defs

/-! The proof-only part of `Definitions.Pouw.PearlC.Defs`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

open Pouw.Fp8Atom Matrix

/-- `x` has a `p`-bit significand at some exponent (BF16 is `p = 8`, E4M3 `p = 4`, and `Representable` is `p = 24`). -/
def HasSig (p : Nat) (x : ℚ) : Prop := ∃ m e : ℤ, |m| < 2 ^ p ∧ x = m * 2 ^ e

/-- A matrix of codes as exact values. -/
def vals {ι κ : Type} (M : Matrix ι κ Code) : Matrix ι κ ℚ := M.map codeVal

/-- An encoder conformance vector: an FP32 word and the code `f32_to_e4m3_sat_word` gives it. -/
structure EncVec where
  u : Nat
  c : Nat

/-- An encoder vector holds for `e4m3Word`, and for `e4m3` on the word's value when that value is finite and nonzero. -/
def EncVec.holds (v : EncVec) : Bool :=
  (e4m3Word v.u).val == v.c &&
    match wordQ v.u with
    | some x => x == 0 || (e4m3 x).val == v.c
    | none => true

/-- An FP32 multiply vector: the words `a`, `b` and the word `p` IEEE binary32 multiplication writes for them. -/
structure F32Vec where
  a : Nat
  b : Nat
  p : Nat

/-- An FP32 multiply vector holds: `a` and `b` are finite and nonzero, and `f32Word` maps the exact product of their
values to `p`. (An exact zero has no sign, so `f32Word 0` is `+0` where a signed zero operand gives `−0`.) -/
def F32Vec.holds (v : F32Vec) : Bool :=
  match wordQ v.a, wordQ v.b with
  | some x, some y => x != 0 && y != 0 && f32Word (x * y) == v.p
  | _, _ => false

variable {m n k r : Nat}

/-- A's noise `N_A = E_A·F_Aᵀ` (`m × k`), on the codes' values. -/
def Noise.NA (z : Noise m n k r) : Matrix (Fin m) (Fin k) ℚ := vals z.EA * (vals z.FA)ᵀ

/-- B's noise `N_B = E_B·F_Bᵀ` (`n × k`), on the codes' values. -/
def Noise.NB (z : Noise m n k r) : Matrix (Fin n) (Fin k) ℚ := vals z.EB * (vals z.FB)ᵀ

/-- The `r × r` Gram factor `F_Bᵀ·F_A`, through which the Gram order forms `N_B·F_A = E_B·(F_Bᵀ·F_A)`. -/
def Noise.gram (z : Noise m n k r) : Matrix (Fin r) (Fin r) ℚ := (vals z.FB)ᵀ * vals z.FA

/-- `β ⊙ M`: row `i` of `M` scaled by `β i` (#311's `β⊙`). -/
def rowScale {p q : Nat} (β : Fin p → ℚ) (M : Matrix (Fin p) (Fin q) ℚ) : Matrix (Fin p) (Fin q) ℚ := diagonal β * M

/-- The noise scale of a row, `β = α·δ·ρ`, for noise of unit RMS, where `ρ = root (st k x)` and `root` is the square
root in use (the FP32 square root, once pinned). §5 folds the noise lines' norm into `β` as well. -/
def betaOf (st : Stat) (root : ℚ → ℚ) (δ α : ℚ) (x : Fin k → ℚ) : ℚ := α * δ * root (st k x)

/-- **The exact chain**: `α·x + β·(e·f)` on the codes' values, so the forming is `e4m3(α·x + β·n)` with `n = E·Fᵀ`
exact. -/
def FormChain.exact (r : Nat) : FormChain r := ⟨fun α x β e f => α * x + β * ∑ t, codeVal (e t) * codeVal (f t)⟩

/-- **#311's BF16 chain**: the noise wgmma over the noise lines `β·e` and `f` (on the codes' values), whose FP32 value
is `noise` (its accumulation order is not pinned, so it is a parameter), one f32→bf16 conversion of it, and one HFMA2
`fma(α, x, bf16(n))` rounded to BF16, which the cast reads. `α` and `x` are the HFMA2's BF16 operands. §1's step list
fixes the conversion and the HFMA2 but not where `β` enters; here it scales the noise lines, as in §5. -/
def FormChain.bf16 (r : Nat) (noise : (Fin r → ℚ) → (Fin r → ℚ) → ℚ) : FormChain r :=
  ⟨fun α x β e f => bf16Rn (α * x + bf16Rn (noise (fun t => β * codeVal (e t)) (fun t => codeVal (f t))))⟩

/-- §5's noise lines of a `p`-row operand: row `i` is `E′_i = lineCodes (β i) (E i)`, the codes the atom-mix reads. -/
def castLines {p : Nat} (β : Fin p → ℚ) (E : Matrix (Fin p) (Fin r) Code) : Matrix (Fin p) (Fin r) Code :=
  fun i => lineCodes (β i) (E i)

/-- The noise as §5's peel reuses it: `E_A` and `E_B` replaced by their noise lines `E′_A` and `E′_B`, `F_A` and `F_B`
unchanged, so at unit `β` the peel's `N_A` is `E′_A·F_Aᵀ`, the noise the atom-mix added up to its accumulator's rounding. -/
def Noise.atomLines (βA : Fin m → ℚ) (βB : Fin n → ℚ) (z : Noise m n k r) : Noise m n k r :=
  ⟨castLines βA z.EA, z.FA, castLines βB z.EB, z.FB⟩

/-- An atom-mix conformance vector: the FP32 words of `α`, `x` and `β`; the `E` row's 32 FP32 words, packed 32 bits
each; the `F` row's 32 E4M3 codes, packed 8 bits each (`codes`); and the code `verity.ml.tc` gives `A′`. -/
structure MixVec where
  α : Nat
  x : Nat
  β : Nat
  e : Nat
  f : Nat
  c : Nat

/-- An atom-mix vector holds: `α`, `x` and `β` are finite, and the atom-mix's step and cast give the code `c` on the `E`
row's FP32 values, `E′_t = e4m3Word (f32Word (β·e_t))` (`lineCodes` where `e_t` is a nonzero code's value), and the
`F` row's codes. -/
def MixVec.holds (v : MixVec) : Bool :=
  match wordQ v.α, wordQ v.x, wordQ v.β with
  | some α, some x, some β =>
    (e4m3 (wordVal (hopperStep (f32Word (α * x))
      (fun t => e4m3Word (f32Word (β * wordVal ((v.e >>> (32 * t.val)) % 2 ^ 32)))) (codes v.f)))).val == v.c
  | _, _, _ => false

/-- `C̃` in exact arithmetic: `A′·B̃ᵀ` on the codes' values. -/
def ctExact (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) : Matrix (Fin m) (Fin n) ℚ :=
  vals Aq * (vals Bq)ᵀ

/-- #311's left factor `[β·E_A ‖ A′·F_B]` (`m × 2r`); `A'` is `A′`'s values. -/
def peelLeft (βA : Fin m → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ :=
  fromCols (rowScale βA (vals z.EA)) (A' * vals z.FB)

/-- #311's right factor `[(β·N_B − B̃)·F_A ‖ −β·E_B]` (`n × 2r`); `B'` is `B̃`'s values. -/
def peelRight (βB : Fin n → ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ :=
  fromCols ((rowScale βB z.NB - B') * vals z.FA) (-rowScale βB (vals z.EB))

/-- #311's form of U: `C̃ + [β·E_A ‖ A′·F_B]·[(β·N_B − B̃)·F_A ‖ −β·E_B]ᵀ`. -/
def peelU (Ct : Matrix (Fin m) (Fin n) ℚ) (βA : Fin m → ℚ) (βB : Fin n → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ)
    (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin m) (Fin n) ℚ :=
  Ct + peelLeft βA A' z * (peelRight βB B' z)ᵀ

/-- The whitepaper's raw-A form of U: `C̃ − A·N_Bᵀ − N_A·Bᵀ − N_A·N_Bᵀ`, on the raw `A`, `B` and the noise as added. -/
def rawU (Ct : Matrix (Fin m) (Fin n) ℚ) (A : Matrix (Fin m) (Fin k) ℚ) (B : Matrix (Fin n) (Fin k) ℚ)
    (NA : Matrix (Fin m) (Fin k) ℚ) (NB : Matrix (Fin n) (Fin k) ℚ) : Matrix (Fin m) (Fin n) ℚ :=
  Ct - A * NBᵀ - NA * Bᵀ - NA * NBᵀ

end Pouw.PearlC
