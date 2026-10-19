import Definitions.Pouw.RelativeSecurity

/-!
# FP-native PoUW: design A's prefix words and its row floor (statement only)

FP-NATIVE's design A (note:20261019T0525Z-draft-fp-native, G2, Lemma R2). An FP8 or FP4 entry's dither moves it by
its format's gap, a power of two, so every pair coefficient of a prefix word is even when each side has one coin, and
the mixed floor `K₂(i)` is then ½ and tight. Design A gives the left operand `P̃` one coin per entry, with gap `α`,
and the right operand `X̃`, whose entry `(k, l)` only column `l`'s words read, two coins per entry, with gaps `γ₀` and
`γ₁`. A word's bit-pair coefficients are `α γ₀` and `α γ₁` on the cross pairs and `0` on each entry's coin pair
`(b₀, b₁)`, and that `0` is what lifts the row's chain bound to `c = 1`:

* `fpPrefix` is the committed prefix word `T_t(i, l) = Σ_{k < (t+1)L} P̃_ik X̃_kl` of design A, over a ring `R`.
* `FpChainBound` is the record's `ChainBound` (`Proofs.Pouw.RelativeSecurity`) for these words, with the gaps
  universally quantified under an admissibility condition `Adm`: `T` certified prefix words of row `i` at distinct
  (column, block) pairs need `T L ≤ c S` products of the allowed form.
* `FpRowFloorWrap`: over `Z/2^m`, AffProd products (any two bit-affine operands, mixed or not) give `c = 1` when
  `v₂(α γ₀) ≤ e`, `v₂(α γ₁) ≤ e'` everywhere and `e + e' < m` (`AdmWrap`). At `m = e + e'` it fails: two words
  with `L = 2`, `α = 2`, `γ = (2, 2)` are certified by 3 products mod 16 (FP-NATIVE's SAT, r20261019-060454-f899).
* `FpRowFloorInt`: over `ℤ` (exact sums), the same with every gap product nonzero (`AdmInt`).

Both are statements for red team (Daniel, 17 Oct: statement first). Neither is proved, and no guarantee reads them. One
coin a side with even gaps is capped at ½ (`K₂(i)`), and two coins on the shared side `P̃` (design B) give only the
flattening bound.
-/

namespace Pouw.FpNative

open Finset Pouw.RelativeSecurity

/-- Design A's bits: `P̃`'s coin `β(i, t, j)` at entry `(i, tL + j)`, and `X̃`'s two coins `b(t, j, l, q)`, `q < 2`, at
entry `(tL + j, l)`. -/
abbrev FBits (n L B n' : ℕ) : Type := (Fin n × Fin B × Fin L) ⊕ (Fin B × Fin L × Fin n' × Fin 2)

section Design

variable {R : Type} [CommRing R] {n L B n' : ℕ}

/-- `P̃_(i, tL+j) = P_(i, tL+j) + α(i, t, j) β(i, t, j)` (`P` in `pc`). -/
def fpP (pc α : Fin n → Fin B → Fin L → R) (i : Fin n) (t : Fin B) (j : Fin L) : BitAff (FBits n L B n') R :=
  ⟨pc i t j, Sum.elim (fun a => if a = (i, t, j) then α i t j else 0) fun _ => 0⟩

/-- `X̃_(tL+j, l) = X_(tL+j, l) + γ(t, j, l, 0) b(t, j, l, 0) + γ(t, j, l, 1) b(t, j, l, 1)` (`X` in `vq`). -/
def fpX (vq : Fin B → Fin L → Fin n' → R) (γ : Fin B → Fin L → Fin n' → Fin 2 → R) (t : Fin B) (j : Fin L)
    (l : Fin n') : BitAff (FBits n L B n') R :=
  ⟨vq t j l, Sum.elim (fun _ => 0) fun a => if (a.1, a.2.1, a.2.2.1) = (t, j, l) then γ t j l a.2.2.2 else 0⟩

/-- Cell-block `(i, l, t)`'s increment `Σ_j P̃_(i, tL+j) X̃_(tL+j, l)`. -/
def fpIncr (pc α : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (γ : Fin B → Fin L → Fin n' → Fin 2 → R) (i : Fin n) (l : Fin n') (t : Fin B) : (FBits n L B n' → R) → R :=
  fun z => ∑ j, (fpP pc α i t j).eval z * (fpX vq γ t j l).eval z

/-- The committed prefix word `T_t(i, l) = Σ_{k < (t+1)L} P̃_ik X̃_kl`. -/
def fpPrefix (pc α : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (γ : Fin B → Fin L → Fin n' → Fin 2 → R) (i : Fin n) (l : Fin n') (t : ℕ) : (FBits n L B n' → R) → R :=
  fun z => ∑ t' : Fin B, if t'.val ≤ t then fpIncr pc α vq γ i l t' z else 0

end Design

/-- **The chain bound for design A**: products of the form `Ok`, indexed by `S`, that certify `T` prefix words of row
`i` at distinct (column, block) pairs (`cl` injective), each word a combination of the products plus a bit-affine
residual that agrees with its target at every bit point with at most two ones, number at least `T L / c`, for every
gap assignment `Adm` admits. The record's `ChainBound` with design A's words and the gaps quantified. -/
def FpChainBound (R : Type) [CommRing R] (n L B n' : ℕ) (Ok : ((FBits n L B n' → R) → R) → Prop)
    (Adm : (Fin n → Fin B → Fin L → R) → (Fin B → Fin L → Fin n' → Fin 2 → R) → Prop) (c : ℕ) : Prop :=
  ∀ (ι τ : Type) [Fintype τ] [DecidableEq τ] (S : Finset ι) (π : ι → (FBits n L B n' → R) → R),
    (∀ k ∈ S, Ok (π k)) → ∀ (pc α : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (γ : Fin B → Fin L → Fin n' → Fin 2 → R), Adm α γ → ∀ (i : Fin n) (cl : τ → Fin n' × Fin B),
    Function.Injective cl → ∀ (coef : τ → ι → R) (r : τ → BitAff (FBits n L B n') R),
    (∀ t (U : Finset (FBits n L B n')), U.card ≤ 2 →
      ∑ k ∈ S, coef t k * π k (bitInd U) + (r t).eval (bitInd U) =
        fpPrefix pc α vq γ i (cl t).1 (cl t).2.val (bitInd U)) →
    L * Fintype.card τ ≤ c * S.card

/-- Gaps admissible mod `2^m`: some `e`, `e'` with `e + e' < m` bound the 2-adic valuations of every `α γ₀` by `e`
and of every `α γ₁` by `e'` (`2^(m-1-e) x ≠ 0` in `Z/2^m` is `v₂(x) ≤ e`). -/
def AdmWrap (m : ℕ) {n L B n' : ℕ} (α : Fin n → Fin B → Fin L → ZMod (2 ^ m))
    (γ : Fin B → Fin L → Fin n' → Fin 2 → ZMod (2 ^ m)) : Prop :=
  ∃ e e' : ℕ, e + e' < m ∧ ∀ i t j l,
    (2 : ZMod (2 ^ m)) ^ (m - 1 - e) * (α i t j * γ t j l 0) ≠ 0 ∧
      (2 : ZMod (2 ^ m)) ^ (m - 1 - e') * (α i t j * γ t j l 1) ≠ 0

/-- Gaps admissible over `ℤ`: every gap product is nonzero. -/
def AdmInt {n L B n' : ℕ} (α : Fin n → Fin B → Fin L → ℤ) (γ : Fin B → Fin L → Fin n' → Fin 2 → ℤ) : Prop :=
  ∀ i t j l q, α i t j * γ t j l q ≠ 0

/-- **Design A's row floor mod `2^m`** (FP-NATIVE's Lemma R2, wrap form): AffProd products, `c = 1`, at admissible
gaps. Open: stated for red team, not proved. -/
def FpRowFloorWrap : Prop := ∀ m n L B n' : ℕ, FpChainBound (ZMod (2 ^ m)) n L B n' AffProd (AdmWrap m) 1

/-- **Design A's row floor over `ℤ`** (FP-NATIVE's Lemma R2, exact form): AffProd products, `c = 1`, at nonzero gaps.
Open: stated for red team, not proved. -/
def FpRowFloorInt : Prop := ∀ n L B n' : ℕ, FpChainBound ℤ n L B n' AffProd AdmInt 1

end Pouw.FpNative
