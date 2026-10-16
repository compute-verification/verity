import Mathlib

/-!
# The decoder's polynomial representations

The counted decoder computes on plain data: a univariate polynomial is its coefficient list, lowest degree first
(`toPolyL`, evaluated by Horner's rule, `horner`), and a bivariate polynomial `Q(X, Y)` is its coefficient table
`P i j` (the coefficient of `X^i Y^j`) on the box `i ≤ D`, `j ≤ L` (`toBi`). These are the readings into Mathlib's
`F[X]` and `F[X][X]` that the correctness proofs use.
-/

namespace Claims.GS

open Polynomial

/-- The polynomial with coefficient list `cs`, lowest degree first. -/
noncomputable def toPolyL {F : Type} [Field F] : List F → F[X]
  | [] => 0
  | c :: cs => C c + X * toPolyL cs

theorem coeff_toPolyL {F : Type} [Field F] : ∀ (cs : List F) (i : ℕ), (toPolyL cs).coeff i = cs.getD i 0
  | [], i => by simp [toPolyL]
  | c :: cs, 0 => by simp [toPolyL]
  | c :: cs, i + 1 => by simp [toPolyL, coeff_X_mul, coeff_toPolyL cs i]

theorem natDegree_toPolyL_lt {F : Type} [Field F] (cs : List F) (h : cs ≠ []) :
    (toPolyL cs).natDegree < cs.length := by
  have hlen : 0 < cs.length := List.length_pos_iff.2 h
  by_cases h0 : toPolyL cs = 0
  · simp [h0, hlen]
  · refine lt_of_not_ge fun hle => ?_
    have := coeff_toPolyL cs (toPolyL cs).natDegree
    rw [List.getD_eq_default _ _ hle] at this
    exact h0 (leadingCoeff_eq_zero.1 this)

/-- Coefficient lists of one length read as one polynomial only if they are equal. -/
theorem toPolyL_inj {F : Type} [Field F] {cs cs' : List F} (hl : cs.length = cs'.length)
    (h : toPolyL cs = toPolyL cs') : cs = cs' := by
  refine List.ext_getElem hl fun i h1 h2 => ?_
  have := congrArg (fun p => p.coeff i) h
  simp only [coeff_toPolyL] at this
  rwa [List.getD_eq_getElem _ _ h1, List.getD_eq_getElem _ _ h2] at this

/-- The first `k` coefficients of `p`. -/
def coeffList {F : Type} [Field F] (k : ℕ) (p : F[X]) : List F := (List.range k).map p.coeff

theorem length_coeffList {F : Type} [Field F] (k : ℕ) (p : F[X]) : (coeffList k p).length = k := by
  simp [coeffList]

theorem toPolyL_coeffList {F : Type} [Field F] (k : ℕ) (p : F[X]) (hp : p.natDegree < k) :
    toPolyL (coeffList k p) = p := by
  ext i
  rw [coeff_toPolyL]
  by_cases hi : i < k
  · simp [coeffList, List.getElem?_range hi]
  · rw [List.getD_eq_default _ _ (by simp [coeffList]; omega)]
    exact (coeff_eq_zero_of_natDegree_lt (by omega)).symm

/-- Horner's rule on a coefficient list: `c₀ + x·(c₁ + x·(…))`, a product and a sum per coefficient. -/
def horner {F : Type} [Field F] (cs : List F) (x : F) : F := cs.foldr (fun c acc => c + x * acc) 0

theorem horner_eq {F : Type} [Field F] : ∀ (cs : List F) (x : F), horner cs x = (toPolyL cs).eval x
  | [], x => by simp [horner, toPolyL]
  | c :: cs, x => by
    have := horner_eq cs x
    simp only [horner, List.foldr_cons] at this ⊢
    rw [this]
    simp [toPolyL]

/-- Row `j` of the table `P` as a polynomial in `X`, on `i ≤ D`. -/
noncomputable def rowX {F : Type} [Field F] (P : ℕ → ℕ → F) (D j : ℕ) : F[X] :=
  ∑ i ∈ Finset.range (D + 1), monomial i (P i j)

/-- **The bivariate polynomial of a table**: `Σ_{i ≤ D, j ≤ L} P i j · X^i Y^j`, `Y` the outer variable. -/
noncomputable def toBi {F : Type} [Field F] (P : ℕ → ℕ → F) (D L : ℕ) : F[X][X] :=
  ∑ j ∈ Finset.range (L + 1), monomial j (rowX P D j)

theorem coeff_rowX {F : Type} [Field F] (P : ℕ → ℕ → F) (D j i : ℕ) :
    (rowX P D j).coeff i = if i ≤ D then P i j else 0 := by
  simp only [rowX, finsetSum_coeff, coeff_monomial]
  rw [Finset.sum_ite_eq']
  simp

theorem coeff_toBi {F : Type} [Field F] (P : ℕ → ℕ → F) (D L i j : ℕ) :
    ((toBi P D L).coeff j).coeff i = if i ≤ D ∧ j ≤ L then P i j else 0 := by
  simp only [toBi, finsetSum_coeff, coeff_monomial, apply_ite (fun p : F[X] => p.coeff i), coeff_zero]
  rw [Finset.sum_ite_eq']
  simp only [Finset.mem_range, Nat.lt_succ_iff, coeff_rowX]
  split_ifs <;> simp_all

/-- The table vanishes off the box `i ≤ D`, `j ≤ L`. -/
def Boxed {F : Type} [Field F] (P : ℕ → ℕ → F) (D L : ℕ) : Prop := ∀ i j, P i j ≠ 0 → i ≤ D ∧ j ≤ L

theorem coeff_toBi_of_boxed {F : Type} [Field F] {P : ℕ → ℕ → F} {D L : ℕ} (h : Boxed P D L) (i j : ℕ) :
    ((toBi P D L).coeff j).coeff i = P i j := by
  rw [coeff_toBi]
  split_ifs with h'
  · rfl
  · by_contra hne
    exact h' (h i j (Ne.symm hne))

theorem eval_toBi {F : Type} [Field F] (P : ℕ → ℕ → F) (D L : ℕ) (g : F[X]) :
    (toBi P D L).eval g = ∑ j ∈ Finset.range (L + 1), rowX P D j * g ^ j := by
  simp [toBi, eval_finsetSum, eval_monomial]

end Claims.GS
