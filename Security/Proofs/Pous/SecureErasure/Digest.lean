import Mathlib.Algebra.Polynomial.Roots
import Proofs.Pous.Statements

/-!
# The digest is almost universal

`Pous.Guarantees.DigestAU` from root counting: a nonzero polynomial of degree at most `d` over a field vanishes at at
most `d` points (`Polynomial.card_roots'`). Ported from PR #1010 (`PousBandwidth` there), the digest's lemmas only.
-/

namespace PousDigest

open Pous Pous.Digest Polynomial
open scoped ENNReal

/-- `pr P ≤ n / m` from a count: `|{P}| ≤ n · d` out of `m · d`. -/
theorem pr_le_of_card {α : Type} [Fintype α] (P : α → Prop) [DecidablePred P] (n m d : ℕ) (hd : d ≠ 0)
    (hα : Fintype.card α = m * d) (h : (Finset.univ.filter P).card ≤ n * d) : pr P ≤ (n : ℝ≥0∞) / m := by
  unfold pr
  rw [Finset.filter_congr_decidable, hα]
  calc ((Finset.univ.filter P).card : ℝ≥0∞) / ((m * d : ℕ) : ℝ≥0∞)
      ≤ ((n * d : ℕ) : ℝ≥0∞) / ((m * d : ℕ) : ℝ≥0∞) := by gcongr
    _ = (n : ℝ≥0∞) / m := by
        push_cast
        exact ENNReal.mul_div_mul_right _ _ (by exact_mod_cast hd) (ENNReal.natCast_ne_top d)

/-- A filter on a product, counted slice by slice along the first factor. -/
theorem card_filter_prod {α β : Type} [Fintype α] [Fintype β] (P : α × β → Prop) [DecidablePred P] :
    (Finset.univ.filter P).card = ∑ a, (Finset.univ.filter fun b => P (a, b)).card := by
  rw [Finset.card_filter, Fintype.sum_prod_type]
  exact Finset.sum_congr rfl fun a _ => (Finset.card_filter _ _).symm

/-- A filter on a product, counted slice by slice along the second factor. -/
theorem card_filter_prod_right {α β : Type} [Fintype α] [Fintype β] (P : α × β → Prop) [DecidablePred P] :
    (Finset.univ.filter P).card = ∑ b, (Finset.univ.filter fun a => P (a, b)).card := by
  rw [Finset.card_filter, Fintype.sum_prod_type_right]
  exact Finset.sum_congr rfl fun b _ => (Finset.card_filter _ _).symm

variable {F : Type} [Field F]

theorem blockHash_sub {W : ℕ} (g : F) (x y : Blk W F) :
    blockHash g x - blockHash g y = ∑ j : Fin W, (x j - y j) * g ^ (W - 1 - (j : ℕ)) := by
  simp [blockHash, ← Finset.sum_sub_distrib, sub_mul]

theorem digest_sub {R W : ℕ} (g z : F) (xs ys : Fin R → Blk W F) :
    digest g z xs - digest g z ys = ∑ r : Fin R, (blockHash g (xs r) - blockHash g (ys r)) * z ^ (r : ℕ) := by
  simp [digest, ← Finset.sum_sub_distrib, sub_mul]

variable [Fintype F] [DecidableEq F]

/-- A nonzero sum `Σ_i a_i x^(e_i)` with distinct exponents, each at most `d`, vanishes at at most `d` points. -/
theorem card_zeros_le {n : ℕ} (a : Fin n → F) (e : Fin n → ℕ) (he : Function.Injective e) (d : ℕ)
    (hd : ∀ i, e i ≤ d) (ha : a ≠ 0) :
    (Finset.univ.filter fun x : F => ∑ i, a i * x ^ e i = 0).card ≤ d := by
  classical
  set P : F[X] := ∑ i, C (a i) * X ^ e i with hP
  have heval : ∀ x, P.eval x = ∑ i, a i * x ^ e i := by
    intro x
    simp [hP, eval_finsetSum]
  obtain ⟨i0, hi0⟩ : ∃ i, a i ≠ 0 := by
    by_contra h
    push Not at h
    exact ha (funext h)
  have hcoeff : P.coeff (e i0) = a i0 := by
    rw [hP, finsetSum_coeff, Finset.sum_eq_single i0]
    · simp
    · intro j _ hj
      simp [he.ne hj.symm]
    · simp
  have hP0 : P ≠ 0 := by
    intro h
    apply hi0
    rw [← hcoeff, h, coeff_zero]
  have hdeg : P.natDegree ≤ d :=
    natDegree_sum_le_of_forall_le _ _ fun i _ => (natDegree_C_mul_X_pow_le _ _).trans (hd i)
  have hsub : (Finset.univ.filter fun x : F => ∑ i, a i * x ^ e i = 0) ⊆ P.roots.toFinset := by
    intro x hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
    rw [Multiset.mem_toFinset, mem_roots hP0, IsRoot, heval]
    exact hx
  exact (Finset.card_le_card hsub).trans ((Multiset.toFinset_card_le _).trans ((card_roots' P).trans hdeg))

/-- Two different blocks hash alike at at most `W − 1` points `g`. -/
theorem card_blockHash_eq_le {W : ℕ} (x y : Blk W F) (hxy : x ≠ y) :
    (Finset.univ.filter fun g : F => blockHash g x = blockHash g y).card ≤ W - 1 := by
  classical
  refine le_trans (Finset.card_le_card fun g hg => ?_)
    (card_zeros_le (fun j : Fin W => x j - y j) (fun j => W - 1 - (j : ℕ)) (fun i j h => ?_) (W - 1)
      (fun j => Nat.sub_le _ _) fun h => hxy (funext fun j => sub_eq_zero.1 (congrFun h j)))
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hg ⊢
    rw [← blockHash_sub, hg, sub_self]
  · have := i.isLt
    have := j.isLt
    exact Fin.ext (by simp only at h; omega)

/-- If some block hash differs at `g`, the digests agree at at most `R − 1` points `z`. -/
theorem card_digest_eq_le {R W : ℕ} (g : F) (xs ys : Fin R → Blk W F)
    (h : ∃ r, blockHash g (xs r) ≠ blockHash g (ys r)) :
    (Finset.univ.filter fun z : F => digest g z xs = digest g z ys).card ≤ R - 1 := by
  classical
  obtain ⟨r, hr⟩ := h
  refine le_trans (Finset.card_le_card fun z hz => ?_)
    (card_zeros_le (fun r : Fin R => blockHash g (xs r) - blockHash g (ys r)) (fun r => (r : ℕ)) Fin.val_injective
      (R - 1) (fun r => by have := r.isLt; omega) fun h0 => hr (sub_eq_zero.1 (congrFun h0 r)))
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hz ⊢
  rw [← digest_sub, hz, sub_self]

theorem digestAU : Pous.Guarantees.DigestAU := by
  intro R W F _ _ xs ys hne
  classical
  obtain ⟨r0, hr0⟩ : ∃ r, xs r ≠ ys r := by
    by_contra h
    push Not at h
    exact hne (funext h)
  set G := Finset.univ.filter fun g : F => blockHash g (xs r0) = blockHash g (ys r0) with hGdef
  have hG : G.card ≤ W - 1 := card_blockHash_eq_le _ _ hr0
  have hslice : ∀ g : F, (Finset.univ.filter fun z : F => digest g z xs = digest g z ys).card ≤
      (if g ∈ G then Fintype.card F else 0) + (R - 1) := by
    intro g
    by_cases hall : ∀ r, blockHash g (xs r) = blockHash g (ys r)
    · have hg : g ∈ G := by simp [hGdef, hall r0]
      simp only [hg, ite_true]
      exact le_add_right (Finset.card_le_univ _)
    · push Not at hall
      exact le_add_left (card_digest_eq_le g xs ys hall)
  have hcount : (Finset.univ.filter fun k : Key F => digest k.1 k.2 xs = digest k.1 k.2 ys).card ≤
      (W - 1 + (R - 1)) * Fintype.card F := by
    rw [card_filter_prod]
    calc ∑ g : F, (Finset.univ.filter fun z : F => digest (g, z).1 (g, z).2 xs = digest (g, z).1 (g, z).2 ys).card
        ≤ ∑ g : F, ((if g ∈ G then Fintype.card F else 0) + (R - 1)) := Finset.sum_le_sum fun g _ => hslice g
      _ = G.card * Fintype.card F + Fintype.card F * (R - 1) := by
        rw [Finset.sum_add_distrib, Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_const, Finset.sum_const,
          smul_eq_mul, smul_eq_mul, Finset.card_univ]
      _ ≤ (W - 1) * Fintype.card F + Fintype.card F * (R - 1) := by gcongr
      _ = (W - 1 + (R - 1)) * Fintype.card F := by ring
  exact pr_le_of_card _ _ _ _ Fintype.card_ne_zero (by rw [Fintype.card_prod]) hcount

end PousDigest
