module

public import Definitions.Core.Guarantees.MemCheck
public import Mathlib.Algebra.MvPolynomial.Degrees
public import Mathlib.Algebra.Polynomial.Degree.Defs
public import Mathlib.Algebra.Polynomial.Eval.Defs
public import Mathlib.FieldTheory.Finite.GaloisField
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.MvPolynomial.CommRing
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Data.Fin.VecNotation
import Mathlib.Logic.Equiv.Fin.Basic
import Proofs.Core.Game.Basic
import Proofs.Core.Game.Poly
import Proofs.Core.Game.Prob
import Proofs.Core.Game.Tape

/-!
# What a fingerprint collision costs a memory check: proofs

`FingerprintCollision`. A multiset's fingerprint is the value at `(α, γ)` of the polynomial
`∏ c, (X 0 + Σ_i C(c_i) X 1^i)` (`fingerprint2`), of total degree at most `|A| · max 1 (m - 1)`. Two different multisets
give different polynomials (`fingerprint2_injective`): sent to `F[γ][α]`, each is `∏ c, (α - g c)` with
`g c = -Σ_i C(c_i) γ^i`, injective in `c`, and the roots of that product are the multiset of the `g c`. So the difference
is a nonzero polynomial of total degree at most `max |A| |B| · max 1 (m - 1)`, and Schwartz–Zippel
(`Game.prCoin_mvEval_eq_zero_le`) bounds the coins at which it vanishes.

`PrintCollision`. Two different keys' prints differ by a nonzero polynomial of degree at most `m - 1` in `δ`
(`chunkPoly`), which vanishes at a uniform `δ` with probability at most `(m - 1) / |F|`; a union bound over the
`C(|S|, 2)` pairs, by induction on `S`, gives the rest.

Witnesses, at GF(2^128) (Mathlib's `GaloisField 2 128`): `FingerprintCollision_honest`, an honest log's two sides (one
multiset of two-chunk records listed in two orders) have equal fingerprints at every coin, and against a different
multiset the bound at two records of two chunks holds, `2 / 2^128`; `FingerprintCollision_nontrivial`, a check that
accepts every pair of multisets accepts two different ones with probability 1, above the bound `1 / 2^128`.
`PrintCollision_honest`, two keys of four chunks share a print with probability at most `3 / 2^128`;
`PrintCollision_nontrivial`, a check that takes every two keys to share a print does so with probability 1, above
`1 / 2^128`.
-/

@[expose] public section

namespace Verity.MemCheck

open Polynomial
open scoped ENNReal

section

variable {F : Type} [Field F]

/-- A record's chunks as a polynomial in `γ`: `Σ_i C(c_i) X^i`. -/
noncomputable def chunkPoly {m : ℕ} (c : Fin m → F) : F[X] :=
  ∑ i : Fin m, C (c i) * X ^ (i : ℕ)

theorem eval_chunkPoly {m : ℕ} (c : Fin m → F) (γ : F) : (chunkPoly c).eval γ = element γ c := by
  simp only [chunkPoly, element, eval_finsetSum, eval_mul, eval_C, eval_pow, eval_X]
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

theorem coeff_chunkPoly {m : ℕ} (c : Fin m → F) (j : Fin m) : (chunkPoly c).coeff j = c j := by
  classical
  rw [chunkPoly, finsetSum_coeff, Finset.sum_eq_single j]
  · simp
  · intro i _ hij
    simp [Fin.val_inj, Ne.symm hij]
  · simp

theorem chunkPoly_injective {m : ℕ} : Function.Injective (chunkPoly (F := F) (m := m)) := by
  intro c c' h
  funext j
  rw [← coeff_chunkPoly c j, ← coeff_chunkPoly c' j, h]

theorem natDegree_chunkPoly_le {m : ℕ} (c : Fin m → F) : (chunkPoly c).natDegree ≤ m - 1 :=
  natDegree_sum_le_of_forall_le _ _ fun i _ =>
    (natDegree_C_mul_X_pow_le _ _).trans (by have := i.isLt; omega)

/-- A record's factor `α + Σ_i γ^i c_i` as a polynomial in `(α, γ) = (X 0, X 1)`. -/
noncomputable def factor2 {m : ℕ} (c : Fin m → F) : MvPolynomial (Fin 2) F :=
  MvPolynomial.X 0 + ∑ i : Fin m, MvPolynomial.C (c i) * MvPolynomial.X 1 ^ (i : ℕ)

/-- A multiset's fingerprint as a polynomial in `(α, γ) = (X 0, X 1)`. -/
noncomputable def fingerprint2 {m : ℕ} (A : Multiset (Fin m → F)) : MvPolynomial (Fin 2) F :=
  (A.map factor2).prod

theorem eval_factor2 {m : ℕ} (c : Fin m → F) (f : Fin 2 → F) :
    MvPolynomial.eval f (factor2 c) = f 0 + element (f 1) c := by
  simp only [factor2, element, map_add, map_sum, map_mul, map_pow, MvPolynomial.eval_X, MvPolynomial.eval_C]
  congr 1
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

theorem eval_fingerprint2 {m : ℕ} (A : Multiset (Fin m → F)) (f : Fin 2 → F) :
    MvPolynomial.eval f (fingerprint2 A) = fingerprint (f 0) (f 1) A := by
  simp only [fingerprint2, fingerprint, map_multiset_prod, Multiset.map_map, Function.comp_def, eval_factor2]

theorem totalDegree_factor2_le {m : ℕ} (c : Fin m → F) : (factor2 c).totalDegree ≤ max 1 (m - 1) := by
  refine (MvPolynomial.totalDegree_add _ _).trans (max_le_max ?_ ?_)
  · exact (MvPolynomial.totalDegree_X _).le
  · refine (MvPolynomial.totalDegree_finsetSum _ _).trans (Finset.sup_le fun i _ => ?_)
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_C, MvPolynomial.totalDegree_X_pow, zero_add]
    have := i.isLt
    omega

theorem totalDegree_fingerprint2_le {m : ℕ} (A : Multiset (Fin m → F)) :
    (fingerprint2 A).totalDegree ≤ A.card * max 1 (m - 1) := by
  refine (MvPolynomial.totalDegree_multiset_prod _).trans ?_
  have h := Multiset.sum_le_card_nsmul ((A.map factor2).map MvPolynomial.totalDegree) (max 1 (m - 1)) (by
    simp only [Multiset.map_map, Function.comp_def, Multiset.mem_map]
    rintro _ ⟨c, -, rfl⟩
    exact totalDegree_factor2_le _)
  simpa using h

/-- Two different multisets have different fingerprint polynomials: sent to `F[γ][α]`, each is the product of
`α - (-Σ_i C(c_i) γ^i)` over its records, whose roots are the multiset of the `-Σ_i C(c_i) γ^i`. -/
theorem fingerprint2_injective {m : ℕ} : Function.Injective (fingerprint2 (F := F) (m := m)) := by
  intro A B h
  let φ : MvPolynomial (Fin 2) F →+* Polynomial (Polynomial F) :=
    MvPolynomial.eval₂Hom (Polynomial.C.comp Polynomial.C) ![Polynomial.X, Polynomial.C Polynomial.X]
  let g : (Fin m → F) → Polynomial F := fun c => -chunkPoly c
  have hg : Function.Injective g := fun c c' hcc' => chunkPoly_injective (neg_injective hcc')
  have hφ : ∀ A : Multiset (Fin m → F),
      φ (fingerprint2 A) = ((A.map g).map fun a => Polynomial.X - Polynomial.C a).prod := by
    intro A
    simp only [fingerprint2, map_multiset_prod, Multiset.map_map, Function.comp_def]
    congr 1
    refine Multiset.map_congr rfl fun c _ => ?_
    simp only [φ, g, factor2, chunkPoly, map_add, map_sum, map_mul, map_pow, MvPolynomial.eval₂Hom_X',
      MvPolynomial.eval₂Hom_C, RingHom.coe_comp, Function.comp_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
      sub_neg_eq_add, map_neg]
  have := congrArg (fun p => (φ p).roots) h
  simp only [hφ, Polynomial.roots_multiset_prod_X_sub_C] at this
  exact Multiset.map_injective hg this

end

/-- Relabelling a uniform coin by an equivalence keeps an event's probability. -/
theorem prCoin_equiv {C D : Type} [Fintype C] [Fintype D] (e : C ≃ D) (p : D → Prop) :
    Game.prCoin (fun c => p (e c)) = Game.prCoin p := by
  classical
  unfold Game.prCoin
  exact Game.Tape.avg_equiv e fun d => if p d then 1 else 0

theorem prCoin_of_forall {C : Type} [Fintype C] [Nonempty C] {p : C → Prop} (h : ∀ c, p c) : Game.prCoin p = 1 := by
  unfold Game.prCoin
  simp only [h, ite_true]
  exact Game.avg_const _

variable {F : Type} [Field F] [Fintype F]

/-- Two different packed keys share a print at a uniform `δ` with probability at most `(m - 1) / |F|`. -/
theorem prCoin_element_eq_le {m : ℕ} (c c' : Fin m → F) (h : c ≠ c') :
    Game.prCoin (fun δ : F => element δ c = element δ c') ≤ ((m - 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
  classical
  have hne : chunkPoly c - chunkPoly c' ≠ 0 := sub_ne_zero.2 fun e => h (chunkPoly_injective e)
  calc Game.prCoin (fun δ : F => element δ c = element δ c')
      = Game.prCoin (fun δ : F => (chunkPoly c - chunkPoly c').eval δ = 0) := by
        congr 1
        funext δ
        simp [eval_chunkPoly, sub_eq_zero]
    _ ≤ ((chunkPoly c - chunkPoly c').natDegree : ℝ≥0∞) / Fintype.card F :=
        Game.prCoin_eval_eq_zero_le _ hne
    _ ≤ ((m - 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
        gcongr
        exact (natDegree_sub_le _ _).trans (max_le (natDegree_chunkPoly_le c) (natDegree_chunkPoly_le c'))

theorem card_gf128 : Fintype.card (GaloisField 2 128) = 2 ^ 128 := by
  rw [Fintype.card_eq_nat_card, GaloisField.card 2 128 (by norm_num)]

end Verity.MemCheck

namespace Verity.SecurityProofs

open Verity.MemCheck Game
open scoped ENNReal

theorem FingerprintCollision : Guarantees.FingerprintCollision := by
  intro F _ _ m A B
  classical
  by_cases hAB : A = B
  · refine le_trans (prCoin_mono (q := fun _ => False) fun c hc => hc.1 hAB) ?_
    rw [prCoin_false]
    exact zero_le
  have hne : fingerprint2 A - fingerprint2 B ≠ 0 := sub_ne_zero.2 fun h => hAB (fingerprint2_injective h)
  calc prCoin (fun c : F × F => A ≠ B ∧ fingerprint c.1 c.2 A = fingerprint c.1 c.2 B)
      = prCoin (fun f : Fin 2 → F => MvPolynomial.eval f (fingerprint2 A - fingerprint2 B) = 0) := by
        rw [← prCoin_equiv (finTwoArrowEquiv F)]
        congr 1
        funext f
        simp [eval_fingerprint2, sub_eq_zero, hAB, finTwoArrowEquiv]
    _ ≤ ((fingerprint2 A - fingerprint2 B).totalDegree : ℝ≥0∞) / Fintype.card F :=
        prCoin_mvEval_eq_zero_le _ hne
    _ ≤ ((max A.card B.card * max 1 (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F := by
        gcongr
        refine (MvPolynomial.totalDegree_sub _ _).trans (max_le ?_ ?_)
        · exact (totalDegree_fingerprint2_le A).trans (Nat.mul_le_mul_right _ (le_max_left _ _))
        · exact (totalDegree_fingerprint2_le B).trans (Nat.mul_le_mul_right _ (le_max_right _ _))

theorem PrintCollision : Guarantees.PrintCollision := by
  intro F _ _ m S
  classical
  induction S using Finset.induction_on with
  | empty => simp [prCoin_false]
  | insert a S ha ih =>
    calc prCoin (fun δ : F => ∃ k ∈ insert a S, ∃ k' ∈ insert a S, k ≠ k' ∧ element δ k = element δ k')
        ≤ prCoin (fun δ : F => (∃ k ∈ S, ∃ k' ∈ S, k ≠ k' ∧ element δ k = element δ k') ∨
            ∃ k' ∈ S, element δ a = element δ k') := by
          refine prCoin_mono fun δ => ?_
          rintro ⟨k, hk, k', hk', hkk', he⟩
          rw [Finset.mem_insert] at hk hk'
          rcases hk with rfl | hk <;> rcases hk' with rfl | hk'
          · exact absurd rfl hkk'
          · exact Or.inr ⟨k', hk', he⟩
          · exact Or.inr ⟨k, hk, he.symm⟩
          · exact Or.inl ⟨k, hk, k', hk', hkk', he⟩
      _ ≤ ((S.card.choose 2 * (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F +
            ∑ k' ∈ S, ((m - 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
          refine (prCoin_or_le _ _).trans (add_le_add ih ((prCoin_exists_le _ _).trans ?_))
          exact Finset.sum_le_sum fun k' hk' => prCoin_element_eq_le _ _ fun h => ha (h ▸ hk')
      _ = (((insert a S).card.choose 2 * (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F := by
          rw [Finset.sum_const, nsmul_eq_mul, ← mul_div_assoc, ENNReal.div_add_div_same,
            Finset.card_insert_of_notMem ha, Nat.choose_succ_succ, Nat.choose_one_right]
          push_cast
          ring_nf

/-- **`FingerprintCollision`'s honest lemma.** At GF(2^128), with records of two chunks: the two sides of an honest log,
one multiset listed in two orders, have equal fingerprints at every coin, so the check accepts it with probability 1;
and against a different multiset of two records the collision probability is at most the bound at two records of two
chunks, `2 / 2^128`. -/
theorem FingerprintCollision_honest :
    prCoin (fun c : GaloisField 2 128 × GaloisField 2 128 =>
        fingerprint (m := 2) c.1 c.2 {1, 0} = fingerprint (m := 2) c.1 c.2 {0, 1}) = 1 ∧
      prCoin (fun c : GaloisField 2 128 × GaloisField 2 128 =>
        ({1, 0} : Multiset (Fin 2 → GaloisField 2 128)) ≠ {1, 1} ∧
          fingerprint (m := 2) c.1 c.2 {1, 0} = fingerprint (m := 2) c.1 c.2 {1, 1}) ≤ 2 / 2 ^ 128 := by
  refine ⟨prCoin_of_forall fun c => by rw [Multiset.pair_comm], ?_⟩
  have := FingerprintCollision (F := GaloisField 2 128) 2 {1, 0} {1, 1}
  rw [card_gf128] at this
  exact this.trans_eq (by norm_num)

/-- **`FingerprintCollision`'s nontrivial lemma: its bound fails at a check that accepts everything.** At GF(2^128), a
check that accepts every pair of multisets accepts the different multisets `{1}` and `{0}` of one-chunk records with
probability 1, above the bound at one record of one chunk, `1 / 2^128`. -/
theorem FingerprintCollision_nontrivial :
    ¬ prCoin (fun _ : GaloisField 2 128 × GaloisField 2 128 =>
        ({1} : Multiset (Fin 1 → GaloisField 2 128)) ≠ {0} ∧ True) ≤ 1 / 2 ^ 128 := by
  rw [prCoin_of_forall fun _ => ⟨by simp, trivial⟩, one_div, not_le]
  exact ENNReal.inv_lt_one.2 (by norm_num)

open Classical in
/-- **`PrintCollision`'s honest lemma.** At GF(2^128), the two keys `0` and `1` of four chunks share a print at a
uniform `δ` with probability at most the bound at two keys of four chunks, `3 / 2^128`. -/
theorem PrintCollision_honest :
    prCoin (fun δ : GaloisField 2 128 => ∃ k ∈ ({0, 1} : Finset (Fin 4 → GaloisField 2 128)),
        ∃ k' ∈ ({0, 1} : Finset (Fin 4 → GaloisField 2 128)), k ≠ k' ∧ element δ k = element δ k') ≤ 3 / 2 ^ 128 := by
  have := PrintCollision (F := GaloisField 2 128) 4 {0, 1}
  rw [card_gf128, Finset.card_pair zero_ne_one] at this
  exact this.trans_eq (by norm_num)

open Classical in
/-- **`PrintCollision`'s nontrivial lemma: its bound fails at a check that takes every two keys to share a print.** At
GF(2^128), such a check finds the two keys `0` and `1` of two chunks sharing a print with probability 1, above the
bound at two keys of two chunks, `1 / 2^128`. -/
theorem PrintCollision_nontrivial :
    ¬ prCoin (fun _ : GaloisField 2 128 => ∃ k ∈ ({0, 1} : Finset (Fin 2 → GaloisField 2 128)),
        ∃ k' ∈ ({0, 1} : Finset (Fin 2 → GaloisField 2 128)), k ≠ k' ∧ True) ≤ 1 / 2 ^ 128 := by
  rw [prCoin_of_forall fun _ => ⟨0, by simp, 1, by simp, zero_ne_one, trivial⟩, one_div, not_le]
  exact ENNReal.inv_lt_one.2 (by norm_num)

end Verity.SecurityProofs
