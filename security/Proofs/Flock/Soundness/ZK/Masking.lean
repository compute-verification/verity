module

public import Proofs.Flock.Soundness.ListSize
import Proofs.Flock.Soundness.NovelBasis
import VCVio.Prelude.Core

/-! Zero knowledge: masked and padded openings are uniform (T2–T4, `padded_openings_uniform`), and T7's numeric step. -/

@[expose] public section

namespace FlockSoundness.ZK

open Finset Polynomial Model

theorem card_translate_fiber {A B : Type*} [AddCommGroup A] [Fintype A] [AddCommGroup B] [Fintype B] [DecidableEq B]
    (Φ : A →+ B) (hΦ : Function.Surjective Φ) (base y : B) :
    #{a | base + Φ a = y} * Fintype.card B = Fintype.card A := by
  have hfib : ∀ y' : B, #{a | Φ a = y'} = #{a | Φ a = 0} := fun y' =>
    AddMonoidHom.card_fiber_eq_of_mem_range Φ (hΦ y') (hΦ 0)
  have hsum : Fintype.card A = ∑ y' : B, #{a | Φ a = y'} := by
    rw [← Finset.card_univ]
    exact Finset.card_eq_sum_card_fiberwise (fun a _ => Finset.mem_univ (Φ a))
  have hshift : #{a | base + Φ a = y} = #{a | Φ a = y - base} := by
    congr 1
    ext a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · intro h
      rw [← h]
      abel
    · intro h
      rw [h]
      abel
  rw [hsum, hshift, hfib, Finset.sum_congr rfl (fun y' _ => hfib y'), Finset.sum_const, Finset.card_univ,
    smul_eq_mul, mul_comm]

noncomputable def paddingMap {F ι : Type*} [Field F] {t : ℕ} (b : Fin t → F[X]) (x g : ι → F) :
    (Fin t → F) →+ (ι → F) where
  toFun μ i := g i * (∑ k, C (μ k) * b k).eval (x i)
  map_zero' := by
    ext i
    simp
  map_add' μ ν := by
    ext i
    simp only [Pi.add_apply, map_add, add_mul, Finset.sum_add_distrib, eval_add, mul_add]

theorem padding_surjective {F ι : Type*} [Field F] [Fintype ι] [DecidableEq ι] {t : ℕ} (b : Fin t → F[X])
    (x : ι → F) (hx : Function.Injective x) (g : ι → F) (hg : ∀ i, g i ≠ 0)
    (hspan : ∀ P : F[X], P.degree < Fintype.card ι → ∃ m : Fin t → F, ∑ k, C (m k) * b k = P) :
    Function.Surjective (paddingMap b x g) := by
  intro y
  have hdeg : (Lagrange.interpolate (Finset.univ : Finset ι) x (fun i => y i / g i)).degree < Fintype.card ι := by
    have := Lagrange.degree_interpolate_lt (s := (Finset.univ : Finset ι)) (fun i => y i / g i) hx.injOn
    simpa [Finset.card_univ] using this
  obtain ⟨m, hm⟩ := hspan _ hdeg
  refine ⟨m, funext fun i => ?_⟩
  show g i * (∑ k, C (m k) * b k).eval (x i) = y i
  rw [hm, Lagrange.eval_interpolate_at_node _ hx.injOn (Finset.mem_univ i)]
  field_simp [hg i]

theorem padded_openings_uniform {F ι : Type*} [Field F] [Fintype F] [DecidableEq F] [Fintype ι] [DecidableEq ι]
    {t : ℕ} (b : Fin t → F[X]) (x : ι → F) (hx : Function.Injective x) (g : ι → F) (hg : ∀ i, g i ≠ 0)
    (hspan : ∀ P : F[X], P.degree < Fintype.card ι → ∃ m : Fin t → F, ∑ k, C (m k) * b k = P)
    (base y : ι → F) :
    #{μ : Fin t → F | ∀ i, base i + g i * (∑ k, C (μ k) * b k).eval (x i) = y i}
        * Fintype.card F ^ Fintype.card ι = Fintype.card F ^ t := by
  have h := card_translate_fiber (paddingMap b x g) (padding_surjective b x hx g hg hspan) base y
  rw [Fintype.card_fun, Fintype.card_fun, Fintype.card_fin] at h
  rw [← h]
  congr 2
  ext μ
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, funext_iff, Pi.add_apply]
  rfl

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

theorem level0_openings_uniform (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) (t : ℕ) (ht : t ≤ 2 ^ L)
    {ι : Type} [Fintype ι] [DecidableEq ι] (hι : Fintype.card ι ≤ t) (x : ι → F) (hx : Function.Injective x)
    (g : ι → F) (hg : ∀ i, g i ≠ 0) (base y : ι → F) :
    #{μ : Fin t → F | ∀ i, base i + g i * (∑ k, C (μ k) * xP A hA L hL (Fin.castLE ht k)).eval (x i) = y i}
        * Fintype.card F ^ Fintype.card ι = Fintype.card F ^ t := by
  refine padded_openings_uniform _ x hx g hg (fun P hP => ?_) base y
  by_cases hP0 : P = 0
  · exact ⟨0, by simp [hP0]⟩
  · have hnat : P.natDegree < t :=
      lt_of_lt_of_le ((Polynomial.natDegree_lt_iff_degree_lt hP0).2 hP) hι
    exact xP_span_lt A hA L hL t ht P hnat

theorem gk_ratio_le_third (p : ℝ) (hp1 : p ≤ 1) :
    3 * (p * (1 - p / 2) ^ 4) ≤ 1 - (1 - p) * (1 - p / 2) ^ 4 := by
  have key : 1 - (1 - p) * (1 - p / 2) ^ 4 - 3 * (p * (1 - p / 2) ^ 4)
      = p ^ 2 / 4 * ((10 - 10 * p) + p ^ 2 * (15 / 4 - p / 2)) := by
    ring
  have h1 : 0 ≤ p ^ 2 / 4 := by positivity
  have h2 : 0 ≤ (10 - 10 * p) + p ^ 2 * (15 / 4 - p / 2) := by
    have : 0 ≤ p ^ 2 * (15 / 4 - p / 2) := mul_nonneg (sq_nonneg p) (by linarith)
    linarith
  nlinarith [mul_nonneg h1 h2]

end FlockSoundness.ZK
