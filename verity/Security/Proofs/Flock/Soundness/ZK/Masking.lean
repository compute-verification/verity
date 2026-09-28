import Proofs.Flock.Soundness.NovelBasis
import Mathlib.LinearAlgebra.Lagrange
import Mathlib.GroupTheory.Index

/-!
# Zero knowledge: masked and padded openings are uniform

The statistical core of the public-circuit zero-knowledge proof of masked Flock (C-Flock ZK; the proof is
`docs/zk-proof-public.md` in the Project store, §4.1's T2–T4 and Lemma B's steps (ii) and (viii)), and the numeric step of
its Goldreich–Kahan abort bound (T7). `README.md` §3 lists the first as to be proved, not assumed: "uniformity of opened
columns of a padded Reed–Solomon codeword when the queries are at most the padding".

* `card_translate_fiber` (T2, T3): for a surjective additive map `Φ` of finite groups, `base + Φ a` takes every value
  equally often, whatever `base` is. Every mask of the protocol is an instance: a pad (`Φ` the identity), the mask slot
  (GF(2)-linear, onto when the prover's rank check passes), an extra-lane pair (`ρ₀ a + ρ₁ b`, invertible when the prover's
  independence check passes), and a padding.
* `padding_surjective` (T4): at distinct points where `g` doesn't vanish, a padding drawn from a family that spans the
  degrees below the number of points reaches every vector of values.
* `padded_openings_uniform`: the two together. Every vector of opened values is reached by the same number of paddings,
  whatever the message part is.
* `level0_openings_uniform`: the instance at C-Flock ZK's level 0, whose padding is `(X_L + κ)·μ` with `μ` a combination of the first
  `t` novel-basis polynomials (`flock-zk-b684b12.patch`, `zk::lane_codeword`): at most `t` distinct opened positions of a
  lane are uniform when `X_L + κ` vanishes at none of them.
* `gk_ratio_le_third`: `3·p·b ≤ 1 − (1 − p)·b` for `b = (1 − p/2)^4` and `p ≤ 1`, the step of T7 after the negative
  binomial's generating function.
-/

namespace FlockSoundness.ZK

open Finset Polynomial Model

/-- **T2 and T3.** For a surjective additive map `Φ : A → B` of finite groups, each value of `a ↦ base + Φ a` is taken by
exactly `|A| / |B|` elements `a`, whatever `base` is: a value masked by the image of uniform randomness under a surjective
map is uniform. -/
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

/-- A padding's values at the points `x`, scaled by `g`: a linear function of the padding's coefficients in the family
`b`. -/
noncomputable def paddingMap {F ι : Type*} [Field F] {t : ℕ} (b : Fin t → F[X]) (x g : ι → F) :
    (Fin t → F) →+ (ι → F) where
  toFun μ i := g i * (∑ k, C (μ k) * b k).eval (x i)
  map_zero' := by
    ext i
    simp
  map_add' μ ν := by
    ext i
    simp only [Pi.add_apply, map_add, add_mul, Finset.sum_add_distrib, eval_add, mul_add]

/-- **T4.** At distinct points `x` where `g` doesn't vanish, if the family `b` spans every polynomial of degree below the
number of points, the padding's scaled values reach every vector: the Lagrange interpolant of `y / g` is a combination of
`b`. -/
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

/-- **The key statistical lemma.** At distinct points `x` where `g` doesn't vanish, with a family `b` that spans the degrees
below the number of points, every vector `y` of opened values is reached by exactly `|F|^t / |F|^#points` paddings `μ`,
whatever the message part `base` is. So, for a uniform padding, the opened values are uniform and independent of the
message. -/
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

/-- **C-Flock ZK's level 0.** A lane's padding is `(X_L + κ)·μ`, `μ` a combination of the first `t` novel-basis
polynomials (`zk::lane_codeword`, `t = t_pad`). At at most `t` distinct opened positions `x` where `g = X_L + κ`
vanishes at none, every vector of opened values is reached by `|F|^t / |F|^#positions` paddings, whatever the lane's
message values `base` are. -/
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

/-- **T7's numeric step.** For `p ≤ 1` and `b = (1 − p/2)^4`, `3·p·b ≤ 1 − (1 − p)·b`: the generating function
`p·s / (1 − (1 − p)·s)` of one geometric variable, at `s = (1 − p/2)^4`, is at most `1/3`. So `t` successes cost at least
`4t/p̃` failed rewinds with probability at most `3^-t` whenever the true rate is at least half the estimate. -/
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
