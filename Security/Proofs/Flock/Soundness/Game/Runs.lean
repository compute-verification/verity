import Proofs.Flock.Soundness.Game.Expect

/-!
# Independent runs of a strategy

`expectN g s n F` is the expectation of `F` over `n` independent runs of the strategy `s`, `F`
reading the list of their outputs, first run first. A per-run quantity summed over the runs has `n`
times its expectation (`expectN_sum_map`), and a product over independent runs is the expectation
to the `n` (`expectN_prod_map`).
-/

namespace FlockSoundness

namespace Game

variable {α : Type}

/-- The expectation of `F` over `n` independent runs of `s`. -/
noncomputable def expectN (g : Game α) (s : Strategy g) : ℕ → (List α → ℝ) → ℝ
  | 0, F => F []
  | n + 1, F => expect (fun a => expectN g s n fun as => F (a :: as)) g s

variable (g : Game α) (s : Strategy g)

theorem expectN_const (c : ℝ) : ∀ n, expectN g s n (fun _ => c) = c
  | 0 => rfl
  | n + 1 => by simp only [expectN, expectN_const c n, expect_const]

theorem expectN_mono {F F' : List α → ℝ} (h : ∀ as, F as ≤ F' as) :
    ∀ n, expectN g s n F ≤ expectN g s n F'
  | 0 => h []
  | n + 1 => by
      simp only [expectN]
      exact expect_mono (fun a => expectN_mono (fun as => h (a :: as)) n) g s

theorem expectN_add (F F' : List α → ℝ) :
    ∀ n, expectN g s n (fun as => F as + F' as) = expectN g s n F + expectN g s n F'
  | 0 => rfl
  | n + 1 => by
      simp only [expectN]
      rw [← expect_add]
      exact congrArg (fun f => expect f g s)
        (funext fun a => expectN_add (fun as => F (a :: as)) (fun as => F' (a :: as)) n)

theorem expectN_const_mul (c : ℝ) (F : List α → ℝ) :
    ∀ n, expectN g s n (fun as => c * F as) = c * expectN g s n F
  | 0 => rfl
  | n + 1 => by
      simp only [expectN]
      rw [← expect_const_mul]
      exact congrArg (fun f => expect f g s)
        (funext fun a => expectN_const_mul c (fun as => F (a :: as)) n)

theorem expectN_nonneg {F : List α → ℝ} (h : ∀ as, 0 ≤ F as) : ∀ n, 0 ≤ expectN g s n F
  | 0 => h []
  | n + 1 => by
      simp only [expectN]
      exact expect_nonneg (fun a => expectN_nonneg (fun as => h (a :: as)) n) g s

theorem expectN_sum {ι : Type} (t : Finset ι) (F : ι → List α → ℝ) (n : ℕ) :
    expectN g s n (fun as => ∑ i ∈ t, F i as) = ∑ i ∈ t, expectN g s n (F i) := by
  classical
  induction t using Finset.induction_on with
  | empty => simp only [Finset.sum_empty]; exact expectN_const g s 0 n
  | insert x t hx ih =>
      simp only [Finset.sum_insert hx]
      rw [expectN_add g s (F x) (fun as => ∑ i ∈ t, F i as), ih]

/-- **A per-run quantity summed over `n` runs** has `n` times its expectation. -/
theorem expectN_sum_map (f : α → ℝ) :
    ∀ n, expectN g s n (fun as => (as.map f).sum) = n * expect f g s
  | 0 => by simp [expectN]
  | n + 1 => by
      simp only [expectN, List.map_cons, List.sum_cons]
      have : ∀ a, expectN g s n (fun as => f a + (as.map f).sum) = f a + n * expect f g s :=
        fun a => by rw [expectN_add, expectN_const, expectN_sum_map f n]
      rw [show (fun a => expectN g s n fun as => f a + (as.map f).sum) =
        fun a => f a + n * expect f g s from funext this, expect_add, expect_const]
      push_cast
      ring

/-- **A product over `n` independent runs** is the expectation to the `n`. -/
theorem expectN_prod_map (f : α → ℝ) :
    ∀ n, expectN g s n (fun as => (as.map f).prod) = expect f g s ^ n
  | 0 => by simp [expectN]
  | n + 1 => by
      simp only [expectN, List.map_cons, List.prod_cons]
      have : ∀ a, expectN g s n (fun as => f a * (as.map f).prod) = expect f g s ^ n * f a :=
        fun a => by rw [expectN_const_mul, expectN_prod_map f n, mul_comm]
      rw [show (fun a => expectN g s n fun as => f a * (as.map f).prod) =
        fun a => expect f g s ^ n * f a from funext this, expect_const_mul, pow_succ]

end Game

end FlockSoundness
