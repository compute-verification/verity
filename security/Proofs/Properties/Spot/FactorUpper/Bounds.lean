module

public import Proofs.Properties.Spot.Interval
public import Spot.FactorUpper
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Stirling
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Exponential
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.MeanInequalities
import Mathlib.Data.Nat.Choose.Bounds
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Algebra.BigOperators.Group.Finset.Powerset

/-!
# Rankin's factor bounded above: the checker's arithmetic

What each piece of `Spot.FactorUpper` bounds, before any replay block enters. A logarithm `v` bounds `L ≥ 0`
when `L ≤ e^v`, or `L = 0` at `none` (`LogUB`), through sums (`lse_sound`), products and powers. `lnHead`'s closed form
rests on Stirling's lower bound on `m!` (`Stirling.sqrt_pi_le_stirlingSeq`) with `π > 3.14`. `h` is increasing and
concave, so each chord between breakpoints is below it (`hLower`, `chordsOf`), and `bound` takes the least of its sums
over Rankin's `θ` (`bound_sound`). A group's capped sums and the sums over sets of at most `M` proof blocks are its
binomial sums (`capP`, `lightSets_sound`, `lightPatterns_sound`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs.FactorUpperProof

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Properties.Spot.SecurityProofs (expLowerSound expUpperSound lnLowerSound lnUpperSound)

/-! ## Logarithms -/

/-- `L ≤ e^v`, or `L = 0` at `none`: what a bound on a logarithm says about `L ≥ 0`. -/
def LogUB : Option ℚ → ℝ → Prop
  | none, L => L = 0
  | some v, L => L ≤ Real.exp v

theorem LogUB.mono {o : Option ℚ} {A B : ℝ} (hA : 0 ≤ A) (hAB : A ≤ B) (h : LogUB o B) : LogUB o A := by
  cases o with
  | none => simp only [LogUB] at h ⊢; linarith
  | some v => exact hAB.trans h

theorem LogUB.zero (o : Option ℚ) : LogUB o 0 := by
  cases o with
  | none => rfl
  | some v => exact (Real.exp_pos _).le

theorem LogUB.le_exp {o : Option ℚ} {L : ℝ} {v : ℚ} (h : LogUB o L) (hv : o = some v) : L ≤ Real.exp v := by
  subst hv; exact h

theorem exp_neg_nat_le (p : ℕ) : Real.exp (-(p : ℝ)) ≤ 1 / 2 ^ p := by
  have h2 : (2 : ℝ) ≤ Real.exp 1 := by have := Real.add_one_le_exp (1 : ℝ); linarith
  have : (2 : ℝ) ^ p ≤ Real.exp p := by
    rw [show (p : ℝ) = p * 1 by ring, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by norm_num) h2 p
  rw [Real.exp_neg, one_div]
  exact inv_anti₀ (by positivity) this

theorem exp_le_expUp (x : ℚ) (p : ℕ) : Real.exp x ≤ (expUp x p : ℝ) := by
  unfold expUp
  split_ifs with h
  · have hx : (x : ℝ) ≤ -(p : ℝ) := by exact_mod_cast h
    calc Real.exp x ≤ Real.exp (-(p : ℝ)) := Real.exp_le_exp.2 hx
      _ ≤ 1 / 2 ^ p := exp_neg_nat_le p
      _ = ((1 / 2 ^ p : ℚ) : ℝ) := by push_cast; ring
  · exact expUpperSound x p

theorem expUp_pos (x : ℚ) (p : ℕ) : (0 : ℝ) < (expUp x p : ℝ) := (Real.exp_pos _).trans_le (exp_le_expUp x p)

theorem expLo_le_exp (x : ℚ) (p : ℕ) : (expLo x p : ℝ) ≤ Real.exp x := by
  unfold expLo
  split_ifs
  · push_cast; exact (Real.exp_pos _).le
  · exact expLowerSound x p

theorem log_le_lnUpper {y : ℚ} (_hy : (0 : ℝ) < y) (p : ℕ) : Real.log y ≤ (lnUpper y p : ℝ) :=
  lnUpperSound y p

theorem lnLower_le_log {y : ℚ} (_hy : (0 : ℝ) < y) (p : ℕ) : (lnLower y p : ℝ) ≤ Real.log y :=
  lnLowerSound y p

/-- `x ≤ e^{ln⁺ x}` for an upper bound on `ln x`, `x > 0`. -/
theorem le_exp_lnUpper {y : ℚ} (hy : (0 : ℝ) < y) (p : ℕ) : (y : ℝ) ≤ Real.exp (lnUpper y p) := by
  calc (y : ℝ) = Real.exp (Real.log y) := (Real.exp_log hy).symm
    _ ≤ _ := Real.exp_le_exp.2 (log_le_lnUpper hy p)

theorem exp_lnLower_le {y : ℚ} (hy : (0 : ℝ) < y) (p : ℕ) : Real.exp (lnLower y p) ≤ (y : ℝ) := by
  calc Real.exp (lnLower y p) ≤ Real.exp (Real.log y) := Real.exp_le_exp.2 (lnLower_le_log hy p)
    _ = y := Real.exp_log hy

theorem sum_map_exp_le (ts : List ℚ) (m : ℚ) (p : ℕ) :
    (ts.map fun t : ℚ => Real.exp t).sum ≤ Real.exp m * (((ts.map fun t => expUp (t - m) p).sum : ℚ) : ℝ) := by
  induction ts with
  | nil => simp
  | cons t ts ih =>
    simp only [List.map_cons, List.sum_cons]
    push_cast at ih ⊢
    have : Real.exp t ≤ Real.exp m * (expUp (t - m) p : ℝ) := by
      calc Real.exp t = Real.exp m * Real.exp ((t - m : ℚ) : ℝ) := by
            rw [← Real.exp_add]; push_cast; ring_nf
        _ ≤ _ := mul_le_mul_of_nonneg_left (exp_le_expUp _ p) (Real.exp_pos _).le
    rw [mul_add]
    exact add_le_add this ih

theorem sum_expUp_pos {ts : List ℚ} (hts : ts ≠ []) (m : ℚ) (p : ℕ) :
    (0 : ℝ) < (((ts.map fun t => expUp (t - m) p).sum : ℚ) : ℝ) := by
  obtain ⟨t, ts', rfl⟩ := List.exists_cons_of_ne_nil hts
  have : ∀ l : List ℚ, (0 : ℝ) ≤ (((l.map fun t => expUp (t - m) p).sum : ℚ) : ℝ) := by
    intro l
    induction l with
    | nil => simp
    | cons a l ih =>
      simp only [List.map_cons, List.sum_cons]; push_cast at ih ⊢
      exact add_nonneg (expUp_pos _ p).le ih
  simp only [List.map_cons, List.sum_cons]; push_cast
  have := this ts'; push_cast at this
  exact add_pos_of_pos_of_nonneg (expUp_pos _ p) this

theorem maxOf_ne_nil {ts : List ℚ} {m : ℚ} (h : maxOf ts = some m) : ts ≠ [] := by
  rintro rfl; simp [maxOf] at h

theorem foldl_max_mem (ts : List ℚ) (t : ℚ) : ts.foldl max t ∈ t :: ts := by
  induction ts generalizing t with
  | nil => simp
  | cons a ts ih =>
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 (ih (max t a)) with h | h
    · rcases max_choice t a with h' | h' <;> rw [h, h'] <;> simp
    · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h)

theorem maxOf_mem {ts : List ℚ} {m : ℚ} (h : maxOf ts = some m) : m ∈ ts := by
  cases ts with
  | nil => simp [maxOf] at h
  | cons t ts => simp only [maxOf, Option.some.injEq] at h; subst h; exact foldl_max_mem ts t

/-- **`lse` is above `ln Σ e^t`.** -/
theorem lse_sound {ts : List ℚ} {p : ℕ} {v : ℚ} (h : lse ts p = some v) :
    (ts.map fun t : ℚ => Real.exp t).sum ≤ Real.exp v := by
  unfold lse at h
  cases hm : maxOf ts with
  | none => simp [hm] at h
  | some m =>
    simp only [hm, Option.map_some, Option.some.injEq] at h
    subst h
    have hZ := sum_expUp_pos (maxOf_ne_nil hm) m p
    calc _ ≤ _ := sum_map_exp_le ts m p
      _ ≤ Real.exp m * Real.exp (lnUpper ((ts.map fun t => expUp (t - m) p).sum) p) :=
          mul_le_mul_of_nonneg_left (le_exp_lnUpper hZ p) (Real.exp_pos _).le
      _ = _ := by rw [← Real.exp_add]; norm_cast

theorem lse_none {ts : List ℚ} {p : ℕ} (h : lse ts p = none) : ts = [] := by
  unfold lse at h
  cases hm : maxOf ts with
  | none => cases ts with
    | nil => rfl
    | cons => simp [maxOf] at hm
  | some m => simp [hm] at h

theorem exp_le_sum_map_exp {ts : List ℚ} {m : ℚ} (hm : m ∈ ts) :
    Real.exp m ≤ (ts.map fun t : ℚ => Real.exp t).sum := by
  induction ts with
  | nil => simp at hm
  | cons t ts ih =>
    simp only [List.map_cons, List.sum_cons]
    have hnn : 0 ≤ (ts.map fun t : ℚ => Real.exp t).sum :=
      List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨a, _, rfl⟩; exact (Real.exp_pos _).le)
    rcases List.mem_cons.1 hm with rfl | h
    · linarith
    · linarith [ih h, Real.exp_pos (t : ℝ)]

theorem sum_expLo_le (ts : List ℚ) (m : ℚ) (p : ℕ) :
    (((ts.map fun t => expLo (t - m) p).sum : ℚ) : ℝ) * Real.exp m ≤ (ts.map fun t : ℚ => Real.exp t).sum := by
  induction ts with
  | nil => simp
  | cons t ts ih =>
    simp only [List.map_cons, List.sum_cons]
    push_cast at ih ⊢
    have : (expLo (t - m) p : ℝ) * Real.exp m ≤ Real.exp t := by
      calc (expLo (t - m) p : ℝ) * Real.exp m ≤ Real.exp ((t - m : ℚ) : ℝ) * Real.exp m :=
            mul_le_mul_of_nonneg_right (expLo_le_exp _ p) (Real.exp_pos _).le
        _ = Real.exp t := by rw [← Real.exp_add]; push_cast; ring_nf
    rw [add_mul]
    exact add_le_add this ih

/-- **`lseLower` is below `ln Σ e^t`.** -/
theorem lseLower_sound {ts : List ℚ} {p : ℕ} {v : ℚ} (h : lseLower ts p = some v) :
    Real.exp v ≤ (ts.map fun t : ℚ => Real.exp t).sum := by
  unfold lseLower at h
  cases hm : maxOf ts with
  | none => simp [hm] at h
  | some m =>
    simp only [hm, Option.map_some, Option.some.injEq] at h
    subst h
    have hmem := exp_le_sum_map_exp (maxOf_mem hm)
    split_ifs with hz
    · have hz' : (0 : ℝ) < (((ts.map fun t => expLo (t - m) p).sum : ℚ) : ℝ) := by exact_mod_cast hz
      rcases max_choice m (m + lnLower (ts.map fun t => expLo (t - m) p).sum p) with h | h <;> rw [h]
      · exact hmem
      · calc Real.exp ((m + lnLower (ts.map fun t => expLo (t - m) p).sum p : ℚ) : ℝ)
            = Real.exp m * Real.exp (lnLower (ts.map fun t => expLo (t - m) p).sum p) := by
              push_cast; rw [Real.exp_add]
          _ ≤ Real.exp m * (((ts.map fun t => expLo (t - m) p).sum : ℚ) : ℝ) :=
              mul_le_mul_of_nonneg_left (exp_lnLower_le hz' p) (Real.exp_pos _).le
          _ ≤ _ := by rw [mul_comm]; exact sum_expLo_le ts m p
    · exact hmem

theorem log_one_add_le_softplusUp (x : ℚ) (p : ℕ) : Real.log (1 + Real.exp x) ≤ (softplusUp x p : ℝ) := by
  unfold softplusUp
  split_ifs with h
  · calc Real.log (1 + Real.exp x) ≤ (1 + Real.exp x) - 1 := Real.log_le_sub_one_of_pos (by positivity)
      _ = Real.exp x := by ring
      _ ≤ _ := exp_le_expUp x p
  · have hZ : (0 : ℝ) < ((expUp (-max 0 x) p + expUp (x - max 0 x) p : ℚ) : ℝ) := by
      push_cast; linarith [expUp_pos (-max 0 x) p, expUp_pos (x - max 0 x) p]
    have key : 1 + Real.exp x ≤
        Real.exp (max 0 x : ℚ) * ((expUp (-max 0 x) p + expUp (x - max 0 x) p : ℚ) : ℝ) := by
      calc 1 + Real.exp x = Real.exp (max 0 x : ℚ) * Real.exp ((-max 0 x : ℚ) : ℝ) +
            Real.exp (max 0 x : ℚ) * Real.exp ((x - max 0 x : ℚ) : ℝ) := by
            rw [← Real.exp_add, ← Real.exp_add]; push_cast; simp
        _ ≤ Real.exp (max 0 x : ℚ) * expUp (-max 0 x) p + Real.exp (max 0 x : ℚ) * expUp (x - max 0 x) p := by
            gcongr <;> exact exp_le_expUp _ _
        _ = _ := by push_cast; ring
    calc Real.log (1 + Real.exp x)
        ≤ Real.log (Real.exp (max 0 x : ℚ) * ((expUp (-max 0 x) p + expUp (x - max 0 x) p : ℚ) : ℝ)) :=
          Real.log_le_log (by positivity) key
      _ = (max 0 x : ℚ) + Real.log ((expUp (-max 0 x) p + expUp (x - max 0 x) p : ℚ) : ℝ) := by
          rw [Real.log_mul (Real.exp_pos _).ne' hZ.ne', Real.log_exp]
      _ ≤ (max 0 x : ℚ) + (lnUpper (expUp (-max 0 x) p + expUp (x - max 0 x) p) p : ℝ) := by
          linarith [log_le_lnUpper hZ p]
      _ = _ := by push_cast; ring

theorem log_one_add_le_of_le {L : ℝ} {l : ℚ} (hL : 0 ≤ L) (h : L ≤ Real.exp l) (p : ℕ) :
    Real.log (1 + L) ≤ (softplusUp l p : ℝ) :=
  (Real.log_le_log (by linarith) (by linarith)).trans (log_one_add_le_softplusUp l p)

theorem log_one_add_le_lnOnePlus {L : ℝ} {l : Option ℚ} (hL : 0 ≤ L) (h : LogUB l L) (p : ℕ) :
    Real.log (1 + L) ≤ (lnOnePlus l p : ℝ) := by
  cases l with
  | none => simp only [LogUB] at h; subst h; simp [lnOnePlus]
  | some l => exact log_one_add_le_of_le hL h p

theorem exp_sub_one_le_mul_exp (X : ℝ) : Real.exp X - 1 ≤ X * Real.exp X := by
  have h := Real.add_one_le_exp (-X)
  have hx : Real.exp (-X) * Real.exp X = 1 := by rw [← Real.exp_add]; simp
  nlinarith [Real.exp_pos X]

/-- **`logExpm1` is above `ln(e^X − 1)`** for `0 ≤ X ≤ P`, `X ≤ e^{lP}`. -/
theorem logExpm1_sound {X : ℝ} {P lP : ℚ} (p : ℕ) (hX : 0 ≤ X) (hXP : X ≤ P) (hXl : X ≤ Real.exp lP) :
    LogUB (logExpm1 P lP p) (Real.exp X - 1) := by
  have hsmall : Real.exp X - 1 ≤ Real.exp ((P : ℝ) + lP) := by
    calc Real.exp X - 1 ≤ X * Real.exp X := exp_sub_one_le_mul_exp X
      _ ≤ Real.exp lP * Real.exp P :=
          mul_le_mul hXl (Real.exp_le_exp.2 hXP) (Real.exp_pos _).le (Real.exp_pos _).le
      _ = _ := by rw [← Real.exp_add, add_comm]
  unfold logExpm1
  split_ifs with h0 h1
  · simp only [LogUB]
    have : X = 0 := le_antisymm (hXP.trans (by exact_mod_cast h0)) hX
    rw [this, Real.exp_zero]; ring
  · simp only [LogUB]; push_cast; exact hsmall
  · simp only [LogUB]
    have hP : (0 : ℝ) < P := by push Not at h0; exact_mod_cast h0
    have hE : (expLo (-P) p : ℝ) < 1 :=
      (expLo_le_exp _ p).trans_lt (Real.exp_lt_one_iff.2 (by push_cast; linarith))
    have hY : (0 : ℝ) < ((1 - expLo (-P) p : ℚ) : ℝ) := by push_cast; linarith
    have hbig : Real.exp X - 1 ≤ Real.exp ((P : ℝ) + (lnUpper (1 - expLo (-P) p) p : ℚ)) := by
      calc Real.exp X - 1 ≤ Real.exp P - 1 := by linarith [Real.exp_le_exp.2 hXP]
        _ = Real.exp P * (1 - Real.exp ((-P : ℚ) : ℝ)) := by
            rw [mul_sub, ← Real.exp_add]; push_cast; simp
        _ ≤ Real.exp P * ((1 - expLo (-P) p : ℚ) : ℝ) := by
            apply mul_le_mul_of_nonneg_left _ (Real.exp_pos _).le
            have := expLo_le_exp (-P) p
            push_cast at this ⊢; linarith
        _ ≤ Real.exp P * Real.exp (lnUpper (1 - expLo (-P) p) p) :=
            mul_le_mul_of_nonneg_left (le_exp_lnUpper hY p) (Real.exp_pos _).le
        _ = _ := by rw [← Real.exp_add]
    rcases min_choice lP (lnUpper (1 - expLo (-P) p) p) with h | h <;> rw [h] <;> push_cast
    · exact hsmall
    · exact hbig

theorem le_minExp {X : ℝ} {f lam : ℚ} (p : ℕ) (hf : X ≤ f) (hl : X ≤ Real.exp lam) : X ≤ (minExp f lam p : ℝ) := by
  unfold minExp
  split_ifs
  · push_cast; exact le_min hf (hl.trans (exp_le_expUp lam p))
  · exact hf

/-! ## Factors -/

/-- **`F` bounds the factor `1 + L`**: `L ≥ 0`, `ln(1 + L) ≤ F.f`, and `F.l` bounds `L`'s logarithm. -/
def FacBounds (F : Fac) (L : ℝ) : Prop := 0 ≤ L ∧ Real.log (1 + L) ≤ F.f ∧ LogUB F.l L

theorem one_bounds {x : ℝ} {lx : ℚ} (p : ℕ) (h0 : 0 ≤ x) (h : x ≤ Real.exp lx) : FacBounds (Fac.one lx p) x :=
  ⟨h0, log_one_add_le_of_le h0 h p, h⟩

theorem pow_bounds {F : Fac} {L : ℝ} {n : ℕ} {lnN : ℚ} (p : ℕ) (hF : FacBounds F L)
    (hn : 0 < n → Real.log n ≤ lnN) : FacBounds (F.pow n lnN p) ((1 + L) ^ n - 1) := by
  obtain ⟨hL, hf, hl⟩ := hF
  have h1 : (1 : ℝ) ≤ (1 + L) ^ n := one_le_pow₀ (by linarith)
  have hlog : Real.log (1 + ((1 + L) ^ n - 1)) = n * Real.log (1 + L) := by
    rw [add_sub_cancel, Real.log_pow]
  unfold Fac.pow
  cases hFl : F.l with
  | none =>
    rw [hFl] at hl; simp only [LogUB] at hl; subst hl
    have hf0 : (0 : ℝ) ≤ F.f := by simpa using hf
    exact ⟨by simp, by simp; exact mul_nonneg (Nat.cast_nonneg n) hf0, by simp [LogUB]⟩
  | some l =>
    rw [hFl] at hl
    simp only
    split_ifs with hn0
    · subst hn0; exact ⟨by simp, by simp, by simp [LogUB]⟩
    · have hnpos : 0 < n := Nat.pos_of_ne_zero hn0
      have hXf : (n : ℝ) * Real.log (1 + L) ≤ ((n * F.f : ℚ) : ℝ) := by
        push_cast; exact mul_le_mul_of_nonneg_left hf (Nat.cast_nonneg n)
      refine ⟨by linarith, by rw [hlog]; exact hXf, ?_⟩
      have hX0 : 0 ≤ (n : ℝ) * Real.log (1 + L) :=
        mul_nonneg (Nat.cast_nonneg n) (Real.log_nonneg (by linarith))
      have hXl : (n : ℝ) * Real.log (1 + L) ≤ Real.exp ((l + lnN : ℚ) : ℝ) := by
        calc (n : ℝ) * Real.log (1 + L) ≤ n * L :=
              mul_le_mul_of_nonneg_left
                (by linarith [Real.log_le_sub_one_of_pos (show (0 : ℝ) < 1 + L by linarith)])
                (Nat.cast_nonneg n)
          _ ≤ n * Real.exp l := mul_le_mul_of_nonneg_left hl (Nat.cast_nonneg n)
          _ = Real.exp (Real.log n + l) := by rw [Real.exp_add, Real.exp_log (by exact_mod_cast hnpos)]
          _ ≤ _ := Real.exp_le_exp.2 (by push_cast; linarith [hn hnpos])
      have := logExpm1_sound p hX0 (le_minExp p hXf hXl) hXl
      have heq : Real.exp ((n : ℝ) * Real.log (1 + L)) - 1 = (1 + L) ^ n - 1 := by
        rw [← Real.log_pow, Real.exp_log (by positivity)]
      rwa [heq] at this

theorem prod_aux {Fs : List Fac} {Ls : List ℝ} (h : List.Forall₂ FacBounds Fs Ls) :
    1 ≤ (Ls.map (1 + ·)).prod ∧ Real.log (Ls.map (1 + ·)).prod ≤ (((Fs.map Fac.f).sum : ℚ) : ℝ) ∧
      Real.log (Ls.map (1 + ·)).prod ≤ ((Fs.filterMap Fac.l).map fun t : ℚ => Real.exp t).sum := by
  induction h with
  | nil => simp
  | @cons F L Fs Ls hab _ ih =>
    obtain ⟨hL, hf, hl⟩ := hab
    obtain ⟨i1, i2, i3⟩ := ih
    have hlogL : Real.log (1 + L) ≤ L := by
      linarith [Real.log_le_sub_one_of_pos (show (0 : ℝ) < 1 + L by linarith)]
    simp only [List.map_cons, List.prod_cons, List.sum_cons]
    refine ⟨one_le_mul_of_one_le_of_one_le (by linarith) i1, ?_, ?_⟩
    · rw [Real.log_mul (by linarith) (by linarith)]; push_cast at i2 ⊢; linarith
    · rw [Real.log_mul (by linarith) (by linarith)]
      cases ha : F.l with
      | none =>
        rw [ha] at hl; simp only [LogUB] at hl; subst hl
        simp only [List.filterMap_cons, ha]; simp; linarith
      | some l =>
        rw [ha] at hl
        change L ≤ _ at hl
        simp only [List.filterMap_cons, ha, List.map_cons, List.sum_cons]
        linarith

theorem prod_bounds {Fs : List Fac} {Ls : List ℝ} (p : ℕ) (h : List.Forall₂ FacBounds Fs Ls) :
    FacBounds (Fac.prod Fs p) ((Ls.map (1 + ·)).prod - 1) := by
  obtain ⟨h1, hf, hl⟩ := prod_aux h
  have hPi : Real.exp (Real.log (Ls.map (1 + ·)).prod) - 1 = (Ls.map (1 + ·)).prod - 1 := by
    rw [Real.exp_log (by linarith)]
  have hX0 : 0 ≤ Real.log (Ls.map (1 + ·)).prod := Real.log_nonneg h1
  unfold Fac.prod
  cases hlse : lse (Fs.filterMap Fac.l) p with
  | none =>
    refine ⟨by linarith, by simp only [add_sub_cancel]; exact hf, ?_⟩
    rw [lse_none hlse] at hl
    simp only [List.map_nil, List.sum_nil] at hl
    have : Real.log (Ls.map (1 + ·)).prod = 0 := le_antisymm hl hX0
    simp only [LogUB]
    rw [← hPi, this, Real.exp_zero]; ring
  | some lam =>
    refine ⟨by linarith, by simp only [add_sub_cancel]; exact hf, ?_⟩
    have hs := lse_sound hlse
    have := logExpm1_sound p hX0 (le_minExp p hf (hl.trans hs)) (hl.trans hs)
    rwa [hPi] at this

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Nat

/-! ## Factorials -/

theorem log_factorial_ge {m : ℕ} (hm : m ≠ 0) :
    (m : ℝ) * Real.log m - m + (Real.log (2 * m) + Real.log Real.pi) / 2 ≤ Real.log (m ! : ℝ) := by
  have hs := Stirling.sqrt_pi_le_stirlingSeq hm
  unfold Stirling.stirlingSeq at hs
  have hmpos : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  have hD : 0 < √(2 * m : ℝ) * (m / Real.exp 1) ^ m := by positivity
  rw [le_div_iff₀ hD] at hs
  have := Real.log_le_log (by positivity) hs
  rw [Real.log_mul (by positivity) (by positivity), Real.log_mul (by positivity) (by positivity), Real.log_pow,
    Real.log_div hmpos.ne' (Real.exp_pos _).ne', Real.log_exp, Real.log_sqrt Real.pi_pos.le,
    Real.log_sqrt (by positivity)] at this
  linarith

theorem lnFactLower_le (m p : ℕ) : (lnFactLower m p : ℝ) ≤ Real.log (m ! : ℝ) := by
  unfold lnFactLower
  split_ifs with hm
  · subst hm; simp
  · have hmpos : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
    have h1 := lnLower_le_log (y := (m : ℚ)) (by exact_mod_cast hmpos) p
    have h2 := lnLower_le_log (y := (2 * m : ℚ)) (by push_cast; positivity) p
    have h3 := lnLower_le_log (y := (314 / 100 : ℚ)) (by norm_num) p
    have hpi : Real.log ((314 / 100 : ℚ) : ℝ) ≤ Real.log Real.pi :=
      Real.log_le_log (by norm_num) (by have := Real.pi_gt_d2; push_cast; linarith)
    push_cast at h1 h2 h3 hpi ⊢
    have := log_factorial_ge hm
    nlinarith

/-! ## `lnHead` -/

/-- `Σ_{1≤j≤m} z^j/j!`. -/
noncomputable def head (m : ℕ) (z : ℝ) : ℝ := ∑ j ∈ Finset.range m, z ^ (j + 1) / ((j + 1)! : ℝ)

theorem head_nonneg (m : ℕ) {z : ℝ} (hz : 0 ≤ z) : 0 ≤ head m z :=
  Finset.sum_nonneg fun _ _ => by positivity

theorem head_le_exp (m : ℕ) {z : ℝ} (hz : 0 ≤ z) : head m z ≤ Real.exp z - 1 := by
  have h := Real.sum_le_exp_of_nonneg hz (m + 1)
  rw [Finset.sum_range_succ'] at h
  simp only [pow_zero, factorial_zero, cast_one, div_one] at h
  unfold head; linarith

theorem head_le_pow (m : ℕ) {z : ℝ} (hz : 0 ≤ z) : head m z ≤ (1 + z) ^ m - 1 := by
  have h := (add_pow z 1 m)
  rw [Finset.sum_range_succ'] at h
  simp only [one_pow, mul_one, pow_zero, choose_zero_right, cast_one] at h
  rw [add_comm z 1] at h
  rw [h]
  unfold head
  have : ∀ j ∈ Finset.range m, z ^ (j + 1) / ((j + 1)! : ℝ) ≤ z ^ (j + 1) * (m.choose (j + 1) : ℝ) := by
    intro j hj
    have hc : (1 : ℝ) ≤ m.choose (j + 1) := by
      exact_mod_cast Nat.choose_pos (by simp at hj; omega)
    have hf : (1 : ℝ) ≤ (j + 1)! := by exact_mod_cast Nat.one_le_iff_ne_zero.2 (factorial_ne_zero _)
    calc z ^ (j + 1) / ((j + 1)! : ℝ) ≤ z ^ (j + 1) := div_le_self (by positivity) hf
      _ ≤ _ := le_mul_of_one_le_right (by positivity) hc
  linarith [Finset.sum_le_sum this]

/-- Each term of the head is at most `m/z` of the next, so at most `(m/z)^{m-j}` of the last. -/
theorem head_term_le {m : ℕ} {z : ℝ} (hz : 0 < z) :
    ∀ k, k < m → z ^ (m - k) / ((m - k)! : ℝ) ≤ z ^ m / (m ! : ℝ) * (m / z) ^ k := by
  intro k
  induction k with
  | zero => intro _; simp
  | succ k ih =>
    intro hk
    have ih := ih (by omega)
    have hmk : m - k = (m - (k + 1)) + 1 := by omega
    rw [hmk] at ih
    rw [factorial_succ, pow_succ] at ih
    have hpos : (0 : ℝ) < ((m - (k + 1))! : ℝ) := by exact_mod_cast factorial_pos _
    have hle : ((m - (k + 1) + 1 : ℕ) : ℝ) ≤ m := by exact_mod_cast (by omega : m - (k + 1) + 1 ≤ m)
    have key : z ^ (m - (k + 1)) / ((m - (k + 1))! : ℝ) =
        z ^ (m - (k + 1)) * z / (((m - (k + 1) + 1 : ℕ) : ℝ) * ((m - (k + 1))! : ℝ)) *
          (((m - (k + 1) + 1 : ℕ) : ℝ) / z) := by
      field_simp
    rw [key, pow_succ]
    push_cast at ih hle ⊢
    have hq : ((m - (k + 1) : ℕ) + 1 : ℝ) / z ≤ m / z := div_le_div_of_nonneg_right hle hz.le
    calc _ ≤ z ^ m / (m ! : ℝ) * (m / z) ^ k * (((m - (k + 1) : ℕ) + 1 : ℝ) / z) :=
          mul_le_mul_of_nonneg_right ih (by positivity)
      _ ≤ z ^ m / (m ! : ℝ) * (m / z) ^ k * (m / z) :=
          mul_le_mul_of_nonneg_left hq (by positivity)
      _ = _ := by ring

theorem head_le_last {m : ℕ} {z : ℝ} (hz : 0 < z) :
    head m z ≤ z ^ m / (m ! : ℝ) * ∑ k ∈ Finset.range m, (m / z) ^ k := by
  unfold head
  rw [Finset.mul_sum, ← Finset.sum_range_reflect]
  apply Finset.sum_le_sum
  intro k hk
  simp only [Finset.mem_range] at hk
  have := head_term_le hz k hk
  rwa [show m - 1 - k + 1 = m - k by omega]

theorem geom_le_min {r : ℝ} (m : ℕ) (h0 : 0 ≤ r) (h1 : r < 1) :
    ∑ k ∈ Finset.range m, r ^ k ≤ min (m : ℝ) (1 / (1 - r)) := by
  refine le_min ?_ ?_
  · calc ∑ k ∈ Finset.range m, r ^ k ≤ ∑ _k ∈ Finset.range m, (1 : ℝ) :=
          Finset.sum_le_sum fun k _ => pow_le_one₀ h0 h1.le
      _ = m := by simp
  · have : 0 ≤ r ^ m := pow_nonneg h0 m
    rw [geom_sum_eq h1.ne, show (r ^ m - 1) / (r - 1) = (1 - r ^ m) / (1 - r) by
      rw [← neg_sub, ← neg_sub 1 r, neg_div_neg_eq]]
    exact div_le_div_of_nonneg_right (by linarith) (by linarith)

/-- **`lnHead` is above `ln Σ_{1≤j≤m} z^j/j!`.** -/
theorem lnHead_sound {m : ℕ} {lm lf lz lzLo : ℚ} {z : ℝ} (p : ℕ) (hz : 0 < z) (hlz : Real.log z ≤ lz)
    (hlzLo : (lzLo : ℝ) ≤ Real.log z) (hlm : 0 < m → Real.log m ≤ lm) (hlf : (lf : ℝ) ≤ Real.log (m ! : ℝ)) :
    LogUB (lnHead m lm lf lz lzLo p) (head m z) := by
  have hH := head_nonneg m hz.le
  have hzexp : z ≤ Real.exp lz := by
    calc z = Real.exp (Real.log z) := (Real.exp_log hz).symm
      _ ≤ _ := Real.exp_le_exp.2 hlz
  unfold lnHead
  by_cases hm0 : m = 0
  · subst hm0; simp [LogUB, head]
  rw [if_neg hm0]
  have hm : 0 < m := Nat.pos_of_ne_zero hm0
  have hwhole : LogUB (if lz ≤ expCap then logExpm1 (expUp lz p) lz p else some (m * softplusUp lz p))
      (head m z) := by
    by_cases hcap : lz ≤ expCap
    · rw [if_pos hcap]
      exact LogUB.mono hH (head_le_exp m hz.le)
        (logExpm1_sound p hz.le (hzexp.trans (exp_le_expUp lz p)) hzexp)
    · rw [if_neg hcap]
      show head m z ≤ Real.exp ((m * softplusUp lz p : ℚ) : ℝ)
      have hl : Real.log (1 + z) ≤ (softplusUp lz p : ℝ) :=
        (Real.log_le_log (by linarith) (by linarith)).trans (log_one_add_le_softplusUp lz p)
      calc head m z ≤ (1 + z) ^ m - 1 := head_le_pow m hz.le
        _ ≤ (1 + z) ^ m := by linarith
        _ = Real.exp (m * Real.log (1 + z)) := by
            rw [← Real.log_pow, Real.exp_log (by positivity)]
        _ ≤ _ := Real.exp_le_exp.2 (by push_cast; exact mul_le_mul_of_nonneg_left hl (by positivity))
  have hlast : lm < lzLo → head m z ≤ Real.exp ((m * lz - lf + (if expUp (lm - lzLo) p < 1 then
      min lm (-lnLower (1 - expUp (lm - lzLo) p) p) else lm) : ℚ) : ℝ) := by
    intro hlt
    have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
    have hr0 : 0 ≤ (m : ℝ) / z := by positivity
    have hrlt : (m : ℝ) / z < 1 := by
      rw [div_lt_one hz]
      have : Real.log m < Real.log z := by
        have := hlm hm; have : (lm : ℝ) < lzLo := by exact_mod_cast hlt
        linarith
      exact (Real.log_lt_log_iff hmpos hz).1 this
    have hmz : (m : ℝ) / z ≤ (expUp (lm - lzLo) p : ℝ) := by
      calc (m : ℝ) / z = Real.exp (Real.log m - Real.log z) := by
            rw [Real.exp_sub, Real.exp_log hmpos, Real.exp_log hz]
        _ ≤ Real.exp ((lm - lzLo : ℚ) : ℝ) := Real.exp_le_exp.2 (by push_cast; linarith [hlm hm])
        _ ≤ _ := exp_le_expUp _ p
    have hT : z ^ m / (m ! : ℝ) ≤ Real.exp ((m * lz - lf : ℚ) : ℝ) := by
      have hfpos : (0 : ℝ) < (m ! : ℝ) := by exact_mod_cast factorial_pos m
      calc z ^ m / (m ! : ℝ) = Real.exp (m * Real.log z - Real.log (m ! : ℝ)) := by
            rw [Real.exp_sub, Real.exp_log hfpos, ← Real.log_pow, Real.exp_log (by positivity)]
        _ ≤ _ := Real.exp_le_exp.2 (by
            push_cast; nlinarith [hlz, hlf, (show (0 : ℝ) ≤ m by positivity)])
    have hG := geom_le_min m hr0 hrlt
    have hmexp : (m : ℝ) ≤ Real.exp lm := by
      calc (m : ℝ) = Real.exp (Real.log m) := (Real.exp_log hmpos).symm
        _ ≤ _ := Real.exp_le_exp.2 (hlm hm)
    have hmin : min (m : ℝ) (1 / (1 - m / z)) ≤ Real.exp ((if expUp (lm - lzLo) p < 1 then
        min lm (-lnLower (1 - expUp (lm - lzLo) p) p) else lm : ℚ) : ℝ) := by
      by_cases he : expUp (lm - lzLo) p < 1
      · rw [if_pos he]
        have he' : (expUp (lm - lzLo) p : ℝ) < 1 := by exact_mod_cast he
        have hpos : (0 : ℝ) < ((1 - expUp (lm - lzLo) p : ℚ) : ℝ) := by push_cast; linarith
        have hgeo : 1 / (1 - m / z) ≤ Real.exp ((-lnLower (1 - expUp (lm - lzLo) p) p : ℚ) : ℝ) := by
          push_cast
          rw [Real.exp_neg, ← one_div]
          have := exp_lnLower_le hpos p
          push_cast at this
          exact one_div_le_one_div_of_le (Real.exp_pos _) (this.trans (by linarith))
        rcases min_choice lm (-lnLower (1 - expUp (lm - lzLo) p) p) with h | h
        · rw [h]; exact (min_le_left _ _).trans hmexp
        · rw [h]; exact (min_le_right _ _).trans hgeo
      · rw [if_neg he]; exact (min_le_left _ _).trans hmexp
    calc head m z ≤ z ^ m / (m ! : ℝ) * ∑ k ∈ Finset.range m, (m / z) ^ k := head_le_last hz
      _ ≤ Real.exp ((m * lz - lf : ℚ) : ℝ) * Real.exp ((if expUp (lm - lzLo) p < 1 then
          min lm (-lnLower (1 - expUp (lm - lzLo) p) p) else lm : ℚ) : ℝ) :=
          mul_le_mul hT (hG.trans hmin) (Finset.sum_nonneg fun _ _ => by positivity) (Real.exp_pos _).le
      _ = _ := by rw [← Real.exp_add]; push_cast; ring_nf
  dsimp only
  generalize (if lz ≤ expCap then logExpm1 (expUp lz p) lz p else some (m * softplusUp lz p)) = W at hwhole ⊢
  by_cases hlt : lm < lzLo
  · rw [if_pos hlt]
    cases W with
    | none => exact hlast hlt
    | some w =>
      show head m z ≤ Real.exp ((min w _ : ℚ) : ℝ)
      rcases min_choice w (m * lz - lf + (if expUp (lm - lzLo) p < 1 then
          min lm (-lnLower (1 - expUp (lm - lzLo) p) p) else lm)) with h | h
      · rw [h]; exact hwhole
      · rw [h]; exact hlast hlt
  · rw [if_neg hlt]; exact hwhole

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper

/-! ## `h` and its chords -/

/-- `φ(y) = 1 − q(1 − e^{−y}) = e^{−h(y)}`. -/
noncomputable def phi (q y : ℝ) : ℝ := 1 - q + q * Real.exp (-y)

theorem phi_pos {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (y : ℝ) : 0 < phi q y := by
  unfold phi
  rcases h0.eq_or_lt with rfl | hq
  · simp
  · nlinarith [Real.exp_pos (-y)]

theorem phi_anti {q : ℝ} (h0 : 0 ≤ q) {y y' : ℝ} (h : y ≤ y') : phi q y' ≤ phi q y := by
  unfold phi
  have := Real.exp_le_exp.2 (neg_le_neg h)
  nlinarith

theorem phi_le_one {q : ℝ} (h0 : 0 ≤ q) {y : ℝ} (hy : 0 ≤ y) : phi q y ≤ 1 := by
  have := phi_anti h0 hy
  simpa [phi] using this

theorem two_term_holder {x₁ x₂ y₁ y₂ l : ℝ} (hx₁ : 0 ≤ x₁) (hx₂ : 0 ≤ x₂) (hy₁ : 0 ≤ y₁) (hy₂ : 0 ≤ y₂)
    (hA : 0 < x₁ + x₂) (hB : 0 < y₁ + y₂) (hl0 : 0 ≤ l) (hl1 : l ≤ 1) :
    x₁ ^ l * y₁ ^ (1 - l) + x₂ ^ l * y₂ ^ (1 - l) ≤ (x₁ + x₂) ^ l * (y₁ + y₂) ^ (1 - l) := by
  set A := x₁ + x₂
  set B := y₁ + y₂
  have hAB : 0 < A ^ l * B ^ (1 - l) := by positivity
  have key : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y →
      x ^ l * y ^ (1 - l) = A ^ l * B ^ (1 - l) * ((x / A) ^ l * (y / B) ^ (1 - l)) := by
    intro x y hx hy
    rw [Real.div_rpow hx hA.le, Real.div_rpow hy hB.le]
    field_simp
  have g₁ := Real.geom_mean_le_arith_mean2_weighted hl0 (by linarith : 0 ≤ 1 - l) (div_nonneg hx₁ hA.le)
    (div_nonneg hy₁ hB.le) (by ring)
  have g₂ := Real.geom_mean_le_arith_mean2_weighted hl0 (by linarith : 0 ≤ 1 - l) (div_nonneg hx₂ hA.le)
    (div_nonneg hy₂ hB.le) (by ring)
  rw [key x₁ y₁ hx₁ hy₁, key x₂ y₂ hx₂ hy₂, ← mul_add]
  have hsum : l * (x₁ / A) + (1 - l) * (y₁ / B) + (l * (x₂ / A) + (1 - l) * (y₂ / B)) = 1 := by
    field_simp; ring
  calc A ^ l * B ^ (1 - l) * ((x₁ / A) ^ l * (y₁ / B) ^ (1 - l) + (x₂ / A) ^ l * (y₂ / B) ^ (1 - l))
      ≤ A ^ l * B ^ (1 - l) * 1 := mul_le_mul_of_nonneg_left (by linarith) hAB.le
    _ = _ := mul_one _

/-- **`φ` is log-convex**, so `h = −ln φ` is concave. -/
theorem phi_logconvex {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (a b l : ℝ) (hl0 : 0 ≤ l) (hl1 : l ≤ 1) :
    phi q (l * a + (1 - l) * b) ≤ phi q a ^ l * phi q b ^ (1 - l) := by
  have hq1 : 0 ≤ 1 - q := by linarith
  have e₁ : (1 - q) ^ l * (1 - q) ^ (1 - l) = 1 - q := by
    rw [← Real.rpow_add' hq1 (by norm_num : l + (1 - l) ≠ 0)]; norm_num
  have e₂ : (q * Real.exp (-a)) ^ l * (q * Real.exp (-b)) ^ (1 - l) = q * Real.exp (-(l * a + (1 - l) * b)) := by
    rw [Real.mul_rpow h0 (Real.exp_pos _).le, Real.mul_rpow h0 (Real.exp_pos _).le, ← Real.exp_mul, ← Real.exp_mul]
    rw [show q ^ l * Real.exp (-a * l) * (q ^ (1 - l) * Real.exp (-b * (1 - l))) =
      (q ^ l * q ^ (1 - l)) * (Real.exp (-a * l) * Real.exp (-b * (1 - l))) by ring]
    rw [← Real.rpow_add' h0 (by norm_num : l + (1 - l) ≠ 0), ← Real.exp_add]
    norm_num; left; ring_nf
  have := two_term_holder hq1 (mul_nonneg h0 (Real.exp_pos (-a)).le) hq1 (mul_nonneg h0 (Real.exp_pos (-b)).le)
    (by have := phi_pos h0 h1 a; unfold phi at this; linarith)
    (by have := phi_pos h0 h1 b; unfold phi at this; linarith) hl0 hl1
  rw [e₁, e₂] at this
  unfold phi
  linarith

theorem hLower_le {q : ℚ} (h0 : (0 : ℝ) ≤ q) (h1 : (q : ℝ) ≤ 1) (y : ℚ) (p : ℕ) :
    (hLower q y p : ℝ) ≤ -Real.log (phi q y) := by
  unfold hLower
  have hphi := phi_pos h0 h1 (y : ℝ)
  have hX : phi q y ≤ ((1 - q + q * expUp (-y) p : ℚ) : ℝ) := by
    unfold phi; push_cast
    have := exp_le_expUp (-y) p; push_cast at this
    nlinarith
  have := log_le_lnUpper (hphi.trans_le hX) p
  have := Real.log_le_log hphi hX
  push_cast at *
  linarith

theorem phi_le_exp_neg_hLower {q : ℚ} (h0 : (0 : ℝ) ≤ q) (h1 : (q : ℝ) ≤ 1) (y : ℚ) (p : ℕ) :
    phi q y ≤ Real.exp (-(hLower q y p : ℝ)) := by
  calc phi q y = Real.exp (Real.log (phi q y)) := (Real.exp_log (phi_pos h0 h1 _)).symm
    _ ≤ _ := Real.exp_le_exp.2 (by linarith [hLower_le h0 h1 y p])

theorem rpow_le_exp_mul {x l θ : ℝ} (hx : 0 < x) (h : x ≤ Real.exp (-l)) (hθ : 0 ≤ θ) :
    x ^ θ ≤ Real.exp (-(θ * l)) := by
  calc x ^ θ ≤ Real.exp (-l) ^ θ := Real.rpow_le_rpow hx.le h hθ
    _ = _ := by rw [← Real.exp_mul]; ring_nf

/-- **A chord**: on `[y₀, y₁]`, `φ(y)^θ ≤ e^{−θ(a + s y)}` for the chord `(s, a)` through `hLower` at the ends. -/
theorem chord_le {q : ℚ} (h0 : (0 : ℝ) ≤ q) (h1 : (q : ℝ) ≤ 1) {θ : ℝ} (hθ : 0 ≤ θ) (p : ℕ) {y₀ y₁ : ℚ}
    (hlt : y₀ < y₁) {y : ℝ} (hy₀ : (y₀ : ℝ) ≤ y) (hy₁ : y ≤ y₁) :
    phi q y ^ θ ≤ Real.exp (-(θ * (((hLower q y₀ p - ((hLower q y₁ p - hLower q y₀ p) / (y₁ - y₀)) * y₀ : ℚ) : ℝ) +
      (((hLower q y₁ p - hLower q y₀ p) / (y₁ - y₀) : ℚ) : ℝ) * y))) := by
  have hlt' : (y₀ : ℝ) < y₁ := by exact_mod_cast hlt
  have hne : (y₁ : ℝ) - y₀ ≠ 0 := by linarith
  set l := ((y₁ : ℝ) - y) / (y₁ - y₀)
  have hl0 : 0 ≤ l := div_nonneg (by linarith) (by linarith)
  have hl1 : l ≤ 1 := (div_le_one (by linarith)).2 (by linarith)
  have hyl : y = l * y₀ + (1 - l) * y₁ := by
    simp only [l]; field_simp; ring
  have hc := phi_logconvex h0 h1 (y₀ : ℝ) (y₁ : ℝ) l hl0 hl1
  rw [← hyl] at hc
  have ha := phi_le_exp_neg_hLower h0 h1 y₀ p
  have hb := phi_le_exp_neg_hLower h0 h1 y₁ p
  have hφ := phi_pos h0 h1 y
  apply rpow_le_exp_mul hφ _ hθ
  calc phi q y ≤ phi q y₀ ^ l * phi q y₁ ^ (1 - l) := hc
    _ ≤ Real.exp (-(hLower q y₀ p : ℝ)) ^ l * Real.exp (-(hLower q y₁ p : ℝ)) ^ (1 - l) :=
        mul_le_mul (Real.rpow_le_rpow (phi_pos h0 h1 _).le ha hl0)
          (Real.rpow_le_rpow (phi_pos h0 h1 _).le hb (by linarith)) (Real.rpow_nonneg (phi_pos h0 h1 _).le _)
          (Real.rpow_nonneg (Real.exp_pos _).le _)
    _ = _ := by
        rw [← Real.exp_mul, ← Real.exp_mul, ← Real.exp_add]
        congr 1
        push_cast
        simp only [l]
        field_simp
        ring

/-- **The chords and the tail cover `y`**: for breakpoints increasing from `y₀ ≤ y`, `φ(y)^θ` is at most the sum of the
chords' weights and the last breakpoint's. -/
theorem cover {q : ℚ} (h0 : (0 : ℝ) ≤ q) (h1 : (q : ℝ) ≤ 1) {θ : ℝ} (hθ : 0 ≤ θ) (p : ℕ) :
    ∀ (ys : List ℚ) (y₀ : ℚ), increasing (y₀ :: ys) = true → ∀ {y : ℝ}, (y₀ : ℝ) ≤ y →
      phi q y ^ θ ≤ ((chordsOf q p (y₀ :: ys)).map fun c => Real.exp (-(θ * ((c.2 : ℝ) + c.1 * y)))).sum +
        Real.exp (-(θ * (hLower q ((y₀ :: ys).getLast (by simp)) p : ℝ))) := by
  intro ys
  induction ys with
  | nil =>
    intro y₀ _ y hy
    simp only [chordsOf, List.map_nil, List.sum_nil, List.getLast_singleton, zero_add]
    have hφ := phi_pos h0 h1 y
    apply rpow_le_exp_mul hφ _ hθ
    exact (phi_anti h0 hy).trans (phi_le_exp_neg_hLower h0 h1 y₀ p)
  | cons y₁ ys ih =>
    intro y₀ hinc y hy
    simp only [increasing, Bool.and_eq_true, decide_eq_true_eq] at hinc
    obtain ⟨hlt, hinc⟩ := hinc
    simp only [chordsOf, List.map_cons, List.sum_cons]
    rw [List.getLast_cons (by simp)]
    have hnn : 0 ≤ ((chordsOf q p (y₁ :: ys)).map fun c => Real.exp (-(θ * ((c.2 : ℝ) + c.1 * y)))).sum :=
      List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨_, _, rfl⟩; exact (Real.exp_pos _).le)
    by_cases hy₁ : y ≤ y₁
    · have := chord_le h0 h1 hθ p hlt hy hy₁
      have := Real.exp_pos (-(θ * (hLower q ((y₁ :: ys).getLast (by simp)) p : ℝ)))
      linarith
    · have := ih y₁ hinc (le_of_not_ge hy₁)
      linarith [Real.exp_pos (-(θ * ((((hLower q y₀ p - ((hLower q y₁ p - hLower q y₀ p) / (y₁ - y₀)) * y₀ : ℚ) :
        ℝ)) + (((hLower q y₁ p - hLower q y₀ p) / (y₁ - y₀) : ℚ) : ℝ) * y)))]

/-! ## The bound from its sums -/

theorem sum_list_comm {α ι : Type*} (cs : List α) (X : Finset ι) (g : α → ι → ℝ) :
    ∑ i ∈ X, (cs.map fun c => g c i).sum = (cs.map fun c => ∑ i ∈ X, g c i).sum := by
  induction cs with
  | nil => simp
  | cons c cs ih => simp only [List.map_cons, List.sum_cons, Finset.sum_add_distrib, ih]

/-- A list of `LogUB`-bounded quantities sums to at most the `e^t` of the bounds present. -/
theorem sum_le_filterMap {α : Type*} (l : List α) (g : α → Option ℚ) (R : α → ℝ) (hR : ∀ a ∈ l, LogUB (g a) (R a)) :
    (l.map R).sum ≤ ((l.filterMap g).map fun t : ℚ => Real.exp t).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.filterMap_cons]
    have ih := ih fun b hb => hR b (List.mem_cons_of_mem _ hb)
    have ha := hR a (List.mem_cons_self ..)
    cases hg : g a with
    | none => rw [hg] at ha; simp only [LogUB] at ha; rw [ha]; simpa using ih
    | some t => rw [hg] at ha; simp only [List.map_cons, List.sum_cons]; exact add_le_add ha ih

theorem Params.valid_parts {P : Params} (h : P.valid = true) :
    (0 : ℝ) ≤ P.q ∧ (P.q : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ P.θ ∧ ∃ ys, P.ys = 0 :: ys ∧ increasing (0 :: ys) = true := by
  simp only [Params.valid, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨hq0, hq1⟩, hθ⟩, hhead⟩, hinc⟩ := h
  refine ⟨by exact_mod_cast hq0, by exact_mod_cast hq1, by exact_mod_cast hθ, ?_⟩
  cases hys : P.ys with
  | nil => rw [hys] at hhead; simp at hhead
  | cons y ys =>
    rw [hys] at hhead hinc
    simp only [List.head?_cons, Option.some.injEq] at hhead
    subst hhead
    exact ⟨ys, rfl, hinc⟩

/-- **`bound` is sound**: when, at every slope `s`, `term s` bounds `Σ_i W_i e^{−θ s y_i}`, and `whole` bounds `Wh`, and
each `F_i ≤ φ(y_i)^θ`, then `1 + Σ_i W_i F_i + Wh ≤ e^v`. -/
theorem bound_sound {P : Params} (hP : P.valid = true) {term : ℚ → Option ℚ} {whole : Option ℚ} {v : ℚ}
    (hv : bound P term whole = some v) {ι : Type*} (X : Finset ι) (W y F : ι → ℝ)
    (hW : ∀ i ∈ X, 0 ≤ W i) (hy : ∀ i ∈ X, 0 ≤ y i) (hF : ∀ i ∈ X, F i ≤ phi P.q (y i) ^ (P.θ : ℝ))
    (hterm : ∀ s : ℚ, LogUB (term s) (∑ i ∈ X, W i * Real.exp (-((P.θ : ℝ) * s * y i))))
    {Wh : ℝ} (hWh : LogUB whole Wh) :
    1 + ∑ i ∈ X, W i * F i + Wh ≤ Real.exp v := by
  obtain ⟨hq0, hq1, hθ, ys, hys, hinc⟩ := Params.valid_parts hP
  unfold bound at hv
  have hl := lse_sound hv
  set θ : ℝ := (P.θ : ℝ)
  set cs := chordsOf P.q P.pθ P.ys
  set last := P.ys.getLast?
  set yJ := (0 :: ys).getLast (List.cons_ne_nil _ _)
  have hcov : ∀ i ∈ X, F i ≤ (cs.map fun c => Real.exp (-(θ * ((c.2 : ℝ) + c.1 * y i)))).sum +
      Real.exp (-(θ * (hLower P.q yJ P.pθ : ℝ))) := by
    intro i hi
    have := cover hq0 hq1 hθ P.pθ ys 0 hinc (y := y i) (by simpa using hy i hi)
    simp only [cs]; rw [hys]
    exact (hF i hi).trans this
  have hlast : last = some yJ := by simp only [last, hys]; exact List.getLast?_eq_getLast _
  -- the chord sums
  have hchord : (cs.map fun c => Real.exp (-(θ * c.2)) * ∑ i ∈ X, W i * Real.exp (-(θ * c.1 * y i))).sum ≤
      (((cs.filterMap fun c => (term c.1).map (-(P.θ * c.2) + ·))).map fun t : ℚ => Real.exp t).sum := by
    apply sum_le_filterMap
    intro c _
    have ht := hterm c.1
    cases htc : term c.1 with
    | none =>
      rw [htc] at ht; simp only [LogUB] at ht
      show Real.exp (-(θ * c.2)) * ∑ i ∈ X, W i * Real.exp (-(θ * c.1 * y i)) = 0
      rw [ht, mul_zero]
    | some t =>
      rw [htc] at ht
      show Real.exp (-(θ * c.2)) * ∑ i ∈ X, W i * Real.exp (-(θ * c.1 * y i)) ≤
        Real.exp ((-(P.θ * c.2) + t : ℚ) : ℝ)
      push_cast; rw [Real.exp_add]
      exact mul_le_mul_of_nonneg_left ht (Real.exp_pos _).le
  have htail : Real.exp (-(θ * (hLower P.q yJ P.pθ : ℝ))) * ∑ i ∈ X, W i ≤
      (([last.bind fun y => (term 0).map (-(P.θ * hLower P.q y P.pθ) + ·)].filterMap id).map
        fun t : ℚ => Real.exp t).sum := by
    rw [hlast]
    have ht := hterm 0
    simp only [Rat.cast_zero, mul_zero, zero_mul, neg_zero, Real.exp_zero, mul_one] at ht
    cases htc : term 0 with
    | none =>
      rw [htc] at ht; simp only [LogUB] at ht
      rw [ht, mul_zero]; simp
    | some t =>
      rw [htc] at ht
      simp only [Option.bind_some, Option.map_some, List.filterMap_cons, id, List.filterMap_nil,
        List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
      push_cast; rw [Real.exp_add]
      exact mul_le_mul_of_nonneg_left ht (Real.exp_pos _).le
  have hwh : Wh ≤ (([whole].filterMap id).map fun t : ℚ => Real.exp t).sum := by
    cases hw : whole with
    | none => rw [hw] at hWh; simp only [LogUB] at hWh; simp [hWh]
    | some t =>
      rw [hw] at hWh
      simp only [List.filterMap_cons, id, List.filterMap_nil, List.map_cons, List.map_nil, List.sum_cons,
        List.sum_nil, add_zero]
      exact hWh
  set T := Real.exp (-(θ * (hLower P.q yJ P.pθ : ℝ)))
  have hsum : ∑ i ∈ X, W i * F i ≤
      (cs.map fun c => Real.exp (-(θ * c.2)) * ∑ i ∈ X, W i * Real.exp (-(θ * c.1 * y i))).sum + T * ∑ i ∈ X, W i := by
    calc ∑ i ∈ X, W i * F i
        ≤ ∑ i ∈ X, ((cs.map fun c => W i * Real.exp (-(θ * ((c.2 : ℝ) + c.1 * y i)))).sum + T * W i) := by
          refine Finset.sum_le_sum fun i hi => ?_
          rw [List.sum_map_mul_left]
          nlinarith [hcov i hi, hW i hi]
      _ = (cs.map fun c => ∑ i ∈ X, W i * Real.exp (-(θ * ((c.2 : ℝ) + c.1 * y i)))).sum + T * ∑ i ∈ X, W i := by
          rw [Finset.sum_add_distrib, sum_list_comm, Finset.mul_sum]
      _ = _ := by
          congr 1
          congr 1
          apply List.map_congr_left
          intro c _
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [show -(θ * ((c.2 : ℝ) + c.1 * y i)) = -(θ * c.2) + -(θ * c.1 * y i) by ring, Real.exp_add]
          ring
  have hWs : 0 ≤ ∑ i ∈ X, W i := Finset.sum_nonneg hW
  simp only [List.map_cons, List.sum_cons, List.map_append, List.sum_append, Rat.cast_zero, Real.exp_zero] at hl
  have hsplit : ([last.bind fun y => (term 0).map (-(P.θ * hLower P.q y P.pθ) + ·), whole].filterMap id) =
      ([last.bind fun y => (term 0).map (-(P.θ * hLower P.q y P.pθ) + ·)].filterMap id) ++ ([whole].filterMap id) := by
    simp only [List.filterMap_cons, List.filterMap_nil]
    cases last.bind fun y => (term 0).map (-(P.θ * hLower P.q y P.pθ) + ·) <;> cases whole <;> rfl
  rw [hsplit, List.map_append, List.sum_append] at hl
  have : T * ∑ i ∈ X, W i = Real.exp (-(θ * (hLower P.q yJ P.pθ : ℝ))) *
      ∑ i ∈ X, W i := rfl
  linarith

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Nat

/-! ## Capped sums over a group -/

/-- `Σ_{j≤m} C(n, j) x^j`: a group of `n` proof blocks' sets of at most `m`, each block at `x`. -/
noncomputable def capP (m n : ℕ) (x : ℝ) : ℝ := ∑ j ∈ Finset.range (m + 1), (n.choose j : ℝ) * x ^ j

theorem capP_zero_left (n : ℕ) (x : ℝ) : capP 0 n x = 1 := by simp [capP]

theorem capP_zero_right (m : ℕ) (x : ℝ) : capP m 0 x = 1 := by
  unfold capP
  rw [Finset.sum_range_succ']
  simp only [choose_zero_right, cast_one, pow_zero, mul_one]
  rw [Finset.sum_eq_zero (fun j _ => by simp)]; ring

theorem capP_pascal (m n : ℕ) (x : ℝ) : capP (m + 1) (n + 1) x = capP (m + 1) n x + x * capP m n x := by
  unfold capP
  rw [Finset.sum_range_succ' _ (m + 1), Finset.sum_range_succ' _ (m + 1)]
  simp only [choose_succ_succ, cast_add, add_mul, Finset.sum_add_distrib, choose_zero_right, cast_one, pow_zero,
    mul_one, Finset.mul_sum]
  have : ∀ j ∈ Finset.range (m + 1), (n.choose j : ℝ) * x ^ (j + 1) = x * ((n.choose j : ℝ) * x ^ j) := by
    intro j _; ring
  rw [Finset.sum_congr rfl this]
  ring

theorem capP_nonneg (m n : ℕ) {x : ℝ} (hx : 0 ≤ x) : 0 ≤ capP m n x :=
  Finset.sum_nonneg fun _ _ => by positivity

theorem one_le_capP (m n : ℕ) {x : ℝ} (hx : 0 ≤ x) : 1 ≤ capP m n x := by
  unfold capP
  rw [Finset.sum_range_succ']
  simp only [choose_zero_right, cast_one, pow_zero, mul_one]
  have := Finset.sum_nonneg (s := Finset.range m) (f := fun j => (n.choose (j + 1) : ℝ) * x ^ (j + 1))
    (fun _ _ => by positivity)
  linarith

theorem capP_mono {m n n' : ℕ} (h : n ≤ n') {x : ℝ} (hx : 0 ≤ x) : capP m n x ≤ capP m n' x :=
  Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (by exact_mod_cast choose_le_choose j h) (by positivity)

theorem capP_of_le {m n : ℕ} (h : n ≤ m) (x : ℝ) : capP m n x = (1 + x) ^ n := by
  unfold capP
  rw [show (1 + x) = x + 1 by ring, add_pow]
  simp only [one_pow, mul_one]
  have hsub : Finset.range (n + 1) ⊆ Finset.range (m + 1) := Finset.range_subset_range.2 (by omega)
  rw [← Finset.sum_subset hsub]
  · exact Finset.sum_congr rfl fun j _ => by ring
  · intro j _ hj
    simp only [Finset.mem_range, not_lt] at hj
    simp [choose_eq_zero_of_lt (by omega : n < j)]

theorem capP_le_pow (m n : ℕ) {x : ℝ} (hx : 0 ≤ x) : capP m n x ≤ (1 + x) ^ n := by
  rcases le_total n m with h | h
  · rw [capP_of_le h]
  · rw [← capP_of_le (le_refl n)]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 (by omega))
      (fun _ _ _ => by positivity)

theorem capP_sub_one_le_head (m n : ℕ) {x : ℝ} (hx : 0 ≤ x) : capP m n x - 1 ≤ head m (n * x) := by
  unfold capP head
  rw [Finset.sum_range_succ']
  simp only [choose_zero_right, cast_one, pow_zero, mul_one, add_sub_cancel_right]
  refine Finset.sum_le_sum fun j _ => ?_
  have := Nat.choose_le_pow_div (α := ℝ) (j + 1) n
  calc (n.choose (j + 1) : ℝ) * x ^ (j + 1) ≤ (n : ℝ) ^ (j + 1) / ((j + 1)! : ℝ) * x ^ (j + 1) :=
        mul_le_mul_of_nonneg_right this (by positivity)
    _ = _ := by rw [mul_pow]; ring

/-! ## Sums over sets, grouped by key -/

/-- The proof blocks of `A` of key `k`. -/
def cntOf {G : Type*} [DecidableEq G] (κ : ℕ → G) (A : Finset ℕ) (k : G) : ℕ := (A.filter (κ · = k)).card

theorem cntOf_insert_self {G : Type*} [DecidableEq G] {κ : ℕ → G} {A : Finset ℕ} {v : ℕ} (hv : v ∉ A) :
    cntOf κ (insert v A) (κ v) = cntOf κ A (κ v) + 1 := by
  unfold cntOf; rw [Finset.filter_insert, if_pos rfl, Finset.card_insert_of_notMem (by simp [hv])]

theorem cntOf_insert_ne {G : Type*} [DecidableEq G] {κ : ℕ → G} {A : Finset ℕ} {v : ℕ} {k : G} (hk : κ v ≠ k) :
    cntOf κ (insert v A) k = cntOf κ A k := by
  unfold cntOf; rw [Finset.filter_insert, if_neg hk]

/-- **The capped sum factors over the keys**: the sets of `D` with at most `m k` proof blocks of each key `k`, each
block at `x(κ v)`, sum to `∏_k Σ_{j≤m k} C(n_k, j) x_k^j`, `n_k` the blocks of key `k`. -/
theorem capped_sum_eq {G : Type*} [DecidableEq G] (K : Finset G) (κ : ℕ → G) (x : G → ℝ) (D : Finset ℕ)
    (hD : ∀ v ∈ D, κ v ∈ K) (m : G → ℕ) :
    ∑ A ∈ D.powerset with ∀ k ∈ K, cntOf κ A k ≤ m k, ∏ v ∈ A, x (κ v) =
      ∏ k ∈ K, capP (m k) (cntOf κ D k) (x k) := by
  induction D using Finset.induction_on generalizing m with
  | empty =>
    rw [Finset.sum_filter]
    simp [cntOf, capP_zero_right]
  | @insert v D hv ih =>
    have hD' : ∀ w ∈ D, κ w ∈ K := fun w hw => hD w (Finset.mem_insert_of_mem hw)
    have hk0 : κ v ∈ K := hD v (Finset.mem_insert_self _ _)
    rw [Finset.sum_filter, Finset.sum_powerset_insert hv, ← Finset.sum_filter]
    have hsplit : ∀ A ∈ D.powerset, (if ∀ k ∈ K, cntOf κ (insert v A) k ≤ m k then ∏ w ∈ insert v A, x (κ w) else 0) =
        x (κ v) * (if 1 ≤ m (κ v) ∧ ∀ k ∈ K, cntOf κ A k ≤ Function.update m (κ v) (m (κ v) - 1) k then
          ∏ w ∈ A, x (κ w) else 0) := by
      intro A hA
      have hvA : v ∉ A := fun h => hv (Finset.mem_powerset.1 hA h)
      rw [Finset.prod_insert hvA]
      have hiff : (∀ k ∈ K, cntOf κ (insert v A) k ≤ m k) ↔
          (1 ≤ m (κ v) ∧ ∀ k ∈ K, cntOf κ A k ≤ Function.update m (κ v) (m (κ v) - 1) k) := by
        constructor
        · intro h
          have h0 := h (κ v) hk0
          rw [cntOf_insert_self hvA] at h0
          refine ⟨by omega, fun k hk => ?_⟩
          by_cases hkv : k = κ v
          · subst hkv; rw [Function.update_self]; omega
          · rw [Function.update_of_ne hkv]
            have := h k hk; rwa [cntOf_insert_ne (Ne.symm hkv)] at this
        · rintro ⟨h1, h⟩ k hk
          by_cases hkv : k = κ v
          · subst hkv; have := h _ hk; rw [Function.update_self] at this
            rw [cntOf_insert_self hvA]; omega
          · have := h k hk; rw [Function.update_of_ne hkv] at this
            rwa [cntOf_insert_ne (Ne.symm hkv)]
      by_cases hc : ∀ k ∈ K, cntOf κ (insert v A) k ≤ m k
      · rw [if_pos hc, if_pos (hiff.1 hc)]
      · rw [if_neg hc, if_neg (fun h => hc (hiff.2 h)), mul_zero]
    rw [Finset.sum_congr rfl hsplit, ← Finset.mul_sum, ih hD' m]
    have hcnt : ∀ k ∈ K, k ≠ κ v → cntOf κ (insert v D) k = cntOf κ D k := fun k _ hk =>
      cntOf_insert_ne (Ne.symm hk)
    set E := ∏ k ∈ K.erase (κ v), capP (m k) (cntOf κ D k) (x k)
    have hR : ∏ k ∈ K, capP (m k) (cntOf κ (insert v D) k) (x k) =
        capP (m (κ v)) (cntOf κ D (κ v) + 1) (x (κ v)) * E := by
      rw [← Finset.mul_prod_erase K _ hk0, cntOf_insert_self hv]
      congr 1
      exact Finset.prod_congr rfl fun k hk => by
        rw [hcnt k (Finset.mem_of_mem_erase hk) (Finset.ne_of_mem_erase hk)]
    have hL : ∏ k ∈ K, capP (m k) (cntOf κ D k) (x k) = capP (m (κ v)) (cntOf κ D (κ v)) (x (κ v)) * E :=
      (Finset.mul_prod_erase K _ hk0).symm
    rw [hR, hL]
    by_cases hm : 1 ≤ m (κ v)
    · simp only [hm, true_and]
      rw [← Finset.sum_filter, ih hD' (Function.update m (κ v) (m (κ v) - 1)), ← Finset.mul_prod_erase K _ hk0,
        Function.update_self]
      have hE : ∏ k ∈ K.erase (κ v), capP (Function.update m (κ v) (m (κ v) - 1) k) (cntOf κ D k) (x k) = E :=
        Finset.prod_congr rfl fun k hk => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hk)]
      rw [hE]
      obtain ⟨m', hm'⟩ : ∃ m', m (κ v) = m' + 1 := ⟨m (κ v) - 1, by omega⟩
      rw [hm', capP_pascal, Nat.add_sub_cancel]
      ring
    · have hm0 : m (κ v) = 0 := by omega
      rw [Finset.sum_eq_zero fun A _ => if_neg fun h => hm h.1, hm0, capP_zero_left, capP_zero_left]
      ring

/-! ## The sets of at most `M` proof blocks -/

/-- `Σ_{j≤M} W^j/j!`. -/
noncomputable def expT (M : ℕ) (W : ℝ) : ℝ := ∑ j ∈ Finset.range (M + 1), W ^ j / (j ! : ℝ)

theorem expT_eq (M : ℕ) (W : ℝ) : expT M W = 1 + head M W := by
  unfold expT head; rw [Finset.sum_range_succ']; simp; ring

theorem expT_nonneg (M : ℕ) {W : ℝ} (hW : 0 ≤ W) : 0 ≤ expT M W := Finset.sum_nonneg fun _ _ => by positivity

theorem pow_add_ge (j : ℕ) {W x : ℝ} (hW : 0 ≤ W) (hx : 0 ≤ x) :
    W ^ (j + 1) + (j + 1) * x * W ^ j ≤ (W + x) ^ (j + 1) := by
  induction j with
  | zero => simp
  | succ j ih =>
    have h1 : 0 ≤ (W + x) := by linarith
    have hexp : (W + x) * (W ^ (j + 1) + (j + 1) * x * W ^ j) =
        W ^ (j + 1 + 1) + ((j + 1 : ℕ) + 1 : ℝ) * x * W ^ (j + 1) + (j + 1) * x * x * W ^ j := by
      push_cast; ring
    have hnn : 0 ≤ ((j : ℝ) + 1) * x * x * W ^ j := by positivity
    calc W ^ (j + 1 + 1) + ((j + 1 : ℕ) + 1 : ℝ) * x * W ^ (j + 1)
        ≤ (W + x) * (W ^ (j + 1) + (j + 1) * x * W ^ j) := by rw [hexp]; linarith
      _ ≤ (W + x) * (W + x) ^ (j + 1) := mul_le_mul_of_nonneg_left ih h1
      _ = _ := by ring

theorem expT_step (M : ℕ) {W x : ℝ} (hW : 0 ≤ W) (hx : 0 ≤ x) :
    expT (M + 1) W + x * expT M W ≤ expT (M + 1) (W + x) := by
  unfold expT
  rw [Finset.sum_range_succ' _ (M + 1), Finset.sum_range_succ' _ (M + 1), Finset.mul_sum]
  simp only [pow_zero, factorial_zero, cast_one, div_one]
  have : ∀ j ∈ Finset.range (M + 1), W ^ (j + 1) / ((j + 1)! : ℝ) + x * (W ^ j / (j ! : ℝ)) ≤
      (W + x) ^ (j + 1) / ((j + 1)! : ℝ) := by
    intro j _
    have hf : (0 : ℝ) < (j ! : ℝ) := by exact_mod_cast factorial_pos j
    rw [factorial_succ, cast_mul]
    have := pow_add_ge j hW hx
    have hj : (0 : ℝ) < ((j + 1 : ℕ) : ℝ) := by positivity
    have heq : W ^ (j + 1) / (((j + 1 : ℕ) : ℝ) * (j ! : ℝ)) + x * (W ^ j / (j ! : ℝ)) =
        (W ^ (j + 1) + (j + 1) * x * W ^ j) / (((j + 1 : ℕ) : ℝ) * (j ! : ℝ)) := by
      field_simp; push_cast; ring
    rw [heq]
    exact div_le_div_of_nonneg_right this (by positivity)
  linarith [Finset.sum_le_sum this, Finset.sum_add_distrib (s := Finset.range (M + 1))
    (f := fun j => W ^ (j + 1) / ((j + 1)! : ℝ)) (g := fun j => x * (W ^ j / (j ! : ℝ)))]

/-- **The sets of at most `M` proof blocks**: `Σ_{|A|≤M} ∏_{v∈A} w_v ≤ Σ_{j≤M} W^j/j!`, `W = Σ_v w_v`. -/
theorem small_sum_le (w : ℕ → ℝ) (hw : ∀ v, 0 ≤ w v) (D : Finset ℕ) :
    ∀ M, ∑ A ∈ D.powerset with A.card ≤ M, ∏ v ∈ A, w v ≤ expT M (∑ v ∈ D, w v) := by
  induction D using Finset.induction_on with
  | empty =>
    intro M; rw [Finset.sum_filter]; simp [expT, Finset.sum_range_succ']
  | @insert v D hv ih =>
    intro M
    have hW : 0 ≤ ∑ u ∈ D, w u := Finset.sum_nonneg fun u _ => hw u
    rw [Finset.sum_filter, Finset.sum_powerset_insert hv, Finset.sum_insert hv, ← Finset.sum_filter]
    have hsplit : ∀ A ∈ D.powerset, (if (insert v A).card ≤ M then ∏ u ∈ insert v A, w u else 0) =
        w v * (if A.card + 1 ≤ M then ∏ u ∈ A, w u else 0) := by
      intro A hA
      have hvA : v ∉ A := fun h => hv (Finset.mem_powerset.1 hA h)
      rw [Finset.card_insert_of_notMem hvA, Finset.prod_insert hvA]
      split_ifs <;> simp
    rw [Finset.sum_congr rfl hsplit, ← Finset.mul_sum]
    cases M with
    | zero =>
      rw [Finset.sum_eq_zero fun A _ => if_neg (by omega), mul_zero, add_zero]
      simpa [expT] using ih 0
    | succ M =>
      have h1 := ih (M + 1)
      have h2 := ih M
      simp only [add_le_add_iff_right] at *
      rw [← Finset.sum_filter]
      calc _ ≤ expT (M + 1) (∑ u ∈ D, w u) + w v * expT M (∑ u ∈ D, w u) :=
            add_le_add h1 (mul_le_mul_of_nonneg_left h2 (hw v))
        _ ≤ expT (M + 1) (∑ u ∈ D, w u + w v) := expT_step M hW (hw v)
        _ = _ := by rw [add_comm (∑ u ∈ D, w u) (w v)]

theorem expT_mono (M : ℕ) {W W' : ℝ} (h0 : 0 ≤ W) (h : W ≤ W') : expT M W ≤ expT M W' :=
  Finset.sum_le_sum fun j _ => div_le_div_of_nonneg_right (pow_le_pow_left₀ h0 h j) (by positivity)

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Nat

/-! ## Option-valued logarithms -/

theorem LogUB.omin {a b : Option ℚ} {X : ℝ} (ha : LogUB a X) (hb : LogUB b X) : LogUB (omin a b) X := by
  cases a with
  | none => exact ha
  | some a =>
    cases b with
    | none => exact hb
    | some b =>
      show X ≤ Real.exp ((min a b : ℚ) : ℝ)
      rcases le_total a b with h | h
      · rw [min_eq_left h]; exact ha
      · rw [min_eq_right h]; exact hb

theorem LogUB.scale {o : Option ℚ} {Y : ℝ} (h : LogUB o Y) (hY : 0 ≤ Y) {k : ℝ} {c : ℚ}
    (hkc : k ≤ Real.exp c) : LogUB (o.map (c + ·)) (k * Y) := by
  cases o with
  | none => show k * Y = 0; rw [show Y = 0 from h, mul_zero]
  | some v =>
    show k * Y ≤ Real.exp ((c + v : ℚ) : ℝ)
    push_cast; rw [Real.exp_add]
    exact mul_le_mul hkc h hY (Real.exp_pos _).le

theorem LogUB.nonneg_eq {o : Option ℚ} {X : ℝ} (h : LogUB o X) (ho : o = none) : X = 0 := by
  subst ho; exact h

theorem lse_filterMap_sound {α : Type*} (l : List α) (g : α → Option ℚ) (R : α → ℝ)
    (hR : ∀ a ∈ l, LogUB (g a) (R a)) (h0 : ∀ a ∈ l, 0 ≤ R a) (p : ℕ) :
    LogUB (lse (l.filterMap g) p) (l.map R).sum := by
  have hs := sum_le_filterMap l g R hR
  have hn : 0 ≤ (l.map R).sum :=
    List.sum_nonneg fun x hx => by obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hx; exact h0 a ha
  cases hv : lse (l.filterMap g) p with
  | none =>
    have := lse_none hv
    rw [this] at hs
    exact le_antisymm (by simpa using hs) hn
  | some v => exact hs.trans (lse_sound hv)

theorem lse_two {a b : Option ℚ} {A B : ℝ} (ha : LogUB a A) (hb : LogUB b B) (hA : 0 ≤ A) (hB : 0 ≤ B) (p : ℕ) :
    LogUB (lse ([a, b].filterMap id) p) (A + B) := by
  have := lse_filterMap_sound [(a, A), (b, B)] Prod.fst Prod.snd
    (by simp only [List.mem_cons, List.not_mem_nil, or_false]; rintro x (rfl | rfl) <;> assumption)
    (by simp only [List.mem_cons, List.not_mem_nil, or_false]; rintro x (rfl | rfl) <;> assumption) p
  have he : [(a, A), (b, B)].filterMap Prod.fst = [a, b].filterMap id := by
    simp only [List.filterMap_cons, List.filterMap_nil, id]
  rw [he] at this
  simpa using this

theorem le_exp_of_bounds {F : Fac} {L : ℝ} (h : FacBounds F L) : 1 + L ≤ Real.exp F.f := by
  have h1 : 0 < 1 + L := by linarith [h.1]
  rw [← Real.exp_log h1]; exact Real.exp_le_exp.2 h.2.1

/-! ## `ln 2` and the groups' logarithms -/

/-- `L` brackets `ln 2`. -/
def ln2Ok (L : Ln2) : Prop := (L.lo : ℝ) ≤ Real.log 2 ∧ Real.log 2 ≤ L.up

theorem ln2Ok_of (p : ℕ) : ln2Ok (Ln2.of p) := by
  refine ⟨?_, ?_⟩
  · show (lnLower 2 p : ℝ) ≤ _
    have := lnLower_le_log (y := 2) (by norm_num) p; simpa using this
  · show _ ≤ (lnUpper 2 p : ℝ)
    have := log_le_lnUpper (y := 2) (by norm_num) p; simpa using this

theorem two_pow_le_exp {L : Ln2} (hL : ln2Ok L) (b : ℕ) : (2 : ℝ) ^ b ≤ Real.exp (b * L.up) := by
  rw [← Real.exp_log (by positivity : (0 : ℝ) < 2 ^ b), Real.log_pow]
  exact Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left hL.2 (Nat.cast_nonneg b))

/-- A group holds a proof block or more, and its logarithms bracket `ln n`, and at `m = ⌊S/b⌋` bound `ln m` above and
`ln m!` below. -/
def groupOk (g : Group) (S : ℕ) : Prop :=
  0 < g.n ∧ Real.log g.n ≤ g.lnN ∧ (g.lnNLo : ℝ) ≤ Real.log g.n ∧
    (0 < S / g.b → Real.log ((S / g.b : ℕ) : ℝ) ≤ g.lm) ∧ (g.lf : ℝ) ≤ Real.log ((S / g.b)! : ℝ)

theorem n_le_exp {g : Group} {S : ℕ} (hg : groupOk g S) : (g.n : ℝ) ≤ Real.exp g.lnN := by
  rw [← Real.exp_log (by exact_mod_cast hg.1 : (0 : ℝ) < g.n)]; exact Real.exp_le_exp.2 hg.2.1

theorem exp_le_n {g : Group} {S : ℕ} (hg : groupOk g S) : Real.exp g.lnNLo ≤ (g.n : ℝ) := by
  rw [← Real.exp_log (by exact_mod_cast hg.1 : (0 : ℝ) < g.n)]; exact Real.exp_le_exp.2 hg.2.2.1

/-! ## A group's sums -/

theorem wholeFac_bounds {L : Ln2} (hL : ln2Ok L) {g : Group} {S : ℕ} (hg : groupOk g S) (lw : ℚ) (p : ℕ) :
    FacBounds (wholeFac L g lw p) ((1 + 2 ^ g.b * Real.exp lw) ^ g.n - 1) := by
  refine pow_bounds p (one_bounds p (by positivity) ?_) fun _ => hg.2.1
  push_cast; rw [Real.exp_add]
  exact mul_le_mul_of_nonneg_right (two_pow_le_exp hL g.b) (Real.exp_pos _).le

/-- A group's sum over its sets of at most `⌊S/b⌋` proof blocks (all of them, at `b = 0`), each block at `x`. -/
noncomputable def capVal (g : Group) (S : ℕ) (x : ℝ) : ℝ := if g.b = 0 then (1 + x) ^ g.n else capP (S / g.b) g.n x

theorem capVal_le_pow (g : Group) (S : ℕ) {x : ℝ} (hx : 0 ≤ x) : capVal g S x ≤ (1 + x) ^ g.n := by
  unfold capVal; split_ifs
  · exact le_rfl
  · exact capP_le_pow _ _ hx

theorem one_le_capVal (g : Group) (S : ℕ) {x : ℝ} (hx : 0 ≤ x) : 1 ≤ capVal g S x := by
  unfold capVal; split_ifs
  · exact one_le_pow₀ (by linarith)
  · exact one_le_capP _ _ hx

theorem cappedFac_bounds {L : Ln2} (hL : ln2Ok L) {g : Group} {S : ℕ} (hg : groupOk g S) (lw : ℚ) (p : ℕ) :
    FacBounds (cappedFac L g lw S p) (capVal g S (2 ^ g.b * Real.exp lw) - 1) := by
  have hW := wholeFac_bounds hL hg lw p
  set x : ℝ := 2 ^ g.b * Real.exp lw with hxdef
  have hx : 0 < x := by positivity
  unfold cappedFac
  by_cases hc : g.b = 0 ∨ g.n ≤ S / g.b
  · rw [if_pos hc]
    have : capVal g S x = (1 + x) ^ g.n := by
      unfold capVal
      rcases hc with h | h
      · rw [if_pos h]
      · split_ifs
        · rfl
        · exact capP_of_le h x
    rw [this]; exact hW
  · rw [if_neg hc]
    simp only [not_or, not_le] at hc
    obtain ⟨hb, hn⟩ := hc
    have hcap : capVal g S x = capP (S / g.b) g.n x := if_neg hb
    rw [hcap]
    have h0 : 0 ≤ capP (S / g.b) g.n x - 1 := by linarith [one_le_capP (S / g.b) g.n hx.le]
    have hn0 : (0 : ℝ) < g.n := by exact_mod_cast hg.1
    have hz : 0 < (g.n : ℝ) * x := mul_pos hn0 hx
    have hlogz : Real.log ((g.n : ℝ) * x) = Real.log g.n + g.b * Real.log 2 + lw := by
      rw [Real.log_mul hn0.ne' hx.ne', hxdef, Real.log_mul (by positivity) (Real.exp_pos _).ne', Real.log_pow,
        Real.log_exp]; ring
    have hhead := lnHead_sound (m := S / g.b) (lm := g.lm) (lf := g.lf) (lz := g.lnN + g.b * L.up + lw)
      (lzLo := g.lnNLo + g.b * L.lo + lw) p hz
      (by rw [hlogz]; push_cast; linarith [hg.2.1, mul_le_mul_of_nonneg_left hL.2 (Nat.cast_nonneg (α := ℝ) g.b)])
      (by rw [hlogz]; push_cast; linarith [hg.2.2.1, mul_le_mul_of_nonneg_left hL.1 (Nat.cast_nonneg (α := ℝ) g.b)])
      hg.2.2.2.1 hg.2.2.2.2
    have hlog : LogUB (omin (wholeFac L g lw p).l (lnHead (S / g.b) g.lm g.lf (g.lnN + g.b * L.up + lw)
        (g.lnNLo + g.b * L.lo + lw) p)) (capP (S / g.b) g.n x - 1) :=
      LogUB.omin (hW.2.2.mono h0 (by linarith [capP_le_pow (S / g.b) g.n hx.le]))
        (hhead.mono h0 (capP_sub_one_le_head _ _ hx.le))
    refine ⟨h0, ?_, hlog⟩
    have hc1 : (1 : ℝ) + (capP (S / g.b) g.n x - 1) = capP (S / g.b) g.n x := by ring
    show Real.log (1 + (capP (S / g.b) g.n x - 1)) ≤ ((min (wholeFac L g lw p).f (lnOnePlus _ p) : ℚ) : ℝ)
    push_cast
    refine le_min ?_ ?_
    · have := hW.2.1
      rw [hc1]
      rw [show (1 : ℝ) + ((1 + x) ^ g.n - 1) = (1 + x) ^ g.n by ring] at this
      exact (Real.log_le_log (by linarith [one_le_capP (S / g.b) g.n hx.le]) (capP_le_pow _ _ hx.le)).trans this
    · exact log_one_add_le_lnOnePlus h0 hlog p

theorem forall₂_map {α : Type*} (l : List α) (F : α → Fac) (V : α → ℝ) (h : ∀ a ∈ l, FacBounds (F a) (V a)) :
    List.Forall₂ FacBounds (l.map F) (l.map V) := by
  induction l with
  | nil => exact List.Forall₂.nil
  | cons a l ih =>
    exact List.Forall₂.cons (h a List.mem_cons_self) (ih fun b hb => h b (List.mem_cons_of_mem _ hb))

theorem prod_map_bounds {α : Type*} (l : List α) (F : α → Fac) (V : α → ℝ) (h : ∀ a ∈ l, FacBounds (F a) (V a))
    (p : ℕ) : FacBounds (Fac.prod (l.map F) p) ((l.map fun a => 1 + V a).prod - 1) := by
  have := prod_bounds p (forall₂_map l F V h)
  simpa [List.map_map, Function.comp_def] using this

/-- A product of factors `1 + V_a` with `V_a = P_a − 1`. -/
theorem prod_map_bounds' {α : Type*} (l : List α) (F : α → Fac) (P : α → ℝ)
    (h : ∀ a ∈ l, FacBounds (F a) (P a - 1)) (p : ℕ) : FacBounds (Fac.prod (l.map F) p) ((l.map P).prod - 1) := by
  have := prod_map_bounds l F (fun a => P a - 1) h p
  simpa using this

/-! ## The sums over sets -/

theorem lightSets_sound {L : Ln2} (hL : ln2Ok L) (gs : List (Group × ℚ)) {S : ℕ} (hg : ∀ x ∈ gs, groupOk x.1 S)
    (p : ℕ) :
    LogUB (lightSets L gs p) ((gs.map fun x => (1 + 2 ^ x.1.b * Real.exp x.2) ^ x.1.n).prod - 1) :=
  (prod_map_bounds' gs _ _ (fun x hx => wholeFac_bounds hL (hg x hx) x.2 p) p).2.2

/-- The least of the groups' widths, `S + 1` for none. -/
def bminOf (gs : List (Group × ℚ)) (S : ℕ) : ℕ := (gs.map fun x => x.1.b).foldl min (S + 1)

theorem lightPatterns_sound {L : Ln2} (hL : ln2Ok L) (gs : List (Group × ℚ)) {S : ℕ} (hg : ∀ x ∈ gs, groupOk x.1 S)
    (p : ℕ) {X : ℝ} (hX0 : 0 ≤ X)
    (hprod : X ≤ (gs.map fun x => capVal x.1 S (2 ^ x.1.b * Real.exp x.2)).prod - 1)
    (hhead : 0 < bminOf gs S →
      X ≤ 2 ^ S * head (S / bminOf gs S) (gs.map fun x => (x.1.n : ℝ) * Real.exp x.2).sum) :
    LogUB (lightPatterns L gs S p) X := by
  have hP : LogUB (Fac.prod (gs.map fun x => cappedFac L x.1 x.2 S p) p).l X :=
    (prod_map_bounds' gs _ _ (fun x hx => cappedFac_bounds hL (hg x hx) x.2 p) p).2.2.mono hX0 hprod
  unfold lightPatterns
  simp only
  rw [show (gs.map fun x => x.1.b).foldl min (S + 1) = bminOf gs S from rfl]
  by_cases hb : bminOf gs S = 0
  · rw [if_pos hb]; exact hP
  rw [if_neg hb]
  set W : ℝ := (gs.map fun x => (x.1.n : ℝ) * Real.exp x.2).sum
  cases hlW : lse (gs.map fun x => x.1.lnN + x.2) p with
  | none => exact hP
  | some lW =>
    cases hlWLo : lseLower (gs.map fun x => x.1.lnNLo + x.2) p with
    | none => exact hP
    | some lWLo =>
      simp only
      refine LogUB.omin hP ?_
      have hne : gs ≠ [] := by rintro rfl; simp [lse, maxOf] at hlW
      have hWpos : 0 < W := by
        obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil gs hne
        have hterm : 0 < (x.1.n : ℝ) * Real.exp x.2 := mul_pos (by exact_mod_cast (hg x hx).1) (Real.exp_pos _)
        have := List.single_le_sum (l := gs.map fun x => (x.1.n : ℝ) * Real.exp x.2)
          (fun y hy => by obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hy; exact mul_nonneg (by positivity) (by positivity))
          _ (List.mem_map_of_mem hx)
        linarith
      have hWup : W ≤ Real.exp lW := by
        refine le_trans ?_ (lse_sound hlW)
        rw [List.map_map]
        refine List.sum_le_sum fun x hx => ?_
        show (x.1.n : ℝ) * Real.exp x.2 ≤ Real.exp ((x.1.lnN + x.2 : ℚ) : ℝ)
        push_cast; rw [Real.exp_add]
        exact mul_le_mul_of_nonneg_right (n_le_exp (hg x hx)) (Real.exp_pos _).le
      have hWlo : Real.exp lWLo ≤ W := by
        refine (lseLower_sound hlWLo).trans ?_
        rw [List.map_map]
        refine List.sum_le_sum fun x hx => ?_
        show Real.exp ((x.1.lnNLo + x.2 : ℚ) : ℝ) ≤ (x.1.n : ℝ) * Real.exp x.2
        push_cast; rw [Real.exp_add]
        exact mul_le_mul_of_nonneg_right (exp_le_n (hg x hx)) (Real.exp_pos _).le
      have hh := lnHead_sound (m := S / bminOf gs S) (lm := lnUpper ((S / bminOf gs S : ℕ) : ℚ) p)
        (lf := lnFactLower (S / bminOf gs S) p) (lz := lW) (lzLo := lWLo) p hWpos
        (by rw [← Real.log_exp lW]; exact Real.log_le_log hWpos hWup)
        (by rw [← Real.log_exp lWLo]; exact Real.log_le_log (Real.exp_pos _) hWlo)
        (fun hm => by
          have := log_le_lnUpper (y := ((S / bminOf gs S : ℕ) : ℚ)) (by exact_mod_cast hm) p
          simpa using this)
        (lnFactLower_le _ p)
      refine LogUB.mono hX0 (hhead (Nat.pos_of_ne_zero hb)) (LogUB.scale hh (head_nonneg _ hWpos.le)
        ?_)
      have := two_pow_le_exp hL S
      push_cast; exact this

/-! ## A request's factor, and the patterns meeting a cell sampled whole -/

theorem requestFac_bounds {L : Ln2} {fin : List Group} {yS : Option ℚ} {θs : ℚ} {S p : ℕ} {Light Heavy : ℝ}
    (hLight : LogUB (lightPatterns L (fin.map fun g => (g, -(θs * g.r))) S p) Light) (hL0 : 0 ≤ Light)
    (hHeavy : LogUB (yS.map fun y => (S : ℚ) * L.up - θs * y) Heavy) (hH0 : 0 ≤ Heavy) :
    FacBounds (requestFac L fin yS θs S p) (Light + Heavy) := by
  have hl := lse_two hLight hHeavy hL0 hH0 p
  exact ⟨add_nonneg hL0 hH0, log_one_add_le_lnOnePlus (add_nonneg hL0 hH0) hl p, hl⟩

theorem touchCount_sound {L : Ln2} {fin inf : List Group} {N S p : ℕ} {Pf Pi H X : ℝ}
    (hf : FacBounds (Fac.prod (fin.map (cappedFac L · 0 S p)) p) (Pf - 1))
    (hi : FacBounds (Fac.prod (inf.map (cappedFac L · 0 S p)) p) (Pi - 1))
    (hH : LogUB (if ((fin ++ inf).map fun g => g.n * g.b).sum < S then none else some ((S : ℚ) * L.up)) H)
    (hH0 : 0 ≤ H) (hX0 : 0 ≤ X) (hX1 : X ≤ N * (Pf * (Pi - 1) + H) * (Pf * Pi + H) ^ (N - 1))
    (hX2 : X ≤ (Pf * Pi + H) ^ N) :
    LogUB (touchCount L fin inf N S p) X := by
  unfold touchCount
  by_cases hN : N = 0
  · rw [if_pos hN]; subst hN; simp only [CharP.cast_eq_zero, zero_mul] at hX1; exact le_antisymm hX1 hX0
  rw [if_neg hN]
  dsimp only
  set finF := Fac.prod (fin.map (cappedFac L · 0 S p)) p
  set infF := Fac.prod (inf.map (cappedFac L · 0 S p)) p
  set heavy : Option ℚ := if ((fin ++ inf).map fun g => g.n * g.b).sum < S then none else some ((S : ℚ) * L.up)
  have hPf1 : 1 ≤ Pf := by linarith [hf.1]
  have hPi1 : 1 ≤ Pi := by linarith [hi.1]
  have hPf : Pf ≤ Real.exp finF.f := by have := le_exp_of_bounds hf; linarith
  have hPi : Pi ≤ Real.exp infF.f := by have := le_exp_of_bounds hi; linarith
  set A := Pf * Pi + H
  set T := Pf * (Pi - 1) + H
  have hA0 : 0 ≤ A := by positivity
  have hT0 : 0 ≤ T := by have : 0 ≤ Pi - 1 := by linarith
                         positivity
  -- `A ≤ e^{lA}`
  have hAl : LogUB (lse ([some (finF.f + infF.f), heavy].filterMap id) p) A := by
    refine lse_two (a := some (finF.f + infF.f)) ?_ hH (by positivity) hH0 p
    show Pf * Pi ≤ Real.exp ((finF.f + infF.f : ℚ) : ℝ)
    push_cast; rw [Real.exp_add]; exact mul_le_mul hPf hPi (by linarith) (Real.exp_pos _).le
  have hlA : A ≤ Real.exp ((lse ([some (finF.f + infF.f), heavy].filterMap id) p).getD 0) := by
    cases hl : lse ([some (finF.f + infF.f), heavy].filterMap id) p with
    | none => exact absurd (lse_none hl) (by simp)
    | some v => rw [hl] at hAl; exact hAl
  set lA := (lse ([some (finF.f + infF.f), heavy].filterMap id) p).getD 0
  have hTl : LogUB (lse ([infF.l.map (finF.f + ·), heavy].filterMap id) p) T := by
    refine lse_two ?_ hH (by have : 0 ≤ Pi - 1 := by linarith
                             positivity) hH0 p
    have := hi.2.2
    have h := LogUB.scale this (by linarith) (c := finF.f) hPf
    exact h
  cases hlT : lse ([infF.l.map (finF.f + ·), heavy].filterMap id) p with
  | none =>
    rw [hlT] at hTl
    have hT : T = 0 := hTl
    have : X ≤ 0 := by rw [hT] at hX1; simpa using hX1
    show X = 0
    exact le_antisymm this hX0
  | some lT =>
    rw [hlT] at hTl
    show X ≤ Real.exp ((min (lnUpper N p + lT + (N - 1 : ℕ) * lA) (N * lA) : ℚ) : ℝ)
    have hNpos : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero hN
    have hNl : (N : ℝ) ≤ Real.exp (lnUpper N p) := by
      have := le_exp_lnUpper (y := (N : ℚ)) (by exact_mod_cast hNpos) p; simpa using this
    have hpow : ∀ k : ℕ, A ^ k ≤ Real.exp (k * lA) := fun k => by
      rw [← Real.exp_log (by positivity : (0 : ℝ) < Real.exp lA)] at hlA
      calc A ^ k ≤ Real.exp lA ^ k := pow_le_pow_left₀ hA0 (by simpa using hlA) k
        _ = Real.exp (k * lA) := by rw [← Real.exp_nat_mul]
    push_cast
    rcases min_choice (lnUpper N p + lT + ((N - 1 : ℕ) : ℝ) * lA : ℝ) ((N : ℝ) * lA) with h | h
    · rw [h]
      refine hX1.trans ?_
      rw [Real.exp_add, Real.exp_add]
      exact mul_le_mul (mul_le_mul hNl hTl hT0 (Real.exp_pos _).le) (hpow (N - 1)) (by positivity)
        (by positivity)
    · rw [h]; exact hX2.trans (hpow N)

end

end Properties.Spot.SecurityProofs.FactorUpperProof
