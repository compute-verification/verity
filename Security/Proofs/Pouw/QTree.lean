import Definitions.Pouw.Basic.QTree
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Logic.Function.Basic

/-!
# Oracle programs: sequencing, querying a list, and a transcript built unit by unit
-/

namespace Pouw.QTree

open Finset

variable {Q R α β : Type}

/-- Run `t`, then the program `f` of its result. -/
def bind : QTree Q R α → (α → QTree Q R β) → QTree Q R β
  | done a, f => f a
  | query q k, f => query q fun r => bind (k r) f

theorem eval_bind (H : Q → R) (t : QTree Q R α) (f : α → QTree Q R β) :
    (t.bind f).eval H = (f (t.eval H)).eval H := by
  induction t with
  | done a => rfl
  | query q k ih => exact ih (H q)

theorem calls_bind (H : Q → R) (t : QTree Q R α) (f : α → QTree Q R β) :
    (t.bind f).calls H = t.calls H + (f (t.eval H)).calls H := by
  induction t with
  | done a => simp [bind, calls, eval]
  | query q k ih =>
    simp only [bind, calls, eval]
    rw [ih (H q)]
    omega

/-- Query every point of a list, in order. -/
def queryAll : List Q → QTree Q R (List R)
  | [] => done []
  | q :: qs => query q fun r => (queryAll qs).bind fun rs => done (r :: rs)

theorem eval_queryAll (H : Q → R) (qs : List Q) : (queryAll qs).eval H = qs.map H := by
  induction qs with
  | nil => rfl
  | cons q qs ih => simp only [queryAll, eval, eval_bind, ih, List.map_cons]

theorem calls_queryAll (H : Q → R) (qs : List Q) : (queryAll (R := R) qs).calls H = qs.length := by
  induction qs with
  | nil => rfl
  | cons q qs ih => simp only [queryAll, calls, calls_bind, ih, List.length_cons]

/-- Units `0, …, n − 1` committed in order, unit `u` by the program `f u`; every other unit blank. -/
def upTo {C : Type} (f : ℕ → QTree Q R C) : ℕ → QTree Q R (ℕ → Option C)
  | 0 => done fun _ => none
  | n + 1 => (upTo f n).bind fun τ => (f n).bind fun c => done (Function.update τ n (some c))

theorem eval_upTo {C : Type} (H : Q → R) (f : ℕ → QTree Q R C) (n : ℕ) :
    (upTo f n).eval H = fun u => if u < n then some ((f u).eval H) else none := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [upTo, eval_bind, ih, eval]
    funext u
    by_cases hu : u = n
    · subst hu; simp
    · rw [Function.update_of_ne hu]
      by_cases h : u < n
      · simp [h, show u < n + 1 by omega]
      · simp [h, show ¬ u < n + 1 by omega]

theorem calls_upTo {C : Type} (H : Q → R) (f : ℕ → QTree Q R C) (n : ℕ) :
    (upTo f n).calls H = ∑ u ∈ range n, (f u).calls H := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [upTo, calls_bind, ih, calls, Finset.sum_range_succ]
    omega

theorem calls_upTo_le {C : Type} (H : Q → R) (f : ℕ → QTree Q R C) (n c : ℕ) (h : ∀ u < n, (f u).calls H ≤ c) :
    (upTo f n).calls H ≤ n * c := by
  rw [calls_upTo]
  have := Finset.sum_le_card_nsmul (range n) (fun u => (f u).calls H) c fun u hu => h u (Finset.mem_range.1 hu)
  simpa using this

/-- A program that makes no queries against `H` returns the same value against every oracle. -/
theorem eval_of_calls_eq_zero {H : Q → R} {t : QTree Q R α} (h : t.calls H = 0) (H' : Q → R) :
    t.eval H' = t.eval H := by
  cases t with
  | done a => rfl
  | query q k => simp [calls] at h

end Pouw.QTree
