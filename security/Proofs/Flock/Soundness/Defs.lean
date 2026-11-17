module

public import Definitions.Flock.Defs
public import Definitions.Flock.Session
public import Proofs.Flock.Soundness.Accounting.Numbers
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Positivity.Finset

/-! The soundness statement's shared definitions: `Arith.Correct`, `Committed` and `RepDoomed`. -/

@[expose] public section

namespace FlockSoundness

open Game Model Accounting
variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

theorem fast100_levels_fit : ∀ m s, fast100 m = some s → ∀ l ∈ s.levels, l.Fits := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  have key : ∀ m', 22 ≤ m' → m' ≤ 35 → ∀ s', fast100 m' = some s' →
      (s'.levels.all fun l => decide l.Fits) = true := by
    intro m' h1' h2' s' hs'
    interval_cases m' <;> (simp only [fast100, Option.some.injEq] at hs'; subst hs'; decide +kernel)
  intro l hl
  have := List.all_eq_true.1 (key m h1 h2 s h) l hl
  simpa using this

theorem Model.Arith.Correct.lagS_interp {A : Arith F K} (hA : A.Correct) :
    ∃ node : Fin 64 → F, Function.Injective node ∧
      ∀ p : Polynomial F, p.natDegree < 64 → ∀ ζ, ∑ s, A.lagS ζ s * p.eval (node s) = p.eval ζ := by
  obtain ⟨nS, _, hS, _, _, hlag, _, _⟩ := hA.nodes
  exact ⟨nS, hS, hlag⟩

omit [Fintype F] [DecidableEq F] in
theorem extraClaims_length (S : Statement) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → F) :
    (extraClaims S ptLocal mPts pts).length = 2 * S.regions.length := by
  unfold extraClaims
  induction S.regions with
  | nil => simp
  | cons R rs ih => simp [List.flatMap_cons, ih]; ring

theorem repDoomed_iff (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) :
    RepDoomed A S sch C extra ↔ RepDoomedW A S sch (closeMsgs A sch C) extra := Iff.rfl

theorem dZc_eq (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) (o : Option (ZcOut F S.m)) :
    DZc A S sch C extra o = DZcW A S sch (closeMsgs A sch C) extra o := by
  cases o <;> rfl

theorem dLc_eq (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcOut F S.m) (o : Option (Claim F S.m)) :
    DLc A S sch C extra zc o = DLcW A S sch (closeMsgs A sch C) extra zc o := by
  cases o <;> rfl

theorem dOp_eq (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (o : Option (OpenOut F)) : DOp A S sch C o = DOpW S sch (closeMsgs A sch C) o := by
  cases o <;> rfl

end FlockSoundness
