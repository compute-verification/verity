import FlockSoundness.Protocol.LigeritoMid

/-!
# Ligerito is sound (PROTOCOL.md §13)

From a claim `Σ_y f(y)·b(y) = T` false for every message near the committed level-0 table,
Ligerito accepts with probability at most `εLig`:
* level 0: the OOD claim's batching coin `β₀` (`L₀/|F|`), then the level-0 folds (`phaseB0`);
* every next level: `phaseA` from `PostFold` to `PreFold`, then its folds (`phaseB`);
* the last level: its queries against the clear residual and the final check (`finalStage`).
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## The schedules' shape -/

/-- A level the proof applies to. -/
def levelGood (l : Level) : Bool :=
  decide l.Ok && decide l.Fits && decide (1 ≤ l.lanes) && decide (l.queries ≤ 2 ^ 64)

/-- The levels chain: every level good, each level's columns plus the split coordinate are the
next level's variables, and the last level's columns are the clear residual's. -/
def levelsChained : List Level → ℕ → Bool
  | [], _ => false
  | [l], ext => levelGood l && decide (l.logCols = ext)
  | l :: l' :: rest, ext =>
      levelGood l && decide (l.logCols + 1 = l'.lanes + l'.logCols) && levelsChained (l' :: rest) ext

/-- The shape of every `fast100` schedule. -/
theorem fast100_shape : ∀ m s, fast100 m = some s →
    2 ≤ s.levels.length ∧ (s.levels.headD default).lanes = s.k0 ∧
      s.m - 7 = s.k0 + (s.levels.headD default).logCols ∧ levelsChained s.levels s.ext = true := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  have key : ∀ m', 22 ≤ m' → m' ≤ 35 → ∀ s', fast100 m' = some s' →
      (decide (2 ≤ s'.levels.length) && decide ((s'.levels.headD default).lanes = s'.k0) &&
        decide (s'.m - 7 = s'.k0 + (s'.levels.headD default).logCols) &&
        levelsChained s'.levels s'.ext) = true := by
    intro m' h1' h2' s' hs'
    interval_cases m' <;> (simp only [fast100, Option.some.injEq] at hs'; subst hs'; decide +kernel)
  have := key m h1 h2 s h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at this
  exact ⟨this.1.1.1, this.1.1.2, this.1.2, this.2⟩

/-! ## Level 0 -/

/-! ## The last level -/

/-! ## Every level -/

/-! ## Ligerito -/

end FlockSoundness
