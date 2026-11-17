module

public import Definitions.Flock.Compiled
public import Flock.Schedule
public import Proofs.Flock.Soundness.Model.Compiled
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset

/-! The executable's `fast100` schedule read as the model's (`schedOf`), and the facts the Ligerito loop relies on
(`LevelSeam`, `DimSeam`). -/

@[expose] public section

namespace FlockSoundness.Refine

open Model Accounting

/-- An executable level as the model's. -/
def levelOf (lv : Flock.Level) : Accounting.Level :=
  { logInvRate := lv.rate, queries := lv.q, logCols := lv.cols, lanes := lv.lanes }

/-- An executable schedule as the model's. -/
def schedOf (s : Flock.Schedule) : Accounting.Schedule :=
  { m := s.m, k0 := s.k0, levels := s.levels.toList.map levelOf, ext := s.ext }

def LevelSeam (lv : Flock.Level) : Prop :=
  lv.strata.toList = strata lv.q lv.d ∧ lv.cap = capDepth (levelOf lv) ∧
    lv.d - lv.cap = pathLen (levelOf lv) ∧ Nat.log2 (lv.q - 1) + 1 = ceilLog2 lv.q ∧
    lv.d = (levelOf lv).logLen

instance instDecidableLevelSeam (lv : Flock.Level) : Decidable (LevelSeam lv) := by unfold LevelSeam; infer_instance

def DimSeam (s : Flock.Schedule) : Prop :=
  s.n = s.m - 7 ∧ 2 ≤ s.levels.size ∧ (s.levels.getD 0 default).lanes = s.k0 ∧
    (s.levels.getD 0 default).cols = s.n - s.k0 ∧
    (∀ i < s.levels.size, 1 ≤ i → (s.levels.getD i default).lanes = 4) ∧
    (∀ i < s.levels.size - 1, (s.levels.getD (i + 1) default).cols + 3 = (s.levels.getD i default).cols) ∧
    s.ext = (s.levels.getD (s.levels.size - 1) default).cols

instance instDecidableDimSeam (s : Flock.Schedule) : Decidable (DimSeam s) := by unfold DimSeam; infer_instance

end FlockSoundness.Refine
