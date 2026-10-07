import Proofs.Flock.Soundness.Model.Compiled
import Flock.Schedule

/-!
# The executable's schedule, read as the model's

`schedOf` reads an executable `fast100` schedule (`Flock.Schedule`, PROTOCOL.md §15) as the model's
(`Accounting.Schedule`). `LevelSeam` and `DimSeam` are the facts about one schedule that the executable's Ligerito loop
relies on and the model states differently: the stratified queries (§13.5), the Merkle cap depth and path length (§14),
the consistency-coin count, and the dimensions of the rounds (§13.3). `Refine.Seam` proves them for every `fast100`
schedule.
-/

namespace FlockSoundness.Refine

open Model Accounting

/-- An executable level as the model's. -/
def levelOf (lv : Flock.Level) : Accounting.Level :=
  { logInvRate := lv.rate, queries := lv.q, logCols := lv.cols, lanes := lv.lanes }

/-- An executable schedule as the model's. -/
def schedOf (s : Flock.Schedule) : Accounting.Schedule :=
  { m := s.m, k0 := s.k0, levels := s.levels.toList.map levelOf, ext := s.ext }

/-- One level's executable quantities are the model's: the strata of its queries (`strata`), its cap depth
(`capDepth`), its path length (`pathLen`), its consistency-coin count (`ceilLog2`), and its codeword length
(`logLen`). -/
def LevelSeam (lv : Flock.Level) : Prop :=
  lv.strata.toList = strata lv.q lv.d ∧ lv.cap = capDepth (levelOf lv) ∧
    lv.d - lv.cap = pathLen (levelOf lv) ∧ Nat.log2 (lv.q - 1) + 1 = ceilLog2 lv.q ∧
    lv.d = (levelOf lv).logLen

instance instDecidableLevelSeam (lv : Flock.Level) : Decidable (LevelSeam lv) := by unfold LevelSeam; infer_instance

/-- The dimensions the executable's Ligerito loop tracks: `n = m − 7`; at least two levels; level 0 has `k0` lanes and
`n − k0` columns; every later level has 4 lanes and 3 columns fewer than the level before; `ext` is the last level's
columns. -/
def DimSeam (s : Flock.Schedule) : Prop :=
  s.n = s.m - 7 ∧ 2 ≤ s.levels.size ∧ (s.levels.getD 0 default).lanes = s.k0 ∧
    (s.levels.getD 0 default).cols = s.n - s.k0 ∧
    (∀ i < s.levels.size, 1 ≤ i → (s.levels.getD i default).lanes = 4) ∧
    (∀ i < s.levels.size - 1, (s.levels.getD (i + 1) default).cols + 3 = (s.levels.getD i default).cols) ∧
    s.ext = (s.levels.getD (s.levels.size - 1) default).cols

instance instDecidableDimSeam (s : Flock.Schedule) : Decidable (DimSeam s) := by unfold DimSeam; infer_instance

end FlockSoundness.Refine
