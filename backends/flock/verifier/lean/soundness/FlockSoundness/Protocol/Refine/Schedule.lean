import Flock.Schedule
import FlockSoundness.Protocol.Model.Compiled

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

end FlockSoundness.Refine
