import Mathlib.Data.List.Basic
import Mathlib.Tactic.DeriveFintype
import FlockSoundness.Protocol.Accounting.Schedule

/-!
# The `fast100` schedule (PROTOCOL.md §15)

The verifier of record pins Flock's `fast100` Ligerito profile and runs it twice on one root.
This file is the schedule table of §15 as data, for every `m` the profile supports (22 to 35).
Level `ℓ` of a schedule is a Reed–Solomon code of rate `2^-logInvRate` on messages of
`2^logCols` columns (codeword length `2^(logCols + logInvRate)`), interleaved over `2^lanes`
rows; the verifier makes `queries` consistency queries on it. The analysis slack is
`η = 1/200` at every level (Johnson regime); the §15 value `0.02` is a diagnostic the verifier never
reads.
-/

namespace FlockSoundness.Accounting

namespace Level

end Level

/-- The §15 table's own consistency checks: the codeword lengths `d` it lists, and the shape
identity `k0 + 3·R + ext = m − 7` (`R` = levels − 1). -/
def Schedule.WellShaped (s : Schedule) (ds : List ℕ) : Prop :=
  s.levels.map Level.logLen = ds ∧ s.k0 + 3 * (s.levels.length - 1) + s.ext = s.m - 7

instance (s : Schedule) (ds : List ℕ) : Decidable (s.WellShaped ds) := by
  unfold Schedule.WellShaped; infer_instance

/-- The `d` column of §15, checked against the schedule. -/
example : (fast100 22).get! |>.WellShaped [10, 8, 7] := by decide
example : (fast100 23).get! |>.WellShaped [11, 9, 7] := by decide
example : (fast100 24).get! |>.WellShaped [12, 10, 8] := by decide
example : (fast100 25).get! |>.WellShaped [13, 11, 9, 7] := by decide
example : (fast100 26).get! |>.WellShaped [14, 12, 10, 8] := by decide
example : (fast100 27).get! |>.WellShaped [15, 13, 11, 9] := by decide
example : (fast100 28).get! |>.WellShaped [18, 16, 14, 12, 10] := by decide
example : (fast100 29).get! |>.WellShaped [18, 16, 14, 12, 10] := by decide
example : (fast100 30).get! |>.WellShaped [18, 16, 14, 12, 10] := by decide
example : (fast100 31).get! |>.WellShaped [19, 17, 15, 13, 11, 9] := by decide
example : (fast100 32).get! |>.WellShaped [20, 18, 16, 14, 12, 10] := by decide
example : (fast100 33).get! |>.WellShaped [21, 19, 17, 15, 13, 11] := by decide
example : (fast100 34).get! |>.WellShaped [22, 20, 18, 16, 14, 12, 10] := by decide
example : (fast100 35).get! |>.WellShaped [23, 21, 19, 17, 15, 13, 11] := by decide

end FlockSoundness.Accounting
