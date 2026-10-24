import Mathlib.Data.List.Basic
import Mathlib.Tactic.DeriveFintype

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

/-- One Ligerito level. -/
structure Level where
  /-- The code rate is `2^-logInvRate`. -/
  logInvRate : ℕ
  /-- Consistency queries made on this level's commitment. -/
  queries : ℕ
  /-- `log2` of the number of message columns. -/
  logCols : ℕ
  /-- `log2` of the number of interleaved rows (lanes): `k0` at level 0, 4 at every other level. -/
  lanes : ℕ
  deriving DecidableEq, Repr, Inhabited

namespace Level

/-- `log2` of the codeword length (the spec's `d`). -/
def logLen (l : Level) : ℕ := l.logCols + l.logInvRate

end Level

/-- A `fast100` schedule for one witness size. -/
structure Schedule where
  /-- `log2` of the number of witness bits. -/
  m : ℕ
  /-- Lanes folded at level 0 (`log_batch_size`). -/
  k0 : ℕ
  /-- The levels, level 0 first. -/
  levels : List Level
  /-- Length of the clear residual: `|yr| = 2^(ext+1)`. -/
  ext : ℕ
  deriving DecidableEq, Repr, Inhabited

/-- Queries per level of `fast100` (PROTOCOL.md §15, first table), by rate exponent. -/
def queriesAt : ℕ → ℕ
  | 1 => 218
  | 2 => 106
  | 3 => 71
  | 4 => 53
  | 5 => 43
  | 6 => 36
  | 7 => 32
  | _ => 0

/-- Build a schedule from the second table of §15: `k0`, the columns per level and `ext`.
Level `ℓ` has rate `2^-(ℓ+1)`, except that `m = 22` has rate `1/16` at level 2. -/
def mk (m k0 : ℕ) (cols : List ℕ) (ext : ℕ) : Schedule where
  m := m
  k0 := k0
  levels := (cols.zipIdx).map fun (c, ℓ) =>
    let r := if m = 22 ∧ ℓ = 2 then 4 else ℓ + 1
    { logInvRate := r, queries := queriesAt r, logCols := c, lanes := if ℓ = 0 then k0 else 4 }
  ext := ext

/-- The `fast100` schedule for `m` (PROTOCOL.md §15), or `none` outside `22 ≤ m ≤ 35`. -/
def fast100 : ℕ → Option Schedule
  | 22 => some (mk 22 6 [9, 6, 3] 3)
  | 23 => some (mk 23 6 [10, 7, 4] 4)
  | 24 => some (mk 24 6 [11, 8, 5] 5)
  | 25 => some (mk 25 6 [12, 9, 6, 3] 3)
  | 26 => some (mk 26 6 [13, 10, 7, 4] 4)
  | 27 => some (mk 27 6 [14, 11, 8, 5] 5)
  | 28 => some (mk 28 4 [17, 14, 11, 8, 5] 5)
  | 29 => some (mk 29 5 [17, 14, 11, 8, 5] 5)
  | 30 => some (mk 30 6 [17, 14, 11, 8, 5] 5)
  | 31 => some (mk 31 6 [18, 15, 12, 9, 6, 3] 3)
  | 32 => some (mk 32 6 [19, 16, 13, 10, 7, 4] 4)
  | 33 => some (mk 33 6 [20, 17, 14, 11, 8, 5] 5)
  | 34 => some (mk 34 6 [21, 18, 15, 12, 9, 6, 3] 3)
  | 35 => some (mk 35 6 [22, 19, 16, 13, 10, 7, 4] 4)
  | _ => none

/-- The §15 table's own consistency checks: the codeword lengths `d` it lists, and the shape
identity `k0 + 3·R + ext = m − 7` (`R` = levels − 1). -/
def Schedule.WellShaped (s : Schedule) (ds : List ℕ) : Prop :=
  s.levels.map Level.logLen = ds ∧ s.k0 + 3 * (s.levels.length - 1) + s.ext = s.m - 7

instance instDecidableWellShaped (s : Schedule) (ds : List ℕ) : Decidable (s.WellShaped ds) := by
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
