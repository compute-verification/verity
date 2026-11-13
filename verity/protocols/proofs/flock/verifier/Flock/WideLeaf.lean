/-
The wide-leaf schedule, as data: `fast100` with level 0's codeword held at `2^19` symbols.
-/
module

public import Flock.Schedule
meta import Flock.Schedule

@[expose] public section

namespace Flock

/-- The wide-leaf schedule for `m ∈ [22, 35]`; none otherwise. A table of `2^m` bits with `32 ≤ m ≤ 35` takes the wide
leaf: level 0 folds `k_0 = m − 25` lanes where `fast100` folds 6, so its codeword stays at `2^19` symbols where
`fast100`'s has `2^{m − 12}`, a level-0 row holds `2^{m − 25}` witness symbols where `fast100`'s holds 64, and the
levels are those of `m = 31`. For `m ≤ 31` it is `fast100 m`.

`fast100` stays beside it, unchanged: the two are the schedules of two statement versions, and a statement names one
(`Tags.wideLeaf`). The `--zk` verifier reads this one at a wide-leaf statement (`Zk.schedule`); without `--zk` such a
statement is refused. -/
def fast100Wide (m : Nat) : Option Schedule :=
  if m ≤ 31 then fast100 m
  else if m ≤ 35 then do
    let s ← fast100 31
    let l0 := s.levels[0]!
    return { s with m, n := m - 7, k0 := m - 25, levels := s.levels.set! 0 { l0 with lanes := m - 25 } }
  else none

/-- The schedule of a statement version's tables: the wide-leaf schedule at a wide-leaf statement (`wide`,
`Tags.wideLeaf`), `fast100` otherwise. -/
def fast100At (wide : Bool) (m : Nat) : Option Schedule := if wide then fast100Wide m else fast100 m

-- up to m = 31 the two schedules are one
#guard (List.range 32).all fun m => reprStr (fast100Wide m) == reprStr (fast100 m)
-- 32 to 35: m = 31's levels, level 0 with m − 25 lanes
#guard [32, 33, 34, 35].all fun m => (fast100Wide m).any fun s =>
  s.m == m && s.n == m - 7 && s.k0 == m - 25 && s.ext == 3 &&
    s.levels.map (fun l => (l.rate, l.q, l.cols, l.d, l.lanes)) ==
      #[(1, 218, 18, 19, m - 25), (2, 106, 15, 17, 4), (3, 71, 12, 15, 4), (4, 53, 9, 13, 4), (5, 43, 6, 11, 4),
        (6, 36, 3, 9, 4)]
#guard [36, 37].all fun m => (fast100Wide m).isNone

end Flock
