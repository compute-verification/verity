/-
The wide-leaf schedule, as data: `fast100` with level 0's codeword held at `2^19` symbols.
-/
module

public import Flock.Zk
meta import Flock.Zk

@[expose] public section

namespace Flock

/-- The wide-leaf schedule for `m ∈ [22, 35]`; none otherwise. A table of `2^m` bits with `32 ≤ m ≤ 35` takes the wide
leaf: level 0 folds `k_0 = m − 25` lanes where `fast100` folds 6, so its codeword stays at `2^19` symbols where
`fast100`'s has `2^{m − 12}`, a level-0 row holds `2^{m − 25}` witness symbols where `fast100`'s holds 64, and the
levels are those of `m = 31`. For `m ≤ 31` it is `fast100 m`.

`fast100` stays beside it, unchanged: the two are the schedules of two statement versions, and a statement names one.
No verifier reads this one yet. -/
def fast100Wide (m : Nat) : Option Schedule :=
  if m ≤ 31 then fast100 m
  else if m ≤ 35 then do
    let s ← fast100 31
    let l0 := s.levels[0]!
    return { s with m, n := m - 7, k0 := m - 25, levels := s.levels.set! 0 { l0 with lanes := m - 25 } }
  else none

namespace Zk

/-- Level 0's query count on the wide-leaf schedule, by `queries0`'s rule at `fast100Wide`'s level 0: `queries0 m` for
`m ≤ 31`, and 219 for `32 ≤ m ≤ 35`, where `L = 2^18` as at `m = 31` (0.43 bits over the threshold; 218 is 0.03
under). -/
def queries0Wide (m : Nat) : V Nat :=
  if 32 ≤ m && m ≤ 35 then pure 219 else queries0 m

/-- The `--zk` schedule over `fast100Wide`, as `schedule` is over `fast100`: level 0 with `queries0Wide`'s `q_0`
queries (restratified, its cap the first stratum), and `t_pad = 2 q_0`. -/
def scheduleWide (m : Nat) : V (Schedule × Nat) := do
  let q0 ← queries0Wide m
  let some sch := fast100Wide m | throw s!"m = {m}: no fast100 schedule"
  let l0 := sch.levels[0]!
  let strata := decompose q0 l0.d
  return ({ sch with levels := sch.levels.set! 0 { l0 with q := q0, strata, cap := strata.getD 0 0 } }, 2 * q0)

end Zk

-- up to m = 31 the two schedules are one
#guard (List.range 32).all fun m => reprStr (fast100Wide m) == reprStr (fast100 m)
#guard (List.range 32).all fun m => reprStr (Zk.scheduleWide m) == reprStr (Zk.schedule m)
-- 32 to 35: m = 31's levels, level 0 with m − 25 lanes
#guard [32, 33, 34, 35].all fun m => (fast100Wide m).any fun s =>
  s.m == m && s.n == m - 7 && s.k0 == m - 25 && s.ext == 3 &&
    s.levels.map (fun l => (l.rate, l.q, l.cols, l.d, l.lanes)) ==
      #[(1, 218, 18, 19, m - 25), (2, 106, 15, 17, 4), (3, 71, 12, 15, 4), (4, 53, 9, 13, 4), (5, 43, 6, 11, 4),
        (6, 36, 3, 9, 4)]
#guard [32, 33, 34, 35].all fun m => match Zk.scheduleWide m with
  | .ok (s, tPad) => tPad == 438 && s.k0 == m - 25 && s.levels.map (·.q) == #[219, 106, 71, 53, 43, 36] &&
      s.levels[0]!.strata == decompose 219 19 && s.levels[0]!.cap == 7
  | .error _ => false
#guard [36, 37].all fun m => (fast100Wide m).isNone && (Zk.scheduleWide m).toOption.isNone
-- `queries0`'s rule at `L = 2^18`: 219 is the least count at 100 bits
#guard [218, 219].map (fun q : Nat =>
  let rho := (2 ^ 18 + 2 * q).toFloat / (2 ^ 19 : Nat).toFloat
  decide (q.toFloat * Float.log2 (1 / (Float.sqrt rho + 0.02)) ≥ 100)) == [false, true]

end Flock
