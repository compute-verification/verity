import Finders.Defs

/-! The two core rows' finders compute on a concrete input. The kernel cannot evaluate a `while` (`repeatM` runs through a
`partial def`), so this is an evaluation, which the lock lists under `compile_time`. -/

#guard Finders.loopedOut 5 == some (10, 5)
#guard Finders.parsedOut 7 == some (7, 8)
