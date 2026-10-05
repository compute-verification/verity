import FlockSoundness.Protocol.Refine.Rep

/-!
# A tactic that walks a successful executable computation

`walk_step` takes one step through `h : x = .ok v` for a computation `x` in the executable's monad `V`, chosen by `x`'s
head: a bind is split into its halves (`ebind_ok`), an `if` or `dite` is split on its condition, a `match` is split, and a
`throw` closes the goal. It fails at a `pure`, so `repeat' walk_step` reaches every return. Choosing the step by the head,
rather than trying each and backtracking, keeps Lean from unfolding the executable to look for a bind that isn't there.
The kernel checks every proof it builds, as it does every other proof in the package.
-/

namespace FlockSoundness.Refine

open Flock

theorem throw_ne_ok {α : Type} (e : String) (a : α) : (throw e : V α) ≠ .ok a := fun h => by cases h

end FlockSoundness.Refine
