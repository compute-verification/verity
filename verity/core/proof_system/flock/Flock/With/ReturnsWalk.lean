/-
`returns_walk`: walks a do block of `V` to prove `Returns` (`Flock.With.Returns`). A macro is code run while compiling, so
it is alone here, the module `compile_time` lists, apart from the proofs that use it (`Flock.With.Deployed`, `FlockVerify`).
-/
import Flock.With.Returns

namespace Flock.With

/-- Walks a do block of `V` to a `Returns` goal at each `pure`. A bind keeps what its action returns when one of `ls`
covers the action or the action returns statements (each then shown to have a registry scheme), and forgets it
otherwise; a `throw` returns nothing; a branch is split, and a join point's call unfolded where it branches. A goal one of
`ls` states is closed by it, and a `pure` by a scheme the registry found, an assumption, or a one-table session of a
statement with one. -/
syntax "returns_walk" (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| returns_walk) => `(tactic| returns_walk [])
  | `(tactic| returns_walk [$ls,*]) => `(tactic| repeat' (first
    | extract_lets
    $[| (with_reducible exact $ls)]*
    $[| (with_reducible apply Returns.bindQ $ls); intro _ _]*
    | (with_reducible refine Returns.bindNamedMapM (fun _ => ?_) fun _ _ => ?_)
    | (with_reducible refine Returns.bindNamed ?_ fun _ _ => ?_)
    | (with_reducible apply Returns.bind'); intro _
    | (with_reducible apply Returns.throw')
    | (with_reducible apply Returns.pure');
      first | exact named_of_find? (by assumption) | assumption | exact named_singleton (by assumption)
    | (with_reducible apply Returns.ite) <;> intro _
    | (with_reducible apply Returns.dite) <;> intro _
    | split
    | (with_reducible conv => arg 1; whnf); split))

end Flock.With
