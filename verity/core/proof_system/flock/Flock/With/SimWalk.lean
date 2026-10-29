/-
`sim_walk`: walks two copies of one definition in step under a simulation (`Flock.With.Sim`). A macro is code run while
compiling, so it is alone here, the module `compile_time` lists, apart from the proofs that use it (`Flock.With.SimVerify`,
`Flock.With.SimZk`).
-/
import Flock.With.Sim

namespace Flock.With

/-- One step of walking two copies of a definition: a `bind`, `pure`, `throw` or lift of either side's monad, a branch, a
loop or traversal, a state action, a `let` (moved to the context, so a `do` block's join point is walked at each of
its calls but never copied into the goal), or one of `ls` (each relating two calls of a `…With` function). -/
syntax "sim_walk" (" [" term,* "]")? : tactic

macro_rules
  | `(tactic| sim_walk) => `(tactic| sim_walk [])
  | `(tactic| sim_walk [$ls,*]) => `(tactic| repeat (first
    | extract_lets
    | (with_reducible apply Sim.bind)
    | (with_reducible intro _)
    | (with_reducible apply Sim.pure)
    | (with_reducible apply Sim.throw)
    | (with_reducible apply Sim.lift)
    | (with_reducible apply Sim.liftTM)
    | (with_reducible apply Sim.monadLift_st2)
    | (with_reducible apply Sim.monadLift_st)
    | (with_reducible apply Sim.lift_st)
    | (with_reducible apply Sim.run)
    | (with_reducible apply Sim.get)
    | (with_reducible apply Sim.set)
    | (with_reducible apply Sim.modify)
    | (with_reducible apply Sim.modifyGet)
    | (with_reducible apply Sim.forIn_range)
    | (with_reducible apply Sim.forIn_array)
    | (with_reducible apply Sim.forIn_list)
    | (with_reducible apply Sim.mapM_array)
    | (with_reducible apply Sim.mapIdxM_array)
    | (with_reducible apply Sim.findM?_list)
    | (with_reducible apply Sim.ite)
    | (with_reducible apply Sim.dite)
    | assumption
    | (with_reducible apply Sim.Hashes.sha512); assumption
    | (with_reducible apply Sim.Hashes.sha256); assumption
    $[| (with_reducible apply $ls)]*
    | split))

end Flock.With
