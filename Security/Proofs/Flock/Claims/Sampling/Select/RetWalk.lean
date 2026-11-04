import Proofs.Flock.Soundness.Refine.Walk

/-!
# Walking a computation that builds its return in a `let`

`ret_walk` (`FlockSoundness.Refine.Walk`) generalizes a `let` that is not a join point and forgets its value, so a return
built in one (`let p : γ := { … }; …; pure p`) says nothing of what it holds. `ret_walk_ret` walks as `ret_walk` does,
with a `let` of the computation's return type `γ` substituted into its body instead, so the goal at that return reads
the values the return was built from. Every other `let` is `ret_walk`'s, so nothing else is inlined.
-/

namespace Properties.SampledProofs.SecurityProofs.RetWalk

open FlockSoundness.Refine

open Lean Elab Tactic Meta in
/-- One step of `ret_walk_ret` on the goal `Ret m Q`: a `let` of the return type is substituted into its body; every
other step is `retStep`'s. -/
def retRetStep (keep : Array Name) (g : MVarId) : TacticM (Option (List MVarId)) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``Ret 3 do return none
  let γ := ty.getArg! 0
  match (ty.getArg! 1).headBeta.consumeMData with
  | .letE _ t v b _ =>
    if ← isDefEq t γ then
      return some [← g.replaceTargetDefEq (mkApp3 (mkConst ``Ret) γ (b.instantiate1 v) (ty.getArg! 2))]
    else retStep keep g
  | _ => retStep keep g

open Lean Elab Tactic Meta in
/-- `ret_walk_ret [c, …]`: `ret_walk [c, …]`, with each `let` of the return type substituted (`retRetStep`). -/
elab "ret_walk_ret" "[" cs:ident,* "]" : tactic => do
  let mut keep := #[]
  for c in cs.getElems do keep := keep.push (← realizeGlobalConstNoOverloadWithInfo c)
  let mut todo := (← getGoals)
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← retRetStep keep g with
      | some new => todo := new ++ rest
      | none => left := left.push g; todo := rest
  setGoals left.toList

end Properties.SampledProofs.SecurityProofs.RetWalk
