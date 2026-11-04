import Proofs.Flock.Soundness.Refine.Walk

/-!
# Walking a long computation that builds its return in a `let`

`ret_walk_ret` walks as `ret_walk` does (`FlockSoundness.Refine.Walk`), with two differences for a long `do` block:

* A `let` of the computation's return type `γ` is substituted into its body, not generalized, so the goal at the return
  built in it (`let p : γ := { … }; …; pure p`) reads the values it was built from.
* A join point's two halves are joined through `id`. `ret_walk` applies the walk of its continuation, a metavariable
  `?hb`, to the join point and the walk of its body (`?hb v he`), and instantiating that beta-reduces, substituting the
  body's proof at every jump, so nested join points (one per `let x ← match …`) multiply it. Under `id` the application
  isn't a redex, and the body's proof appears once.
-/

namespace Properties.SampledProofs.SecurityProofs.RetWalk

open FlockSoundness.Refine

open Lean Elab Tactic Meta in
/-- One step of `ret_walk_ret` on the goal `Ret m Q`: a `let` of the return type is substituted into its body, a join
point is split into its body and its continuation joined through `id`, and every other step is `retStep`'s. -/
def retRetStep (keep : Array Name) (g : MVarId) : TacticM (Option (List MVarId)) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``Ret 3 do return none
  let γ := ty.getArg! 0
  let q := ty.getArg! 2
  let ret (m : Expr) : Expr := mkApp3 (mkConst ``Ret) γ m q
  match (ty.getArg! 1).headBeta.consumeMData with
  | .letE n t v b _ =>
    let isJp ← forallTelescopeReducing t fun _ r =>
      isDefEq r (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) γ)
    if isJp && v.consumeMData.isLambda then
      let heTy ← forallTelescopeReducing t fun xs _ => do mkForallFVars xs (ret (mkAppN v xs).headBeta)
      let hbTy ← withLocalDeclD n t fun j => do
        let hj ← forallTelescopeReducing t fun xs _ => do mkForallFVars xs (ret (mkAppN j xs))
        withLocalDeclD `hj hj fun h => do mkForallFVars #[j, h] (ret (b.instantiate1 j))
      let he ← mkFreshExprSyntheticOpaqueMVar heTy
      let hb ← mkFreshExprSyntheticOpaqueMVar hbTy
      g.assign (mkApp2 (mkApp2 (mkConst ``id [Level.zero]) hbTy hb) v he)
      let intro (e : Expr) : MetaM MVarId := do
        let mv := e.mvarId!
        let k ← forallTelescope (← mv.getType) fun xs _ => pure xs.size
        return (← mv.introNP k).2
      return some [← intro he, ← intro hb]
    else if ← isDefEq t γ then
      return some [← g.replaceTargetDefEq (ret (b.instantiate1 v))]
    else retStep keep g
  | _ => retStep keep g

open Lean Elab Tactic Meta in
/-- `ret_walk_ret [c, …]`: `ret_walk [c, …]` with a `let` of the return type substituted and each join point joined
through `id` (`retRetStep`). -/
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
