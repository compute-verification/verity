import Proofs.Flock.Soundness.Refine.Walk

/-!
# Walking a computation that builds its return in a `let`

`ret_walk` (`FlockSoundness.Refine.Walk`) generalizes a `let` that is not a join point and forgets its value, so a return
built in one (`let p := { … }; …; pure p`) says nothing of what it holds. `ret_walk_let` walks as `ret_walk` does, with
such a `let` generalized beside the equation of its value (`ret_let_eq`), so the goal at each return reads the values the
return was built from once the equations are substituted.
-/

namespace Properties.SampledProofs.SecurityProofs.LetWalk

open FlockSoundness.Refine

/-- A `let` that is not a join point, generalized with the equation of its value. -/
theorem ret_let_eq {α γ : Type} (e : α) (b : α → Flock.V γ) (Q : γ → Prop) (h : ∀ x, x = e → Ret (b x) Q) :
    Ret (b e) Q := h e rfl

open Lean Elab Tactic Meta in
/-- One step of `ret_walk_let` on the goal `Ret m Q`: a `let` that is not a join point is generalized with its value's
equation (`ret_let_eq`); every other step is `retStep`'s. -/
def retLetStep (keep : Array Name) (g : MVarId) : TacticM (Option (List MVarId)) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``Ret 3 do return none
  let γ := ty.getArg! 0
  let q := ty.getArg! 2
  match (ty.getArg! 1).headBeta.consumeMData with
  | .letE n t v b _ =>
    let isJp ← forallTelescopeReducing t fun _ r =>
      isDefEq r (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) γ)
    if isJp && v.consumeMData.isLambda then retStep keep g
    else
      let h ← mkFreshExprSyntheticOpaqueMVar (← withLocalDeclD n t fun x => do
        withLocalDeclD `h_let (← mkEq x v) fun hx => do
          mkForallFVars #[x, hx] (mkApp3 (mkConst ``Ret) γ (b.instantiate1 x) q))
      g.assign (mkAppN (mkConst ``ret_let_eq) #[t, γ, v, .lam n t b .default, q, h])
      return some [(← h.mvarId!.introNP 2).2]
  | _ => retStep keep g

open Lean Elab Tactic Meta in
/-- `ret_walk_let [c, …]`: `ret_walk [c, …]`, with each `let` that is not a join point kept as its value's equation
(`retLetStep`). -/
elab "ret_walk_let" "[" cs:ident,* "]" : tactic => do
  let mut keep := #[]
  for c in cs.getElems do keep := keep.push (← realizeGlobalConstNoOverloadWithInfo c)
  let mut todo := (← getGoals)
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← retLetStep keep g with
      | some new => todo := new ++ rest
      | none => left := left.push g; todo := rest
  setGoals left.toList

end Properties.SampledProofs.SecurityProofs.LetWalk
