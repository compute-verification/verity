module

public import Proofs.Flock.Soundness.Refine.Walk

/-!
# Walking every return, reading the values bound before it

`ret_walk_zeta` proves `Ret m Q` (`Walk.lean`) where `Q` reads values `m` binds with `let` before it returns, as a
parser does that returns a structure of the fields it read. It differs from `ret_walk` in three ways:

* a `let` of a value that is not a function is substituted into the rest of the computation, so a return's `Q` reads
  the value; a `let` of a function of one argument is a join point, walked once into a hypothesis that each jump to it
  uses (`ret_jp`);
* a `match` at the head is split at that `match` (`Lean.Meta.Split.splitMatch`), never at one inside it;
* each step applies its lemma to the goal. At a join point the walk of the rest is `ret_jp`'s argument `hb`, never a
  function applied to the join point's body, so the proof term never substitutes the body into the rest.

A bind forgets what its first half returned, and a `pure a` whose `Q a` is not closed by `rfl` is left as a goal.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace FlockSoundness.Refine

open Flock

/-- **A join point, walked once**: if every return of the join point `v` satisfies `Q`, and `body j` returns only what
satisfies `Q` at every `j` whose returns all do, then so does `body v`. -/
theorem ret_jp {α γ : Type} (Q : γ → Prop) (v : α → V γ) (body : (α → V γ) → V γ)
    (hv : ∀ a, Ret (v a) Q) (hb : ∀ jp : α → V γ, (∀ a, Ret (jp a) Q) → Ret (body jp) Q) : Ret (body v) Q :=
  hb v hv

open Lean Elab Tactic Meta

private meta def applyIntros (g : MVarId) (e : Expr) : MetaM (List MVarId) := do
  (← g.apply e).mapM fun g => return (← g.intros).2

/-- One step of `ret_walk_zeta` on the goal `Ret m Q`, by `m`'s head; `none` when the goal is not a `Ret` or no step
applies. -/
meta def retZetaStep (g : MVarId) : MetaM (Option (List MVarId)) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``Ret 3 do return none
  let at_ (m : Expr) : MetaM MVarId := g.replaceTargetDefEq (mkApp3 ty.getAppFn (ty.getArg! 0) m (ty.getArg! 2))
  let m := (ty.getArg! 1).headBeta.consumeMData
  if m != ty.getArg! 1 then return some [← at_ m]
  let lem (n : Name) : MetaM Expr := mkConstWithFreshMVarLevels n
  match m with
  | .letE n t v b _ =>
    if t.isForall then
      let body := Expr.lam n t b .default
      some <$> applyIntros g (← mkAppOptM ``ret_jp #[t.bindingDomain!, ty.getArg! 0, ty.getArg! 2, v, body])
    else return some [← at_ (b.instantiate1 v)]
  | _ =>
  if m.isAppOfArity ``Bind.bind 6 then return some (← applyIntros g (← lem ``ret_bind'))
  if m.isAppOfArity ``Pure.pure 4 then
    let gs ← g.apply (← lem ``ret_pure)
    return some (← gs.flatMapM fun g => g.withContext do
      let g ← g.replaceTargetDefEq (← instantiateMVars (← g.getType)).headBeta
      try g.refl; return [] catch _ => return [g])
  if m.isAppOfArity ``MonadExcept.throw 5 then return some (← g.apply (← lem ``ret_throw))
  if m.isAppOfArity ``ite 5 then return some (← applyIntros g (← lem ``ret_ite))
  if m.isAppOfArity ``dite 5 then return some (← applyIntros g (← lem ``ret_dite))
  if m.getAppFn.isFVar then
    for d in ← getLCtx do
      let found ← forallTelescope (← instantiateMVars d.type) fun _ r =>
        pure (r.isAppOfArity ``Ret 3 && (r.getArg! 1).getAppFn == m.getAppFn)
      if found then
        let pf := mkAppN d.toExpr m.getAppArgs
        unless ← isDefEq (← inferType pf) ty do return none
        g.assign pf
        return some []
    return none
  if ← isMatcherApp m then return some (← Split.splitMatch g m)
  return none

/-- `ret_walk_zeta` walks the goal `Ret m Q` to every return of `m` (see the module docstring), leaving the goal `Q a`
at each `pure a` that `rfl` doesn't close, and each goal it has no step for. -/
elab "ret_walk_zeta" : tactic => do
  let mut todo := (← getGoals)
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← retZetaStep g with
      | none => left := left.push g; todo := rest
      | some gs => todo := gs ++ rest
  replaceMainGoal left.toList

end FlockSoundness.Refine
