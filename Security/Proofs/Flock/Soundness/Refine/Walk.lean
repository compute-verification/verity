import Proofs.Flock.Soundness.Refine.Rep

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

open Lean Elab Tactic Meta in
/-- One step through `h : x = .ok v`, by `x`'s head (see the module docstring). -/
elab "walk_step" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let some d := (← getLCtx).findFromUserName? `h | throwError "walk_step: no h"
    let ty ← instantiateMVars d.type
    let some (_, lhs, _) := ty.eq? | throwError "walk_step: h is not an equation"
    let lhs := lhs.headBeta.consumeMData
    let h := mkIdent `h
    if lhs.isAppOfArity ``Bind.bind 6 then
      evalTactic (← `(tactic| obtain ⟨_, _, $h:ident⟩ := ebind_ok $h))
    else if lhs.isAppOfArity ``ite 5 then
      let cs ← Term.exprToSyntax (lhs.getArg! 1)
      evalTactic (← `(tactic| by_cases hc : $cs <;>
        first | rw [ite_eq_left hc] at $h:ident | rw [ite_eq_right hc] at $h:ident))
    else if lhs.isAppOfArity ``dite 5 then
      let cs ← Term.exprToSyntax (lhs.getArg! 1)
      evalTactic (← `(tactic| by_cases hc : $cs <;>
        first | rw [dite_eq_left hc] at $h:ident | rw [dite_eq_right hc] at $h:ident))
    else if lhs.isAppOfArity ``MonadExcept.throw 5 || lhs.isAppOfArity ``throwThe 5 then
      evalTactic (← `(tactic| exact absurd $h (throw_ne_ok _ _)))
    else if (← Lean.Meta.isMatcherApp lhs) then
      evalTactic (← `(tactic| split at $h:ident))
    else throwError "walk_step: stop"

/-! ### Walking every return

`walk_step` rewrites `h` and so substitutes each step's result into the rest of the computation; a `do` block's join
points (`let __do_jp := fun … => rest`, one per `if` or `match` whose arms both continue) are inlined into both arms, and
the walk doubles at each. `ret_walk` proves `Ret x Q`, that every value `x` returns satisfies `Q`, by building the proof
term directly: each lemma below is applied with its arguments given, so nothing is unified and no `let` is inlined. A
join point's body is walked once, into a hypothesis that each jump to it uses. -/

/-- Every value `m` returns satisfies `Q`. -/
def Ret {γ : Type} (m : V γ) (Q : γ → Prop) : Prop := ∀ v, m = .ok v → Q v

theorem ret_bind {α γ : Type} (x : V α) (f : α → V γ) (Q : γ → Prop) (h : ∀ a, x = .ok a → Ret (f a) Q) :
    Ret (x >>= f) Q := fun v hv => by
  cases hx : x with
  | error e => rw [hx] at hv; cases hv
  | ok a => rw [hx] at hv; exact h a hx v hv

theorem ret_bind' {α γ : Type} (x : V α) (f : α → V γ) (Q : γ → Prop) (h : ∀ a, Ret (f a) Q) : Ret (x >>= f) Q :=
  ret_bind x f Q fun a _ => h a

theorem ret_ite {γ : Type} (c : Prop) [Decidable c] (t e : V γ) (Q : γ → Prop) (ht : Ret t Q) (he : Ret e Q) :
    Ret (ite c t e) Q := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact ht
  · rw [ite_eq_right hc]; exact he

theorem ret_dite {γ : Type} (c : Prop) [Decidable c] (t : c → V γ) (e : ¬c → V γ) (Q : γ → Prop)
    (ht : ∀ h, Ret (t h) Q) (he : ∀ h, Ret (e h) Q) : Ret (dite c t e) Q := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact ht hc
  · rw [dite_eq_right hc]; exact he hc

theorem ret_throw {γ : Type} (e : String) (Q : γ → Prop) : Ret (throw e : V γ) Q := fun _ h => by cases h

theorem ret_pure {γ : Type} (a : γ) (Q : γ → Prop) (h : Q a) : Ret (pure a : V γ) Q := fun _ hv => by
  cases hv; exact h

/-- A `let` that is not a join point, generalized. -/
theorem ret_let {α γ : Type} (e : α) (b : α → V γ) (Q : γ → Prop) (h : ∀ x, Ret (b x) Q) : Ret (b e) Q := h e

open Lean Elab Tactic Meta in
/-- One step of `ret_walk` on the goal `Ret m Q`, by `m`'s head; `none` when the goal is not a `Ret`. A bind of a
computation headed by a constant in `keep` keeps its equation `x = .ok a`; every other step forgets what it passed. -/
def retStep (keep : Array Name) (g : MVarId) : TacticM (Option (List MVarId)) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``Ret 3 do return none
  let γ := ty.getArg! 0
  let q := ty.getArg! 2
  let m := (ty.getArg! 1).headBeta.consumeMData
  let ret (m : Expr) : Expr := mkApp3 (mkConst ``Ret) γ m q
  let vγ := mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) γ
  let close (pf : Expr) (gs : List Expr) : MetaM (List MVarId) := do
    g.assign pf
    gs.mapM fun e => do
      let mv := e.mvarId!
      let n ← forallTelescope (← mv.getType) fun xs _ => pure xs.size
      return (← mv.introNP n).2
  let goal (ty : Expr) : MetaM Expr := mkFreshExprSyntheticOpaqueMVar ty
  match m with
  | .letE n t v b _ =>
      let isJp ← forallTelescopeReducing t fun _ r => isDefEq r vγ
      if isJp && v.consumeMData.isLambda then
        -- he : ∀ xs, Ret (v xs) Q;  hb : ∀ j, (∀ xs, Ret (j xs) Q) → Ret (b j) Q;  the step is `hb v he`
        let he ← goal (← forallTelescopeReducing t fun xs _ => do mkForallFVars xs (ret (mkAppN v xs).headBeta))
        let hb ← goal (← withLocalDeclD n t fun j => do
          let hj ← forallTelescopeReducing t fun xs _ => do mkForallFVars xs (ret (mkAppN j xs))
          withLocalDeclD `hj hj fun h => do mkForallFVars #[j, h] (ret (b.instantiate1 j)))
        some <$> close (mkApp2 hb v he) [he, hb]
      else
        let h ← goal (← withLocalDeclD n t fun x => do mkForallFVars #[x] (ret (b.instantiate1 x)))
        some <$> close (mkAppN (mkConst ``ret_let) #[t, γ, v, .lam n t b .default, q, h]) [h]
  | _ =>
  if m.isAppOfArity ``Bind.bind 6 then
    let α := m.getArg! 2
    let x := m.getArg! 4
    let f := m.getArg! 5
    if keep.any x.getAppFn.isConstOf then
      let h ← goal (← withLocalDeclD `a α fun a => do
        let eq ← mkEq x (mkApp3 (mkConst ``Except.ok [0, 0]) (mkConst ``String) α a)
        withLocalDeclD `hx eq fun hx => do mkForallFVars #[a, hx] (ret (mkApp f a).headBeta))
      some <$> close (mkAppN (mkConst ``ret_bind) #[α, γ, x, f, q, h]) [h]
    else
      let h ← goal (← withLocalDeclD `a α fun a => do mkForallFVars #[a] (ret (mkApp f a).headBeta))
      some <$> close (mkAppN (mkConst ``ret_bind') #[α, γ, x, f, q, h]) [h]
  else if m.isAppOfArity ``ite 5 then
    let ht ← goal (ret (m.getArg! 3))
    let he ← goal (ret (m.getArg! 4))
    some <$> close (mkAppN (mkConst ``ret_ite) #[γ, m.getArg! 1, m.getArg! 2, m.getArg! 3, m.getArg! 4, q, ht, he])
      [ht, he]
  else if m.isAppOfArity ``dite 5 then
    let c := m.getArg! 1
    let ht ← goal (← withLocalDeclD `h c fun h => do mkForallFVars #[h] (ret (mkApp (m.getArg! 3) h).headBeta))
    let he ← goal (← withLocalDeclD `h (mkNot c) fun h => do
      mkForallFVars #[h] (ret (mkApp (m.getArg! 4) h).headBeta))
    some <$> close (mkAppN (mkConst ``ret_dite) #[γ, c, m.getArg! 2, m.getArg! 3, m.getArg! 4, q, ht, he]) [ht, he]
  else if m.isAppOfArity ``MonadExcept.throw 5 then
    some <$> close (mkAppN (mkConst ``ret_throw) #[γ, m.getArg! 4, q]) []
  else if m.isAppOfArity ``Pure.pure 4 then
    let h ← goal (mkApp q (m.getArg! 3)).headBeta
    some <$> close (mkAppN (mkConst ``ret_pure) #[γ, m.getArg! 3, q, h]) [h]
  else if m.getAppFn.isFVar then
    -- a jump to a join point: the hypothesis its body was walked into
    let j := m.getAppFn
    for d in ← getLCtx do
      let found ← forallTelescope (← instantiateMVars d.type) fun _ r =>
        pure (r.isAppOfArity ``Ret 3 && (r.getArg! 1).getAppFn == j)
      if found then
        g.assign (mkAppN d.toExpr m.getAppArgs)
        return some []
    throwError "ret_walk: a jump to {j} with no hypothesis"
  else if ← isMatcherApp m then
    return some (← Lean.Elab.Tactic.run g (evalTactic (← `(tactic| split))))
  else throwError "ret_walk: no step for {m.getAppFn}"

open Lean Elab Tactic Meta in
/-- `ret_walk [c, …]` walks the goal `Ret m Q` to every return of `m`, leaving the goal `Q a` at each `pure a`; a bind of
a computation headed by one of the constants keeps its equation as a hypothesis (see `retStep`). -/
elab "ret_walk" "[" cs:ident,* "]" : tactic => do
  let keep ← cs.getElems.mapM realizeGlobalConstNoOverloadWithInfo
  let mut todo := (← getGoals)
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← retStep keep g with
      | some new => todo := new ++ rest
      | none => left := left.push g; todo := rest
  setGoals left.toList

end FlockSoundness.Refine
