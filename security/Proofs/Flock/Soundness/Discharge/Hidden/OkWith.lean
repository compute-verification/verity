module

public import Proofs.Flock.Soundness.Refine.Walk
public meta import Lean.Elab.Tactic

/-!
# Tactics that build a computation's return

`walk_step` (`Refine/Walk.lean`) reads what a computation that returned checked. `ok_step` goes the other way: it proves
`OkWith x Q`, that the executable's computation `x` returns a value that satisfies `Q`, one step at a time by `x`'s
head, from hypotheses that each of its steps returns. With `walk_step` on one run and `ok_step` on another, a proof
carries a run that returned to a second run of the same computation that differs in an argument, where each step
either returns the same value or is shown to return on its own.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

open Flock

/-! ## A computation that returns -/

theorem ok_bind {α β : Type} (a : α) (f : α → V β) : (Except.ok a >>= f) = f a := rfl

/-- `m` returns a value that satisfies `Q`. -/
def OkWith {γ : Type} (m : V γ) (Q : γ → Prop) : Prop := ∃ v, m = .ok v ∧ Q v

theorem okWith_bind {α γ : Type} {x : V α} {a : α} {f : α → V γ} {Q : γ → Prop} (hx : x = .ok a)
    (h : OkWith (f a) Q) : OkWith (x >>= f) Q := by
  obtain ⟨v, hv, hq⟩ := h
  refine ⟨v, ?_, hq⟩
  rw [hx]
  exact hv

theorem okWith_bind' {α γ : Type} {x : V α} {f : α → V γ} {Q : γ → Prop} (hx : OkWith x fun _ => True)
    (h : ∀ a, x = .ok a → OkWith (f a) Q) : OkWith (x >>= f) Q := by
  obtain ⟨a, ha, -⟩ := hx
  exact okWith_bind ha (h a ha)

theorem okWith_ite_pos {γ : Type} {c : Prop} [Decidable c] {t e : V γ} {Q : γ → Prop} (hc : c) (h : OkWith t Q) :
    OkWith (ite c t e) Q := by
  rw [ite_eq_left hc]
  exact h

theorem okWith_ite_neg {γ : Type} {c : Prop} [Decidable c] {t e : V γ} {Q : γ → Prop} (hc : ¬c) (h : OkWith e Q) :
    OkWith (ite c t e) Q := by
  rw [ite_eq_right hc]
  exact h

theorem okWith_pure {γ : Type} {a : γ} {Q : γ → Prop} (h : Q a) : OkWith (pure a : V γ) Q := ⟨a, rfl, h⟩

theorem okWith_of_eq {γ : Type} {m : V γ} {a : γ} (h : m = .ok a) : OkWith m fun _ => True := ⟨a, h, trivial⟩

theorem okWith_of_isOk {γ : Type} {m : V γ} (h : m.isOk = true) : OkWith m fun _ => True := by
  cases m with
  | error e => cases h
  | ok a => exact ⟨a, rfl, trivial⟩

/-- A loop whose body only updates its state returns. -/
theorem okWith_forIn_array {α β : Type} (xs : Array α) (init : β) (f : α → β → β) :
    OkWith (forIn xs init fun a b => (pure (ForInStep.yield (f a b)) : V (ForInStep β))) fun _ => True :=
  ⟨_, Array.forIn_pure_yield_eq_foldl .., trivial⟩

/-- A loop over no element returns. -/
theorem okWith_forIn_empty {α β : Type} (init : β) (f : α → β → V (ForInStep β)) :
    OkWith (forIn (#[] : Array α) init f) fun _ => True :=
  ⟨init, (by simp only [List.forIn_toArray, List.forIn_nil]; rfl), trivial⟩

open Lean Elab Tactic Meta in
/-- Closes the goal from a hypothesis `∀ xs, a = b → … → False` whose premises hold by `rfl`: the arm a `match`'s
split leaves that an earlier split of the same `match` ruled out. -/
elab "absurd_step" : tactic => do
  let g ← getMainGoal
  g.withContext do
    for d in ← getLCtx do
      if d.isImplementationDetail then continue
      let t ← instantiateMVars d.type
      unless t.isForall do continue
      unless ← forallTelescope t fun _ r => pure (r.isConstOf ``False) do continue
      let hs ← Term.exprToSyntax d.toExpr
      let s ← saveState
      try
        evalTactic (← `(tactic| (exfalso; apply $hs <;> rfl)))
        return
      catch _ => s.restore
    throwError "absurd_step: no hypothesis"

open Lean Elab Tactic Meta in
/-- One step towards `OkWith m Q`, by `m`'s head. A bind whose computation a hypothesis `x = .ok a` gives continues at
`a`; any other bind leaves `OkWith x fun _ => True` as a goal of its own and continues at whatever `x` returns. An `if`
takes the arm a hypothesis decides, a `match` is reduced, rewritten by a hypothesis on a discriminant, or split, and a
`pure` leaves `Q` of its value. -/
elab "ok_step" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let ty := (← instantiateMVars (← g.getType)).consumeMData
    unless ty.isAppOfArity ``OkWith 3 do throwError "ok_step: the goal is not OkWith"
    let m := (ty.getArg! 1).headBeta.consumeMData
    let hyps := (← getLCtx).foldl (init := #[]) fun acc d => if d.isImplementationDetail then acc else acc.push d
    if m.isAppOfArity ``Bind.bind 6 then
      let x := m.getArg! 4
      for d in hyps.reverse do
        let t ← instantiateMVars d.type
        let some (_, l, r) := t.eq? | continue
        unless r.isAppOf ``Except.ok do continue
        if ← withReducible (isDefEq l x) then
          let hs ← Term.exprToSyntax d.toExpr
          evalTactic (← `(tactic| refine okWith_bind $hs ?_))
          return
      evalTactic (← `(tactic| refine okWith_bind' ?_ (fun _ _ => ?_)))
    else if m.isAppOfArity ``ite 5 then
      let c := m.getArg! 1
      for d in hyps.reverse do
        let t ← instantiateMVars d.type
        if ← withReducible (isDefEq t c) then
          let hs ← Term.exprToSyntax d.toExpr
          evalTactic (← `(tactic| refine okWith_ite_pos $hs ?_))
          return
        if ← withReducible (isDefEq t (mkNot c)) then
          let hs ← Term.exprToSyntax d.toExpr
          evalTactic (← `(tactic| refine okWith_ite_neg $hs ?_))
          return
      throwError "ok_step: no hypothesis decides {c}"
    else if m.isAppOfArity ``Pure.pure 4 then
      evalTactic (← `(tactic| refine okWith_pure ?_))
    else if (← Lean.Meta.isMatcherApp m) then
      let m' ← whnfCore m
      if m' != m then
        let g' ← g.replaceTargetDefEq (mkApp3 ty.getAppFn (ty.getArg! 0) m' (ty.getArg! 2))
        replaceMainGoal [g']
        return
      let some app ← Lean.Meta.matchMatcherApp? m | throwError "ok_step: no matcher"
      for disc in app.discrs do
        for d in hyps.reverse do
          let t ← instantiateMVars d.type
          let some (_, l, _) := t.eq? | continue
          if ← withReducible (isDefEq l disc) then
            let hs ← Term.exprToSyntax d.toExpr
            evalTactic (← `(tactic| rw [$hs:term]))
            return
      evalTactic (← `(tactic| split <;> try absurd_step))
    else throwError "ok_step: stop at {m.getAppFn}"

open Lean Elab Tactic Meta in
/-- `name_ok h as x`: the variable `h : _ = .ok _` returns gets the name `x`. -/
elab "name_ok " h:term " as " x:ident : tactic => do
  let g ← getMainGoal
  g.withContext do
    let e ← Tactic.elabTerm h none
    let ty ← instantiateMVars (← inferType e)
    let some (_, _, rhs) := ty.consumeMData.eq? | throwError "name_ok: not an equation"
    let v := rhs.consumeMData.appArg!
    unless v.isFVar do throwError "name_ok: {v} is not a variable"
    replaceMainGoal [← g.rename v.fvarId! x.getId]

open Lean Elab Tactic Meta in
/-- Clears the hypotheses `walk_step` left behind: each earlier `h`, the computation still to walk at that step. -/
elab "clear_walked" : tactic => do
  for d in ← (← getMainGoal).withContext getLCtx do
    if d.userName.eraseMacroScopes == `h then
      try liftMetaTactic fun g => return [← g.clear d.fvarId] catch _ => pure ()

end FlockSoundness.Discharge.Hidden
