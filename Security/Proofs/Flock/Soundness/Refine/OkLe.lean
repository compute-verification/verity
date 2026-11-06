module

public import Proofs.Flock.Soundness.Refine.Rep

/-!
# Two runs of one executable, where one returns what the other returns

`OkLe x y` says that `y` returns whatever `x` returns. `ok_le_walk` proves it for two runs of one executable whose
arguments differ, by walking both at once: a step that both sides take alike (a bind, an `if` on one condition, a `match`
on one scrutinee, a `for` loop over one collection, a `let`) is taken on both, a computation both sides share is closed by
`okLe_refl`, a `throw` on the left closes the goal, and a join point's bodies are walked once into a hypothesis each jump
to it uses. It leaves the goals where the two sides differ otherwise. It runs only inside the elaborator, and the kernel
checks every proof it builds.
-/

@[expose] public section

namespace FlockSoundness.Refine

open Flock

/-- `y` returns whatever `x` returns. -/
def OkLe {α : Type} (x y : V α) : Prop := ∀ a, x = .ok a → y = .ok a

theorem okLe_refl {α : Type} (x : V α) : OkLe x x := fun _ h => h

theorem okLe_bind {α γ : Type} (x x' : V α) (f f' : α → V γ) (hx : OkLe x x') (hf : ∀ a, OkLe (f a) (f' a)) :
    OkLe (x >>= f) (x' >>= f') := fun v hv => by
  cases h : x with
  | error e => rw [h] at hv; cases hv
  | ok a => rw [h] at hv; rw [hx a h]; exact hf a v hv

theorem okLe_ite {γ : Type} (c : Prop) [Decidable c] (t e t' e' : V γ) (ht : c → OkLe t t') (he : ¬c → OkLe e e') :
    OkLe (ite c t e) (ite c t' e') := by
  by_cases hc : c
  · rw [ite_eq_left hc, ite_eq_left hc]; exact ht hc
  · rw [ite_eq_right hc, ite_eq_right hc]; exact he hc

theorem okLe_dite {γ : Type} (c : Prop) [Decidable c] (t t' : c → V γ) (e e' : ¬c → V γ)
    (ht : ∀ h, OkLe (t h) (t' h)) (he : ∀ h, OkLe (e h) (e' h)) : OkLe (dite c t e) (dite c t' e') := by
  by_cases hc : c
  · rw [dite_eq_left hc, dite_eq_left hc]; exact ht hc
  · rw [dite_eq_right hc, dite_eq_right hc]; exact he hc

theorem okLe_throw {γ : Type} (e : String) (y : V γ) : OkLe (throw e : V γ) y := fun _ h => by cases h

theorem okLe_bind_throw {α γ : Type} (e : String) (f : α → V γ) (y : V γ) : OkLe ((throw e : V α) >>= f) y :=
  fun _ h => by cases h

/-- A `let` both sides bind to one value, generalized. -/
theorem okLe_let {α γ : Type} (v : α) (b b' : α → V γ) (h : ∀ x, OkLe (b x) (b' x)) : OkLe (b v) (b' v) := h v

/-- A join point on each side, related by `P`: its bodies `v` and `v'`, and the rest `b` and `b'` at any two related
join points. -/
theorem okLe_jp {β γ : Type} (v v' : β) (b b' : β → V γ) (P : β → β → Prop) (hv : P v v')
    (hb : ∀ j j', P j j' → OkLe (b j) (b' j')) : OkLe (b v) (b' v') := hb v v' hv

theorem okLe_forIn_list {α β : Type} (xs : List α) (init : β) (f g : α → β → V (ForInStep β))
    (h : ∀ a b, OkLe (f a b) (g a b)) : OkLe (forIn xs init f) (forIn xs init g) := by
  induction xs generalizing init with
  | nil => exact okLe_refl _
  | cons a as ih =>
    simp only [List.forIn_cons]
    refine okLe_bind _ _ _ _ (h a init) fun s => ?_
    cases s with
    | done b => exact okLe_refl _
    | yield b => exact ih b

theorem okLe_forIn_array {α β : Type} (xs : Array α) (init : β) (f g : α → β → V (ForInStep β))
    (h : ∀ a b, OkLe (f a b) (g a b)) : OkLe (forIn xs init f) (forIn xs init g) := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact okLe_forIn_list _ _ _ _ h

/-- Where `x` returns, what `y` returns `x` returns. -/
theorem okLe_back {α : Type} {x y : V α} {c : α} (hx : x = .ok c) (h : OkLe x y) : OkLe y x := fun a ha => by
  rw [h c hx] at ha; cases ha; exact hx

open Lean Elab Tactic Meta in
/-- One step of `ok_le_walk` on the goal `OkLe m m'`, by the heads of `m` and `m'`; `none` where it takes none. -/
meta def okLeStep (g : MVarId) : TacticM (Option (List MVarId)) := g.withContext do
  let ty := (← instantiateMVars (← g.getType)).consumeMData
  unless ty.isAppOfArity ``OkLe 3 do return none
  let γ := ty.getArg! 0
  let m := (ty.getArg! 1).headBeta.consumeMData
  let m' := (ty.getArg! 2).headBeta.consumeMData
  let le (γ m m' : Expr) : Expr := mkApp3 (mkConst ``OkLe) γ m m'
  let close (pf : Expr) (gs : List Expr) : MetaM (List MVarId) := do
    g.assign pf
    gs.mapM fun e => do
      let mv := e.mvarId!
      let n ← forallTelescope (← mv.getType) fun xs _ => pure xs.size
      return (← mv.introNP n).2
  let goal (ty : Expr) : MetaM Expr := mkFreshExprSyntheticOpaqueMVar ty
  -- `∀ xs, OkLe (j xs) (j' xs)` for join points `j`, `j'` of type `t`
  let jumps (t j j' : Expr) : MetaM Expr := forallTelescopeReducing t fun xs r => do
    mkForallFVars xs (le ((← whnf r).getArg! 1) (mkAppN j xs).headBeta (mkAppN j' xs).headBeta)
  if m == m' then
    return some (← close (mkApp2 (mkConst ``okLe_refl) γ m) [])
  match m, m' with
  | .letE n t v b _, .letE _ t' v' b' _ =>
      unless t == t' do return none
      let isJp ← forallTelescopeReducing t fun _ r => return (← whnf r).isAppOfArity ``Except 2
      if isJp && v.consumeMData.isLambda && n.eraseMacroScopes == `__do_jp then
        let hv ← goal (← jumps t v v')
        let hb ← goal (← withLocalDeclD n t fun j => withLocalDeclD n t fun j' => do
          withLocalDeclD `hj (← jumps t j j') fun h => do
            mkForallFVars #[j, j', h] (le γ (b.instantiate1 j) (b'.instantiate1 j')))
        let P ← withLocalDeclD n t fun j => withLocalDeclD n t fun j' => do mkLambdaFVars #[j, j'] (← jumps t j j')
        some <$> close (mkAppN (mkConst ``okLe_jp) #[t, γ, v, v', .lam n t b .default, .lam n t b' .default, P, hv, hb])
          [hv, hb]
      else if v == v' then
        let h ← goal (← withLocalDeclD n t fun x => do mkForallFVars #[x] (le γ (b.instantiate1 x) (b'.instantiate1 x)))
        some <$> close (mkAppN (mkConst ``okLe_let) #[t, γ, v, .lam n t b .default, .lam n t b' .default, h]) [h]
      else
        -- a local function the sides define differently: inlined, so each use is walked where it is used
        let h ← goal (le γ (b.instantiate1 v) (b'.instantiate1 v'))
        some <$> close h [h]
  | _, _ =>
  if m.isAppOfArity ``Bind.bind 6 && m'.isAppOfArity ``Bind.bind 6 then
    let α := m.getArg! 2
    let x := m.getArg! 4
    let x' := m'.getArg! 4
    let f := m.getArg! 5
    let f' := m'.getArg! 5
    let hx ← if x == x' then pure (mkApp2 (mkConst ``okLe_refl) α x) else goal (le α x x')
    let hf ← goal (← withLocalDeclD `a α fun a => do
      mkForallFVars #[a] (le γ (mkApp f a).headBeta (mkApp f' a).headBeta))
    some <$> close (mkAppN (mkConst ``okLe_bind) #[α, γ, x, x', f, f', hx, hf])
      (if x == x' then [hf] else [hx, hf])
  else if m.isAppOfArity ``ite 5 && m'.isAppOfArity ``ite 5 && m.getArg! 1 == m'.getArg! 1 then
    let c := m.getArg! 1
    let ht ← goal (← withLocalDeclD `hc c fun h => do mkForallFVars #[h] (le γ (m.getArg! 3) (m'.getArg! 3)))
    let he ← goal (← withLocalDeclD `hc (mkNot c) fun h => do mkForallFVars #[h] (le γ (m.getArg! 4) (m'.getArg! 4)))
    some <$> close (mkAppN (mkConst ``okLe_ite) #[γ, c, m.getArg! 2, m.getArg! 3, m.getArg! 4, m'.getArg! 3,
      m'.getArg! 4, ht, he]) [ht, he]
  else if m.isAppOfArity ``dite 5 && m'.isAppOfArity ``dite 5 && m.getArg! 1 == m'.getArg! 1 then
    let c := m.getArg! 1
    let ht ← goal (← withLocalDeclD `h c fun h => do
      mkForallFVars #[h] (le γ (mkApp (m.getArg! 3) h).headBeta (mkApp (m'.getArg! 3) h).headBeta))
    let he ← goal (← withLocalDeclD `h (mkNot c) fun h => do
      mkForallFVars #[h] (le γ (mkApp (m.getArg! 4) h).headBeta (mkApp (m'.getArg! 4) h).headBeta))
    some <$> close (mkAppN (mkConst ``okLe_dite) #[γ, c, m.getArg! 2, m.getArg! 3, m'.getArg! 3, m.getArg! 4,
      m'.getArg! 4, ht, he]) [ht, he]
  else if m.isAppOfArity ``Bind.bind 6 && (m.getArg! 4).isAppOfArity ``MonadExcept.throw 5 then
    some <$> close (mkAppN (mkConst ``okLe_bind_throw) #[m.getArg! 2, γ, (m.getArg! 4).getArg! 4, m.getArg! 5, m']) []
  else if m.isAppOfArity ``MonadExcept.throw 5 then
    some <$> close (mkAppN (mkConst ``okLe_throw) #[γ, m.getArg! 4, m']) []
  else if m.getAppFn.isFVar && m'.getAppFn.isFVar && m.getAppArgs == m'.getAppArgs then
    -- a jump to a join point on each side: the hypothesis their bodies were walked into
    let j := m.getAppFn
    let j' := m'.getAppFn
    for d in ← getLCtx do
      let found ← forallTelescope (← instantiateMVars d.type) fun _ r => do
        let r := r.consumeMData
        pure (r.isAppOfArity ``OkLe 3 && (r.getArg! 1).getAppFn == j && (r.getArg! 2).getAppFn == j')
      if found then
        g.assign (mkAppN d.toExpr m.getAppArgs)
        return some []
    return none
  else if m.isAppOfArity ``ForIn.forIn 8 && m'.isAppOfArity ``ForIn.forIn 8 then
    return some (← Lean.Elab.Tactic.run g (evalTactic (← `(tactic|
      first
      | refine okLe_forIn_array _ _ _ _ fun _ _ => ?_
      | refine okLe_forIn_list _ _ _ _ fun _ _ => ?_))))
  else if let (some a, some a') := (← matchMatcherApp? m, ← matchMatcherApp? m') then
    unless a.matcherName == a'.matcherName && a.discrs == a'.discrs do return none
    return some (← Lean.Elab.Tactic.run g (evalTactic (← `(tactic| split))))
  else return none

open Lean Elab Tactic Meta in
/-- `ok_le_walk` walks every goal `OkLe m m'` as far as both sides step alike (see the module docstring), leaving the
goals where they differ otherwise. -/
elab "ok_le_walk" : tactic => do
  let mut todo := (← getGoals)
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← okLeStep g with
      | some new => todo := new ++ rest
      | none => left := left.push g; todo := rest
  setGoals left.toList

end FlockSoundness.Refine
