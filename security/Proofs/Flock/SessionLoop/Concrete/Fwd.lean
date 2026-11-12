module

public import Lean
public meta import Lean

@[expose] public section

/-!
# A forward walk through an executable computation

`fwd_walk [rules] block [consts]` proves `m = r` for a computation `m : Except ε γ`, `r` usually `.ok ?v`, by stepping
through `m` in order: a `let` goes into the context (a value with a free variable of the goal is substituted instead,
after the rules rewrite it), a jump to a join point takes the join point's body, a bind's computation is evaluated (by a
rule, by the kernel's `whnf`, or by walking it), an `if` takes the branch its condition decides, a `match` the branch its
discriminants' values select, and the walk ends at `pure a`, with `?v := a`. Each step's proof is a term the kernel
rechecks; nothing is unfolded into both arms of a branch. A constant in `block`, or one whose definition reads one, is
never handed to the kernel's `whnf`: its application is unfolded and walked instead.
-/

namespace Fwd

open Lean Meta Elab Tactic

universe u v

/-- The step a bind of a computation that returned takes, stated once, so that no proof asks the kernel to unfold
`bind (.ok a) f` against `f a`, which can send it into evaluating `f a`'s own first computation. -/
theorem bind_ok {ε : Type u} {α β : Type v} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- A loop whose body always yields is a fold: the walk proves the body once, over fresh variables. -/
theorem forIn_list_yield {α β ε : Type} (xs : List α) (b : β) (f : α → β → Except ε (ForInStep β)) (g : β → α → β)
    (h : ∀ a r, f a r = .ok (.yield (g r a))) : forIn xs b f = .ok (xs.foldl g b) := by
  induction xs generalizing b with
  | nil => rfl
  | cons x xs ih => rw [List.forIn_cons, h]; exact ih _

theorem forIn_array_yield {α β ε : Type} (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β)) (g : β → α → β)
    (h : ∀ a r, f a r = .ok (.yield (g r a))) : forIn xs b f = .ok (xs.foldl g b) := by
  rw [← Array.forIn_toList, forIn_list_yield _ _ _ g h, Array.foldl_toList]

meta structure Cfg where
  rules : Array AbstractMVarsResult
  blocked : NameSet
  shallow : NameSet
  taint : IO.Ref (NameMap Bool)
  ftaint : IO.Ref (Std.HashMap FVarId Bool)
  trace : Bool

abbrev M := ReaderT Cfg MetaM

meta partial def constTainted (c : Name) : M Bool := do
  let cfg ← read
  if cfg.blocked.contains c then return true
  if let some b := (← cfg.taint.get).find? c then return b
  cfg.taint.modify (·.insert c false)
  let r ← match (← getEnv).find? c with
    | some (.defnInfo d) => d.value.getUsedConstants.anyM constTainted
    | _ => pure false
  cfg.taint.modify (·.insert c r)
  return r

meta partial def exprTainted (e : Expr) : M Bool := do
  let cfg ← read
  if e.getUsedConstants.any cfg.shallow.contains then return true
  if ← e.getUsedConstants.anyM constTainted then return true
  for fv in (collectFVars {} e).fvarIds do
    if let some b := (← cfg.ftaint.get).get? fv then
      if b then return true
    else
      let b ← match ← fv.getValue? with
        | some v => exprTainted (← instantiateMVars v)
        | none => pure false
      cfg.ftaint.modify (·.insert fv b)
      if b then return true
  return false

/-- The first constant that taints `e`, through let-bound variables. -/
meta partial def taintReason (e : Expr) : M (Option Name) := do
  let cfg ← read
  for c in e.getUsedConstants do
    if cfg.shallow.contains c then return some c
    if ← constTainted c then return some c
  for fv in (collectFVars {} e).fvarIds do
    if let some v ← fv.getValue? then
      if let some n ← taintReason (← instantiateMVars v) then return some n
  return none

/-- Whether `e` mentions a variable of the goal (a local without a value). -/
meta def symbolic (e : Expr) : MetaM Bool := do
  let lctx ← getLCtx
  return e.hasAnyFVar fun fv => match lctx.find? fv with
    | some d => !d.isLet
    | none => false

meta def tr (s : String) : MetaM Unit := logInfo s

meta def kwhnf (e : Expr) : MetaM Expr := do
  let e ← instantiateMVars e
  match Kernel.whnf (← getEnv) (← getLCtx) e with
  | .ok r => pure r
  | .error ex => throwError "fwd: the kernel's whnf failed on{indentExpr e}\n{ex.toMessageData (← getOptions)}"

meta def isCtorApp (e : Expr) : MetaM Bool := do
  match e.getAppFn with
  | .const c _ => return (← getEnv).isConstructor c
  | .lit _ => return true
  | _ => return false

meta def hint (e ty : Expr) : MetaM Expr := mkExpectedTypeHint e ty

/-- `s` itself or the value it is let-bound to, through let-bound variables, when that is a constructor application. -/
meta partial def ctorVal? (s : Expr) : MetaM (Option Expr) := do
  let s := (← instantiateMVars s).consumeMData
  if s.isFVar then
    match ← s.fvarId!.getValue? with
    | some v => return ← ctorVal? v
    | none => return none
  match s.getAppFn with
  | .const c lvls =>
    if (← getEnv).isConstructor c then return some s
    if let some (.defnInfo d) := (← getEnv).find? c then
      let b := (d.value.instantiateLevelParams d.levelParams lvls).beta s.getAppArgs
      if let .const c' _ := b.getAppFn then
        if (← getEnv).isConstructor c' then return some b
    return none
  | _ => return none

/-- `e` with each projection of a let-bound constructor application replaced by its field, definitionally. -/
meta def projReduce (e : Expr) : MetaM Expr := do
  if !e.hasFVar then return e
  Meta.transform e (post := fun e => do
    match e with
    | .proj sn i s =>
      if isClass (← getEnv) sn then return .continue
      if let some v ← ctorVal? s then
        let ci ← getConstInfoCtor v.getAppFn.constName!
        if v.getAppNumArgs == ci.numParams + ci.numFields then
          return .visit (v.getArg! (ci.numParams + i))
      return .continue
    | _ =>
      let .const n _ := e.getAppFn | return .continue
      let some info ← getProjectionFnInfo? n | return .continue
      if info.fromClass then return .continue
      let args := e.getAppArgs
      if args.size < info.numParams + 1 then return .continue
      let some v ← ctorVal? args[info.numParams]! | return .continue
      let ci ← getConstInfoCtor v.getAppFn.constName!
      unless v.getAppNumArgs == ci.numParams + ci.numFields do return .continue
      return .visit (mkAppN (v.getArg! (ci.numParams + info.i)) (args.extract (info.numParams + 1) args.size)))

/-- `(e', e = e')` by `projReduce`, `none` if it changes nothing. -/
meta def projStep (e : Expr) : MetaM (Option (Expr × Expr)) := do
  let e' ← projReduce e
  if e' == e then return none
  return some (e', ← hint (← mkEqRefl e) (← mkEq e e'))

meta def okArg? (v : Expr) : Option Expr :=
  if v.isAppOfArity ``Except.ok 3 then some v.appArg! else none

/-- A rule's left side `l` matches `e`, of the same head: argument by argument, reducibly, else (an argument with no
metavariable left, neither side tainted) by the kernel. -/
meta def matchArgs (l e : Expr) : M Bool := do
  if ← withReducible (isDefEq l e) then return true
  let la := l.getAppArgs
  let ea := e.getAppArgs
  unless la.size == ea.size do return false
  for a in la, b in ea do
    if ← withReducible (isDefEq a b) then continue
    let a ← instantiateMVars a
    let b ← instantiateMVars b
    if a.hasMVar || (← exprTainted a) || (← exprTainted b) then return false
    if (← read).trace then tr s!"fwd: the kernel compares an argument of {← a.numObjs} objects with one of {← b.numObjs}"
    unless Kernel.isDefEqGuarded (← getEnv) (← getLCtx) a b do return false
  return true

mutual

/-- Close a side condition of a rule: an assumption, an equation the kernel checks (after the rules rewrite both
sides), or a decidable proposition the kernel decides. -/
meta partial def closeSide (mv : MVarId) : M Bool := do
  let t ← instantiateMVars (← mv.getType)
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    if ← withReducible (isDefEq d.type t) then
      mv.assign d.toExpr
      return true
  let kdefeq (a b : Expr) : M Bool := do
    if (← exprTainted a) || (← exprTainted b) then return false
    return Kernel.isDefEqGuarded (← getEnv) (← getLCtx) a b
  if let some (_, a, b) := t.eq? then
    if !a.hasMVar && !b.hasMVar then
      if ← kdefeq a b then
        mv.assign (← hint (← mkEqRefl a) t)
        return true
      let (a', pa) ← match ← rewriteIn a with
        | some (a', p) => pure (a', p)
        | none => pure (a, ← mkEqRefl a)
      let (b', pb) ← match ← rewriteIn b with
        | some (b', p) => pure (b', p)
        | none => pure (b, ← mkEqRefl b)
      if ← kdefeq a' b' then
        mv.assign (← mkEqTrans pa (← mkEqSymm pb))
        return true
  if !t.hasMVar && !(← exprTainted t) then
    if let some inst ← (synthInstance? (mkApp (mkConst ``Decidable) t)) then
      let d ← kwhnf (mkApp2 (mkConst ``Decidable.decide) t inst)
      if d.isConstOf ``Bool.true then
        mv.assign (mkApp3 (mkConst ``of_decide_eq_true) t inst (← mkEqRefl (mkConst ``Bool.true)))
        return true
  return false

/-- Discharge a rule's arguments after its left side matched: a side condition by `closeSide`, an equation of a
computation by walking it (which assigns the arguments its right side names), in the order they are bound. -/
meta partial def discharge (mvs : Array Expr) : M Bool := do
  for mv in mvs do
    let m := mv.mvarId!
    if ← m.isAssigned then continue
    let t ← instantiateMVars (← m.getType)
    if !(← isProp t) then continue
    if ← closeSide m then continue
    if let some (T, _, _) := t.eq? then
      let Tw ← whnfR T
      if Tw.isAppOfArity ``Except 2 then
        let g ← mkFreshExprSyntheticOpaqueMVar t
        walk g.mvarId!
        m.assign (← instantiateMVars g)
        continue
    if (← read).trace then tr s!"fwd: cannot discharge {((toString (← ppExpr t)).take 400).toString}"
    return false
  for mv in mvs do
    unless ← mv.mvarId!.isAssigned do return false
  return true

/-- A rule's instance at `e`: `(r, p : e = r)`. -/
meta partial def ruleAt (e : Expr) : M (Option (Expr × Expr)) := do
  for ar in (← read).rules do
    let s : Meta.SavedState ← Meta.saveState
    try
      let (_, _, p) ← openAbstractMVarsResult ar
      let (mvs, _, ty) ← forallMetaTelescopeReducing (← inferType p)
      let some (_, l, r) := ty.eq? | do s.restore; continue
      unless l.getAppFn.isConst && e.getAppFn.isConst && l.getAppFn.constName! == e.getAppFn.constName! do
        s.restore; continue
      if (← read).trace then tr s!"fwd: a rule on {l.getAppFn} is tried"
      if ← matchArgs l e then
        if (← read).trace then tr s!"fwd: its left side matches"
        if ← discharge mvs then return some (← instantiateMVars r, ← instantiateMVars (mkAppN p mvs))
      if (← read).trace then tr s!"fwd: it does not apply"
      s.restore
    catch ex =>
      if (← read).trace then tr s!"fwd: a rule failed: {((toString (← ex.toMessageData.toString)).take 300).toString}"
      s.restore
  return none

/-- Rewrite `e` with the rules wherever they apply: `(e', p : e = e')`, `none` if none applies. -/
meta partial def rewriteIn (e : Expr) : M (Option (Expr × Expr)) := do
  let mut e := e
  let mut pf : Option Expr := none
  let mut progress := true
  let mut n := 0
  while progress && n < 64 do
    progress := false
    n := n + 1
    for ar in (← read).rules do
      let s : Meta.SavedState ← Meta.saveState
      try
        let (_, _, p) ← openAbstractMVarsResult ar
        let (mvs, _, ty) ← forallMetaTelescopeReducing (← inferType p)
        let some (_, l, r) := ty.eq? | do s.restore; continue
        let ea ← withReducible (kabstract e l)
        if !ea.hasLooseBVars then
          s.restore; continue
        unless ← discharge mvs do s.restore; continue
        let r ← instantiateMVars r
        let h ← instantiateMVars (mkAppN p mvs)
        let ea ← instantiateMVars ea
        let eNew := ea.instantiate1 r
        let motive := Expr.lam `y (← inferType r) ea .default
        let st ← mkCongrArg motive h
        pf := some (← match pf with | some q => mkEqTrans q st | none => pure st)
        e := eNew
        progress := true
        break
      catch ex =>
        if (← read).trace then tr s!"fwd: a rewrite failed: {((toString (← ex.toMessageData.toString)).take 300).toString}"
        s.restore
  return pf.map (e, ·)

/-- A `forIn` over a list or an array whose body, walked once over fresh variables (a pair's two fields), yields: its
value is the fold of what the body yields (`forIn_list_yield`), with no iteration. `none` when the body does not walk
to a `yield`. -/
meta partial def summarize (x : Expr) : M (Option (Expr × Expr)) := do
  let args := x.getAppArgs
  let (ρ, α, β, xs, b, f) := (args[1]!, args[2]!, args[4]!, args[5]!, args[6]!, args[7]!)
  let ρw ← whnfR ρ
  let lem ← if ρw.isAppOf ``List then pure ``Fwd.forIn_list_yield
    else if ρw.isAppOf ``Array then pure ``Fwd.forIn_array_yield else return none
  let αw ← whnfR α
  let s : Meta.SavedState ← Meta.saveState
  try
    let r ← withLocalDeclD `r β fun r => do
      let bodyAt (a : Expr) : M (Expr × Expr) := do
        let lhs := mkApp2 f a r
        let tyW ← whnfR (← inferType lhs)
        unless tyW.isAppOfArity ``Except 2 do throwError "fwd: the loop body is not an Except computation"
        let w ← mkFreshExprMVar tyW.appArg!
        let okW := mkApp3 (mkConst ``Except.ok tyW.getAppFn.constLevels!) tyW.appFn!.appArg! tyW.appArg! w
        let g ← mkFreshExprSyntheticOpaqueMVar (← mkEq lhs okW)
        walk g.mvarId!
        let w ← instantiateMVars w
        unless w.isAppOfArity ``ForInStep.yield 2 do throwError "fwd: the loop body does not yield"
        return (w.appArg!, ← instantiateMVars g)
      if αw.isAppOfArity ``Prod 2 then
        withLocalDeclD `a1 αw.appFn!.appArg! fun a1 => withLocalDeclD `a2 αw.appArg! fun a2 => do
          let (t, pf) ← bodyAt (← mkAppM ``Prod.mk #[a1, a2])
          let tL ← mkLambdaFVars #[a1, a2] t
          let pL ← mkLambdaFVars #[a1, a2, r] pf
          let G ← withLocalDeclD `x α fun x => do
            mkLambdaFVars #[r, x] (mkApp2 tL (← mkAppM ``Prod.fst #[x]) (← mkAppM ``Prod.snd #[x])).headBeta
          let P ← withLocalDeclD `x α fun x => withLocalDeclD `r' β fun r' => do
            mkLambdaFVars #[x, r'] (mkApp3 pL (← mkAppM ``Prod.fst #[x]) (← mkAppM ``Prod.snd #[x]) r')
          pure (G, P)
      else
        withLocalDeclD `a α fun a => do
          let (t, pf) ← bodyAt a
          pure (← mkLambdaFVars #[r, a] t, ← mkLambdaFVars #[a, r] pf)
    let (G, P) := r
    let pr ← mkAppM lem #[xs, b, f, G, P]
    let some (_, lhsL, rhsL) := (← inferType pr).eq? | throwError "fwd: summarize"
    unless ← isDefEq lhsL x do throwError "fwd: the summarized loop is not the loop"
    if (← read).trace then tr "fwd: a loop summarized as a fold"
    return some (rhsL, ← hint pr (← mkEq x rhsL))
  catch ex =>
    if (← read).trace then tr s!"fwd: a loop not summarized: {((toString (← ex.toMessageData.toString)).take 300).toString}"
    s.restore
    return none

/-- `x`'s value, a constructor application, with `x = v`. -/
meta partial def evalVal (x : Expr) : M (Expr × Expr) := do
  let x ← instantiateMVars x
  if let some (v, p) ← ruleAt x then
    if ← isCtorApp v then return (v, p)
    let (w, q) ← evalVal v
    return (w, ← mkEqTrans p q)
  let xt ← whnfR (← inferType x)
  let walkIt (_ : Unit) : M (Expr × Expr) := do
    let a ← mkFreshExprMVar xt.appArg!
    let okA := mkApp3 (mkConst ``Except.ok xt.getAppFn.constLevels!) xt.appFn!.appArg! xt.appArg! a
    let g ← mkFreshExprSyntheticOpaqueMVar (← mkEq x okA)
    walk g.mvarId!
    return (← instantiateMVars okA, ← instantiateMVars g)
  let isExcept := xt.isAppOfArity ``Except 2
  let x' := x.headBeta.consumeMData
  if isExcept && x'.isAppOfArity ``ForIn.forIn 8 then
    if let some r ← summarize x' then return r
  let jump ← if x'.getAppFn.isFVar then pure (← x'.getAppFn.fvarId!.getValue?).isSome else pure false
  let structural := x'.isLet || x'.isAppOfArity ``Bind.bind 6 || x'.isAppOfArity ``ite 5 || x'.isAppOfArity ``dite 5 ||
    (← isMatcherApp x') || jump
  if isExcept && structural && (← symbolic x) then return ← walkIt ()
  if let some (x2, p) ← projStep x then
    let (w, q) ← evalVal x2
    return (w, ← mkEqTrans p q)
  if !(← exprTainted x) then
    let v ← kwhnf x
    if ← isCtorApp v then return (v, ← hint (← mkEqRefl x) (← mkEq x v))
  if let some (x2, p) ← rewriteIn x then
    let (w, q) ← evalVal x2
    return (w, ← mkEqTrans p q)
  if isExcept then return ← walkIt ()
  throwError "fwd: no value for{indentExpr x}"

/-- Walk the goal `m = r` to its end. -/
meta partial def walk (g0 : MVarId) : M Unit := do
  let mut g := g0
  let mut fuel := 100000
  while fuel > 0 do
    fuel := fuel - 1
    match ← g.withContext (step g) with
    | none => return
    | some g' => g := g'
  throwError "fwd: out of steps"

meta partial def step (g : MVarId) : M (Option MVarId) := do
  let ty ← instantiateMVars (← g.getType)
  let some (T, lhs0, rhs) := ty.eq? | throwError "fwd: the goal is not an equation{indentExpr ty}"
  let lvl := ty.getAppFn.constLevels![0]!
  let eqT := fun (a b : Expr) => mkApp3 (mkConst ``Eq [lvl]) T a b
  let lhs := lhs0.headBeta.consumeMData
  let next (pf : Expr) (lhs' : Expr) : MetaM (Option MVarId) := do
    let q ← mkFreshExprSyntheticOpaqueMVar (eqT lhs' rhs)
    g.assign (mkApp6 (mkConst ``Eq.trans [lvl]) T lhs lhs' rhs pf q)
    return some q.mvarId!
  if (← read).trace then
    let t ← if lhs.approxDepth.toNat < 12 then pure (toString (← ppExpr lhs)) else pure s!"{lhs.getAppFn} …"
    tr s!"fwd step: {(t.take 300).toString}"
  if let some (v, p) ← ruleAt lhs then
    return ← next p v
  if let .letE n t v b _ := lhs then
    let v ← instantiateMVars v
    if v.consumeMData.isLambda then
      let g1 ← g.replaceTargetDefEq (.letE n t v (eqT b rhs) false)
      let (_, g2) ← g1.intro1P
      return some g2
    let ctor ← isCtorApp v.consumeMData
    let (v', pv) ← if ctor then pure (v, none) else match ← rewriteIn v with
      | some (v', p) => pure (v', some p)
      | none => pure (v, none)
    if !ctor && ((← symbolic v') || (← exprTainted v')) then
      let body := b.instantiate1 v'
      match pv with
      | some p =>
        let motive := Expr.lam n t b .default
        let c ← mkCongrArg motive p
        return ← next c body
      | none =>
        return some (← g.replaceTargetDefEq (eqT body rhs))
    else
      let tgt := Expr.letE n t v' (eqT b rhs) false
      let g1 ← match pv with
        | some p =>
          let motive := Expr.lam n t b .default
          let c ← mkCongrArg motive p
          let q ← mkFreshExprSyntheticOpaqueMVar tgt
          g.assign (mkApp6 (mkConst ``Eq.trans [lvl]) T lhs (b.instantiate1 v') rhs c q)
          pure q.mvarId!
        | none => g.replaceTargetDefEq tgt
      let (_, g2) ← g1.intro1P
      return some g2
  let fn := lhs.getAppFn
  if fn.isFVar then
    if let some v ← fn.fvarId!.getValue? then
      return some (← g.replaceTargetDefEq (eqT (v.betaRev lhs.getAppRevArgs) rhs))
  if lhs.isAppOfArity ``Bind.bind 6 then
    let x := lhs.getArg! 4
    let f := lhs.getArg! 5
    if (← read).trace then tr s!"fwd: a bind of {x.headBeta.getAppFn} with {x.headBeta.getAppNumArgs} args"
    let (v, hx) ← evalVal x
    let some a := okArg? v | throwError "fwd: the bound computation{indentExpr x}\nevaluates to{indentExpr v}"
    let (v, hx, a) ← match ← rewriteIn a with
      | some (a', p) => pure (mkApp v.appFn! a', ← mkEqTrans hx (← mkCongrArg v.appFn! p), a')
      | none => pure (v, hx, a)
    let args := lhs.getAppArgs
    let mid := mkAppN fn (args.set! 4 v)
    let motive ← withLocalDeclD `y (← inferType x) fun y => mkLambdaFVars #[y] (mkAppN fn (args.set! 4 y))
    let c ← mkCongrArg motive hx
    let fa := f.beta #[a]
    let q ← mkFreshExprSyntheticOpaqueMVar (eqT fa rhs)
    match ← (try some <$> mkAppM ``Fwd.bind_ok #[a, f] catch _ => pure none) with
    | some bo =>
      let q2 := mkApp6 (mkConst ``Eq.trans [lvl]) T mid fa rhs bo q
      g.assign (mkApp6 (mkConst ``Eq.trans [lvl]) T lhs mid rhs c q2)
    | none =>
      if (← read).trace then tr "fwd: bind_ok does not apply"
      g.assign (mkApp6 (mkConst ``Eq.trans [lvl]) T lhs mid rhs c q)
    return some q.mvarId!
  if lhs.isAppOfArity ``ite 5 || lhs.isAppOfArity ``dite 5 then
    let c := lhs.getArg! 1
    let inst := lhs.getArg! 2
    let (pos, hc) ← decideCond c inst
    let lα := fn.constLevels!
    let α := lhs.getArg! 0
    let t := lhs.getArg! 3
    let e := lhs.getArg! 4
    if lhs.isAppOfArity ``ite 5 then
      if pos then return ← next (mkAppN (mkConst ``if_pos lα) #[c, inst, hc, α, t, e]) t
      else return ← next (mkAppN (mkConst ``if_neg lα) #[c, inst, hc, α, t, e]) e
    else
      if pos then return ← next (mkAppN (mkConst ``dif_pos lα) #[c, inst, hc, α, t, e]) (t.beta #[hc])
      else return ← next (mkAppN (mkConst ``dif_neg lα) #[c, inst, hc, α, t, e]) (e.beta #[hc])
  if let some mi ← matchMatcherApp? lhs then
    let mut mi := mi
    let mut cur := lhs
    let mut pf : Option Expr := none
    for i in [0:mi.discrs.size] do
      let d ← instantiateMVars mi.discrs[i]!
      unless ← isCtorApp d do
        let (v, hd) ← evalVal d
        let m0 := mi
        let motive ← withLocalDeclD `y (← inferType d) fun y =>
          mkLambdaFVars #[y] ({ m0 with discrs := m0.discrs.set! i y }).toExpr
        let c ← mkCongrArg motive hd
        mi := { mi with discrs := mi.discrs.set! i v }
        cur := mi.toExpr
        pf := some (← match pf with | some q => mkEqTrans q c | none => pure c)
    match ← reduceMatcher? cur with
    | .reduced e =>
      match pf with
      | some p => return ← next p e
      | none =>
        return some (← g.replaceTargetDefEq (eqT e rhs))
    | _ => throwError "fwd: the match does not reduce{indentExpr cur}"
  if lhs.isAppOfArity ``Pure.pure 4 || lhs.isAppOfArity ``Except.ok 3 || lhs.isAppOfArity ``Except.pure 3 then
    if let some (a', p) ← rewriteIn (← instantiateMVars lhs.appArg!) then
      return ← next (← mkCongrArg lhs.appFn! p) (mkApp lhs.appFn! a')
    let tt ← whnfR T
    let a ← zetaReduce (← instantiateMVars lhs.appArg!)
    if (← read).trace then tr s!"fwd: returns a value of {← a.numObjs} objects"
    let okA := mkApp3 (mkConst ``Except.ok tt.getAppFn.constLevels!) tt.appFn!.appArg! tt.appArg! a
    if rhs.hasMVar then
      unless ← withAssignableSyntheticOpaque (isDefEq rhs okA) do
        throwError "fwd: returns{indentExpr a}\nnot{indentExpr rhs}"
    else
      unless Kernel.isDefEqGuarded (← getEnv) (← getLCtx) okA rhs do
        throwError "fwd: returns{indentExpr a}\nnot{indentExpr rhs}"
    g.assign (← hint (← mkEqRefl lhs) (eqT lhs rhs))
    return none
  if lhs.isAppOfArity ``MonadExcept.throw 5 || lhs.isAppOfArity ``Except.error 3 then
    throwError "fwd: reached a throw{indentExpr lhs.appArg!}"
  if ← isCtorApp lhs then
    let a ← zetaReduce (← instantiateMVars lhs)
    unless ← withAssignableSyntheticOpaque (isDefEq rhs a) do
      throwError "fwd: ends at{indentExpr a}\nnot{indentExpr rhs}"
    g.assign (← hint (← mkEqRefl lhs) (eqT lhs rhs))
    return none
  if !(← exprTainted lhs) then
    let v ← kwhnf lhs
    if okArg? v |>.isSome then
      let tt ← whnfR T
      if rhs.hasMVar then
        unless ← withAssignableSyntheticOpaque (isDefEq rhs (← zetaReduce v)) do throwError "fwd: returns{indentExpr v}\nnot{indentExpr rhs}"
      g.assign (← hint (← mkEqRefl lhs) (eqT lhs rhs))
      let _ := tt
      return none
    if v.isAppOfArity ``Except.error 3 then
      throwError "fwd: evaluates to an error{indentExpr v}"
  if let some e' ← withTransparency .all (unfoldDefinition? lhs) then
    return some (← g.replaceTargetDefEq (eqT e' rhs))
  let why ← taintReason lhs
  throwError "fwd: stuck at{indentExpr lhs}\ntainted by {why}"

/-- Decide `c`: a hypothesis, a rule, or the kernel. -/
meta partial def decideCond (c inst : Expr) : M (Bool × Expr) := do
  let c ← instantiateMVars c
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    if ← withReducible (isDefEq d.type c) then return (true, d.toExpr)
    if ← withReducible (isDefEq d.type (mkNot c)) then return (false, d.toExpr)
  let attempt (c' : Expr) (pc : Option Expr) : M (Option (Bool × Expr)) := do
    if ← exprTainted c' then return none
    let inst' ← if pc.isSome then synthInstance (mkApp (mkConst ``Decidable) c') else pure inst
    let d ← kwhnf (mkApp2 (mkConst ``Decidable.decide) c' inst')
    let fix (h : Expr) (pos : Bool) : MetaM Expr := do
      match pc with
      | none => pure h
      | some p => if pos then mkEqMPR p h else mkEqMPR (← mkCongrArg (mkConst ``Not) p) h
    if d.isConstOf ``Bool.true then
      return some (true, ← fix (mkApp3 (mkConst ``of_decide_eq_true) c' inst' (← mkEqRefl (mkConst ``Bool.true))) true)
    if d.isConstOf ``Bool.false then
      return some (false, ← fix (mkApp3 (mkConst ``of_decide_eq_false) c' inst' (← mkEqRefl (mkConst ``Bool.false))) false)
    if (← read).trace then
      tr s!"fwd: the condition is stuck {((toString (← ppExpr c')).take 400).toString}"
      tr s!"fwd: its whnf is {d.getAppFn} with {d.getAppNumArgs} args: {((toString (← ppExpr d)).take 1500).toString}"
      let mut x := d
      for _ in [0:6] do
        if x.isApp && x.getAppFn.isConst && x.getAppFn.constName!.isStr && x.getAppFn.constName!.getString! == "rec" then
          let mj ← kwhnf x.appArg!
          tr s!"fwd: its major whnf is {mj.getAppFn} with {mj.getAppNumArgs} args: {((toString (← ppExpr mj)).take 1500).toString}"
          x := mj
        else break
    return none
  let mut cur := c
  let mut pc : Option Expr := none
  for _ in [0:8] do
    if let some r ← attempt cur pc then return r
    let s ← match ← projStep cur with
      | some s => pure (some s)
      | none => rewriteIn cur
    let some (c2, p) := s | break
    pc := some (← match pc with | some q => mkEqTrans q p | none => pure p)
    cur := c2
  throwError "fwd: the condition is stuck{indentExpr cur}"

end

end Fwd

namespace Fwd

open Lean Meta Elab Term in
meta def mkCfg (rs : Array Term) (bs ss : Array Ident) (trace : Bool) : TermElabM Cfg := do
  let mut es := #[]
  for r in rs do
    let c? ← if r.raw.isIdent then
        try some <$> realizeGlobalConstNoOverloadWithInfo r.raw catch _ => pure none
      else pure none
    match c? with
    | some c => es := es.push (← mkConstWithFreshMVarLevels c)
    | none => es := es.push (← elabTerm r none)
  synthesizeSyntheticMVarsNoPostponing
  let mut rules := #[]
  for e in es do rules := rules.push (← abstractMVars (← instantiateMVars e))
  if trace then tr "fwd: rules elaborated"
  let mut blocked : NameSet := {}
  for b in bs do blocked := blocked.insert (← realizeGlobalConstNoOverloadWithInfo b)
  let mut shallow : NameSet := {}
  for b in ss do shallow := shallow.insert (← realizeGlobalConstNoOverloadWithInfo b)
  if trace then tr s!"fwd: shallow {shallow.toList}"
  return { rules, blocked, shallow, taint := ← IO.mkRef {}, ftaint := ← IO.mkRef {}, trace }

end Fwd

open Lean Meta Elab Tactic in
/-- `fwd_walk [h, …] block [c, …] shallow [c, …]` (see the module docstring). -/
elab "fwd_walk" " [" rs:term,* "]" " block" " [" bs:ident,* "]" " shallow" " [" ss:ident,* "]" tr:(" trace")? :
    tactic => do
  let g ← getMainGoal
  g.withContext do
    let cfg ← Fwd.mkCfg rs.getElems bs.getElems ss.getElems tr.isSome
    (Fwd.walk g).run cfg
    if tr.isSome then
      Fwd.tr "fwd: walk done"
      let pf ← instantiateMVars (mkMVar g)
      Fwd.tr s!"fwd: proof instantiated, {← pf.numObjs} objects"
  replaceMainGoal []

open Lean Meta Elab Term Command in
/-- `fwd_def V T binders : m by [h, …] block [c, …] shallow [c, …]` walks `m = .ok ?v` and adds `V`, the value `?v` as a
function of the binders that are data, and the theorem `T : ∀ binders, m = .ok (V data)`. -/
elab "fwd_def " vn:ident tn:ident bs:bracketedBinder* " : " m:term " by" " [" rs:term,* "]" " block" " [" bl:ident,* "]"
    " shallow" " [" ss:ident,* "]" tr:(" trace")? : command => do
  let ns ← getCurrNamespace
  let vName := ns ++ vn.getId
  let tName := ns ++ tn.getId
  liftTermElabM do
    elabBinders bs fun xs => do
      let mE ← elabTerm m none
      synthesizeSyntheticMVarsNoPostponing
      let mE ← instantiateMVars mE
      let T ← whnfR (← inferType mE)
      unless T.isAppOfArity ``Except 2 do throwError "fwd_def: {mE} is not an Except computation"
      let lv := T.getAppFn.constLevels!
      let ε := T.appFn!.appArg!
      let γ := T.appArg!
      let vM ← mkFreshExprMVar γ
      let g ← mkFreshExprSyntheticOpaqueMVar (← mkEq mE (mkApp3 (mkConst ``Except.ok lv) ε γ vM))
      let cfg ← Fwd.mkCfg rs.getElems bl.getElems ss.getElems tr.isSome
      (Fwd.walk g.mvarId!).run cfg
      let v ← instantiateMVars vM
      let pf ← instantiateMVars g
      if v.hasMVar || pf.hasMVar then throwError "fwd_def: the walk left metavariables"
      let mut data := #[]
      for x in xs do
        if ← isProp (← inferType x) then
          if v.containsFVar x.fvarId! then throwError "fwd_def: the value reads the hypothesis {x}"
        else data := data.push x
      let vTy ← instantiateMVars (← mkForallFVars data γ)
      let vVal ← instantiateMVars (← mkLambdaFVars data v)
      if vTy.hasMVar || vVal.hasMVar then throwError "fwd_def: the value's type or body has metavariables"
      addDecl (.defnDecl (mkDefinitionValEx vName [] vTy vVal ReducibilityHints.abbrev DefinitionSafety.safe [vName])) (forceExpose := true)
      let rhs := mkApp3 (mkConst ``Except.ok lv) ε γ (mkAppN (mkConst vName) data)
      let tTy ← instantiateMVars (← mkForallFVars xs (← mkEq mE rhs))
      let tVal ← instantiateMVars (← mkLambdaFVars xs pf)
      if tTy.hasMVar then throwError "fwd_def: the theorem's type has metavariables{indentExpr tTy}"
      if tVal.hasMVar then throwError "fwd_def: the proof has metavariables"
      addDecl (.thmDecl (mkTheoremValEx tName [] tTy tVal [tName]))
      if tr.isSome then Fwd.tr s!"fwd_def: added {vName} and {tName}"

open Lean Meta Elab Tactic in
/-- `kernel_rfl`, on a goal a `simp` left under metadata: closes `a = b` with `Eq.refl a` and leaves `a ≡ b` to the
kernel. -/
elab "kernel_rfl'" : tactic => do
  let g ← getMainGoal
  let some (_, a, _) := (← instantiateMVars (← g.getType)).consumeMData.eq? | throwError "kernel_rfl': not an equation"
  g.assign (← mkEqRefl a)
  replaceMainGoal []

open Lean Meta Elab Tactic in
/-- On `ψ (match d with …) = match d with …` (or `ψ (if c then …) = if c then …`), case `d` (or `c`) once for both sides. -/
elab "case_both" : tactic => withMainContext do
  let g ← getMainGoal
  let t ← instantiateMVars (← g.getType)
  let some (_, lhs, _) := t.eq? | throwError "case_both: not an equation"
  let x := lhs.appArg!
  if x.isAppOfArity ``ite 5 then
    let c := x.getArg! 1
    let cs ← `(term| $(← Term.exprToSyntax c))
    evalTactic (← `(tactic| by_cases hc : $cs <;> first | simp only [if_pos hc] | simp only [if_neg hc]))
    return
  let some app ← matchMatcherApp? x | throwError "case_both: no match under the map"
  let d := app.discrs[0]!
  let d ← instantiateMVars d
  let target ← if d.isFVar then pure d else
    match d.getAppFn.constName? with
    | some n =>
      if (← getEnv).isConstructor n then
        match d.getAppArgs.reverse.find? (·.isFVar) with
        | some a => pure a
        | none => throwError "case_both: the discriminant is a closed constructor"
      else pure d
    | none => pure d
  if target.isFVar then
    let gs ← g.cases target.fvarId!
    replaceMainGoal (gs.map (·.mvarId)).toList
    evalTactic (← `(tactic| all_goals try dsimp only))
  else
    let (fvs, g') ← g.generalize #[{ expr := target, xName? := `z }]
    let gs ← g'.cases fvs[0]!
    replaceMainGoal (gs.map (·.mvarId)).toList
    evalTactic (← `(tactic| all_goals try dsimp only))
