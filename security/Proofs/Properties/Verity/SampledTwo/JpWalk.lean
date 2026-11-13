module

public import Proofs.Flock.SpotFlock.HeldOf
public import Lean
public meta import Lean

/-!
# Walking a registration check's `do` block

`Registration.check` and `checkRegistered` are each one `do` block whose `if`s and `match`es each leave a join point, a
function of the codes so far, for the rest of the block. `jp_mono` proves once per join point, deepest first, that it
keeps a nonempty code list nonempty (`jp_body`: down the join point's spine, an `if` or a `match` split, a jump the
proof of the join point jumped to, a loop by a step that keeps a nonempty list nonempty); `jp_walk` then follows the one
path on which no code is added, unfolding each join point once. So a proof is linear in the block, where unfolding the
block into both arms of each branch doubles at every one.
-/

@[expose] public section

namespace SpotFlock.RecordNames

open Lean (Json)
open Spot.Registration

/-! ## Code lists -/

theorem isEmpty_addCode (c : Array String) (x : String) : (Spot.addCode c x).isEmpty = false := by
  unfold Spot.addCode
  split
  · rename_i h
    cases c using Array.rec with
    | mk l => cases l <;> simp_all
  · simp

theorem isEmpty_kept (b : Bool) (c : Array String) (x : String) : (kept b c x).isEmpty = (b && c.isEmpty) := by
  unfold kept
  cases b <;> simp [isEmpty_addCode]

theorem isEmpty_forIn_list {α : Type} (l : List α) (c : Array String)
    (f : α → Array String → Id (ForInStep (Array String))) (hc : c.isEmpty = false)
    (hf : ∀ x d, d.isEmpty = false → (f x d).value.isEmpty = false) :
    Array.isEmpty (forIn (m := Id) l c f) = false := by
  induction l generalizing c with
  | nil => exact hc
  | cons a l ih =>
    have e1 := hf a c hc
    simp only [List.forIn_cons]
    cases h : f a c with
    | done d => rw [h] at e1; exact e1
    | yield d => rw [h] at e1; exact ih d e1

theorem isEmpty_forIn {α : Type} (xs : Array α) (c : Array String)
    (f : α → Array String → Id (ForInStep (Array String))) (hc : c.isEmpty = false)
    (hf : ∀ x d, d.isEmpty = false → (f x d).value.isEmpty = false) :
    Array.isEmpty (forIn (m := Id) xs c f) = false := by
  rw [← Array.forIn_toList]
  exact isEmpty_forIn_list _ c f hc hf

theorem isEmpty_singleton (x : String) : (#[x] : Array String).isEmpty = false := rfl

theorem ite_false_eq_true (p : Prop) [Decidable p] (x : Bool) :
    ((if p then false else x) = true) ↔ (¬p ∧ x = true) := by
  by_cases h : p <;> simp [h]

theorem ite_eq_true_false (p : Prop) [Decidable p] (x : Bool) :
    ((if p then x else false) = true) ↔ (p ∧ x = true) := by
  by_cases h : p <;> simp [h]

theorem isEmpty_kept_of (b : Bool) (c : Array String) (x : String) (hc : c.isEmpty = false) :
    (kept b c x).isEmpty = false := by
  rw [isEmpty_kept, hc, Bool.and_false]

theorem isEmpty_ite {p : Prop} [Decidable p] {a b : Array String} (ha : a.isEmpty = false)
    (hb : b.isEmpty = false) : (if p then a else b).isEmpty = false := by
  split <;> assumption

theorem value_ite {p : Prop} [Decidable p] {a b : ForInStep (Array String)} (ha : a.value.isEmpty = false)
    (hb : b.value.isEmpty = false) : (if p then a else b).value.isEmpty = false := by
  split <;> assumption

theorem value_yield {c : Array String} (hc : c.isEmpty = false) : (ForInStep.yield c).value.isEmpty = false := hc

theorem value_done {c : Array String} (hc : c.isEmpty = false) : (ForInStep.done c).value.isEmpty = false := hc


theorem foldl_inv {α β : Type} (P : β → Prop) (f : β → α → β) (hf : ∀ b a, P b → P (f b a)) :
    ∀ (l : List α) (b : β), P b → P (l.foldl f b)
  | [], _, h => h
  | a :: l, b, h => foldl_inv P f hf l (f b a) (hf b a h)

/-! ## Walking a `do` block's join points -/

open Lean Meta in
/-- `some false` at a join point's type, `Unit → Array String → Id (Array String)`, and `some true` at a loop
body's, `Unit → Array String → Id (ForInStep (Array String))`. -/
meta def jpKind? (ty : Expr) : MetaM (Option Bool) := do
  let arr := mkApp (mkConst ``Array [0]) (mkConst ``String)
  let ty₁ ← mkArrow (mkConst ``Unit) (← mkArrow arr (mkApp (mkConst ``Id [0]) arr))
  if ← isDefEq ty ty₁ then return some false
  let ty₂ ← mkArrow (mkConst ``Unit)
    (← mkArrow arr (mkApp (mkConst ``Id [0]) (mkApp (mkConst ``ForInStep [0]) arr)))
  if ← isDefEq ty ty₂ then return some true
  return none

open Lean Meta Elab Tactic in
/-- Names the context's `let`s: each `do` join point `jp0, jp1, …` and each code list `cv0, cv1, …`, in context
order, and every other `l_<its name>`. Run again after more `let`s come into the context, it keeps the names it gave. -/
elab "name_lets" : tactic => withMainContext do
  let mut g ← getMainGoal
  let mut k := 0
  let mut j := 0
  let arr := mkApp (mkConst ``Array [0]) (mkConst ``String)
  for decl in (← getLCtx) do
    if decl.isLet && !decl.isImplementationDetail then
      let n := decl.userName.eraseMacroScopes
      if n == `__do_jp || (n.toString.startsWith "jp" && (← jpKind? decl.type).isSome) then
        g ← g.rename decl.fvarId (Name.mkSimple s!"jp{k}")
        k := k + 1
      else if ← isDefEq decl.type arr then
        g ← g.rename decl.fvarId (Name.mkSimple s!"cv{j}")
        j := j + 1
      else if !n.toString.startsWith "l_" then
        g ← g.rename decl.fvarId (Name.mkSimple s!"l_{n}")
  replaceMainGoal [g]

open Lean Meta Elab Tactic in
/-- The context's `let`s whose name starts with `pfx`, as simp lemmas, in context order; with `jp`, only join points. -/
meta def letLemmas (pfx : String) (jp : Bool := false) : TacticM (Array Ident) := withMainContext do
  let mut out := #[]
  for decl in (← getLCtx) do
    if decl.isLet && decl.userName.toString.startsWith pfx then
      if !jp || (← jpKind? decl.type).isSome then out := out.push (mkIdent decl.userName)
  return out

open Lean Meta Elab Tactic in
/-- The context's hypotheses whose name starts with `pfx`, in context order: a join point's `mjpK` stays one after the
join point is cleared of its value. -/
meta def hypsWith (pfx : String) : TacticM (Array Ident) := withMainContext do
  let mut out := #[]
  for decl in (← getLCtx) do
    if !decl.isLet && !decl.isImplementationDetail && decl.userName.toString.startsWith pfx then
      out := out.push (mkIdent decl.userName)
  return out

open Lean Meta Elab Tactic in
/-- The main goal's hypothesis named `n`. -/
meta def findHyp (n : Name) : TacticM (Option LocalDecl) := withMainContext do
  return (← getLCtx).findFromUserName? n

/-- Closes `(…).isEmpty = false`, a code list's being nonempty, by its spine (`jp_body`). -/
syntax "jp_disch" : tactic
/-- `jp_disch`'s lemmas about the helpers a block calls; a file adds its own with `macro_rules`. -/
syntax "jp_ext" : tactic
/-- `(…).isEmpty = false` or `(ForInStep.value …).isEmpty = false`, down its spine, linearly: an `if` or a `match`
splits, a join point jumped to is its `mjp`, a loop is `isEmpty_forIn` over a step that keeps a nonempty list nonempty,
and a `let` comes into the context, its join points each proved once (`jp_mono`); `xs` are further simp lemmas. -/
syntax "jp_body" " [" Lean.Parser.Tactic.simpLemma,* "]" : tactic

macro_rules
  | `(tactic| jp_ext) => `(tactic| fail "jp_ext: no helper lemma applies")

macro_rules
  | `(tactic| jp_disch) => `(tactic| first | assumption | jp_body [])

/-- `simp` over code lists: in the goal without `zeta`, and at `h` pushing `isEmpty` into branches, a branch that adds
a code nonempty (`jp_disch`), and `ts`. -/
syntax "jp_core" " [" Lean.Parser.Tactic.simpLemma,* "]" (" at " ident)? : tactic
macro_rules
  | `(tactic| jp_core [$ts,*]) => `(tactic|
      simp (config := { failIfUnchanged := false, zeta := false }) (disch := jp_disch) only
        [Id.run, bind, pure, $ts,*])
  | `(tactic| jp_core [$ts,*] at $h) => `(tactic|
      simp (config := { failIfUnchanged := false }) (disch := jp_disch) only
        [Id.run, bind, pure, apply_ite Array.isEmpty, isEmpty_addCode, isEmpty_kept, ↓isEmpty_forIn,
          ↓isEmpty_forIn_list, ite_self, Bool.and_false, Bool.false_and, ite_false_eq_true, ite_eq_true_false,
          isEmpty_singleton, Bool.and_eq_true, Bool.false_eq_true, false_and, and_false, $ts,*] at $h:ident)

open Lean Meta Elab Tactic in
/-- `mjpK : ∀ d, d.isEmpty = false → (jpK () d).isEmpty = false` (a loop body's: `(jpK () d).value`) into the
context, by `jp_body` over `jpK`'s body. -/
meta def proveMono (jp : Ident) (xs : Array (TSyntax ``Lean.Parser.Tactic.simpLemma)) : TacticM Unit := do
  let nm := mkIdent (Name.mkSimple s!"m{jp.getId}")
  let some d ← findHyp jp.getId | throwError "jp_mono: no join point {jp}"
  let some step ← withMainContext (jpKind? d.type) | throwError "jp_mono: {jp} is no join point"
  if step then
    evalTactic (← `(tactic|
      have $nm : ∀ d : Array String, d.isEmpty = false → Array.isEmpty (ForInStep.value ($jp () d)) = false := by
        intro d hd
        simp (config := { zeta := false }) only [$jp:ident]
        jp_body [$xs,*]))
  else
    evalTactic (← `(tactic|
      have $nm : ∀ d : Array String, d.isEmpty = false → Array.isEmpty ($jp () d) = false := by
        intro d hd
        simp (config := { zeta := false }) only [$jp:ident]
        jp_body [$xs,*]))

open Lean Meta Elab Tactic in
/-- `mjpK` (`proveMono`) for each join point in the context that has none, in context order, so each from the ones
before it; `xs` are further simp lemmas. -/
elab "jp_mono" " [" xs:Lean.Parser.Tactic.simpLemma,* "]" : tactic => do
  for jp in ← letLemmas "jp" true do
    if (← findHyp (Name.mkSimple s!"m{jp.getId}")).isNone then proveMono jp xs.getElems

open Lean Meta in
/-- `e` under `Id.run` and `pure`. -/
meta def jpStrip (e : Expr) : Expr := Id.run do
  let mut e := e.consumeMData.headBeta
  for _ in [0:8] do
    if e.isAppOfArity ``Id.run 2 then e := (e.getArg! 1).consumeMData.headBeta
    else if e.isAppOfArity ``Pure.pure 4 then e := (e.getArg! 3).consumeMData.headBeta
    else break
  return e

open Lean Meta Elab Tactic in
/-- One step of `jp_body`, then `jp_body` on each goal it leaves. -/
meta partial def jpBody (xs : Array (TSyntax ``Lean.Parser.Tactic.simpLemma)) (fuel : Nat) : TacticM Unit := do
  if (← getGoals).isEmpty then return
  if fuel == 0 then throwError "jp_body: out of fuel"
  let sub (t : TSyntax `tactic) : TacticM Unit := do
    evalTactic t
    let gs ← getGoals
    let mut out := #[]
    for g in gs do
      setGoals [g]
      jpBody xs (fuel - 1)
      out := out ++ (← getGoals)
    setGoals out.toList
  if (← withMainContext do return (← whnfR (← getMainTarget)).isForall) then
    sub (← `(tactic| (intro _ _ hd'; try dsimp (config := { zeta := false }) only [])))
    return
  evalTactic (← `(tactic| try dsimp (config := { zeta := false }) only []))
  evalTactic (← `(tactic| try extract_lets))
  evalTactic (← `(tactic| try dsimp (config := { zeta := false }) only []))
  evalTactic (← `(tactic| name_lets))
  let ty ← withMainContext do instantiateMVars (← getMainTarget)
  let some (_, lhs, _) := ty.cleanupAnnotations.eq? | throwError "jp_body: {ty} is not an equation"
  unless lhs.isAppOfArity ``Array.isEmpty 2 do throwError "jp_body: {lhs} is not a code list's isEmpty"
  let a := jpStrip (lhs.getArg! 1)
  let (step, x) := if a.isAppOfArity ``ForInStep.value 2 then (true, jpStrip (a.getArg! 1)) else (false, a)
  if x.isAppOfArity ``ite 5 then
    sub (← if step then `(tactic| apply value_ite) else `(tactic| apply isEmpty_ite))
    return
  if ← withMainContext (isMatcherApp x) then
    sub (← `(tactic| split))
    return
  if step && x.isAppOfArity ``ForInStep.yield 2 then
    sub (← `(tactic| apply value_yield))
    return
  if step && x.isAppOfArity ``ForInStep.done 2 then
    sub (← `(tactic| apply value_done))
    return
  let f := x.getAppFn
  if f.isFVar then
    let d ← withMainContext f.fvarId!.getDecl
    if d.isLet && d.userName.toString.startsWith "jp" then
      let m := Name.mkSimple s!"m{d.userName}"
      if (← findHyp m).isNone then proveMono (mkIdent d.userName) xs
      sub (← `(tactic| apply $(mkIdent m)))
    else if d.isLet then
      sub (← `(tactic| simp (config := { zeta := false }) only [$(mkIdent d.userName):ident]))
    else
      evalTactic (← `(tactic| assumption))
    return
  if x.isAppOf ``ForIn.forIn then
    sub (← `(tactic| first | apply isEmpty_forIn | apply isEmpty_forIn_list))
    return
  if ← tryTactic (evalTactic (← `(tactic| exact isEmpty_addCode _ _))) then return
  if ← tryTactic (evalTactic (← `(tactic| assumption))) then return
  if ← tryTactic (evalTactic (← `(tactic| jp_ext))) then return
  if ← tryTactic (evalTactic (← `(tactic| apply isEmpty_kept_of))) then
    sub (← `(tactic| skip))
    return
  let before ← getMainTarget
  evalTactic (← `(tactic| jp_core [$xs,*]))
  if (← getGoals).isEmpty then return
  if (← instantiateMVars (← getMainTarget)) == (← instantiateMVars before) then
    throwError "jp_body: stuck at {x}"
  jpBody xs (fuel - 1)

open Lean Meta Elab Tactic in
elab_rules : tactic
  | `(tactic| jp_body [$xs,*]) => jpBody xs.getElems 400

/-- Closes `(… ).isEmpty = false` from the join points' `mjp`s and `xs` (`jp_body`). -/
macro "jp_close" " [" xs:Lean.Parser.Tactic.simpLemma,* "]" : tactic => `(tactic| jp_body [$xs,*])

open Lean Meta Elab Tactic in
/-- One goal's walk of `h : (…).isEmpty = true` (see `jp_walk`). -/
meta partial def walkGoal (h : Ident) (ts : Array (TSyntax ``Lean.Parser.Tactic.simpLemma)) (n : IO.Ref Nat) :
    TacticM Unit := do
  evalTactic (← `(tactic| jp_core [$ts,*] at $h))
  if (← getGoals).isEmpty then return
  repeat
    let some d ← findHyp h.getId | return
    let ty ← instantiateMVars d.type
    unless ty.isAppOfArity ``And 2 do break
    let k ← n.modifyGet fun k => (k, k + 1)
    let hc := mkIdent (Name.mkSimple s!"hc{k}")
    evalTactic (← `(tactic| obtain ⟨$hc, $h⟩ := $h))
  let some d ← findHyp h.getId | return
  let ty ← instantiateMVars d.type
  let some (_, lhs, _) := ty.eq? | return
  if lhs.isAppOfArity ``ite 5 then
    evalTactic (← `(tactic| split at $h:ident))
    let gs ← getGoals
    let mut out := #[]
    for g in gs do
      setGoals [g]
      let k ← n.modifyGet fun k => (k, k + 1)
      evalTactic (← `(tactic| try rename_i $(mkIdent (Name.mkSimple s!"hc{k}")):ident))
      walkGoal h ts n
      out := out ++ (← getGoals)
    setGoals out.toList
    return
  unless lhs.isAppOfArity ``Array.isEmpty 2 do return
  let x := (lhs.getArg! 1).headBeta.consumeMData
  let f := x.getAppFn
  if f.isFVar then
    let fd ← withMainContext f.fvarId!.getDecl
    if fd.isLet && fd.userName.toString.startsWith "jp" then
      evalTactic (← `(tactic| simp only [$(mkIdent fd.userName):ident] at $h:ident))
      walkGoal h ts n
    return
  if ← withMainContext (Lean.Meta.isMatcherApp x) then
    evalTactic (← `(tactic| split at $h:ident))
    let gs ← getGoals
    let mut out := #[]
    for g in gs do
      setGoals [g]
      walkGoal h ts n
      out := out ++ (← getGoals)
    setGoals out.toList

open Lean Meta Elab Tactic in
/-- `jp_walk h [xs]` walks `h : (…).isEmpty = true` down the one path on which no code is added: each branch that adds
one is dropped (`mjpK`, `isEmpty_addCode`), each condition on the path goes into the context as `hcN`, a join point
jumped to is unfolded once (one cleared of its value is not), and a `match` on the path is split. An `if` neither
branch of which adds a code is split too, its condition `hcN` in each branch. -/
elab "jp_walk" h:ident " [" xs:Lean.Parser.Tactic.simpLemma,* "]" : tactic => do
  let cvs ← letLemmas "cv"
  let ms ← hypsWith "mjp"
  let ts ← (cvs ++ ms).mapM fun i => `(Lean.Parser.Tactic.simpLemma| $i:ident)
  let ts := ts ++ xs.getElems
  let n ← IO.mkRef 0
  let gs ← getGoals
  let mut out := #[]
  for g in gs do
    setGoals [g]
    walkGoal h ts n
    out := out ++ (← getGoals)
  setGoals out.toList

/-! ## The helpers' codes -/

theorem applyOutputs_fst (c : Array String) (mine : Array Spot.Registered.Entry) (held : Array (String × String))
    (outputs : Array (String × String × Earlier)) (log : Array Json) (hc : c.isEmpty = false) :
    (applyOutputs c mine held outputs log).1.isEmpty = false := by
  unfold applyOutputs
  rw [← Array.foldl_toList]
  refine foldl_inv (fun (s : Array String × Array Spot.Registered.Entry × Array (String × String)) =>
    s.1.isEmpty = false) _ (fun s o hs => ?_) _ _ hc
  obtain ⟨c, mine, held⟩ := s
  obtain ⟨v, d, p⟩ := o
  simp only [applyOutput]
  split
  · exact hs
  · split
    · exact hs
    · exact isEmpty_addCode _ _

theorem applyOutputs_values (c : Array String) (mine : Array Spot.Registered.Entry) (held : Array (String × String))
    (outputs : Array (String × String × Earlier)) (log : Array Json) :
    (applyOutputs c mine held outputs log).2.1.map (·.value) = mine.map (·.value) := by
  unfold applyOutputs
  rw [← Array.foldl_toList]
  refine foldl_inv (fun (s : Array String × Array Spot.Registered.Entry × Array (String × String)) =>
    s.2.1.map (·.value) = mine.map (·.value)) _ (fun s o hs => ?_) _ _ rfl
  obtain ⟨c, mine', held⟩ := s
  obtain ⟨v, d, p⟩ := o
  simp only [applyOutput]
  split
  · exact hs
  · split
    · simp only [Array.map_map] at hs ⊢
      rw [← hs]
      congr 1
      funext e
      simp only [Function.comp]
      split <;> rfl
    · exact hs

theorem isEmpty_checkRegistered (c : Array String) (got : Json) (mine : Flock.V (Array Spot.Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Spot.Supplied.Rows))
    (rootOf : String → Spot.Supplied.Rows → Spot.Registered.Entry → Flock.V String) (hc : c.isEmpty = false) :
    (checkRegistered c got mine held supplied rootOf).isEmpty = false := by
  unfold checkRegistered
  jp_close [hc]

theorem isEmpty_checkOutputs (c : Array String) (got : Json) (mine : Flock.V (Array Spot.Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Spot.Supplied.Rows))
    (rootOf : String → Spot.Supplied.Rows → Spot.Registered.Entry → Flock.V String)
    (outputs : Array (String × String × Earlier)) (log : Array Json) (hc : c.isEmpty = false) :
    (checkOutputs c got mine held supplied rootOf outputs log).isEmpty = false := by
  unfold checkOutputs
  split
  · exact isEmpty_addCode _ _
  · exact isEmpty_checkRegistered _ _ _ _ _ _ (applyOutputs_fst _ _ _ _ _ hc)

macro_rules
  | `(tactic| jp_ext) => `(tactic| (apply isEmpty_checkOutputs; jp_disch))

end SpotFlock.RecordNames
