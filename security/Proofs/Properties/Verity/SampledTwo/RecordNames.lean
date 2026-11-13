module

public import Proofs.Flock.SpotFlock.HeldOf
public import Proofs.Flock.SpotFlock.Jeq
public import Proofs.Flock.SpotFlock.RunsAs.Outcome
public import Proofs.Flock.Soundness.ExecDraw
public import Lean
public meta import Lean

/-!
# A registration check with no code names its registration's names

`Registration.check rec v` collects R1–R8's failure codes for the record `rec` at the verifier `v`. Where it collects
none and `v` lists each port once, `rec` names `v`'s names (`recordOk_of_check`): its roots' ports are `v`'s ports and
its registered values `v`'s registered values, each once. R3-ports holds the roots' distinct ports to `v`'s ports by
count and membership, and R7-values the registered values to the verifier's own entries the same way; a record with no
registered value has no `registered` key, since its keys are exactly R1's `KEYS`.

`check` is one `do` block whose `if`s and `match`es each leave a join point, a function of the codes so far, for the
rest of the block. `jp_mono` proves once per join point, deepest first, that it keeps a nonempty code list nonempty;
`jp_walk` then follows the one path on which no code is added, unfolding each join point once. So the proof is linear
in the block, where unfolding the block into both arms of each branch doubles at every one.
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

/-! ## Walking a `do` block's join points -/

open Lean Meta Elab Tactic in
/-- Names the context's `let`s: each `do` join point `jp0, jp1, …` and each code list `cv0, cv1, …`, in context order,
and every other `l_<its name>`. -/
elab "name_lets" : tactic => withMainContext do
  let mut g ← getMainGoal
  let mut k := 0
  let mut j := 0
  let arr := mkApp (mkConst ``Array [0]) (mkConst ``String)
  for decl in (← getLCtx) do
    if decl.isLet && !decl.isImplementationDetail then
      if decl.userName.eraseMacroScopes == `__do_jp then
        g ← g.rename decl.fvarId (Name.mkSimple s!"jp{k}")
        k := k + 1
      else if ← isDefEq decl.type arr then
        g ← g.rename decl.fvarId (Name.mkSimple s!"cv{j}")
        j := j + 1
      else
        g ← g.rename decl.fvarId (Name.mkSimple s!"l_{decl.userName.eraseMacroScopes}")
  replaceMainGoal [g]

open Lean Meta Elab Tactic in
/-- The context's `let`s whose name starts with `pfx`, as simp lemmas, in context order; join points only of the type
`Unit → Array String → Id (Array String)` (`jp`). -/
meta def letLemmas (pfx : String) (jp : Bool := false) : TacticM (Array Ident) := withMainContext do
  let arr := mkApp (mkConst ``Array [0]) (mkConst ``String)
  let jpTy ← mkArrow (mkConst ``Unit) (← mkArrow arr (mkApp (mkConst ``Id [0]) arr))
  let mut out := #[]
  for decl in (← getLCtx) do
    if decl.isLet && decl.userName.toString.startsWith pfx then
      if !jp || (← isDefEq decl.type jpTy) then out := out.push (mkIdent decl.userName)
  return out

/-- A loop body's step keeps a nonempty code list nonempty. -/
macro "jp_loop" : tactic => `(tactic| (
    intro _ _ hd'
    simp (config := { failIfUnchanged := false }) only [Id.run, pure, bind, apply_ite ForInStep.value,
      apply_ite Array.isEmpty, isEmpty_addCode, isEmpty_kept, ite_self, Bool.and_false, Bool.false_and, hd']
    repeat' split
    all_goals (simp only [ForInStep.value, Id.run, pure, bind, apply_ite Array.isEmpty, apply_ite ForInStep.value,
      isEmpty_addCode, isEmpty_kept, ite_self, Bool.and_false, Bool.false_and, hd'])))

syntax "jp_disch" : tactic
macro_rules
  | `(tactic| jp_disch) => `(tactic| first
      | assumption
      | jp_loop
      | (simp (disch := jp_disch) only [isEmpty_addCode, isEmpty_kept, isEmpty_forIn, isEmpty_forIn_list,
          Bool.and_false, Bool.false_and]; done))

/-- `simp` over code lists: `isEmpty` pushed into branches, a branch that adds a code is nonempty, and `ts`. -/
syntax "jp_core" " [" Lean.Parser.Tactic.simpLemma,* "]" (" at " ident)? : tactic
macro_rules
  | `(tactic| jp_core [$ts,*]) => `(tactic|
      simp (config := { failIfUnchanged := false }) (disch := jp_disch) only
        [Id.run, bind, pure, apply_ite Array.isEmpty, isEmpty_addCode, isEmpty_kept, isEmpty_forIn,
          isEmpty_forIn_list, ite_self, Bool.and_false, Bool.false_and, $ts,*])
  | `(tactic| jp_core [$ts,*] at $h) => `(tactic|
      simp (config := { failIfUnchanged := false }) (disch := jp_disch) only
        [Id.run, bind, pure, apply_ite Array.isEmpty, isEmpty_addCode, isEmpty_kept, isEmpty_forIn,
          isEmpty_forIn_list, ite_self, Bool.and_false, Bool.false_and, ite_false_eq_true, ite_eq_true_false,
          isEmpty_singleton,
          Bool.and_eq_true, Bool.false_eq_true, false_and, and_false, $ts,*] at $h:ident)

open Lean Meta Elab Tactic in
/-- `mjpK : ∀ d, d.isEmpty = false → (jpK () d).isEmpty = false` for each join point, deepest first, each from the ones
before it; `xs` are further simp lemmas. -/
elab "jp_mono" " [" xs:Lean.Parser.Tactic.simpLemma,* "]" : tactic => do
  let jps ← letLemmas "jp" true
  let cvs ← letLemmas "cv"
  let mut ms : Array Ident := #[]
  for jp in jps do
    let nm := mkIdent (Name.mkSimple s!"m{jp.getId}")
    let ts ← (cvs ++ ms).mapM fun i => `(Lean.Parser.Tactic.simpLemma| $i:ident)
    let ts := ts ++ xs.getElems
    evalTactic (← `(tactic|
      have $nm : ∀ d : Array String, d.isEmpty = false → Array.isEmpty ($jp () d) = false := by
        intro d hd
        simp only [$jp:ident]
        jp_core [hd, $ts,*]
        repeat' split
        all_goals jp_core [hd, $ts,*]))
    ms := ms.push nm

open Lean Meta Elab Tactic in
/-- Closes `(… ).isEmpty = false` from the join points' `mjp`s and `xs`. -/
elab "jp_close" " [" xs:Lean.Parser.Tactic.simpLemma,* "]" : tactic => do
  let cvs ← letLemmas "cv"
  let ms ← letLemmas "jp" true
  let ms := ms.map fun i => mkIdent (Name.mkSimple s!"m{i.getId}")
  let ts ← (cvs ++ ms).mapM fun i => `(Lean.Parser.Tactic.simpLemma| $i:ident)
  let ts := ts ++ xs.getElems
  evalTactic (← `(tactic| jp_core [$ts,*]))
  evalTactic (← `(tactic| repeat' split))
  evalTactic (← `(tactic| all_goals jp_core [$ts,*]))

open Lean Meta Elab Tactic in
/-- The main goal's hypothesis named `n`. -/
meta def findHyp (n : Name) : TacticM (Option LocalDecl) := withMainContext do
  return (← getLCtx).findFromUserName? n

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
      walkGoal h ts n
      out := out ++ (← getGoals)
    setGoals out.toList
    return
  unless lhs.isAppOfArity ``Array.isEmpty 2 do return
  let x := (lhs.getArg! 1).headBeta.consumeMData
  let f := x.getAppFn
  if f.isFVar then
    let fd ← withMainContext f.fvarId!.getDecl
    if fd.userName.toString.startsWith "jp" then
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
jumped to is unfolded once, and a `match` on the path is split. -/
elab "jp_walk" h:ident " [" xs:Lean.Parser.Tactic.simpLemma,* "]" : tactic => do
  let cvs ← letLemmas "cv"
  let ms ← letLemmas "jp" true
  let ms := ms.map fun i => mkIdent (Name.mkSimple s!"m{i.getId}")
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

/-! ## Counting names -/

theorem perm_of_nodup_subset {α : Type} [DecidableEq α] : ∀ {S D : List α}, S.Nodup → (∀ x ∈ S, x ∈ D) →
    D.length ≤ S.length → S.Perm D
  | [], D, _, _, hl => by
    cases D with
    | nil => exact List.Perm.refl _
    | cons _ _ => simp at hl
  | a :: S, D, hs, hsub, hl => by
    have ha : a ∈ D := hsub a (by simp)
    have hn := List.nodup_cons.1 hs
    have ih : S.Perm (D.erase a) := perm_of_nodup_subset hn.2
      (fun x hx => (List.mem_erase_of_ne (fun e => hn.1 (by subst e; exact hx))).2 (hsub x (by simp [hx])))
      (by rw [List.length_erase_of_mem ha]; simp at hl; omega)
    exact (List.Perm.cons a ih).trans (List.perm_cons_erase ha).symm

theorem length_eraseDups_le {α : Type} [BEq α] :
    ∀ (n : Nat) (l : List α), l.length ≤ n → l.eraseDups.length ≤ l.length
  | _, [], _ => by simp
  | 0, _ :: _, h => by simp at h
  | n + 1, a :: as, h => by
    rw [List.eraseDups_cons]
    have := length_eraseDups_le n (as.filter fun b => !b == a)
      (Nat.le_trans (List.length_filter_le _ _) (by simp at h; omega))
    have := List.length_filter_le (fun b => !b == a) as
    simp; omega

theorem eraseDups_eq_of_length {α : Type} [BEq α] [LawfulBEq α] :
    ∀ (n : Nat) (l : List α), l.length ≤ n → l.eraseDups.length = l.length → l.eraseDups = l ∧ l.Nodup
  | _, [], _, _ => by simp
  | 0, _ :: _, h, _ => by simp at h
  | n + 1, a :: as, h, he => by
    rw [List.eraseDups_cons] at he ⊢
    have h1 := length_eraseDups_le as.length (as.filter fun b => !b == a) (List.length_filter_le _ _)
    have h2 := List.length_filter_le (fun b => !b == a) as
    simp only [List.length_cons, Nat.add_right_cancel_iff] at he
    have hf : (as.filter fun b => !b == a).length = as.length := by omega
    have hfa : as.filter (fun b => !b == a) = as := List.filter_eq_self.2 (List.length_filter_eq_length_iff.1 hf)
    rw [hfa] at he ⊢
    obtain ⟨ih1, ih2⟩ := eraseDups_eq_of_length n as (by simp at h; omega) he
    refine ⟨by rw [ih1], List.nodup_cons.2 ⟨fun ha => ?_, ih2⟩⟩
    have := List.length_filter_eq_length_iff.1 hf a ha
    simp at this

theorem foldl_pushes {α β : Type} (p : Array β → α → Bool) (q : α → β) : ∀ (l : List α) (acc : Array β),
    (l.foldl (fun acc r => if p acc r then acc else acc.push (q r)) acc).size ≤ acc.size + l.length ∧
    ((l.foldl (fun acc r => if p acc r then acc else acc.push (q r)) acc).size = acc.size + l.length →
      (l.foldl (fun acc r => if p acc r then acc else acc.push (q r)) acc).toList = acc.toList ++ l.map q)
  | [], acc => by simp
  | r :: l, acc => by
    simp only [List.foldl_cons, List.length_cons, List.map_cons]
    by_cases hp : p acc r = true
    · simp only [hp, ↓reduceIte]
      have := (foldl_pushes p q l acc).1
      exact ⟨by omega, fun h => by omega⟩
    · simp only [hp, Bool.false_eq_true, ↓reduceIte]
      have ih := foldl_pushes p q l (acc.push (q r))
      simp only [Array.size_push, Array.toList_push] at ih
      refine ⟨by omega, fun h => ?_⟩
      rw [ih.2 (by omega)]
      simp

theorem foldl_pushes_array {α β : Type} (p : Array β → α → Bool) (q : α → β) (xs : Array α)
    (h : (xs.foldl (fun acc r => if p acc r then acc else acc.push (q r)) #[]).size = xs.size) :
    (xs.foldl (fun acc r => if p acc r then acc else acc.push (q r)) #[]).toList = xs.toList.map q := by
  rw [← Array.foldl_toList] at h ⊢
  have := (foldl_pushes p q xs.toList #[]).2 (by simpa using h)
  simpa using this

theorem mapM_some {α β : Type} (f : α → Option β) (g : α → β) :
    ∀ (l : List α), (∀ x ∈ l, f x = some (g x)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | x :: l, h => by
    rw [List.mapM_cons, h x (by simp), mapM_some f g l (fun y hy => h y (by simp [hy]))]
    rfl

theorem mapM_ok_list {α β γ ε : Type} (f : α → Except ε β) (g : β → γ) (k : α → γ)
    (hf : ∀ x y, f x = .ok y → g y = k x) : ∀ (xs : List α) (ys : List β), xs.mapM f = .ok ys → ys.map g = xs.map k
  | [], ys, h => by simp [List.mapM_nil, pure, Except.pure] at h; subst h; rfl
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | error e => rw [hx] at h; cases h
    | ok y =>
      rw [hx] at h
      cases hxs : xs.mapM f with
      | error e => simp [hxs, bind, Except.bind] at h
      | ok ys' =>
        simp only [hxs, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        simp [hf x y hx, mapM_ok_list f g k hf xs ys' hxs]

theorem mapM_ok_array {α β γ ε : Type} (f : α → Except ε β) (g : β → γ) (k : α → γ)
    (hf : ∀ x y, f x = .ok y → g y = k x) (xs : Array α) (ys : Array β) (h : xs.mapM f = .ok ys) :
    ys.map g = xs.map k := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => simp [hl, Functor.map, Except.map] at h
  | ok l =>
    simp only [hl, Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    apply Array.ext'
    simp [mapM_ok_list f g k hf _ _ hl]

theorem foldl_inv {α β : Type} (P : β → Prop) (f : β → α → β) (hf : ∀ b a, P b → P (f b a)) :
    ∀ (l : List α) (b : β), P b → P (l.foldl f b)
  | [], _, h => h
  | a :: l, b, h => foldl_inv P f hf l (f b a) (hf b a h)

/-! ## A key an object's keys don't name -/

open Std.DTreeMap.Internal in
theorem mem_toListModel_of_get? {δ : Type} : ∀ (t : Impl String fun _ => δ) (k : String) (v : δ),
    Impl.Const.get? t k = some v → k ∈ t.toListModel.map (·.1)
  | .leaf, k, v, h => by simp [Impl.Const.get?] at h
  | .inner _ k' v' l r, k, v, h => by
    have hl := mem_toListModel_of_get? l k v
    have hr := mem_toListModel_of_get? r k v
    simp only [Impl.toListModel_inner, List.map_append, List.map_cons, List.mem_append, List.mem_cons]
    unfold Impl.Const.get? at h
    split at h
    · exact Or.inl (hl h)
    · exact Or.inr (Or.inr (hr h))
    · rename_i he
      exact Or.inr (Or.inl (Std.LawfulEqOrd.eq_of_compare he))

theorem toList_foldl_push {δ : Type} : ∀ (l : List ((_ : String) × δ)) (acc : Array (String × δ)),
    (l.foldl (fun acc p => acc.push (p.1, p.2)) acc).toList = acc.toList ++ l.map (fun p => (p.1, p.2))
  | [], acc => by simp
  | p :: l, acc => by simp only [List.foldl_cons, toList_foldl_push l]; simp

open Std.DTreeMap.Internal in
theorem getElem?_none_of_not_key (m : Std.TreeMap.Raw String Json) (k : String)
    (h : k ∉ m.toArray.map (·.1)) : m[k]? = none := by
  cases hk : m[k]? with
  | none => rfl
  | some v =>
    exfalso
    apply h
    have h1 := mem_toListModel_of_get? m.inner.inner k v hk
    rw [Array.mem_def, Array.toList_map]
    simp only [Std.TreeMap.Raw.toArray, Std.DTreeMap.Raw.Const.toArray, Impl.Const.toArray, Impl.foldl_eq_foldl,
      toList_foldl_push]
    simpa [List.map_map] using h1

/-- **A record whose keys are R1's `KEYS` has no `registered` entry.** -/
theorem registered_null {rec : Json} (hk : Spot.keys? rec = some KEYS) : Spot.getD rec "registered" = .null := by
  cases rec with
  | obj m =>
    simp only [Spot.keys?, Option.some.injEq] at hk
    rw [SpotFlock.RunsAs.getD_obj, getElem?_none_of_not_key m "registered" (fun hm => ?_)]
    · rfl
    · have h2 := (FlockSoundness.Audit.DrawExec.mem_qsort _ (· < ·) _).2 hm
      rw [show (m.toArray.map Prod.fst).qsort (· < ·) = KEYS from hk] at h2
      simp [KEYS, Spot.sortStrings, FlockSoundness.Audit.DrawExec.mem_qsort] at h2
  | _ => simp [Spot.keys?] at hk

/-! ## The two names -/

/-- A JSON string's text; none for anything else. -/
def strOf : Json → Option String
  | .str s => some s
  | _ => none

/-- **R3-ports names the ports**: a record whose roots' distinct ports are as many as its roots and as `ports`, each
port among them, names `ports` each once at its roots, where `ports` lists each once. -/
theorem ports_names {rec : Json} {rs : Array Json} {ports : Array (String × Json)}
    (hn : (ports.map (·.1)).toList.Nodup) (hrs : Spot.getD rec "roots" = .arr rs)
    (h1 : (portIndex rs).count = rs.size) (h2 : (portIndex rs).count = ports.size)
    (h3 : ∀ p ∈ ports, (portIndex rs).has p.1 = true) :
    namesEq (namesAt rec "roots" "port") (ports.map (·.1)).toList = true := by
  have h3' : ∀ p ∈ ports, (rs.foldl (fun (acc : Array Json) r =>
      if acc.any (Spot.jeq · (Spot.getD r "port")) then acc else acc.push (Spot.getD r "port")) #[]).any
        (Spot.jeq · (Json.str p.1)) = true := fun p hp => by rw [← portIndex_has]; exact h3 p hp
  rw [portIndex_count] at h1 h2
  have hD := foldl_pushes_array (fun (acc : Array Json) r => acc.any (Spot.jeq · (Spot.getD r "port")))
    (Spot.getD · "port") rs h1
  generalize hDdef : rs.foldl (fun (acc : Array Json) r =>
    if acc.any (Spot.jeq · (Spot.getD r "port")) then acc else acc.push (Spot.getD r "port")) #[] = D at hD h2 h3'
  generalize hP : (ports.map (·.1)).toList = P at hn ⊢
  have hmem : ∀ x ∈ P, Json.str x ∈ D.toList := by
    intro x hx
    rw [← hP, Array.toList_map, List.mem_map] at hx
    obtain ⟨p, hp, rfl⟩ := hx
    obtain ⟨i, hi, hj⟩ := Array.any_eq_true.1 (h3' p (Array.mem_def.2 hp))
    have he := str_of_jeq (jeq_symm hj)
    rw [← Array.mem_def, ← he]
    exact Array.getElem_mem hi
  have hS : (P.map Json.str).Nodup := hn.map (fun a b h => Json.str.inj h)
  have hlen : D.toList.length ≤ (P.map Json.str).length := by
    rw [← hP]
    simp [h2]
  have hperm : (P.map Json.str).Perm D.toList :=
    perm_of_nodup_subset hS (fun x hx => by
      obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
      exact hmem y hy) hlen
  have hstr : ∀ e ∈ rs.toList, ∃ s ∈ P, Spot.getD e "port" = .str s := by
    intro e he
    have : Spot.getD e "port" ∈ D.toList := by rw [hD]; exact List.mem_map_of_mem he
    obtain ⟨s, hs, he'⟩ := List.mem_map.1 (hperm.mem_iff.2 this)
    exact ⟨s, hs, he'.symm⟩
  let g : Json → String := fun e => (strOf (Spot.getD e "port")).getD ""
  have hmap : namesAt rec "roots" "port" = some (rs.toList.map g) := by
    unfold namesAt
    rw [hrs]
    exact mapM_some _ g rs.toList (fun e he => by
      obtain ⟨s, -, hs⟩ := hstr e he
      simp only [g, hs, strOf]
      rfl)
  have hN : (rs.toList.map g).Perm P := by
    have h4 : rs.toList.map g = D.toList.map (fun j => (strOf j).getD "") := by
      rw [hD, List.map_map]
      rfl
    rw [h4]
    have := hperm.symm.map (fun j => (strOf j).getD "")
    simpa [List.map_map, Function.comp_def, strOf] using this
  rw [hmap]
  simp only [namesEq, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.contains_iff_mem]
  exact ⟨⟨hN.nodup_iff.2 hn, fun x hx => hN.mem_iff.1 hx⟩, fun x hx => hN.mem_iff.2 hx⟩

/-- **R7-values names the registered values**: a record whose `registered` entries' values are strings, distinct, as
many as `R` and each in `R`, names `R` each once there. -/
theorem values_names {rec : Json} {g : Array Json} {R : List String}
    (hg : Spot.getD rec "registered" = .arr g)
    (h1 : ((g.map (Spot.getD · "value")).filterMap strOf).size = g.size)
    (h2 : ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups.length = g.size)
    (h3 : R.length ≤ g.size)
    (h4 : ∀ x ∈ ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups, x ∈ R) :
    namesEq (namesAt rec "registered" "value") R = true := by
  generalize hN : ((g.map (Spot.getD · "value")).filterMap strOf).toList = N at h2 h4
  have hNl : N.length = g.size := by rw [← hN, Array.length_toList]; exact h1
  obtain ⟨hE, hNd⟩ := eraseDups_eq_of_length N.length N (Nat.le_refl _) (by rw [h2, hNl])
  have hN' : N = (g.toList.map (Spot.getD · "value")).filterMap strOf := by
    rw [← hN, Array.toList_filterMap, Array.toList_map]
  have hsome : ∀ e ∈ g.toList, (strOf (Spot.getD e "value")).isSome := by
    have : ((g.toList.map (Spot.getD · "value")).filterMap strOf).length =
        (g.toList.map (Spot.getD · "value")).length := by
      rw [← hN', hNl, List.length_map, Array.length_toList]
    intro e he
    exact List.filterMap_length_eq_length.1 this _ (List.mem_map_of_mem he)
  have hlist : ∀ (l : List Json), (∀ e ∈ l, (strOf (Spot.getD e "value")).isSome) →
      l.mapM (fun e => match Spot.getD e "value" with | .str s => some s | _ => none) =
        some ((l.map (Spot.getD · "value")).filterMap strOf) := by
    intro l
    induction l with
    | nil => intro _; rfl
    | cons e l ih =>
      intro h
      obtain ⟨s, hs⟩ := Option.isSome_iff_exists.1 (h e (by simp))
      have he : Spot.getD e "value" = .str s := by
        revert hs
        unfold strOf
        split <;> intro hs <;> simp_all
      rw [List.mapM_cons, ih (fun e' he' => h e' (by simp [he']))]
      simp [he, strOf]
      rfl
  have hmap : namesAt rec "registered" "value" = some N := by
    unfold namesAt
    rw [hg, hN']
    exact hlist _ hsome
  rw [hmap]
  have hperm : N.Perm R := perm_of_nodup_subset hNd (fun x hx => h4 x (by rw [hE]; exact hx)) (by omega)
  simp only [namesEq, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.contains_iff_mem]
  exact ⟨⟨hNd, fun x hx => hperm.mem_iff.1 hx⟩, fun x hx => hperm.mem_iff.2 hx⟩

/-! ## The helpers' codes -/

theorem applyOutputs_fst (c : Array String) (mine : Array Spot.Registered.Entry) (held : Array (String × String))
    (outputs : Array (String × String × Earlier)) (log : Array Json) (hc : c.isEmpty = false) :
    (applyOutputs c mine held outputs log).1.isEmpty = false := by
  unfold applyOutputs
  rw [← Array.foldl_toList]
  refine foldl_inv (fun s => s.1.isEmpty = false) _ (fun s o hs => ?_) _ _ hc
  obtain ⟨c, mine, held⟩ := s
  obtain ⟨v, d, p⟩ := o
  unfold applyOutput
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
  refine foldl_inv (fun s => s.2.1.map (·.value) = mine.map (·.value)) _ (fun s o hs => ?_) _ _ rfl
  obtain ⟨c, mine', held⟩ := s
  obtain ⟨v, d, p⟩ := o
  unfold applyOutput
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
  lift_lets
  extract_lets
  name_lets
  jp_mono [hc]
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
  | `(tactic| jp_disch) => `(tactic|
      (simp (disch := jp_disch) only [isEmpty_checkOutputs, isEmpty_addCode, isEmpty_kept, Bool.and_false,
        Bool.false_and]; done))

/-! ## The walks -/

/-- **R7-values passed**: the record's `registered` is a list whose values are strings, distinct, as many as `mine`'s
and each one of `mine`'s values. -/
def ValuesHold (g : Array Json) (mine : Array Spot.Registered.Entry) : Prop :=
  ((g.map (Spot.getD · "value")).filterMap strOf).size = g.size ∧
  ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups.length = g.size ∧
  ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups.length = mine.size ∧
  ∀ x ∈ ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups, ∃ e ∈ mine, e.value = x

theorem checkRegistered_pass {c : Array String} {got : Json} {mine : Array Spot.Registered.Entry}
    {held : Array (String × String)} {supplied : Array (String × Spot.Supplied.Rows)}
    {rootOf : String → Spot.Supplied.Rows → Spot.Registered.Entry → Flock.V String}
    (h : checkRegistered c got (.ok mine) held supplied rootOf = #[]) : ∃ g, got = .arr g ∧ ValuesHold g mine := by
  have h' : (checkRegistered c got (.ok mine) held supplied rootOf).isEmpty = true := by rw [h]; rfl
  unfold checkRegistered at h'
  cases got with
  | arr g =>
    refine ⟨g, rfl, ?_⟩
    lift_lets at h'
    extract_lets at h'
    name_lets
    jp_mono []
    jp_walk h' []
    trace_state
    sorry
  | _ => simp [Id.run, pure, isEmpty_addCode] at h'

/-- **What a check with no code says of the names**: the record's keys are R1's, its roots a list R3-ports holds to the
ports, and with registered values, its `registered` a list R7-values holds to the verifier's own entries. -/
def CheckFacts (rec : Json) (v : Verifier) : Prop :=
  Spot.keys? rec = some (if !v.registered.isEmpty then KEYS_V2 else KEYS) ∧
  (∃ rs, Spot.getD rec "roots" = .arr rs ∧ (portIndex rs).count = rs.size ∧ (portIndex rs).count = v.ports.size ∧
    ∀ p ∈ v.ports, (portIndex rs).has p.1 = true) ∧
  (v.registered.isEmpty = false → ∃ mine g mine', ownEntries v = .ok mine ∧ Spot.getD rec "registered" = .arr g ∧
    mine'.map (·.value) = mine.map (·.value) ∧ ValuesHold g mine')

theorem checkFacts_of_check {rec : Json} {v : Verifier} (h : check rec v = #[]) : CheckFacts rec v := by
  have h' : (check rec v).isEmpty = true := by rw [h]; rfl
  unfold check at h'
  lift_lets at h'
  extract_lets at h'
  name_lets
  cases hv2 : v.registered.isEmpty
  all_goals
    jp_mono [poolMarks, r8Codes, isEmpty_checkOutputs, hv2, l_v2]
    jp_walk h' [poolMarks, r8Codes, isEmpty_checkOutputs, hv2, l_v2]
  trace_state
  all_goals sorry

/-! ## The record names its registration's names -/

theorem entries_values {d : Spot.Registered.Desc} {ps : Array Spot.Registered.Param} {values : Array (String × Nat)}
    {schemas : Array (String × String)} {window : Option String} {fixed : Array String}
    {es : Array Spot.Registered.Entry} (h : Spot.Registered.entries d ps values schemas window fixed = .ok es) :
    es.map (·.value) = values.map (·.1) := by
  unfold Spot.Registered.entries at h
  refine mapM_ok_array _ (·.value) (·.1) (fun x y hx => ?_) _ _ h
  obtain ⟨v, w⟩ := x
  simp only [bind, Except.bind, pure, Except.pure] at hx
  repeat'
    first
    | split at hx
    | focus (cases hx; done)
    | (cases hx; rfl)
    | (simp [throw, throwThe, MonadExceptOf.throw] at hx; done)

theorem recordOk_of_facts {rec : Json} {v : Verifier} (hf : CheckFacts rec v) (hn : (v.ports.map (·.1)).toList.Nodup) :
    recordOk rec (namesOf v) = true := by
  obtain ⟨hk, ⟨rs, hrs, h1, h2, h3⟩, h7⟩ := hf
  unfold recordOk
  rw [Bool.and_eq_true]
  refine ⟨ports_names hn hrs h1 h2 h3, ?_⟩
  cases hv : v.registered.isEmpty
  · obtain ⟨mine, g, mine', hmine, hg, hm', hv1, hv2, hv3, hv4⟩ := h7 hv
    have hvals : mine'.map (·.value) = v.registered.map (·.1) := by rw [hm', entries_values hmine]
    have hR : (v.registered.map (·.1)).toList.length = mine'.size := by
      have := congrArg Array.size hvals
      simpa using this.symm
    refine values_names hg hv1 hv2 (by rw [hR, ← hv3, hv2]) (fun x hx => ?_)
    obtain ⟨e, he, rfl⟩ := hv4 x hx
    have : e.value ∈ mine'.map (·.value) := Array.mem_map_of_mem he
    rw [hvals] at this
    simpa using this
  · simp only [hv, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hk
    have : v.registered = #[] := Array.isEmpty_iff.1 hv
    show namesEq (namesAt rec "registered" "value") (v.registered.map (·.1)).toList = true
    unfold namesAt
    rw [registered_null hk, this]
    rfl

/-- **A registration check with no code names the registration's names**, where the verifier lists each port once. -/
theorem recordOk_of_check {rec : Json} {v : Verifier} (h : check rec v = #[])
    (hn : (v.ports.map (·.1)).toList.Nodup) : recordOk rec (namesOf v) = true :=
  recordOk_of_facts (checkFacts_of_check h) hn

end SpotFlock.RecordNames
