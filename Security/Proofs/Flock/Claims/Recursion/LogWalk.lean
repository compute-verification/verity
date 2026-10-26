import Proofs.Flock.Claims.KS.ConflictWithin
import Proofs.Flock.Soundness.Refine.Walk

/-!
# Walking a logged run

`RetL m l Q`: every run of `m : LogV γ` from the log `l` that returns satisfies `Q` of its value and final log.
`retl_walk` proves it as `ret_walk` proves `Ret` (`FlockSoundness.Refine.Walk`), step by step on `m`'s head: a bind of a
lifted executable step (`liftM v`) forgets the log, which the step leaves alone, and keeps `v = .ok a` when `v`'s head
is one of the given constants; a bind of `pure` substitutes; a bind of a record digest (`dgWith`) appends its input to
the log (`retl_dg_bind`); a join point is inlined; a `let` keeps its value's equation; every other bind keeps its run
`x.go l = (.ok a, l₁)`, so a loop (`mapM_log`, `mapIdxM_log`, `forIn_log`) is read off afterwards. A value takes its
binder's name, its run or equation `h_` and its log `l_` that name.
-/

namespace FlockDeploy.KS

open Flock

theorem go_liftM {α : Type} (v : Flock.V α) (l : Array ByteArray) : (liftM v : LogV α).go l = (v, l) := by
  cases v <;> rfl

theorem go_bind {α β : Type} (x : LogV α) (f : α → LogV β) (l : Array ByteArray) :
    (x >>= f).go l = match x.go l with
      | (.ok a, l₁) => (f a).go l₁
      | (.error e, l₁) => (.error e, l₁) := by
  simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind]
  rcases x.run.run l with ⟨r, l₁⟩
  cases r <;> rfl

theorem go_pure {α : Type} (a : α) (l : Array ByteArray) : (pure a : LogV α).go l = (.ok a, l) := rfl

theorem go_bind_ok {α β : Type} {x : LogV α} {f : α → LogV β} {l l₂ : Array ByteArray} {b : β}
    (h : (x >>= f).go l = (.ok b, l₂)) : ∃ a l₁, x.go l = (.ok a, l₁) ∧ (f a).go l₁ = (.ok b, l₂) := by
  rw [go_bind] at h
  rcases hx : x.go l with ⟨r, l₁⟩
  rw [hx] at h
  cases r with
  | error e => cases h
  | ok a => exact ⟨a, l₁, rfl, h⟩

/-- The log a record digest adds: its input, where the digest is SHA-512. -/
def dgL (spec : SessionSpec) (b : ByteArray) : Array ByteArray := if spec.sha512Digests then #[b] else #[]

theorem go_dg (h : ByteArray → ByteArray) (spec : SessionSpec) (b : ByteArray) (l : Array ByteArray) :
    (spec.dgWith (Hashes.logged h) b).go l =
      (.ok (if spec.sha512Digests then h b else Sha256.hash b), l ++ dgL spec b) := by
  unfold SessionSpec.dgWith dgL
  cases spec.sha512Digests
  · simp [Hashes.logged, Hashes.ofSha512]; rfl
  · simp [Hashes.logged, Hashes.ofSha512]; rfl

/-- `dgL` as a list. -/
def dgl (spec : SessionSpec) (b : ByteArray) : List ByteArray := if spec.sha512Digests then [b] else []

theorem dgL_toList (spec : SessionSpec) (b : ByteArray) : (dgL spec b).toList = dgl spec b := by
  unfold dgL dgl
  split <;> rfl

theorem go_dg_inv {h : ByteArray → ByteArray} {spec : SessionSpec} {b v : ByteArray} {l l' : Array ByteArray}
    (e : (spec.dgWith (Hashes.logged h) b).go l = (.ok v, l')) :
    v = (if spec.sha512Digests then h b else Sha256.hash b) ∧ l' = l ++ dgL spec b := by
  rw [go_dg] at e
  obtain ⟨h1, h2⟩ := Prod.mk.inj e
  exact ⟨(Except.ok.inj h1).symm, h2.symm⟩

/-- A loop that only folds logs nothing. -/
theorem go_forIn_pure_inv {α β : Type} {xs : Array α} {f : α → β → β} {init v : β} {l l' : Array ByteArray}
    (e : (forIn xs init (fun a b => pure (.yield (f a b))) : LogV β).go l = (.ok v, l')) :
    v = xs.foldl (fun b a => f a b) init ∧ l' = l := by
  rw [Array.forIn_pure_yield_eq_foldl] at e
  obtain ⟨h1, h2⟩ := Prod.mk.inj e
  exact ⟨(Except.ok.inj h1).symm, h2.symm⟩

/-- Per-element logs that are a function of the element's result flatten to that function over the results. -/
theorem flatten_snd {α β : Type} {R : α → β → Prop} {F : β → List ByteArray} :
    ∀ {xs : List α} {yds : List (β × List ByteArray)},
      List.Forall₂ (fun x yd => R x yd.1 ∧ yd.2 = F yd.1) xs yds →
        (yds.map Prod.snd).flatten = (yds.map Prod.fst).flatMap F
  | _, _, .nil => rfl
  | _, _, .cons ⟨_, e⟩ h => by simp [e, flatten_snd h]

/-- Per-element logs that are a function of the element flatten to that function over the elements. -/
theorem flatten_eq_flatMap {α : Type} {R : α → List ByteArray → Prop} {F : α → List ByteArray}
    (hR : ∀ x d, R x d → d = F x) : ∀ {xs : List α} {ds : List (List ByteArray)}, List.Forall₂ R xs ds →
      ds.flatten = xs.flatMap F
  | _, _, .nil => rfl
  | _, _, .cons h t => by simp [hR _ _ h, flatten_eq_flatMap hR t]

theorem flatMap_nil' {α : Type} (xs : List α) : xs.flatMap (fun _ => ([] : List ByteArray)) = [] := by
  induction xs <;> simp_all

theorem forall₂_fst_snd {α β δ : Type} {P : β → δ → Prop} :
    ∀ {xs : List α} {yds : List (β × δ)}, List.Forall₂ (fun _ yd => P yd.1 yd.2) xs yds →
      List.Forall₂ P (yds.map Prod.fst) (yds.map Prod.snd)
  | _, _, .nil => .nil
  | _, _, .cons h t => .cons h (forall₂_fst_snd t)

theorem forall₂_exists_right {α β : Type} {R : α → β → Prop} :
    ∀ {xs : List α} {ys : List β}, List.Forall₂ R xs ys → ∀ x ∈ xs, ∃ y, R x y
  | _, _, .nil => fun _ h => by cases h
  | _, _, .cons h t => fun x hx => by
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨_, h⟩
    · exact forall₂_exists_right t x hx

theorem go_pure_inv {α : Type} {a b : α} {l l' : Array ByteArray} (e : (pure a : LogV α).go l = (.ok b, l')) :
    b = a ∧ l' = l := by
  obtain ⟨h1, h2⟩ := Prod.mk.inj e
  exact ⟨(Except.ok.inj h1).symm, h2.symm⟩

/-! ## `RetL` -/

/-- Every run of `m` from the log `l` that returns satisfies `Q` of its value and final log. -/
def RetL {γ : Type} (m : LogV γ) (l : Array ByteArray) (Q : γ → Array ByteArray → Prop) : Prop :=
  ∀ v l', m.go l = (.ok v, l') → Q v l'

theorem retl_bind {α γ : Type} (x : LogV α) (f : α → LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (h : ∀ a l₁, x.go l = (.ok a, l₁) → RetL (f a) l₁ Q) :
    RetL (x >>= f) l Q := fun v l' hv => by
  obtain ⟨a, l₁, hx, hf⟩ := go_bind_ok hv
  exact h a l₁ hx v l' hf

theorem retl_lift {α γ : Type} (x : Flock.V α) (f : α → LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (h : ∀ a, x = .ok a → RetL (f a) l Q) : RetL (liftM x >>= f) l Q :=
  retl_bind _ _ _ _ fun a l₁ hx => by
    rw [go_liftM] at hx
    obtain ⟨h1, rfl⟩ := Prod.mk.inj hx
    exact h a h1

theorem retl_dg_bind {γ : Type} (h : ByteArray → ByteArray) (spec : SessionSpec) (b : ByteArray)
    (f : ByteArray → LogV γ) (l : Array ByteArray) (Q : γ → Array ByteArray → Prop)
    (hf : RetL (f (if spec.sha512Digests then h b else Sha256.hash b)) (l ++ dgL spec b) Q) :
    RetL (spec.dgWith (Hashes.logged h) b >>= f) l Q :=
  retl_bind _ _ _ _ fun _ _ hx => by
    obtain ⟨rfl, rfl⟩ := go_dg_inv hx
    exact hf

theorem retl_lift' {α γ : Type} (x : Flock.V α) (f : α → LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (h : ∀ a, RetL (f a) l Q) : RetL (liftM x >>= f) l Q :=
  retl_lift x f l Q fun a _ => h a

theorem retl_pure_bind {α γ : Type} (a : α) (f : α → LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (h : RetL (f a) l Q) : RetL (pure a >>= f) l Q := by
  rw [pure_bind]
  exact h

theorem retl_map {α γ : Type} (g : α → γ) (x : LogV α) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (h : RetL (x >>= fun a => pure (g a)) l Q) : RetL (g <$> x) l Q := by
  rw [map_eq_bind_pure_comp]
  exact h

theorem retl_ite {γ : Type} (c : Prop) [Decidable c] (t e : LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (ht : c → RetL t l Q) (he : ¬c → RetL e l Q) : RetL (ite c t e) l Q := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact ht hc
  · rw [ite_eq_right hc]; exact he hc

theorem retl_dite {γ : Type} (c : Prop) [Decidable c] (t : c → LogV γ) (e : ¬c → LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (ht : ∀ h, RetL (t h) l Q) (he : ∀ h, RetL (e h) l Q) :
    RetL (dite c t e) l Q := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact ht hc
  · rw [dite_eq_right hc]; exact he hc

theorem retl_throw {γ : Type} (e : String) (l : Array ByteArray) (Q : γ → Array ByteArray → Prop) :
    RetL (throw e : LogV γ) l Q := fun _ _ h => by cases h

theorem retl_throw_bind {α γ : Type} (e : String) (f : α → LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) : RetL (throw e >>= f) l Q := fun _ _ h => by
  obtain ⟨_, _, hx, -⟩ := go_bind_ok h
  cases hx

theorem retl_pure {γ : Type} (a : γ) (l : Array ByteArray) (Q : γ → Array ByteArray → Prop) (h : Q a l) :
    RetL (pure a : LogV γ) l Q := fun _ _ hv => by
  cases hv
  exact h

theorem retl_let {α γ : Type} (e : α) (b : α → LogV γ) (l : Array ByteArray)
    (Q : γ → Array ByteArray → Prop) (h : ∀ x, x = e → RetL (b x) l Q) : RetL (b e) l Q := h e rfl

open Lean Elab Tactic Meta in
/-- One step of `retl_walk` on the goal `RetL m l Q`, by `m`'s head; `none` when the goal is not a `RetL`. -/
def retlStep (keep : Array Name) (g : MVarId) : TacticM (Option (List MVarId)) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``RetL 4 do return none
  let γ := ty.getArg! 0
  let l := ty.getArg! 2
  let q := ty.getArg! 3
  let m := (ty.getArg! 1).headBeta.consumeMData
  let ret (m l : Expr) : Expr := mkApp4 (mkConst ``RetL) γ m l q
  let lγ := mkApp (mkConst ``LogV) γ
  let logT := mkApp (mkConst ``Array [0]) (mkConst ``ByteArray)
  let close (pf : Expr) (gs : List Expr) : MetaM (List MVarId) := do
    g.assign pf
    gs.mapM fun e => do
      let mv := e.mvarId!
      let n ← forallTelescope (← mv.getType) fun xs _ => pure xs.size
      return (← mv.introNP n).2
  let goal (ty : Expr) : MetaM Expr := mkFreshExprSyntheticOpaqueMVar ty
  -- a value takes its binder's name, its run or equation `h_` and its log `l_` that name
  let named (pre : String) (n : Name) : Name :=
    if n.hasMacroScopes then n else .mkSimple (pre ++ n.toString)
  let binder (f : Expr) : Name := match f.consumeMData with
    | .lam n .. => n
    | _ => `a
  match m with
  | .letE n t v b _ =>
      let isJp ← forallTelescope t fun _ r => isDefEq r lγ
      if isJp && v.consumeMData.isLambda then
        -- a join point is inlined: `Q` names the walk's first log, so its body can't be walked from an arbitrary one
        return some [← g.replaceTargetDefEq (ret (b.instantiate1 v) l)]
      else
        let h ← goal (← withLocalDeclD n t fun x => do
          withLocalDeclD (named "h_" n) (← mkEq x v) fun hx => do
            mkForallFVars #[x, hx] (ret (b.instantiate1 x) l))
        some <$> close (mkAppN (mkConst ``retl_let) #[t, γ, v, .lam n t b .default, l, q, h]) [h]
  | _ =>
  if m.isAppOfArity ``Bind.bind 6 then
    let α := m.getArg! 2
    let x := m.getArg! 4
    let f := m.getArg! 5
    let n := binder f
    let xc := x.consumeMData
    if xc.isAppOfArity ``liftM 5 then
      let v := xc.getArg! 4
      if keep.any v.getAppFn.isConstOf then
        let h ← goal (← withLocalDeclD n α fun a => do
          let eq ← mkEq v (mkApp3 (mkConst ``Except.ok [0, 0]) (mkConst ``String) α a)
          withLocalDeclD (named "h_" n) eq fun hx => do mkForallFVars #[a, hx] (ret (mkApp f a).headBeta l))
        some <$> close (mkAppN (mkConst ``retl_lift) #[α, γ, v, f, l, q, h]) [h]
      else
        let h ← goal (← withLocalDeclD n α fun a => do mkForallFVars #[a] (ret (mkApp f a).headBeta l))
        some <$> close (mkAppN (mkConst ``retl_lift') #[α, γ, v, f, l, q, h]) [h]
    else if xc.isAppOfArity ``Pure.pure 4 then
      let a := xc.getArg! 3
      let h ← goal (ret (mkApp f a).headBeta l)
      some <$> close (mkAppN (mkConst ``retl_pure_bind) #[α, γ, a, f, l, q, h]) [h]
    else if xc.isAppOfArity ``MonadExcept.throw 5 then
      some <$> close (mkAppN (mkConst ``retl_throw_bind) #[α, γ, xc.getArg! 4, f, l, q]) []
    else if xc.getAppFn.isConstOf ``SessionSpec.dgWith && (xc.getArg! (xc.getAppNumArgs - 3)).isAppOfArity
        ``Hashes.logged 1 then
      let n := xc.getAppNumArgs
      let hh := (xc.getArg! (n - 3)).getArg! 0
      let spec := xc.getArg! (n - 2)
      let b := xc.getArg! (n - 1)
      let v ← mkAppM ``ite #[← mkEq (← mkAppM ``SessionSpec.sha512Digests #[spec]) (mkConst ``Bool.true),
        mkApp hh b, mkApp (mkConst ``Sha256.hash) b]
      let h ← goal (ret (mkApp f v).headBeta (← mkAppM ``HAppend.hAppend #[l, ← mkAppM ``dgL #[spec, b]]))
      some <$> close (mkAppN (mkConst ``retl_dg_bind) #[γ, hh, spec, b, f, l, q, h]) [h]
    else
      let h ← goal (← withLocalDeclD n α fun a => withLocalDeclD (named "l_" n) logT fun l₁ => do
        let eq ← mkEq (mkApp4 (mkConst ``StV.go) logT α x l)
          (← mkAppM ``Prod.mk #[mkApp3 (mkConst ``Except.ok [0, 0]) (mkConst ``String) α a, l₁])
        withLocalDeclD (named "h_" n) eq fun hx => do mkForallFVars #[a, l₁, hx] (ret (mkApp f a).headBeta l₁))
      some <$> close (mkAppN (mkConst ``retl_bind) #[α, γ, x, f, l, q, h]) [h]
  else if m.isAppOfArity ``Functor.map 6 then
    return some (← g.apply (← mkConstWithFreshMVarLevels ``retl_map))
  else if m.isAppOfArity ``ite 5 then
    let c := m.getArg! 1
    let ht ← goal (← withLocalDeclD `hc c fun h => do mkForallFVars #[h] (ret (m.getArg! 3) l))
    let he ← goal (← withLocalDeclD `hc (mkNot c) fun h => do mkForallFVars #[h] (ret (m.getArg! 4) l))
    some <$> close (mkAppN (mkConst ``retl_ite) #[γ, m.getArg! 1, m.getArg! 2, m.getArg! 3, m.getArg! 4, l, q, ht, he])
      [ht, he]
  else if m.isAppOfArity ``dite 5 then
    let c := m.getArg! 1
    let ht ← goal (← withLocalDeclD `h c fun h => do mkForallFVars #[h] (ret (mkApp (m.getArg! 3) h).headBeta l))
    let he ← goal (← withLocalDeclD `h (mkNot c) fun h => do
      mkForallFVars #[h] (ret (mkApp (m.getArg! 4) h).headBeta l))
    some <$> close (mkAppN (mkConst ``retl_dite) #[γ, c, m.getArg! 2, m.getArg! 3, m.getArg! 4, l, q, ht, he])
      [ht, he]
  else if m.isAppOfArity ``MonadExcept.throw 5 then
    some <$> close (mkAppN (mkConst ``retl_throw) #[γ, m.getArg! 4, l, q]) []
  else if m.isAppOfArity ``Pure.pure 4 then
    let h ← goal (mkApp2 q (m.getArg! 3) l).headBeta
    some <$> close (mkAppN (mkConst ``retl_pure) #[γ, m.getArg! 3, l, q, h]) [h]
  else if ← isMatcherApp m then
    return some (← Lean.Elab.Tactic.run g (evalTactic (← `(tactic| split))))
  else return none

open Lean Elab Tactic Meta in
/-- `retl_walk [c, …]` walks the goal `RetL m l Q` to every return of `m`, leaving the goal `Q a l'` at each `pure a`
(see `retlStep`). -/
elab "retl_walk" "[" cs:ident,* "]" : tactic => do
  let mut keep := #[]
  for c in cs.getElems do keep := keep.push (← realizeGlobalConstNoOverloadWithInfo c)
  let mut todo := (← getGoals)
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← retlStep keep g with
      | some new => todo := new ++ rest
      | none => left := left.push g; todo := rest
  setGoals left.toList

/-! ## Loops -/

theorem go_map {α β : Type} (g : α → β) (x : LogV α) (l : Array ByteArray) {b : β} {l' : Array ByteArray}
    (h : (g <$> x).go l = (.ok b, l')) : ∃ a, x.go l = (.ok a, l') ∧ b = g a := by
  rw [map_eq_bind_pure_comp] at h
  obtain ⟨a, l₁, hx, hp⟩ := go_bind_ok h
  cases hp
  exact ⟨a, hx, rfl⟩

/-- **A list's `mapM` logs its elements' logs in order.** -/
theorem list_mapM_log {α β : Type} {g : α → LogV β} {R : α → β → List ByteArray → Prop}
    (hg : ∀ x l, RetL (g x) l (fun y l' => ∃ d, l'.toList = l.toList ++ d ∧ R x y d)) :
    ∀ (xs : List α) (l : Array ByteArray), RetL (xs.mapM g) l (fun ys l' => ∃ yds : List (β × List ByteArray),
      ys = yds.map Prod.fst ∧ List.Forall₂ (fun x yd => R x yd.1 yd.2) xs yds ∧
        l'.toList = l.toList ++ (yds.map Prod.snd).flatten)
  | [], l => by
    rw [List.mapM_nil]
    exact retl_pure _ _ _ ⟨[], rfl, .nil, by simp⟩
  | x :: xs, l => by
    rw [List.mapM_cons]
    refine retl_bind _ _ _ _ fun y l₁ hy => retl_bind _ _ _ _ fun ys l₂ hys => retl_pure _ _ _ ?_
    obtain ⟨d, hd, hR⟩ := hg x l y l₁ hy
    obtain ⟨yds, rfl, hF, hl⟩ := list_mapM_log hg xs l₁ ys l₂ hys
    exact ⟨(y, d) :: yds, rfl, .cons hR hF, by simp [hl, hd]⟩

/-- **An array's `mapM` logs its elements' logs in order.** -/
theorem mapM_log {α β : Type} {g : α → LogV β} {R : α → β → List ByteArray → Prop}
    (hg : ∀ x l, RetL (g x) l (fun y l' => ∃ d, l'.toList = l.toList ++ d ∧ R x y d))
    {xs : Array α} {l : Array ByteArray} {ys : Array β} {l' : Array ByteArray}
    (h : (xs.mapM g).go l = (.ok ys, l')) : ∃ yds : List (β × List ByteArray),
      ys.toList = yds.map Prod.fst ∧ List.Forall₂ (fun x yd => R x yd.1 yd.2) xs.toList yds ∧
        l'.toList = l.toList ++ (yds.map Prod.snd).flatten := by
  rw [Array.mapM_eq_mapM_toList] at h
  obtain ⟨ys', h', rfl⟩ := go_map _ _ _ h
  obtain ⟨yds, rfl, hF, hl⟩ := list_mapM_log hg xs.toList l ys' l' h'
  exact ⟨yds, by simp, hF, hl⟩

theorem mapIdxM_go_log {α β : Type} {g : ℕ → α → LogV β} {R : α → β → List ByteArray → Prop}
    (hg : ∀ i x l, RetL (g i x) l (fun y l' => ∃ d, l'.toList = l.toList ++ d ∧ R x y d)) :
    ∀ (xs : List α) (acc : Array β) (l : Array ByteArray), RetL (List.mapIdxM.go g xs acc) l
      (fun ys l' => ∃ yds : List (β × List ByteArray), ys = acc.toList ++ yds.map Prod.fst ∧
        List.Forall₂ (fun x yd => R x yd.1 yd.2) xs yds ∧ l'.toList = l.toList ++ (yds.map Prod.snd).flatten)
  | [], acc, l => by
    simp only [List.mapIdxM.go]
    exact retl_pure _ _ _ ⟨[], by simp, .nil, by simp⟩
  | x :: xs, acc, l => by
    simp only [List.mapIdxM.go]
    refine retl_bind _ _ _ _ fun y l₁ hy => fun ys l₂ hys => ?_
    obtain ⟨d, hd, hR⟩ := hg _ x l y l₁ hy
    obtain ⟨yds, rfl, hF, hl⟩ := mapIdxM_go_log hg xs (acc.push y) l₁ ys l₂ hys
    exact ⟨(y, d) :: yds, by simp, .cons hR hF, by simp [hl, hd]⟩

/-- **An array's `mapIdxM` logs its elements' logs in order.** -/
theorem mapIdxM_log {α β : Type} {g : ℕ → α → LogV β} {R : α → β → List ByteArray → Prop}
    (hg : ∀ i x l, RetL (g i x) l (fun y l' => ∃ d, l'.toList = l.toList ++ d ∧ R x y d))
    {xs : Array α} {l : Array ByteArray} {ys : Array β} {l' : Array ByteArray}
    (h : (xs.mapIdxM g).go l = (.ok ys, l')) : ∃ yds : List (β × List ByteArray),
      ys.toList = yds.map Prod.fst ∧ List.Forall₂ (fun x yd => R x yd.1 yd.2) xs.toList yds ∧
        l'.toList = l.toList ++ (yds.map Prod.snd).flatten := by
  have e : (Array.toList <$> xs.mapIdxM g).go l = (.ok ys.toList, l') := by
    rw [map_eq_bind_pure_comp, go_bind, h]
    rfl
  rw [Array.toList_mapIdxM] at e
  obtain ⟨yds, hys, hF, hl⟩ := mapIdxM_go_log hg xs.toList #[] l ys.toList l' e
  exact ⟨yds, by simpa using hys, hF, hl⟩

/-- **A loop over a list that always continues logs its iterations' logs in order.** -/
theorem list_forIn_log {α : Type} {body : α → PUnit → LogV (ForInStep PUnit)}
    {R : α → List ByteArray → Prop} :
    ∀ (xs : List α), (∀ x ∈ xs, ∀ l, RetL (body x ⟨⟩) l
      (fun st l' => st = .yield ⟨⟩ ∧ ∃ d, l'.toList = l.toList ++ d ∧ R x d)) →
    ∀ (l : Array ByteArray), RetL (forIn xs ⟨⟩ body) l
      (fun _ l' => ∃ ds : List (List ByteArray), List.Forall₂ R xs ds ∧ l'.toList = l.toList ++ ds.flatten)
  | [], _, l => by
    rw [List.forIn_nil]
    exact retl_pure _ _ _ ⟨[], .nil, by simp⟩
  | x :: xs, hb, l => by
    rw [List.forIn_cons]
    refine retl_bind _ _ _ _ fun st l₁ hst => ?_
    obtain ⟨rfl, d, hd, hR⟩ := hb x (List.mem_cons_self ..) l st l₁ hst
    intro u l₂ hu
    obtain ⟨ds, hF, hl⟩ := list_forIn_log xs (fun y hy => hb y (List.mem_cons_of_mem _ hy)) l₁ u l₂ hu
    exact ⟨d :: ds, .cons hR hF, by simp [hl, hd]⟩

/-- **A loop over an array that always continues logs its iterations' logs in order.** -/
theorem forIn_log {α : Type} {body : α → PUnit → LogV (ForInStep PUnit)} {R : α → List ByteArray → Prop}
    {xs : Array α} (hb : ∀ x ∈ xs.toList, ∀ l, RetL (body x ⟨⟩) l
      (fun st l' => st = .yield ⟨⟩ ∧ ∃ d, l'.toList = l.toList ++ d ∧ R x d))
    {l l' : Array ByteArray} {u : PUnit} (h : (forIn xs ⟨⟩ body).go l = (.ok u, l')) :
    ∃ ds : List (List ByteArray), List.Forall₂ R xs.toList ds ∧ l'.toList = l.toList ++ ds.flatten := by
  rw [← Array.forIn_toList] at h
  exact list_forIn_log xs.toList hb l u l' h

end FlockDeploy.KS
