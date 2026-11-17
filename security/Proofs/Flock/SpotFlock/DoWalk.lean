module

public import Proofs.Flock.Claims.Sampling.Select.Take
public import Lean
public meta import Lean

/-!
# Walking a `do` block in `Except` by its join points

A reader written as one `do` block leaves a join point at each `match` and `if` whose arms go on to the same rest of
the block: a function of the value just read, which every arm that returns calls. A proof that the block returned
`.ok r` only where some line did has to follow every arm to its jump. Unfolding a join point into its arms copies the
rest of the block into each of them, at every join point, so a block with twenty of them is not walked that way.

The tactics here walk such a block once:

* `jp_step h to r with inv by tac` takes one join point. It names the join point, walks the arms with the rest of the
  block folded in the name, and concludes that the join point returned at some value where `inv` holds, `tac` proving
  `inv` at each arm's jump. `h` becomes the join point's own equation at that value.
* `do_walk h to r` repeats it down the block, with no invariant, to the block's last line, leaving in the context
  what each line returned.
* `head_whnf at h` unfolds a `match` that `split` cannot split (one on an array literal) to the `if`s it is made of,
  and `lit_walk h hn` closes the arms of such a `match` at a value that is not the literal's.

Each tactic only rewrites a hypothesis to one the kernel checks is the same, or closes a goal by `cases`, so a proof
it builds is checked like any other. This module holds the tactics alone: no statement reads it.
-/

@[expose] public section

namespace SpotFlock.DoWalk

open Properties.Spot.SecurityProofs (bind_eq_ok)

open Lean Meta Elab Tactic in
/-- `jp_intro k hk at h`: where `h` is `(have x := v; b) = r` with `v` a function, the join point a `do` block leaves
for the rest of the block, names it `k` with `hk : k = v` and makes `h` the equation `b = r` at `k`. -/
elab "jp_intro " k:ident hk:ident " at " h:ident : tactic => withMainContext do
  let hDecl ← getLocalDeclFromUserName h.getId
  let ty ← instantiateMVars hDecl.type
  let some (_, lhs, rhs) := ty.eq? | throwError "jp_intro: {h} is not an equation"
  let .letE _ t v b _ := lhs.consumeMData | throwError "jp_intro: no `have` at the head of {h}"
  unless (← whnf t).isForall do throwError "jp_intro: the `have` at the head of {h} is not a function"
  let g ← getMainGoal
  let target ← g.getType
  let newTy ← withLocalDeclD k.getId t fun kf => do
    mkForallFVars #[kf] (← mkArrow (← mkEq kf v) (← mkArrow (← mkEq (b.instantiate1 kf) rhs) target))
  let g' ← mkFreshExprSyntheticOpaqueMVar newTy (← g.getTag)
  g.assign (mkApp3 g' v (← mkEqRefl v) (mkFVar hDecl.fvarId))
  let (_, g'') ← g'.mvarId!.introN 3 [k.getId, hk.getId, h.getId]
  replaceMainGoal [← g''.clear hDecl.fvarId]

open Lean Meta Elab Tactic in
/-- `head_zeta at h`: where `h` is `(have x := v; b) = r` with `v` no function, puts `v` for `x`. -/
elab "head_zeta" " at " h:ident : tactic => withMainContext do
  let hDecl ← getLocalDeclFromUserName h.getId
  let ty ← instantiateMVars hDecl.type
  let some (_, lhs, rhs) := ty.eq? | throwError "head_zeta: {h} is not an equation"
  let .letE _ t v b _ := lhs.consumeMData | throwError "head_zeta: no `have` at the head of {h}"
  if (← whnf t).isForall then throwError "head_zeta: the `have` at the head of {h} is a function"
  let g ← (← getMainGoal).replaceLocalDeclDefEq hDecl.fvarId (← mkEq (b.instantiate1 v) rhs)
  replaceMainGoal [g]

open Lean Meta in
/-- `e` with the matchers at its head unfolded, at most `fuel` of them one inside the other, in weak head normal form
without unfolding any other definition. -/
meta def headMatchWhnf : Nat → Expr → MetaM Expr
  | 0, e => whnfCore e.consumeMData
  | fuel + 1, e => do
    let e ← whnfCore e.consumeMData
    let .const n lvls := e.getAppFn | return e
    let aux := match n with
      | .str _ s => s.startsWith "_sparseCasesOn"
      | _ => false
    unless aux || (← isMatcher n) do return e
    let some v := ((← getEnv).find? n).bind (·.value?) | return e
    headMatchWhnf fuel ((v.instantiateLevelParams (← getConstInfo n).levelParams lvls).beta e.getAppArgs)

open Lean Meta Elab Tactic in
/-- `head_whnf at h`: where `h` is `l = r` with a `match` at the head of `l` that `split` cannot split, unfolds the
matcher to the `if`s and casts it is made of. -/
elab "head_whnf" " at " h:ident : tactic => withMainContext do
  let hDecl ← getLocalDeclFromUserName h.getId
  let ty ← instantiateMVars hDecl.type
  let some (_, lhs, rhs) := ty.eq? | throwError "head_whnf: {h} is not an equation"
  let g ← (← getMainGoal).replaceLocalDeclDefEq hDecl.fvarId (← mkEq (← headMatchWhnf 64 lhs) rhs)
  replaceMainGoal [g]

/-- `throws h`: closes a goal whose `h` says a line that throws returned. -/
macro "throws " h:ident : tactic => `(tactic| (cases $h:ident; done))

/-- `jp_step h to r with inv by tac`: one join point of a `do` block. The block from it on returned `r` (`h`), so
the join point did, at a value where `inv` holds: each arm that returns jumps to the join point, and `tac` proves
`inv` there. `h` becomes the join point's own equation at that value. -/
macro "jp_step " h:ident " to " r:term " with " inv:term " by " tac:tacticSeq : tactic => `(tactic|
  (jp_intro k hk at $h:ident
   obtain ⟨_, hw, _⟩ : ∃ w, k w = Except.ok $r ∧ $inv w
   · repeat' first
       | (refine ⟨_, $h, ?_⟩; ($tac); done)
       | (have h' := bind_eq_ok.1 $h; clear $h; obtain ⟨_, _, $h:ident⟩ := h')
       | (cases $h:ident; done)
       | split at $h:ident
       | dsimp only at $h:ident
   clear $h
   subst hk
   dsimp -zeta only at hw
   have $h := hw
   clear hw))

/-- `do_walk h to r`: walks the `do` block of `h : block = .ok r` to its last line, leaving in the context what each
line returned. -/
macro "do_walk " h:ident " to " r:term : tactic => `(tactic|
  repeat' first
    | (jp_step $h to $r with (fun _ => True) by trivial)
    | head_zeta at $h:ident
    | (have h' := bind_eq_ok.1 $h; clear $h; obtain ⟨_, _, $h:ident⟩ := h')
    | (cases $h:ident; done)
    | split at $h:ident)

/-- `lit_walk h hn`: closes a goal whose `h` is a `match` on string and array literals at a value `hn` says is not
the literal's. The matcher is unfolded to its `if`s (`head_whnf`), and each arm throws or contradicts `hn`. -/
macro "lit_walk " h:ident hn:ident : tactic => `(tactic|
  repeat' first
    | (cases $h:ident; done)
    | (apply $hn; assumption)
    | (head_whnf at $h:ident; split at $h:ident))

end SpotFlock.DoWalk
