/-
A simulation between two verdict monads: a relation `R` between computations of `M` and of `N` that `pure`, `bind`,
`throw` and the lift of a `V` action respect. Lean has no parametricity theorem, so `Flock.With.SimVerify` and
`Flock.With.SimZk` prove, once per `…With` function, that a simulation relates its runs under related oracles; a logging
or counting oracle is then one simulation each (`Flock.With.Log`).

Here: the simulation, its lift through `StateT` (the transcript and the Ligerito state), its rules for the loops and
traversals the verifier uses (`forIn` over lists, arrays and ranges, `Array.mapM`, `Array.mapIdxM`, `List.findM?`), related
oracles, and `sim_walk`, which walks two copies of one definition in step.
-/
import Flock.With.Transcript

namespace Flock.With

variable {M N : Type → Type} [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N] [MonadLiftT V M]
  [MonadLiftT V N]

/-- A relation between computations of `M` and of `N` that `pure`, `bind`, `throw` and the lift of a `V` action respect. -/
structure Sim (M N : Type → Type) [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N] [MonadLiftT V M]
    [MonadLiftT V N] where
  R : {α : Type} → M α → N α → Prop
  pure : ∀ {α : Type} (a : α), R (Pure.pure a) (Pure.pure a)
  bind : ∀ {α β : Type} {x : M α} {y : N α} {f : α → M β} {g : α → N β}, R x y → (∀ a, R (f a) (g a)) →
    R (x >>= f) (y >>= g)
  throw : ∀ {α : Type} (e : String), R (throw e : M α) (throw e : N α)
  lift : ∀ {α : Type} (v : V α), R (monadLift v : M α) (monadLift v : N α)

namespace Sim

/-- `S` on the runs from each state of `σ`. -/
def st (S : Sim M N) (σ : Type) : Sim (StateT σ M) (StateT σ N) where
  R x y := ∀ s, S.R (x s) (y s)
  pure a s := S.pure (a, s)
  bind hxy hfg s := S.bind (hxy s) fun p => hfg p.1 p.2
  throw e s := S.bind (S.throw e) fun a => S.pure (a, s)
  lift v s := S.bind (S.lift v) fun a => S.pure (a, s)

theorem st_intro (S : Sim M N) {σ α : Type} {x : StateT σ M α} {y : StateT σ N α} (h : ∀ s, S.R (x s) (y s)) :
    (S.st σ).R x y := h

theorem run (S : Sim M N) {σ α : Type} {x : StateT σ M α} {y : StateT σ N α} (h : (S.st σ).R x y) (s : σ) :
    S.R (x.run s) (y.run s) := h s

theorem lift_st (S : Sim M N) {σ α : Type} {x : M α} {y : N α} (h : S.R x y) :
    (S.st σ).R (StateT.lift x) (StateT.lift y) := fun s => S.bind h fun a => S.pure (a, s)

theorem monadLift_st (S : Sim M N) {σ α : Type} {x : M α} {y : N α} (h : S.R x y) :
    (S.st σ).R (monadLift x : StateT σ M α) (monadLift y : StateT σ N α) := S.lift_st h

theorem monadLift_st2 (S : Sim M N) {σ τ α : Type} {x : M α} {y : N α} (h : S.R x y) :
    ((S.st σ).st τ).R (monadLift x : StateT τ (StateT σ M) α) (monadLift y : StateT τ (StateT σ N) α) :=
  (S.st σ).lift_st (S.lift_st h)

theorem get (S : Sim M N) {σ : Type} : (S.st σ).R (get : StateT σ M σ) get :=
  fun s => S.pure (s, s)

theorem set (S : Sim M N) {σ : Type} (s' : σ) : (S.st σ).R (set s' : StateT σ M PUnit) (set s') :=
  fun _ => S.pure (PUnit.unit, s')

theorem modify (S : Sim M N) {σ : Type} (f : σ → σ) : (S.st σ).R (modify f : StateT σ M PUnit) (modify f) :=
  fun s => S.pure (PUnit.unit, f s)

theorem modifyGet (S : Sim M N) {σ α : Type} (f : σ → α × σ) :
    (S.st σ).R (modifyGet f : StateT σ M α) (modifyGet f) :=
  fun s => S.pure (f s)

theorem ite (S : Sim M N) {α : Type} {c : Prop} [Decidable c] {x x' : M α} {y y' : N α} (h₁ : c → S.R x y)
    (h₂ : ¬c → S.R x' y') : S.R (if c then x else x') (if c then y else y') := by
  by_cases hc : c
  · rw [ite_eq_left hc, ite_eq_left hc]; exact h₁ hc
  · rw [ite_eq_right hc, ite_eq_right hc]; exact h₂ hc

theorem dite (S : Sim M N) {α : Type} {c : Prop} [Decidable c] {x : c → M α} {x' : ¬c → M α} {y : c → N α}
    {y' : ¬c → N α} (h₁ : ∀ h, S.R (x h) (y h)) (h₂ : ∀ h, S.R (x' h) (y' h)) :
    S.R (if h : c then x h else x' h) (if h : c then y h else y' h) := by
  by_cases hc : c
  · rw [dite_eq_left hc, dite_eq_left hc]; exact h₁ hc
  · rw [dite_eq_right hc, dite_eq_right hc]; exact h₂ hc

theorem forIn_list (S : Sim M N) {α β : Type} (l : List α) (b : β) {f : α → β → M (ForInStep β)}
    {g : α → β → N (ForInStep β)} (h : ∀ a b, S.R (f a b) (g a b)) : S.R (forIn l b f) (forIn l b g) := by
  induction l generalizing b with
  | nil => exact S.pure b
  | cons a l ih =>
    rw [List.forIn_cons, List.forIn_cons]
    refine S.bind (h a b) fun r => ?_
    cases r with
    | done b => exact S.pure b
    | yield b => exact ih b

theorem forIn_array (S : Sim M N) {α β : Type} (xs : Array α) (b : β) {f : α → β → M (ForInStep β)}
    {g : α → β → N (ForInStep β)} (h : ∀ a b, S.R (f a b) (g a b)) : S.R (forIn xs b f) (forIn xs b g) := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact S.forIn_list _ b h

theorem forIn_range (S : Sim M N) {β : Type} (r : Std.Legacy.Range) (b : β) {f : Nat → β → M (ForInStep β)}
    {g : Nat → β → N (ForInStep β)} (h : ∀ a b, S.R (f a b) (g a b)) : S.R (forIn r b f) (forIn r b g) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  exact S.forIn_list _ b h

theorem mapM_list (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {α β : Type} (l : List α) {f : α → M β} {g : α → N β}
    (h : ∀ a, S.R (f a) (g a)) : S.R (l.mapM f) (l.mapM g) := by
  induction l with
  | nil => exact S.pure []
  | cons a l ih =>
    rw [List.mapM_cons, List.mapM_cons]
    exact S.bind (h a) fun b => S.bind ih fun bs => S.pure _

theorem mapM_array (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {α β : Type} (xs : Array α) {f : α → M β}
    {g : α → N β} (h : ∀ a, S.R (f a) (g a)) : S.R (xs.mapM f) (xs.mapM g) := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList, map_eq_pure_bind, map_eq_pure_bind]
  exact S.bind (S.mapM_list _ h) fun bs => S.pure _

theorem mapIdxM_array (S : Sim M N) {α β : Type} (xs : Array α) {f : Nat → α → M β} {g : Nat → α → N β}
    (h : ∀ i a, S.R (f i a) (g i a)) : S.R (xs.mapIdxM f) (xs.mapIdxM g) := by
  unfold Array.mapIdxM Array.mapFinIdxM
  suffices hs : ∀ i j (inv : i + j = xs.size) (bs : Array β),
      S.R (Array.mapFinIdxM.map xs (fun i a _ => f i a) i j inv bs) (Array.mapFinIdxM.map xs (fun i a _ => g i a) i j inv bs)
    from hs _ _ _ _
  intro i
  induction i with
  | zero => intro j inv bs; exact S.pure bs
  | succ i ih => intro j inv bs; exact S.bind (h _ _) fun b => ih _ _ _

theorem findM?_list (S : Sim M N) {α : Type} (l : List α) {p : α → M Bool} {q : α → N Bool} (h : ∀ a, S.R (p a) (q a)) :
    S.R (l.findM? p) (l.findM? q) := by
  induction l with
  | nil => exact S.pure none
  | cons a l ih =>
    simp only [List.findM?]
    refine S.bind (h a) fun b => ?_
    cases b
    · exact ih
    · exact S.pure _

/-- Two oracles whose every hash `S` relates. -/
structure Hashes (S : Sim M N) (hs : Flock.Hashes M) (hs' : Flock.Hashes N) : Prop where
  sha512 : ∀ b, S.R (hs.sha512 b) (hs'.sha512 b)
  sha256 : ∀ b, S.R (hs.sha256 b) (hs'.sha256 b)
  blake3 : ∀ b, S.R (hs.blake3 b) (hs'.blake3 b)
  blake3Keyed : ∀ k b, S.R (hs.blake3Keyed k b) (hs'.blake3Keyed k b)

/-- Two Merkle schemes' node and leaf, related by `S`. -/
structure Ops (S : Sim M N) (ops : MerkleOps M) (ops' : MerkleOps N) : Prop where
  node : ∀ l r, S.R (ops.node l r) (ops'.node l r)
  leaf : ∀ row salt, S.R (ops.leaf row salt) (ops'.leaf row salt)

theorem liftTM (S : Sim M N) {α : Type} (x : Transcript.TM α) : (S.st Transcript).R (Flock.liftTM x) (Flock.liftTM x) :=
  fun t => S.lift (x t)

end Sim

/-- One step of walking two copies of a definition: a `bind`, `pure`, `throw` or lift of either side's monad, a branch, a
loop or traversal, a state action, a `let` (moved to the context, so a `do` block's join point is walked at each of
its calls but never copied into the goal), or one of `lemmas` (each relating two calls of a `…With` function). -/
syntax "sim_walk" (" [" term,* "]")? : tactic

macro_rules
  | `(tactic| sim_walk) => `(tactic| sim_walk [])
  | `(tactic| sim_walk [$ls,*]) => `(tactic| repeat (first
    | extract_lets
    | (with_reducible apply Sim.bind)
    | (with_reducible intro _)
    | (with_reducible apply Sim.pure)
    | (with_reducible apply Sim.throw)
    | (with_reducible apply Sim.lift)
    | (with_reducible apply Sim.liftTM)
    | (with_reducible apply Sim.monadLift_st2)
    | (with_reducible apply Sim.monadLift_st)
    | (with_reducible apply Sim.lift_st)
    | (with_reducible apply Sim.run)
    | (with_reducible apply Sim.get)
    | (with_reducible apply Sim.set)
    | (with_reducible apply Sim.modify)
    | (with_reducible apply Sim.modifyGet)
    | (with_reducible apply Sim.forIn_range)
    | (with_reducible apply Sim.forIn_array)
    | (with_reducible apply Sim.forIn_list)
    | (with_reducible apply Sim.mapM_array)
    | (with_reducible apply Sim.mapIdxM_array)
    | (with_reducible apply Sim.findM?_list)
    | (with_reducible apply Sim.ite)
    | (with_reducible apply Sim.dite)
    | assumption
    | (with_reducible apply Sim.Hashes.sha512); assumption
    | (with_reducible apply Sim.Hashes.sha256); assumption
    $[| (with_reducible apply $ls)]*
    | split))

end Flock.With
