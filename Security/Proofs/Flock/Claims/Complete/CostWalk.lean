import Flock.With.Transcript

/-!
# Cost bounds, and the walk that reads them off a definition

A cost bound (`Cost`) on the computations of a verdict monad: `P x n`, which `pure`, `throw` and the lift of a `V`
action keep at `0`, `bind` adds, and a larger `n` keeps; lifted through `StateT` (`Cost.st`), with its rules for
branches, loops, traversals and state actions. `cost_walk` walks a definition with them, the bound left a metavariable
that the walk assigns. A macro is code run while compiling, so it is alone here, the module `compile_time` lists, apart
from the proofs that use it (`FlockComplete`'s `Count`).
-/

namespace FlockComplete

open Flock

/-- **A cost bound** on the computations of a verdict monad: `pure`, `throw` and the lift of a `V` action cost `0`, and a
`bind` the sum of its parts. -/
structure Cost (M : Type → Type) [Monad M] [MonadExceptOf String M] [MonadLiftT V M] where
  P : {α : Type} → M α → Nat → Prop
  mono : ∀ {α : Type} {x : M α} {n m : Nat}, P x n → n ≤ m → P x m
  pure : ∀ {α : Type} (a : α), P (Pure.pure a) 0
  bind : ∀ {α β : Type} {x : M α} {f : α → M β} {n m : Nat}, P x n → (∀ a, P (f a) m) → P (x >>= f) (n + m)
  throw : ∀ {α : Type} (e : String) (n : Nat), P (throw e : M α) n
  lift : ∀ {α : Type} (v : V α), P (monadLift v : M α) 0

namespace Cost

/-- `C` on the runs from each state of `σ`. -/
def st {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) (σ : Type) :
    Cost (StateT σ M) where
  P x n := ∀ s, C.P (x s) n
  mono h hnm s := C.mono (h s) hnm
  pure a s := C.pure (a, s)
  bind hx hf s := C.bind (hx s) fun p => hf p.1 p.2
  throw e n s := C.mono (C.bind (C.throw e n) fun a => C.pure (a, s)) (by omega)
  lift v s := C.bind (C.lift v) fun a => C.pure (a, s)

theorem lift_st {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {σ α : Type}
    {x : M α} {n : Nat} (h : C.P x n) : (C.st σ).P (StateT.lift x) n :=
  fun s => C.bind h fun a => C.pure (a, s)

theorem monadLift_st {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M)
    {σ α : Type} {x : M α} {n : Nat} (h : C.P x n) : (C.st σ).P (monadLift x : StateT σ M α) n :=
  C.lift_st h

theorem get {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {σ : Type} :
    (C.st σ).P (get : StateT σ M σ) 0 :=
  fun s => C.pure (s, s)

theorem set {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {σ : Type}
    (s' : σ) : (C.st σ).P (set s' : StateT σ M PUnit) 0 :=
  fun _ => C.pure (PUnit.unit, s')

theorem modify {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {σ : Type}
    (f : σ → σ) : (C.st σ).P (modify f : StateT σ M PUnit) 0 :=
  fun s => C.pure (PUnit.unit, f s)

theorem ite {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {α : Type}
    {c : Prop} [Decidable c] {x y : M α} {n m : Nat} (hx : c → C.P x n) (hy : ¬c → C.P y m) :
    C.P (if c then x else y) (max n m) := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact C.mono (hx hc) (Nat.le_max_left _ _)
  · rw [ite_eq_right hc]; exact C.mono (hy hc) (Nat.le_max_right _ _)

theorem forIn_list {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {α β : Type}
    (l : List α) (b : β) {f : α → β → M (ForInStep β)} {k : Nat} (h : ∀ a ∈ l, ∀ b, C.P (f a b) k) :
    C.P (forIn l b f) (l.length * k) := by
  induction l generalizing b with
  | nil => exact C.mono (C.pure b) (by omega)
  | cons a l ih =>
    rw [List.forIn_cons]
    refine C.mono (C.bind (h a List.mem_cons_self b) fun r => ?_) (n := k + l.length * k)
      (by rw [List.length_cons, Nat.succ_mul]; omega)
    cases r with
    | done b => exact C.mono (C.pure b) (by omega)
    | yield b => exact ih b fun a ha => h a (List.mem_cons_of_mem _ ha)

theorem forIn_range {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {β : Type}
    (r : Std.Legacy.Range) (b : β) {f : Nat → β → M (ForInStep β)} {k : Nat} (h : ∀ a b, C.P (f a b) k) :
    C.P (forIn r b f) (r.size * k) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := C.forIn_list (List.range' r.start r.size r.step) b fun a _ => h a
  rwa [List.length_range'] at this

/-- A loop over `[0:n]` at `n` stated: the walk takes it only at that `n`. -/
theorem forIn_upto {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {β : Type}
    (n : Nat) (b : β) {f : Nat → β → M (ForInStep β)} {k : Nat} (h : ∀ a b, C.P (f a b) k) :
    C.P (forIn [0:n] b f) (n * k) := by
  have := C.forIn_range [0:n] b h
  simpa [Std.Legacy.Range.size] using this

/-- A loop over `[0:n]` whose last step costs `k'` and each other `k`. -/
theorem forIn_last {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {β : Type}
    (n : Nat) (b : β) {f : Nat → β → M (ForInStep β)} {k k' : Nat} (h : ∀ i, i + 1 < n → ∀ b, C.P (f i b) k)
    (h' : 1 ≤ n → ∀ b, C.P (f (n - 1) b) k') : C.P (forIn [0:n] b f) ((n - 1) * k + k') := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices ∀ j s (b : β), s + j + 1 = n → C.P (forIn (List.range' s (j + 1) 1) b f) (j * k + k') by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · exact C.mono (C.pure b) (Nat.zero_le _)
    · have := this (n - 1) 0 b (by omega)
      rwa [Nat.sub_add_cancel hn] at this
  intro j
  induction j with
  | zero =>
    intro s b hs
    rw [List.range'_succ, List.forIn_cons]
    have hl := h' (by omega) b
    rw [show n - 1 = s by omega] at hl
    refine C.mono (C.bind hl fun r => ?_) (n := k' + 0) (by omega)
    cases r with
    | done b => exact C.pure b
    | yield b => show C.P (forIn [] b f) 0; rw [List.forIn_nil]; exact C.pure b
  | succ j ih =>
    intro s b hs
    rw [List.range'_succ, List.forIn_cons]
    refine C.mono (C.bind (h s (by omega) b) fun r => ?_) (n := k + (j * k + k')) (by rw [Nat.succ_mul]; omega)
    cases r with
    | done b => exact C.mono (C.pure b) (by omega)
    | yield b => exact ih (s + 1) b (by omega)

theorem forIn_array {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {α β : Type}
    (xs : Array α) (b : β) {f : α → β → M (ForInStep β)} {k : Nat} (h : ∀ a ∈ xs, ∀ b, C.P (f a b) k) :
    C.P (forIn xs b f) (xs.size * k) := by
  rw [← Array.forIn_toList]
  have := C.forIn_list xs.toList b fun a ha => h a (Array.mem_toList_iff.1 ha)
  rwa [Array.length_toList] at this

theorem bind0 {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {α β : Type}
    {x : M α} {f : α → M β} (hx : C.P x 0) (hf : ∀ a, C.P (f a) 0) : C.P (x >>= f) 0 :=
  C.bind hx hf

/-- A loop whose steps cost nothing, over a list the cost cannot name (one computed by an earlier step). -/
theorem forIn_list0 {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {α β : Type}
    (l : List α) (b : β) {f : α → β → M (ForInStep β)} (h : ∀ a ∈ l, ∀ b, C.P (f a b) 0) : C.P (forIn l b f) 0 := by
  have := C.forIn_list l b h
  rwa [Nat.mul_zero] at this

theorem forIn_array0 {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {α β : Type}
    (xs : Array α) (b : β) {f : α → β → M (ForInStep β)} (h : ∀ a ∈ xs, ∀ b, C.P (f a b) 0) :
    C.P (forIn xs b f) 0 := by
  have := C.forIn_array xs b h
  rwa [Nat.mul_zero] at this

theorem mapM_list {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] [LawfulMonad M]
    (C : Cost M) {α β : Type} (l : List α) {f : α → M β} (h : ∀ a, C.P (f a) 0) : C.P (l.mapM f) 0 := by
  induction l with
  | nil => exact C.pure []
  | cons a l ih =>
    rw [List.mapM_cons]
    exact C.bind (h a) fun b => C.bind ih fun bs => C.pure _

theorem mapM_array {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] [LawfulMonad M]
    (C : Cost M) {α β : Type} (xs : Array α) {f : α → M β} (h : ∀ a, C.P (f a) 0) : C.P (xs.mapM f) 0 := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  exact C.bind (C.mapM_list _ h) fun bs => C.pure _

theorem run {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {σ α : Type}
    {x : StateT σ M α} {n : Nat} (h : (C.st σ).P x n) (s : σ) : C.P (x.run s) n := h s

theorem liftTM {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M) {α : Type}
    (x : Transcript.TM α) : (C.st Transcript).P (Flock.liftTM x) 0 :=
  fun t => C.lift (x t)

end Cost

/-- One step of walking a definition under a cost bound: a `bind`, one of `ls`, a `pure`, `throw` or lift, a branch, a
loop over a list or array, a traversal, a state action, or a `let` (moved to the context). A loop over a range is one
of `ls` (`Cost.forIn_range`, or `Cost.forIn_upto` at its bound), so a walk can leave one to be bounded by hand
(`Cost.forIn_last`); the walk works on the first goal, so it stops there, and resumes on the goals after it. A bound is
assigned where its step starts, so a loop over a value an earlier step computed costs `0` (`Cost.forIn_array0`). -/
syntax "cost_walk" (" [" term,* "]")? : tactic

macro_rules
  | `(tactic| cost_walk) => `(tactic| cost_walk [])
  | `(tactic| cost_walk [$ls,*]) => `(tactic| repeat (first
    | extract_lets
    | (with_reducible apply Cost.bind)
    | (with_reducible apply Cost.bind0)
    $[| (with_reducible apply $ls)]*
    | (with_reducible intro _)
    | (with_reducible apply Cost.pure)
    | (with_reducible apply Cost.throw)
    | (with_reducible apply Cost.lift)
    | (with_reducible apply Cost.liftTM)
    | (with_reducible apply Cost.monadLift_st)
    | (with_reducible apply Cost.lift_st)
    | (with_reducible apply Cost.get)
    | (with_reducible apply Cost.set)
    | (with_reducible apply Cost.modify)
    | (with_reducible apply Cost.forIn_array)
    | (with_reducible apply Cost.forIn_array0)
    | (with_reducible apply Cost.forIn_list)
    | (with_reducible apply Cost.forIn_list0)
    | (with_reducible apply Cost.mapM_array)
    | (with_reducible apply Cost.run)
    | (with_reducible apply Cost.ite)
    | split))

end FlockComplete
