import Mathlib.Data.Fin.Tuple.Basic

/-!
# Straight-line lane programs and receipt folds (trusted)

A lane program computes on values of one type `α`, a lane's registers. Its first `n` nodes are its inputs, and each
instruction appends one node: any function of `f` earlier nodes. An instruction that reads fewer than `f` repeats a
source. Keys, start states and immediates are constants inside the functions, so a program may be specialised to the
drawn keys. `Prog.run` reads `c` output nodes.

A receipt fold (`foldRun`) absorbs `N` words into a state of `c` registers, word `i` through its own step.
-/

namespace Accounting.Program

/-- A straight-line program on `n` inputs with `m` instructions, each reading `f` earlier nodes: node `i < n` is input
`i`, and node `n + j` is instruction `j`'s result, `op` applied to the nodes `src` names. -/
inductive Prog (α : Type) (n f : ℕ) : ℕ → Type
  | nil : Prog α n f 0
  | snoc {m : ℕ} (P : Prog α n f m) (src : Fin f → Fin (n + m)) (op : (Fin f → α) → α) : Prog α n f (m + 1)

/-- The value of every node of `P` on the inputs `x`. -/
def Prog.eval {α : Type} {n f : ℕ} : {m : ℕ} → Prog α n f m → (Fin n → α) → Fin (n + m) → α
  | _, .nil, x => x
  | _, .snoc P src op, x => Fin.snoc (α := fun _ => α) (P.eval x) (op fun k => P.eval x (src k))

/-- What `P` computes: the values of its `c` output nodes `out` on the inputs `x`. -/
def Prog.run {α : Type} {n f m c : ℕ} (P : Prog α n f m) (out : Fin c → Fin (n + m)) (x : Fin n → α) : Fin c → α :=
  fun o => P.eval x (out o)

/-- Input `i` is essential to `F`: two inputs that differ only at `i` give different results. -/
def Essential {α β : Type} {n : ℕ} (F : (Fin n → α) → β) (i : Fin n) : Prop :=
  ∃ x y : Fin n → α, (∀ j, j ≠ i → x j = y j) ∧ F x ≠ F y

/-- A receipt fold over `N` words: from the state `s` of `c` registers, word `i` is absorbed by its step `g i`. -/
def foldRun {α : Type} {c N : ℕ} (g : Fin N → (Fin c → α) → α → (Fin c → α)) (s : Fin c → α) (d : Fin N → α) :
    Fin c → α :=
  Fin.foldl N (fun t i => g i t (d i)) s

/-- A fold step that binds its word at every state and its state at every word. -/
def BindingStep (c : ℕ) (α : Type) : Type :=
  {g : (Fin c → α) → α → (Fin c → α) // (∀ s, Function.Injective (g s)) ∧ ∀ d, Function.Injective fun s => g s d}

end Accounting.Program
