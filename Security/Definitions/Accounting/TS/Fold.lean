import Mathlib.Data.Fin.Tuple.Basic

/-!
# A fold of words (trusted)

The receipt folds' common form: a run of `N` words absorbed one at a time into a state of `c` registers, word `i` by
its own step. A step binds when it is injective in the word at every state and in the state at every word; duo's and
fx1's steps with odd multipliers do (`Pouw.Guarantees.Receipt.DuoStepBijective`, `Fx1StepBijective`).
-/

namespace Accounting.TS

/-- The state after the words `d` from the state `s`, word `i` absorbed by the step `g i`. -/
def foldRun {α : Type} {c N : ℕ} (g : Fin N → (Fin c → α) → α → (Fin c → α)) (s : Fin c → α) (d : Fin N → α) :
    Fin c → α :=
  Fin.foldl N (fun t i => g i t (d i)) s

/-- A fold step that binds its word at every state and its state at every word. -/
def BindingStep (c : ℕ) (α : Type) : Type :=
  {g : (Fin c → α) → α → (Fin c → α) // (∀ s, Function.Injective (g s)) ∧ ∀ d, Function.Injective fun s => g s d}

end Accounting.TS
