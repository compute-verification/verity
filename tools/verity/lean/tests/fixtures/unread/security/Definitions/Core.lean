namespace Core

structure Box where
  val : Nat

def double (n : Nat) : Nat := n + n

def pick : Nat → Nat
  | 0 => 1
  | n + 1 => n

def pred' : Nat → Nat
  | 0 => 1
  | n + 1 => n

namespace Guarantees

def Spec : Prop := ∀ b : Box, double b.val = 2 * b.val ∧ pred' b.val ≤ b.val + 1

end Guarantees

def Hyp : Prop := ∀ n : Nat, n = n

def unusedHelper (n : Nat) : Nat := n * 3

abbrev Width := Nat

structure Pair where
  a : Nat
  b : Nat

instance : Inhabited Box := ⟨⟨0⟩⟩

structure Wrap where
  n : Nat

def unwrap (w : Wrap) : Nat := w.n

instance : Coe Wrap Nat := ⟨unwrap⟩

theorem double_eq (n : Nat) : double n = n + n := rfl

theorem unusedLemma (n : Nat) : n + 0 = n := Nat.add_zero n

def citedByTool : Nat := 5

end Core
