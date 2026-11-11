import Definitions.Core
import Definitions.Dead

namespace Core

theorem usesDead : Dead.lonely = 7 := by decide

theorem helper (n : Nat) : double n = 2 * n := by
  unfold double
  omega

theorem pred_le : (n : Nat) → pred' n ≤ n + 1
  | 0 => Nat.le_refl 1
  | n + 1 => Nat.le_succ_of_le (Nat.le_succ n)

theorem unusedHere (n : Nat) : double n = n + n := rfl

namespace SecurityProofs

theorem Spec : Guarantees.Spec := fun b => ⟨helper b.val, pred_le b.val⟩

end SecurityProofs

end Core
