import Control.Assumptions
import Control.Defs

namespace Control

theorem main_thm (h : Control.Hyp) (n : Nat) : double n = n + n := by
  unfold double
  omega

theorem helper : double 0 = 0 := rfl

theorem Small_fails : ∃ n, ¬ Small n := ⟨10, by unfold Small; omega⟩

theorem Small_meets : double 1 < 10 := by unfold double; omega

end Control
