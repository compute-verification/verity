module

public import Control.Assumptions
public import Control.Defs

@[expose] public section

namespace Control

theorem main_thm (h : Control.Hyp) (n : Nat) : double n = n + n := by
  unfold double
  omega

theorem helper : double 0 = 0 := rfl

theorem Small_fails : ∃ n, ¬ Small n := ⟨10, by unfold Small; omega⟩

theorem Small_meets : double 1 < 10 := by unfold double; omega

end Control
