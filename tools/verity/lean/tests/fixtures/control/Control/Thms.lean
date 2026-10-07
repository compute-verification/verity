import Control.Assumptions
import Control.Defs

namespace Control

theorem main_thm (h : Control.Hyp) (n : Nat) : double n = n + n := by
  unfold double
  omega

theorem helper : double 0 = 0 := rfl

end Control
