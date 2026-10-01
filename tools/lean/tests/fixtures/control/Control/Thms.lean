import Control.Assumptions
import Control.Defs

namespace Control

theorem main_thm (h : Control.Hyp) (n : Nat) : double n = n + n := by
  unfold double
  omega

theorem helper : double 0 = 0 := rfl

/-- What `claim` states: through this definition, it assumes `Fits`. -/
def Claim : Prop := ∀ n, Fits n → double n = 2 * n

theorem claim : Claim := by
  unfold Claim double
  intros
  omega

end Control
