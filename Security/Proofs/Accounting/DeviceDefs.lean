import Definitions.Accounting.Device

/-! The proof-only part of `Definitions.Accounting.Device`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting

variable {ι α : Type} [Fintype α]

namespace Schedule

variable (sc : Schedule ι α) (w : Window) (a : α)

/-- The bits account `a` took in over side channel `i` over window `w`. -/
def side (i : ι) : ℕ := ∑ t ∈ w.ticks, (sc t).side a i

/-- The on-chip state account `a` held over window `w`, in bit-ticks. -/
def chipHeld : ℕ := ∑ t ∈ w.ticks, (sc t).chip a

end Schedule

end Accounting
