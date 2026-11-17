module

public import Definitions.Slowdown.BitView
public import Definitions.Slowdown.Assumptions.Cover

/-! The parameters at which the cover theorem's per-block count applies to an inference block of a division at `(Γ, α)`, each checkable on `r` and none about the world. -/

@[expose] public section

namespace Slowdown.Cover.Assumptions

/-- **The read cap is below the upstream depth's cost**: `S < b D`, so that an inference block handed at most `S` bits
outside the free bits is handed fewer than `b` bits for each of `D` steps. -/
def CapBelowDepth (S b : ℝ) (D : ℕ) : Prop :=
  S < b * D

/-- **A piece's block fits the block size**: no gate of `r` is wider than `w₀` bits, and `w₀ c + S + width(θ) ≤ s`, so
the bits of an inference block of at most `c` gates, of the at most `S` bits it reads and of `r`'s registered
information are at most `s`. -/
def BlockFits (r : Verity.RegisteredCircuit) (c : ℕ) (S : ℝ) (w₀ s : ℕ) : Prop :=
  (∀ g, r.D.w g ≤ w₀) ∧ (w₀ * c : ℝ) + S + r.D.width r.θ ≤ s

end Slowdown.Cover.Assumptions
