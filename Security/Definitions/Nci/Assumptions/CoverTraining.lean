module

public import Definitions.Nci.BitView
public import Definitions.Nci.Assumptions.Cover

/-!
# The cover theorem on a compliant declaration: the parameters it is taken at

`Nci.Cover.Guarantees.CoverTraining` reads the reference training circuit `r` bit by bit, at a block size `s` and an
upstream depth `D`. These name what those parameters must be for the
cover theorem's per-block count to apply to an inference block of a division at `(Γ, α)`, which has at most `Γ` gates
and reads at most `S = α Γ` bits: each is checkable on `r`, and none is about the world.
-/

@[expose] public section

namespace Nci.Cover.Assumptions

/-- **The read cap is below the upstream depth's cost**: `S < b D`, so that an inference block handed at most `S` bits
outside the free bits is handed fewer than `b` bits for each of `D` steps. -/
def CapBelowDepth (S b : ℝ) (D : ℕ) : Prop :=
  S < b * D

/-- **A piece's block fits the block size**: no gate of `r` is wider than `w₀` bits, and `w₀ c + S + width(θ) ≤ s`, so
the bits of an inference block of at most `c` gates, of the at most `S` bits it reads and of `r`'s registered
information are at most `s`. -/
def BlockFits (r : Verity.RegisteredCircuit) (c : ℕ) (S : ℝ) (w₀ s : ℕ) : Prop :=
  (∀ g, r.D.w g ≤ w₀) ∧ (w₀ * c : ℝ) + S + r.D.width r.θ ≤ s

end Nci.Cover.Assumptions
