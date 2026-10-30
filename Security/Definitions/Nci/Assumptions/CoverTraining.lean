import Definitions.Nci.BitView
import Definitions.Nci.Assumptions.Cover

/-!
# The cover theorem on a compliant declaration: the parameters it is taken at

`Nci.Cover.Guarantees.CoverTraining`, `CoverTrainingAligned` and `CoverTrainingChained` read the reference training
circuit `r` bit by bit, at a block size `s` and an upstream depth `D`. These name what those parameters must be for the
cover theorem's per-block count to apply to an inference block of a division at `(c, k)`: each is checkable on `r`, and
none is about the world.
-/

namespace Nci.Cover.Assumptions

/-- **The read cap is below the upstream depth's cost**: `k < b D`, so that an inference block handed at most `k` bits
outside the free bits is handed fewer than `b` bits for each of `D` steps. -/
def CapBelowDepth (k : ℕ) (b : ℝ) (D : ℕ) : Prop :=
  (k : ℝ) < b * D

/-- **A piece's block fits the block size**: no gate of `r` is wider than `w₀` bits, and `w₀ c + k + width(θ) ≤ s`, so
the bits of an inference block of at most `c` gates, of the at most `k` bits it reads and of `r`'s registered
information are at most `s`. -/
def BlockFits (r : Verity.Declaration) (c k w₀ s : ℕ) : Prop :=
  (∀ g, r.D.w g ≤ w₀) ∧ w₀ * c + k + r.D.width r.θ ≤ s

/-- **The activation regions are whole words at least `f` bits wide**: each use `j` of the multiplication blocks `Mb`
reads its activations from the gates `Aw j` of `r`, each at least `f` bits wide. -/
def RegionsAtLeast {r : Verity.Declaration} {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl r) J L)
    (Aw : J → Finset (Fin r.D.N)) (f : ℕ) : Prop :=
  ∀ j ∈ Mb.uses, ∀ a ∈ Aw j, f ≤ r.D.w a

end Nci.Cover.Assumptions
