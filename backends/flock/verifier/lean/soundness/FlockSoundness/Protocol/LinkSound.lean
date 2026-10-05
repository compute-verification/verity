import Mathlib.Data.Nat.BitIndices
import FlockSoundness.Protocol.LincheckPhase
import FlockSoundness.Protocol.OpeningPhase
import FlockSoundness.Protocol.ListSize

/-!
# The link points (PROTOCOL.md §16.2)

`link_sound`: if the committed values do not satisfy the statement, the two link points leave some
rep start undoomed with probability at most `linkError`. For a codeword near the committed table
whose witness satisfies the R1CS and the pin, some region fails; that region's claim, minus its
public value, is a nonzero polynomial in the link point of degree at most `|free| + nbl ≤ m_pts`
(`regionPoly`). It is nonzero because at the Boolean point of the failing region bit it is the
witness bit minus the public bit (`eval_regionPoly_bool`). Two independent points square the
probability, and a union bound runs over the list and the regions.

The layout of the point (`Statement.LinkLayout`): each region's free bits are distinct and read
link coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` fit inside the point.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The layout of the link points (§16.2): each region's free bits are distinct and read link
coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …` fit inside the point. -/
def Model.Statement.LinkLayout (S : Statement) (ptLocal mPts : ℕ) : Prop :=
  (∀ R ∈ S.regions, R.free.Nodup ∧ R.free.length ≤ ptLocal) ∧ ptLocal + S.nbl ≤ mPts

/-! ## The bits of a region address -/

/-! ## A region claim as a polynomial in the link point -/

/-! ## At the Boolean point of a failing region bit -/

/-! ## The link points -/

end FlockSoundness
