module

public import Proofs.Slowdown.Cover.Paths
public import Proofs.Slowdown.Cover.Charge
public import Proofs.Slowdown.Cover.Area
public import Proofs.Slowdown.Cover.Bits
public import Proofs.Slowdown.Cover.Sum
public import Proofs.Slowdown.Cover.Training

/-!
# Proofs of the cover theorem

Each piece is its own file: paths into a block (`Paths`), what a block pays for the products it finishes (`Charge`), and
the per-block area count and the sum over a cover (`Area`). `Bits` translates a declaration's gates to the bits the cover
theorem counts: the block of a set of gates, what it computes and what it is handed. `Sum` adds the per-block bounds over
a division at the inference-only policy's rate. `Training` joins them to a declaration that complies with the inference-only policy
(`Definitions.Slowdown.Guarantees.CoverTraining`).
-/

@[expose] public section
