module

public import Proofs.Nci.Cover.Paths
public import Proofs.Nci.Cover.Charge
public import Proofs.Nci.Cover.Area
public import Proofs.Nci.Cover.Bits
public import Proofs.Nci.Cover.Sum
public import Proofs.Nci.Cover.Training
public import Proofs.Nci.Cover.Checked
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Tactic.Measurability.Init

/-!
# Proofs of the cover theorem

Each piece is its own file: paths into a block (`Paths`), what a block pays for the products it finishes (`Charge`), and
the per-block area count and the sum over a cover (`Area`). `Bits` translates a declaration's gates to the bits the cover
theorem counts: the block of a set of gates, what it computes and what it is handed. `Sum` adds the per-block bounds over
a division at the NCI policy's rate. `Training` joins them to a declaration that complies with the NCI policy
(`Definitions.Nci.Guarantees.CoverTraining`), and `Checked` carries it to a verdict of the NCI check: the gate-level
check's, and the check of record's (`SecurityProofs.DescAgrees`).
-/

@[expose] public section
