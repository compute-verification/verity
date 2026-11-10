module

public import Proofs.InferenceOnly.Cover.Paths
public import Proofs.InferenceOnly.Cover.Charge
public import Proofs.InferenceOnly.Cover.Area
public import Proofs.InferenceOnly.Cover.Bits
public import Proofs.InferenceOnly.Cover.Sum
public import Proofs.InferenceOnly.Cover.Training
public import Proofs.InferenceOnly.Cover.Checked
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Tactic.Measurability.Init

/-!
# Proofs of the cover theorem

Each piece is its own file: paths into a block (`Paths`), what a block pays for the products it finishes (`Charge`), and
the per-block area count and the sum over a cover (`Area`). `Bits` translates a declaration's gates to the bits the cover
theorem counts: the block of a set of gates, what it computes and what it is handed. `Sum` adds the per-block bounds over
a division at the inference-only policy's rate. `Training` joins them to a declaration that complies with the inference-only policy
(`Definitions.InferenceOnly.Guarantees.CoverTraining`), and `Checked` carries it to a verdict of the inference-only check: the gate-level
check's, and the check of record's (`SecurityProofs.DescAgrees`).
-/

@[expose] public section
