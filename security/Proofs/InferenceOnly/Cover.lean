module

public import Proofs.Slowdown.Cover
public import Proofs.InferenceOnly.Cover.Checked
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Tactic.Measurability.Init

/-!
# The cover theorem at the inference-only check

`Checked` carries the cover theorem (`Proofs.Slowdown.Cover`) to a verdict of the inference-only check: the gate-level
check's, and the check of record's (`SecurityProofs.DescAgrees`).
-/

@[expose] public section
