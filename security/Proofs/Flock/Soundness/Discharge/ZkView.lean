module

public import Proofs.Flock.Soundness.Discharge.ZkView.Fixed
public import Proofs.Flock.Soundness.Discharge.ZkView.Prefinal
public import Proofs.Flock.Soundness.Discharge.ZkView.Decode
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import VCVio.Prelude.Core

/-!
# Zero-knowledge at the transcript `flock-circuit --zk` sends

The statements are in `ZkView/Statements.lean`; the tables at the executable's leaves and the honest prover's messages in
`ZkView/Defs.lean`; the proof file's bytes in `ZkView/Wire.lean`.
-/

@[expose] public section
