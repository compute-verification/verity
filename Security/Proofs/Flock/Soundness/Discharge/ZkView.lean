module

public import Proofs.Flock.Soundness.Discharge.ZkView.Fixed
public import Proofs.Flock.Soundness.Discharge.ZkView.Prefinal
public import Proofs.Flock.Soundness.Discharge.ZkView.Decode
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import VCVio.Prelude.Core

/-!
# Zero-knowledge at the transcript `flock-circuit --zk` sends

The statements are in `ZkView/Statements.lean`; the tables at the executable's leaves and the honest prover's messages in
`ZkView/Defs.lean`; the proof file's bytes in `ZkView/Wire.lean`.
-/

@[expose] public section
