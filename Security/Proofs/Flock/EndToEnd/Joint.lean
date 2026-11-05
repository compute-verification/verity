module

public import Proofs.Flock.EndToEnd.Joint.Bound
public import Proofs.Flock.Soundness.Audit.StratifiedProj
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

/-! The sampled proofs' joint stratified bound, per sampling block and pooled over a period:
`Flock.SecurityProofs.JointStratified : Flock.Guarantees.JointStratified` and
`Flock.SecurityProofs.JointStratifiedHorizon : Flock.Guarantees.JointStratifiedHorizon` (`Joint/Bound.lean`), stated in
`Joint/Statement.lean`. The stratified law one stratum at a time, for a consumer that draws one stratum by itself:
`Law.stratified_proj_escape` (`Soundness/Audit/StratifiedProj.lean`). -/

@[expose] public section
