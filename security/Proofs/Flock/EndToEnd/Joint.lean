module

public import Mathlib.Tactic.Echelon.Zsqrtd
public import Mathlib.Tactic.NormNum.Irrational
public import Mathlib.Tactic.NormNum.IsCoprime
public import Mathlib.Tactic.NormNum.IsSquare
public import Mathlib.Tactic.NormNum.LegendreSymbol
public import Mathlib.Tactic.NormNum.ModEq
public import Mathlib.Tactic.NormNum.NatFib
public import Mathlib.Tactic.NormNum.NatLog
public import Mathlib.Tactic.NormNum.NatSqrt
public import Mathlib.Tactic.NormNum.Ordinal
public import Mathlib.Tactic.NormNum.Parity
public import Mathlib.Tactic.NormNum.Prime
public import Mathlib.Tactic.NormNum.RealSqrt
public import Mathlib.Tactic.ReduceModChar
public import Proofs.Flock.EndToEnd.Joint.Drawn
public import Proofs.Flock.EndToEnd.Joint.Prod
public import Proofs.Flock.EndToEnd.Joint.Statement
public import Proofs.Flock.EndToEnd.Period.Union
public import Proofs.Spot.Sampling.StratifiedProj

/-! SPOT's joint stratified bound, per sampling block and pooled over a period:
`Flock.SecurityProofs.JointStratified : Flock.Guarantees.JointStratified` and
`Flock.SecurityProofs.JointStratifiedPooled : Flock.Guarantees.JointStratifiedPooled` (`Joint/Bound.lean`), stated in
`Joint/Statement.lean`. The stratified sampling design one stratum at a time, for a consumer that samples one stratum by itself:
`SamplingDesign.stratified_proj_missProb` (`Soundness/Audit/StratifiedProj.lean`). -/

@[expose] public section
