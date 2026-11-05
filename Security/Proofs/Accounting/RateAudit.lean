module

public import Proofs.Accounting.RateAudit.Audit
public import Proofs.Accounting.RateAudit.Chain
public import Proofs.Accounting.RateAudit.Conservation
public import Proofs.Accounting.RateAudit.Deployed
public import Proofs.Accounting.RateAudit.DeployedSketch
public import Proofs.Accounting.RateAudit.DeployedSketchSum
public import Proofs.Accounting.RateAudit.DeployedRelay
public import Proofs.Accounting.RateAudit.DeployedRun
public import Proofs.Accounting.RateAudit.Honest
public import Proofs.Accounting.RateAudit.Horizon
public import Proofs.Accounting.RateAudit.Interactive
public import Proofs.Accounting.RateAudit.Joint
public import Proofs.Accounting.RateAudit.Named
public import Proofs.Accounting.RateAudit.Partial
public import Proofs.Accounting.RateAudit.PartialJoint
public import Proofs.Accounting.RateAudit.Relay
public import Proofs.Accounting.RateAudit.Session
public import Proofs.Accounting.RateAudit.Sketch
public import Proofs.Accounting.RateAudit.SketchHonest
public import Proofs.Accounting.RateAudit.SketchInteractive
public import Proofs.Accounting.RateAudit.SketchRank
public import Proofs.Accounting.RateAudit.SketchTable
public import Proofs.Accounting.RateAudit.Windows
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init

/-! The rate audit's proofs: the guarantee in the random-oracle model (`Accounting.SecurityProofs.RateAudit`), its
completeness (`Accounting.SecurityProofs.RateHonest`), its time average over windows
(`Accounting.SecurityProofs.RateAuditWindows`), its reading in a joint run with the rebuilds charged as compute
(`Accounting.SecurityProofs.RateOfAudit`), the guarantee with side sources that answer what the chip sent
(`Accounting.SecurityProofs.RateAuditInteractiveRO` when they ignore the oracle, `RateAuditInteractive` with relayed
tags, and `RateOfAuditInteractive` and `RateOfAuditInteractiveRO` in a joint run), the guarantee for a device that may
complete part of a word (`Accounting.SecurityProofs.RateAuditPartial`, its completeness `RatePartialHonest`, and in a
joint run `RateOfAuditPartial` and its passes form `RateOfAuditPartialPasses`), and the sketch-then-hash variant for
a device that reads sectors verbatim (`Accounting.SecurityProofs.RateAuditSketch`, its completeness
`Accounting.SecurityProofs.RateSketchHonest`, with interactive side sources that ignore the oracle
`Accounting.SecurityProofs.RateAuditSketchInteractiveRO`,
`Accounting.SecurityProofs.SketchBinds`, `Accounting.SecurityProofs.RateAuditTensorSketch`, priced by the signs' rank
`Accounting.SecurityProofs.SketchRankFiber`, `Accounting.SecurityProofs.RateAuditTensorSketchRank`), and outside that class
the table that reads only the digest (`Accounting.SecurityProofs.RateSketchTable`,
`Accounting.SecurityProofs.RateTensorSketchTable`), and the remote protocol's conservation over tiers
(`Accounting.SecurityProofs.RateConservationRO`, `RateConservation`, their capacity forms `RateConservationCapRO` and
`RateConservationCap`, and `RateDisplacement`), chained passes (`Accounting.SecurityProofs.RateChainPassRO`,
`RateChainPass`, `RateChainRO`, `RateChain`), and conservation at every count, for a pass and for a window
(`Accounting.SecurityProofs.RatePaidRO`, `RatePaid`, `RateChainPaidRO`, `RateChainPaid`), and for a window whose roots
are ground against the Fiat–Shamir seed (`Accounting.SecurityProofs.RateGrindPaidRO`, `RateGrindPaid`), and the causal
bound over the monitoring period on the samples (`Accounting.SecurityProofs.HorizonSampling`, `RateHorizon`) with the
route `Sampled` it gives a window (`Accounting.SecurityProofs.RateChainSampledRO`, `RateChainSampled`), and conservation
with the compression naming words, not queries (`Accounting.SecurityProofs.RateConservationNamedRO`, `RatePaidNamedRO`),
with its route `Sampled` (`Accounting.SecurityProofs.RateChainSampledNamedRO`), and the pooled draw of a window and of a
session of windows, over the monitoring period and in every window (`Accounting.SecurityProofs.RateWindowDraw`,
`RateSessionDraw`), with its route `Sampled` (`Accounting.SecurityProofs.RateSessionSampledNamedRO`), and the remote
bandwidth audit as deployed, one guarantee over a session of ground, chained windows of devices with partial completion
(`Accounting.SecurityProofs.RemoteSessionConservation`, at the deployed parameters
`RemoteSessionConservationDeployed`), and the same session as a run of the remote audit, meeting its routes
(`Accounting.SecurityProofs.RemoteSessionRoutes`) and conserving over the monitoring period and in each window at the
same `γ` (`RemoteSessionRunConservation`, `RemoteWindowRunConservation`), and all three with side sources that may relay
receipts (`RemoteSessionConservationRelay`, `RemoteSessionRunConservationRelay`, `RemoteWindowRunConservationRelay`),
and the composed verbatim-sector tensor-sketch session (`RemoteSessionConservationSketch`; at the deployed 16-row
parameters, under the per-pass hash budget, `RemoteSessionConservationSketchDeployed`, with the budgeted session bound
`RemoteSessionConservationSketchSum` and the product form kept as
`RemoteSessionConservationSketchDeployedProduct`). -/

@[expose] public section
