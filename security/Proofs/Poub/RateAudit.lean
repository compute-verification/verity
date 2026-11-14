module

public import Proofs.Poub.RateAudit.SketchWitnesses
public import Proofs.Poub.RateAudit.Witnesses
public import Proofs.Poub.RateAudit.Audit
public import Proofs.Poub.RateAudit.Chain
public import Proofs.Poub.RateAudit.Conservation
public import Proofs.Poub.RateAudit.Deployed
public import Proofs.Poub.RateAudit.DeployedSketch
public import Proofs.Poub.RateAudit.DeployedSketchSum
public import Proofs.Poub.RateAudit.DeployedRelay
public import Proofs.Poub.RateAudit.DeployedRun
public import Proofs.Poub.RateAudit.Honest
public import Proofs.Poub.RateAudit.Interactive
public import Proofs.Poub.RateAudit.Joint
public import Proofs.Poub.RateAudit.Named
public import Proofs.Poub.RateAudit.Partial
public import Proofs.Poub.RateAudit.PartialJoint
public import Proofs.Poub.RateAudit.Pooled
public import Proofs.Poub.RateAudit.Relay
public import Proofs.Poub.RateAudit.Run
public import Proofs.Poub.RateAudit.Sketch
public import Proofs.Poub.RateAudit.SketchHonest
public import Proofs.Poub.RateAudit.SketchInteractive
public import Proofs.Poub.RateAudit.SketchRank
public import Proofs.Poub.RateAudit.SketchTable
public import Proofs.Poub.RateAudit.Windows
public import Definitions.Poub.RateAuditRemote

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
bound over the monitoring period on the samples (`Accounting.SecurityProofs.PooledSampling`, `RatePooled`) with the
route `Sampled` it gives a window (`Accounting.SecurityProofs.RateChainSampledRO`, `RateChainSampled`), and conservation
with the compression naming words, not queries (`Accounting.SecurityProofs.RateConservationNamedRO`, `RatePaidNamedRO`),
with its route `Sampled` (`Accounting.SecurityProofs.RateChainSampledNamedRO`), and the pooled sample of a window and of a
session of windows, over the monitoring period and in every window (`Accounting.SecurityProofs.RateWindowSample`,
`RateRunSample`), with its route `Sampled` (`Accounting.SecurityProofs.RateRunSampledNamedRO`), and PoUB
as deployed, one guarantee over a session of ground, chained windows of devices with partial completion
(`Accounting.SecurityProofs.RemoteRunConservation`, at the deployed parameters
`RemoteRunConservationDeployed`), and the same session as a run of PoUB, meeting its routes
(`Accounting.SecurityProofs.RemoteRunRoutes`) and conserving over the monitoring period and in each window at the
same `γ` (`RemoteMonitoringPeriodConservation`, `RemoteWindowRunConservation`), and all three with side sources that may relay
receipts (`RemoteRunConservationRelay`, `RemoteMonitoringPeriodConservationRelay`, `RemoteWindowRunConservationRelay`),
and the composed verbatim-sector tensor-sketch session (`RemoteRunConservationSketch`; at the deployed 16-row
parameters, under the per-pass hash budget, `RemoteRunConservationSketchDeployed`, with the budgeted session bound
`RemoteRunConservationSketchSum` and the product form kept as
`RemoteRunConservationSketchDeployedProduct`). -/

@[expose] public section
