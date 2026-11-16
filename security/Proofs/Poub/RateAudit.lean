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

/-! The rate audit's proofs: the guarantee in the random-oracle model (`ResourceAccounting.SecurityProofs.RateAudit`), its
completeness (`ResourceAccounting.SecurityProofs.RateHonest`), its time average over windows
(`ResourceAccounting.SecurityProofs.RateAuditWindows`), its reading in a joint run with the rebuilds charged as compute
(`ResourceAccounting.SecurityProofs.RateOfAudit`), the guarantee with side sources that answer what the chip sent
(`ResourceAccounting.SecurityProofs.RateAuditInteractiveRO` when they ignore the oracle, `RateAuditInteractive` with relayed
tags, and `RateOfAuditInteractive` and `RateOfAuditInteractiveRO` in a joint run), the guarantee for a device that may
complete part of a word (`ResourceAccounting.SecurityProofs.RateAuditPartial`, its completeness `RatePartialHonest`, and in a
joint run `RateOfAuditPartial` and its passes form `RateOfAuditPartialPasses`), and the sketch-then-hash variant for
a device that reads sectors verbatim (`ResourceAccounting.SecurityProofs.RateAuditSketch`, its completeness
`ResourceAccounting.SecurityProofs.RateSketchHonest`, with interactive side sources that ignore the oracle
`ResourceAccounting.SecurityProofs.RateAuditSketchInteractiveRO`,
`ResourceAccounting.SecurityProofs.SketchBinds`, `ResourceAccounting.SecurityProofs.RateAuditTensorSketch`, priced by the signs' rank
`ResourceAccounting.SecurityProofs.SketchRankFiber`, `ResourceAccounting.SecurityProofs.RateAuditTensorSketchRank`), and outside that class
the table that reads only the digest (`ResourceAccounting.SecurityProofs.RateSketchTable`,
`ResourceAccounting.SecurityProofs.RateTensorSketchTable`), and the remote protocol's conservation over tiers
(`ResourceAccounting.SecurityProofs.RateConservationRO`, `RateConservation`, their capacity forms `RateConservationCapRO` and
`RateConservationCap`, and `RateDisplacement`), chained passes (`ResourceAccounting.SecurityProofs.RateChainPassRO`,
`RateChainPass`, `RateChainRO`, `RateChain`), and conservation at every count, for a pass and for a window
(`ResourceAccounting.SecurityProofs.RatePaidRO`, `RatePaid`, `RateChainPaidRO`, `RateChainPaid`), and for a window whose roots
are ground against the Fiat–Shamir seed (`ResourceAccounting.SecurityProofs.RateGrindPaidRO`, `RateGrindPaid`), and the causal
bound over the monitoring period on the samples (`ResourceAccounting.SecurityProofs.PooledSampling`, `RatePooled`) with the
route `Sampled` it gives a window (`ResourceAccounting.SecurityProofs.RateChainSampledRO`, `RateChainSampled`), and conservation
with the compression naming words, not queries (`ResourceAccounting.SecurityProofs.RateConservationNamedRO`, `RatePaidNamedRO`),
with its route `Sampled` (`ResourceAccounting.SecurityProofs.RateChainSampledNamedRO`), and the pooled sample of a window and of a
session of windows, over the monitoring period and in every window (`ResourceAccounting.SecurityProofs.RateWindowSample`,
`RateRunSample`), with its route `Sampled` (`ResourceAccounting.SecurityProofs.RateRunSampledNamedRO`), and PoUB
as deployed, one guarantee over a session of ground, chained windows of devices with partial completion
(`ResourceAccounting.SecurityProofs.RemoteRunConservation`, at the deployed parameters
`RemoteRunConservationDeployed`), and the same session as a run of PoUB, meeting its routes
(`ResourceAccounting.SecurityProofs.RemoteRunRoutes`) and conserving over the monitoring period and in each window at the
same `γ` (`RemoteMonitoringPeriodConservation`, `RemoteWindowRunConservation`), and all three with side sources that may relay
receipts (`RemoteRunConservationRelay`, `RemoteMonitoringPeriodConservationRelay`, `RemoteWindowRunConservationRelay`),
and the composed verbatim-sector tensor-sketch session (`RemoteRunConservationSketch`; at the deployed 16-row
parameters, under the per-pass hash budget, `RemoteRunConservationSketchDeployed`, with the budgeted session bound
`RemoteRunConservationSketchSum` and the product form kept as
`RemoteRunConservationSketchDeployedProduct`). -/

@[expose] public section
