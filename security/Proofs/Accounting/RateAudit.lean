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
public import Proofs.Accounting.RateAudit.Interactive
public import Proofs.Accounting.RateAudit.Joint
public import Proofs.Accounting.RateAudit.Named
public import Proofs.Accounting.RateAudit.Partial
public import Proofs.Accounting.RateAudit.PartialJoint
public import Proofs.Accounting.RateAudit.Pooled
public import Proofs.Accounting.RateAudit.Relay
public import Proofs.Accounting.RateAudit.Session
public import Proofs.Accounting.RateAudit.Sketch
public import Proofs.Accounting.RateAudit.SketchHonest
public import Proofs.Accounting.RateAudit.SketchInteractive
public import Proofs.Accounting.RateAudit.SketchRank
public import Proofs.Accounting.RateAudit.SketchTable
public import Proofs.Accounting.RateAudit.Windows
public import Definitions.Accounting.RateAuditRemote

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
`RateSessionSample`), with its route `Sampled` (`Accounting.SecurityProofs.RateSessionSampledNamedRO`), and the remote
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
