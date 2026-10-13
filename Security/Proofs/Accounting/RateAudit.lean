import Proofs.Accounting.RateAudit.Audit
import Proofs.Accounting.RateAudit.Honest
import Proofs.Accounting.RateAudit.Windows
import Proofs.Accounting.RateAudit.Joint
import Proofs.Accounting.RateAudit.Relay
import Proofs.Accounting.RateAudit.Interactive
import Proofs.Accounting.RateAudit.Partial
import Proofs.Accounting.RateAudit.PartialJoint
import Proofs.Accounting.RateAudit.Sketch
import Proofs.Accounting.RateAudit.Conservation
import Proofs.Accounting.RateAudit.Chain
import Proofs.Accounting.RateAudit.Horizon
import Proofs.Accounting.RateAudit.Named
import Proofs.Accounting.RateAudit.Session
import Proofs.Accounting.RateAudit.Deployed
import Proofs.Accounting.RateAudit.SketchHonest
import Proofs.Accounting.RateAudit.SketchInteractive

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
`Accounting.SecurityProofs.SketchBinds`, `Accounting.SecurityProofs.RateAuditTensorSketch`), and the remote protocol's
conservation over tiers (`Accounting.SecurityProofs.RateConservationRO`, `RateConservation`, their capacity forms
`RateConservationCapRO` and `RateConservationCap`, and `RateDisplacement`), chained passes
(`Accounting.SecurityProofs.RateChainPassRO`, `RateChainPass`, `RateChainRO`, `RateChain`), and conservation at every
count, for a pass and for a window (`Accounting.SecurityProofs.RatePaidRO`, `RatePaid`, `RateChainPaidRO`,
`RateChainPaid`), and for a window whose roots are ground against the Fiat–Shamir seed
(`Accounting.SecurityProofs.RateGrindPaidRO`, `RateGrindPaid`), and the causal horizon bound on the samples
(`Accounting.SecurityProofs.HorizonSampling`, `RateHorizon`) with the route `Sampled` it gives a window
(`Accounting.SecurityProofs.RateChainSampledRO`, `RateChainSampled`), and conservation with the compression naming
words, not queries (`Accounting.SecurityProofs.RateConservationNamedRO`, `RatePaidNamedRO`), with its route `Sampled`
(`Accounting.SecurityProofs.RateChainSampledNamedRO`), and the pooled draw of a window and of a session of windows,
over the horizon and in every window (`Accounting.SecurityProofs.RateWindowDraw`, `RateSessionDraw`), with its route
`Sampled` (`Accounting.SecurityProofs.RateSessionSampledNamedRO`), and the remote bandwidth audit as deployed, one
guarantee over a session of ground, chained windows of devices with partial completion
(`Accounting.SecurityProofs.RemoteSessionConservation`, at the deployed parameters
`RemoteSessionConservationDeployed`). -/
