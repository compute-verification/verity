import Definitions.Accounting.Actual
import Definitions.Accounting.Additive
import Definitions.Accounting.Actual.Sampler
import Definitions.Accounting.Bandwidth
import Definitions.Accounting.BandwidthAudit
import Definitions.Accounting.BoundReceipt
import Definitions.Accounting.BoundSketch
import Definitions.Accounting.Bridge
import Definitions.Accounting.Capacity
import Definitions.Accounting.CoinFlow
import Definitions.Accounting.Carrier
import Definitions.Accounting.Compose
import Definitions.Accounting.ComposePeriod
import Definitions.Accounting.Confidence
import Definitions.Accounting.CostModel
import Definitions.Accounting.Device
import Definitions.Accounting.Filler
import Definitions.Accounting.HMu
import Definitions.Accounting.Gates
import Definitions.Accounting.Headline
import Definitions.Accounting.HeadlineDc
import Definitions.Accounting.Joint
import Definitions.Accounting.JointBlocks
import Definitions.Accounting.JointPeriod
import Definitions.Accounting.JointBridge.CreditedWords
import Definitions.Accounting.JointBridge.H200
import Definitions.Accounting.JointBridge.SpaceOfPeriodHeld
import Definitions.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted
import Definitions.Accounting.JointLive
import Definitions.Accounting.Memory
import Definitions.Accounting.PerRun
import Definitions.Accounting.Physical
import Definitions.Accounting.Slowdown
import Definitions.Accounting.Pouw
import Definitions.Accounting.PouwWork
import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditDeployed
import Definitions.Accounting.RateAuditHorizon
import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditJoint
import Definitions.Accounting.RateAuditPartial
import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditSketch
import Definitions.Accounting.RateAuditSketchDeployed
import Definitions.Accounting.RateAuditSketchInteractive
import Definitions.Accounting.RateAuditSketchLayout
import Definitions.Accounting.Remote
import Definitions.Accounting.Remote.Deployed
import Definitions.Accounting.Requirement
import Definitions.Accounting.Requirement.Placed
import Definitions.Accounting.Responder
import Definitions.Accounting.Schedule
import Definitions.Accounting.Served
import Definitions.Accounting.ServingBlocks
import Definitions.Accounting.ServingBlocks.Instances
import Definitions.Accounting.Tag
import Definitions.Accounting.TS
import Definitions.Accounting.TS.Fixed
import Definitions.Accounting.Window
import Definitions.Accounting.Wipe

/-!
# Resource accounting: the device model the resource audits share, and their joint guarantee

Each resource audit reads as an audit of one joint run on the device model, and the joint guarantee bounds the
probability that some audit accepts while its claim fails by the sum of their errors.

Read first `Accounting.Guarantees.JointPeriodLive` (module `Definitions.Accounting.Guarantees.JointLive`), the joint
guarantee over a period played as a causal game, over sampled proofs, PoUW, PoUS, the bandwidth audit and the network
certifier. Security/README.md's table lists `Bandwidth.Guarantees.AuditRO` (`Definitions.Accounting.Guarantees.Bandwidth`)
for the bandwidth audit. The guarantees are `Definitions.Accounting.Guarantees` and the named assumptions
`Definitions.Accounting.Assumptions`.

The modules, by topic:
* The device model and the actual circuit: `Device`, `Additive` (a device's period on 𝓜), `Physical`, `Actual`,
  `Actual.Sampler`, `Bridge` (the device's schedule the actual circuit induces), `Confidence`, `CostModel`, `PerRun`.
* Joint runs and periods: `Joint`, `JointPeriod`, `JointBlocks` (server blocks audited by every protocol), `JointLive`
  (a period as a causal game), and `JointBridge`, which reads each subprotocol's period guarantee as a joint run's.
* Compute: `Pouw`, `PouwWork`, `Compose` and `ComposePeriod` (PoUW as a compile pass over a registered circuit, and
  over a monitoring period), `Filler`, `HMu`, `BoundReceipt`, `BoundSketch`, `CoinFlow`, `Served`, `Carrier`, `TS`.
* Bandwidth: `Bandwidth`, `BandwidthAudit`, `Tag`, the rate audit (`RateAudit` and the `RateAudit*` variants), and
  PoUB (`Remote`, `Remote.Deployed`).
* Capacity and memory: `Capacity`, `Memory`, `Responder`, `Window`, `Wipe`, `Requirement`, `Requirement.Placed`,
  `Schedule`.
* Lower bounds: `Gates` (the gate model), `Slowdown` (tasks, thresholds and the slowdown), `ServingBlocks` (server
  blocks, with an honest layout and counterexamples in `ServingBlocks.Instances`).
* The headline instances' exact data: `Headline`, `HeadlineDc`.
-/
