module

public import Definitions.Accounting.Actual
public import Definitions.Accounting.Additive
public import Definitions.Accounting.Actual.Sampler
public import Definitions.Accounting.Bandwidth
public import Definitions.Accounting.BandwidthAudit
public import Definitions.Accounting.BoundReceipt
public import Definitions.Accounting.BoundSketch
public import Definitions.Accounting.Bridge
public import Definitions.Accounting.Capacity
public import Definitions.Accounting.CoinFlow
public import Definitions.Accounting.Carrier
public import Definitions.Accounting.Compose
public import Definitions.Accounting.ComposePeriod
public import Definitions.Accounting.Confidence
public import Definitions.Accounting.CostModel
public import Definitions.Accounting.Device
public import Definitions.Accounting.Filler
public import Definitions.Accounting.HMu
public import Definitions.Accounting.Gates
public import Definitions.Accounting.Headline
public import Definitions.Accounting.HeadlineDc
public import Definitions.Accounting.Joint
public import Definitions.Accounting.JointBlocks
public import Definitions.Accounting.JointPeriod
public import Definitions.Accounting.JointBridge.CreditedWords
public import Definitions.Accounting.JointBridge.H200
public import Definitions.Accounting.JointBridge.SpaceOfPeriodHeld
public import Definitions.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
public import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted
public import Definitions.Accounting.JointLive
public import Definitions.Accounting.Memory
public import Definitions.Accounting.PerRun
public import Definitions.Accounting.Physical
public import Definitions.Accounting.Slowdown
public import Definitions.Accounting.Pouw
public import Definitions.Accounting.PouwWork
public import Definitions.Accounting.RateAudit
public import Definitions.Accounting.RateAuditDeployed
public import Definitions.Accounting.RateAuditHorizon
public import Definitions.Accounting.RateAuditInteractive
public import Definitions.Accounting.RateAuditJoint
public import Definitions.Accounting.RateAuditPartial
public import Definitions.Accounting.RateAuditRemote
public import Definitions.Accounting.RateAuditSketch
public import Definitions.Accounting.RateAuditSketchDeployed
public import Definitions.Accounting.RateAuditSketchInteractive
public import Definitions.Accounting.RateAuditSketchLayout
public import Definitions.Accounting.Remote
public import Definitions.Accounting.Remote.Deployed
public import Definitions.Accounting.Requirement
public import Definitions.Accounting.Requirement.Placed
public import Definitions.Accounting.Responder
public import Definitions.Accounting.Schedule
public import Definitions.Accounting.Served
public import Definitions.Accounting.ServingBlocks
public import Definitions.Accounting.ServingBlocks.Instances
public import Definitions.Accounting.Tag
public import Definitions.Accounting.TS
public import Definitions.Accounting.TS.Fixed
public import Definitions.Accounting.Window
public import Definitions.Accounting.Wipe

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

@[expose] public section
