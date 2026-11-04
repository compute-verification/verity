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
# Resource accounting's definitions (trusted)

Resource accounting turns the subprotocols' guarantees into bounds on what a device computed, read, held and sent. Its
definitions are the device model every resource audit is about, the actual circuit the device evaluates, a joint
audit's run on it, and each audit's game read as one of that run's audits. This module imports every module of them;
the lock's `layers` lists what they may import:

* **The device and the actual circuit.** `Device` (the device model); `Actual`, `Physical`, `Confidence` and `Bridge`
  (the actual circuit, its physical assumptions, its runs, strategies and confidence set, and the device's schedule it
  induces); `Window` (a location with the ring's oracle locations); `CostModel` (the cost model of the actual circuit's
  provers); `Additive` (a device's period on the machine model 𝓜); `Gates` (the gate model the lower bounds on a
  registered circuit are proved over); `Requirement` (what a registered circuit needs of a device's memory).
* **The joint audit.** `Joint` (a joint audit's run, and what each audit's guarantee says about it); `PerRun` (per-run
  audits); `JointPeriod` and `JointBridge` (a period audited by every subprotocol, and each subprotocol's per-period
  guarantee read as its account); `JointLive` (a period as a causal game, its claims on a fixed extractor's values);
  `JointBlocks` and `Wipe` (server blocks, each opened by a memory wipe, and the streamed-label wipe on the device
  model).
* **Each audit's game.** The compute audit: `Pouw` and `PouwWork` (PoUW's game, and its work on one node of the actual
  circuit), `Compose` and `ComposePeriod` (PoUW as a compile pass over a registered circuit, over one execution phase
  and over a monitoring period), `HMu` and `CoinFlow` (the receipt-luck premise H-μ⁺ and a Program's coin flow),
  `Served`, `Filler`, `BoundReceipt` and `BoundSketch` (receipts bound to the matmul), and `TS`, `Headline` and
  `HeadlineDc` (the tensor–scalar model and the headline instances' exact data). The capacity audit: `Capacity`,
  `Memory` and `Responder`. The bandwidth audit: `Bandwidth`, `BandwidthAudit`, the rate audit's games (`RateAudit*`),
  `Tag`, `Schedule` and `Carrier`. PoUB's node and run: `Remote`.
* **The overhead lower bound's inputs.** `Slowdown` (tasks, classes, thresholds and slowdown) and `ServingBlocks`
  (server blocks and a training class's counts).

## Top guarantees

Each is `Accounting.Guarantees.X`, stated in the file named under `Security/Definitions/Accounting/Guarantees/` and
proved as `Accounting.SecurityProofs.X` in the file of that name under `Security/Proofs/Accounting/` unless another is
named.

* `Joint`, `CovertResidual` (`Joint.lean`): on any device, the probability that the compute, bandwidth and capacity
  audits all accept while some account is below its certificate, or beyond its residual, is at most `εW + εQ + εS`.
* `JointPeriodLive`, `JointHorizonLive`, `JointHorizonBlocksLive` (`JointLive.lean`): the joint guarantee over one
  period, over `m` periods and over `B` server blocks with the memory wipes that open them, as one causal game over the
  five subprotocols' per-period guarantees.
* `ComputeOfPouw` (`Pouw.lean`): PoUW's guarantee is the compute guarantee, at certificate `(1 − γ)(1 − εs)·W`.
* `Compose.CircuitCredit` (`Compose.lean`) and `ComposePeriod.CircuitCreditPeriod` (`ComposePeriod.lean`): PoUW's
  credit on a registered circuit it compiles, over one execution phase and over a monitoring period's accepted prefix.
* `TS.Sm90SheetHeadline` (`HeadlineDc.lean`, proved in `HeadlineDc/Sm90.lean`) and `TS.Sm120SheetHeadline`
  (`Headline.lean`, proved in `Headline/Sm120.lean`): the headline at H200's and sm_120's data sheets.
* `CapacityOfStateBound` (`Capacity.lean`): a state-bounded game's soundness is the capacity guarantee.
* `Memory.Adds` (`Memory.lean`): memory adds against PoUS's guarantee, counting the help.
* `BandwidthOfAudit` (`BandwidthAudit.lean`): the bandwidth audit's guarantee is the bandwidth guarantee; the bandwidth
  audit's own guarantee is `Bandwidth.Guarantees.AuditRO` (`Bandwidth.lean`, proved in `Bandwidth/AuditRO.lean`), in
  the random-oracle model.
* `AccelAdditive` (`Additive.lean`): instructions that different subprotocols issue together use at most a period's
  `ρ_r·T` units of each resource `r`.
* `Actual.Runtime` (`Actual.lean`, proved in `Actual/Runtime.lean`): a prohibited computation takes long on the actual
  circuit.
* `ServingBlocks.ServingBlockBound` (`ServingBlocks.lean`): a training class's counted weight is at most the overhead
  lower bound's sum over server blocks, plus the dark remainder's.

## Where to look

* Named assumptions: `Security/Definitions/Accounting/Assumptions/`; guarantees:
  `Security/Definitions/Accounting/Guarantees.lean` and `Guarantees/`.
* Properties of the code: `Security/Properties/Accounting/`; proofs: `Security/Proofs/Accounting.lean` and
  `Security/Proofs/Accounting/`.
* Locks: `Security/lean-audit.json` (the guarantees) and `Security/Properties/lean-audit.json` (the properties).
* Which of these statements protocol soundness rests on: `Security/README.md`.
-/
