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
public import Definitions.Accounting.ComposeMonitoringPeriod
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
public import Definitions.Accounting.JointBridge.SpaceOfPeriodHeld
public import Definitions.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
public import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted
public import Definitions.Accounting.JointBridge.CommOfPooledLeakAccepted
public import Definitions.Accounting.JointBridge.CommOfPooledLeakAcceptedWitness
public import Definitions.Accounting.JointLive
public import Definitions.Accounting.Memory
public import Definitions.Accounting.PerRun
public import Definitions.Accounting.PeriodBinding
public import Definitions.Accounting.Physical
public import Definitions.Accounting.Slowdown
public import Definitions.Accounting.Pouw
public import Definitions.Accounting.PouwWitness
public import Definitions.Accounting.NetWitness
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
public import Definitions.Accounting.ReadUnit
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
# Resource accounting's definitions (trusted)

Resource accounting bounds what a device computed, read, held and sent, by reading each subprotocol's guarantee as an
account on one device model.

## Read first

The lock's `deployed` guarantees for the area, the bandwidth audit's and PoUB's, each with the module that states it:

* `Accounting.Guarantees.RemoteSessionConservationDeployed` (`Definitions.Accounting.Guarantees.RateAuditDeployed`):
  PoUB at its deployed parameters.
* `Accounting.Guarantees.RemoteSessionConservationSketchDeployed` (the same module): PoUB's sketch session at the
  deployed receipt parameters, under the per-pass hash budget.
* `Accounting.Guarantees.StackedSessionConservationSketchDeployed` (`Definitions.Accounting.Guarantees.Stacking`): PoUB
  with a store per pass. `Accounting.Guarantees.StackedRequirementSketchDeployed` (the same module) reads
  it against a registered circuit's requirement.
* `Accounting.Guarantees.Remote.SketchBoundDeployed` (`Definitions.Accounting.Guarantees.RemoteSketch`): the sketch
  bound at the deployed receipts.
* `Bandwidth.Guarantees.AuditRO` (`Definitions.Accounting.Guarantees.Bandwidth`): the bandwidth audit in the
  random-oracle model. It is the bandwidth audit's row in `security/README.md`.
* `Properties.RemoteAudit.Line.LineHeld` (`Properties.Accounting.Remote.Line`): the memory line of record,
  link-charged. It is PoUB's row in `security/README.md`.

The area's other guarantees are `Accounting.Guarantees.X`, stated in the module named under
`Definitions.Accounting.Guarantees` and proved under `Proofs.Accounting`:

* `Joint` and `CovertResidual` (`Joint`): on any device, the probability that the compute, bandwidth and capacity audits
  all accept while some account is below its certificate, or beyond its residual, is at most `εW + εQ + εS`.
* `JointPeriodLive`, `JointPooledLive` and `JointPooledBlocksLive` (`JointLive`): the joint guarantee over one period,
  over `m` periods and over `B` server blocks, as one causal game over the subprotocols' per-period guarantees.
* `ComputeOfPouw` (`Pouw`): PoUW's guarantee is the compute guarantee, at certificate `(1 − γ)(1 − εs)·W`.
* `Compose.CircuitCredit` (`Compose`) and `ComposeMonitoringPeriod.CircuitCreditPeriod` (`ComposeMonitoringPeriod`):
  PoUW's credit on a registered circuit it compiles, over one execution phase and over a monitoring period.
* `TS.Sm90SheetHeadline` (`HeadlineDc`) and `TS.Sm120SheetHeadline` (`Headline`): the headline at H200's and sm_120's
  data sheets.
* `CapacityOfStateBound` (`Capacity`): a state-bounded game's soundness is the capacity guarantee.
* `Memory.Adds` (`Memory`): memory adds against PoUS's guarantee, counting the help.
* `BandwidthOfAudit` (`BandwidthAudit`): the bandwidth audit's guarantee is the bandwidth guarantee.
* `AccelAdditive` (`Additive`): instructions that different subprotocols issue together use at most a period's
  `ρ_r·T` units of each resource `r`.
* `ServingBlocks.ServingBlockBound` (`ServingBlocks`): a training class's counted weight is at most the overhead lower
  bound's sum over server blocks, plus the dark remainder's.

## Assumptions

The area's premises are stated in the modules under `Definitions.Accounting.Assumptions`. The lock's named assumptions
are in `Definitions.Assumptions`.

## Map

This module imports the area's definitions; the lock's `layers` lists what they may import.

* The device and the actual circuit. `Device` is the device model. `Actual`, `Physical`, `Confidence` and `Bridge` are
  the actual circuit, its physical assumptions, its runs, strategies and confidence set, and the schedule it induces.
  `Window` is a location with the ring's oracle locations, `CostModel` the cost model of the actual circuit's provers,
  and `Additive` a device's period on the machine model 𝓜. `Gates` is the gate model of the lower bounds on a
  registered circuit, and `Requirement` what a registered circuit needs of a device's memory.
* The joint audit. `Joint` is a joint audit's run and what each audit's guarantee says about it, and `PerRun` the
  per-run audits. `JointPeriod` and `JointBridge` are a period audited by every subprotocol, with each per-period
  guarantee read as its account. `JointLive` is a period as a causal game. `NetWitness` is the run the network
  certifier's bridge fails at, and `JointBridge.CommOfPooledLeakAcceptedWitness` an honest run of its monitoring
  period, pooled.
* Server blocks and reads. `JointBlocks` and `Wipe` are server blocks, each opened by a memory wipe, and the
  streamed-label wipe on the device model. `ReadUnit` is a unit that returns one row of a registered table at an index
  another unit picks.
* The compute audit. `Pouw`, `PouwWork` and `PouwWitness` are PoUW's game, its work on one node of the actual circuit
  and the run its witnesses fail at. `Compose` and `ComposeMonitoringPeriod` are PoUW as a compile pass over a
  registered circuit. `HMu` and `CoinFlow` are the receipt-luck premise and a Program's coin flow. `Served`, `Filler`,
  `BoundReceipt` and `BoundSketch` bind receipts to the matmul. `TS`, `Headline` and `HeadlineDc` are the
  tensor–scalar model and the headline instances' exact data.
* The capacity audit: `Capacity`, `Memory` and `Responder`.
* The bandwidth audit. `Bandwidth` is its game in the random-oracle model, and `BandwidthAudit` that game as the
  bandwidth audit of a joint run. `RateAudit` and the modules named after it are the rate audit's games. `Tag` is the
  block tags, `Schedule` a deterministic schedule and `Carrier` device memory whose contents are a circuit's values.
  `Remote` is PoUB's vocabulary: the node, its parameters, a run and its routes.
* The overhead lower bound's inputs. `Slowdown` is tasks, classes, thresholds and slowdown, and `ServingBlocks` server
  blocks and a training class's counts.

## Where to look

* Guarantees: `security/Definitions/Accounting/Guarantees.lean` and `Guarantees/`. Premises:
  `security/Definitions/Accounting/Assumptions/`.
* Properties of the code: `security/Properties/Accounting/`. Proofs: `security/Proofs/Accounting.lean` and
  `security/Proofs/Accounting/`.
* Locks: `security/lean-audit.json` (the guarantees) and `security/Properties/lean-audit.json` (the properties).
* Which of these statements protocol soundness rests on: `security/README.md`.
-/

@[expose] public section
