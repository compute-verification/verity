module

public import Definitions.ResourceAccounting.Actual
public import Definitions.Compose.Additive
public import Definitions.ResourceAccounting.Actual.Sampler
public import Definitions.Poub.Bandwidth
public import Definitions.Poub.BandwidthAudit
public import Definitions.Poub.BoundReceipt
public import Definitions.Poub.BoundSketch
public import Definitions.Certifier.Bridge
public import Definitions.Pous.Accounting.Capacity
public import Definitions.Pouw.CoinFlow
public import Definitions.Poub.Carrier
public import Definitions.Compose.Compose
public import Definitions.Compose.ComposeMonitoringPeriod
public import Definitions.Pouw.Confidence
public import Definitions.Pouw.CostModel
public import Definitions.Registration.Device
public import Definitions.Pouw.Filler
public import Definitions.Pouw.HMu
public import Definitions.ResourceAccounting.Gates
public import Definitions.Pouw.Headline
public import Definitions.Pouw.HeadlineDc
public import Definitions.Compose.Joint
public import Definitions.Compose.JointPeriod
public import Definitions.Certifier.JointBridge.CreditedWords
public import Definitions.Certifier.JointBridge.SpaceOfPeriodHeld
public import Definitions.Certifier.JointBridge.BandwidthOfAuditPerRunPeriod
public import Definitions.Certifier.JointBridge.CommOfMonitoringPeriodLeakAccepted
public import Definitions.Certifier.JointBridge.CommOfPooledLeakAccepted
public import Definitions.Certifier.JointBridge.CommOfPooledLeakAcceptedWitness
public import Definitions.Compose.JointLive
public import Definitions.Pous.Accounting.Memory
public import Definitions.Pouw.PerRun
public import Definitions.Compose.PeriodBinding
public import Definitions.Registration.Physical
public import Definitions.Pouw.Pouw
public import Definitions.Pouw.PouwWitness
public import Definitions.Certifier.NetWitness
public import Definitions.Pouw.PouwWork
public import Definitions.Poub.RateAudit
public import Definitions.Poub.RateAuditDeployed
public import Definitions.Poub.RateAuditInteractive
public import Definitions.Poub.RateAuditJoint
public import Definitions.Poub.RateAuditPartial
public import Definitions.Poub.RateAuditPooled
public import Definitions.Poub.RateAuditRemote
public import Definitions.Poub.RateAuditSketch
public import Definitions.Poub.RateAuditSketchDeployed
public import Definitions.Poub.RateAuditSketchInteractive
public import Definitions.Poub.RateAuditSketchLayout
public import Definitions.ResourceAccounting.ReadUnit
public import Definitions.ResourceAccounting.Records
public import Definitions.ResourceAccounting.RecordsToy
public import Definitions.Poub.Remote
public import Definitions.Poub.Remote.Deployed
public import Definitions.Poub.Remote.DrawInstances
public import Definitions.ResourceAccounting.Requirement
public import Definitions.ResourceAccounting.Requirement.Placed
public import Definitions.Pous.Accounting.Responder
public import Definitions.ResourceAccounting.Schedule
public import Definitions.ResourceAccounting.Served
public import Definitions.Slowdown.ServingBlocks
public import Definitions.Slowdown.ServingBlocks.Instances
public import Definitions.Poub.Tag
public import Definitions.Pouw.TS
public import Definitions.Pouw.TS.Fixed
public import Definitions.Pous.Accounting.Window
public import Definitions.Pous.Accounting.Wipe

/-!
# Resource accounting's definitions (trusted)

Resource accounting bounds what a device computed, read, held and sent, by reading each subprotocol's guarantee as an
account on one device model.

## Read first

The lock's `deployed` guarantees for the area, the bandwidth audit's and PoUB's, each with the module that states it:

* `Accounting.Guarantees.RemoteRunConservationDeployed` (`Definitions.Poub.Guarantees.RateAuditDeployed`):
  PoUB at its deployed parameters.
* `Accounting.Guarantees.RemoteRunConservationSketchDeployed` (the same module): PoUB's sketch session at the
  deployed receipt parameters, under the per-pass hash budget.
* `Accounting.Guarantees.StackedRunConservationSketchDeployed` (`Definitions.Poub.Guarantees.Stacking`): PoUB
  with a store per pass. `Accounting.Guarantees.StackedRequirementSketchDeployed` (the same module) reads
  it against a registered circuit's requirement.
* `Accounting.Guarantees.Remote.SketchBoundDeployed` (`Definitions.Poub.Guarantees.RemoteSketch`): the sketch
  bound at the deployed receipts.
* `Accounting.Guarantees.Remote.AuditRO` (`Definitions.Poub.Guarantees.RemotePass`): the bandwidth audit in the
  random-oracle model. It is the bandwidth audit's row in `security/README.md`.
* `Properties.Poub.Line.LineHeld` (`Properties.Poub.Remote.Line`): the memory line of record,
  link-charged. It is PoUB's row in `security/README.md`.

The area's other guarantees are `Accounting.Guarantees.X`, stated in the module named under
`Definitions.ResourceAccounting.Guarantees` and proved under `Proofs.ResourceAccounting`:

* `Joint` and `CovertResidual` (`Joint`): on any device, the probability that the compute, bandwidth and capacity audits
  all accept while some account is below its certificate, or beyond its residual, is at most `εW + εQ + εS`.
* `JointPeriodLive`, `JointPooledLive` and `JointPooledBlocksLive` (`JointLive`): the joint guarantee over one period,
  over `m` periods and over `B` server blocks, as one causal game over the subprotocols' per-period guarantees.
* `ComputeOfPouw` (`Pouw`): PoUW's guarantee is the compute guarantee, at certificate `(1 − γ)(1 − εs)·W`.
* `Compose.CircuitCredit` (`Compose`) and `ComposeMonitoringPeriod.CircuitCreditMonitoringPeriod` (`ComposeMonitoringPeriod`):
  PoUW's credit on a registered circuit it compiles, over one execution phase and over a monitoring period.
* `TS.Sm90SheetHeadline` (`HeadlineDc`) and `TS.Sm120SheetHeadline` (`Headline`): the headline at H200's and sm_120's
  data sheets.
* `CapacityOfStateBound` (`Capacity`): a state-bounded game's soundness is the capacity guarantee.
* `Memory.Adds` (`Memory`): memory adds against PoUS's guarantee, counting the help.
* `BandwidthOfAudit` (`BandwidthAudit`): the bandwidth audit's guarantee is the bandwidth guarantee.
* `Verity.Guarantees.RegistrationSound` (`Registration.Guarantees`, the view and the prefix heads in
  `Registration.View`): registration is an accounting protocol, and a log it accepts holds the registered facts the
  theorems read.
* `AccelAdditive` (`Additive`): instructions that different subprotocols issue together use at most a period's
  `ρ_r·T` units of each resource `r`.
* `ServingBlocks.ServingBlockBound` (`ServingBlocks`): a training class's counted weight is at most the overhead lower
  bound's sum over server blocks, plus the dark remainder's.

## Assumptions

The area's premises are stated in the modules under `Definitions.ResourceAccounting.Assumptions`. The lock's named assumptions
are in `Definitions.Assumptions`.

## Map

This module imports the area's definitions; the lock's `layers` lists what they may import.

* The device and the actual circuit. `Device` is the device model. `Actual`, `Physical`, `Confidence` and `Bridge` are
  the actual circuit, its physical assumptions, its runs, strategies and confidence set, and the schedule it induces.
  `Window` is a location with the ring's oracle locations, `CostModel` the cost model of the actual circuit's provers,
  and `Additive` a device's period on the machine model 𝓜. `Gates` is the gate model of the lower bounds on a
  registered circuit, and `Requirement` what a registered circuit needs of a device's memory. `Records` is a trusted
  device's record limit, and `RecordsToy` the runs its honest lemma is stated at.
* The joint audit. `Joint` is a joint audit's run and what each audit's guarantee says about it, and `PerRun` the
  per-run audits. `JointPeriod` and `JointBridge` are a period audited by every subprotocol, with each per-period
  guarantee read as its account. `JointLive` is a period as a causal game. `NetWitness` is the run the network
  attester's bridge fails at, and `JointBridge.CommOfPooledLeakAcceptedWitness` an honest run of its monitoring
  period, pooled.
* The wipe and reads. `Wipe` is the streamed-label wipe on the device model. `ReadUnit` is a unit that returns one row
  of a registered table at an index another unit picks.
* The compute audit. `Pouw`, `PouwWork` and `PouwWitness` are PoUW's game, its work on one node of the actual circuit
  and the run its witnesses fail at. `Compose` and `ComposeMonitoringPeriod` are PoUW as a compile pass over a
  registered circuit. `HMu` and `CoinFlow` are the receipt-luck premise and a Program's coin flow. `Served`, `Filler`,
  `BoundReceipt` and `BoundSketch` bind receipts to the matmul. `TS`, `Headline` and `HeadlineDc` are the
  tensor–scalar model and the headline instances' exact data.
* The capacity audit: `Capacity`, `Memory` and `Responder`.
* The bandwidth audit. `Bandwidth` is its game in the random-oracle model, and `BandwidthAudit` that game as the
  bandwidth audit of a joint run. `RateAudit` and the modules named after it are the rate audit's games. `Tag` is the
  block tags, `Schedule` a deterministic schedule and `Carrier` device memory whose contents are a circuit's values.
  `Remote` is PoUB's vocabulary: the node, its parameters, a run and its routes. `Remote.DrawInstances` holds the
  instances its sampled draw's witnesses are stated at.
* The overhead lower bound's inputs. `ServingBlocks` is server blocks and a training class's counts.

## Where to look

* Guarantees: `security/Definitions/ResourceAccounting/Guarantees.lean` and `Guarantees/`. Premises:
  `security/Definitions/ResourceAccounting/Assumptions/`.
* Properties of the code: `security/Properties/ResourceAccounting/`. Proofs: `security/Proofs/ResourceAccounting.lean` and
  `security/Proofs/ResourceAccounting/`.
* Locks: `security/lean-audit.json` (the guarantees) and `security/Properties/lean-audit.json` (the properties).
* Which of these statements protocol soundness rests on: `security/README.md`.
-/

@[expose] public section
