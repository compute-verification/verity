module

public import Proofs.Accounting.Defs
public import Proofs.Accounting.Memory
public import Proofs.Accounting.MemoryShare
public import Proofs.Accounting.Device
public import Proofs.Accounting.Gates
public import Proofs.Accounting.FillerLuck
public import Proofs.Accounting.Headline
public import Proofs.Accounting.HeadlineDc
public import Proofs.Accounting.HeadlineDc.Meets
public import Proofs.Accounting.Gates.Rows
public import Proofs.Accounting.Gates.Bitwise
public import Proofs.Accounting.Gates.ToCircuit
public import Proofs.Accounting.Bandwidth
public import Proofs.Accounting.ChainWork
public import Proofs.Accounting.Prob
public import Proofs.Accounting.Joint
public import Proofs.Accounting.JointBridge.WorkOfCreditedWords
public import Proofs.Accounting.JointBridge.SpaceOfPeriodHeld
public import Proofs.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
public import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted
public import Proofs.Accounting.JointLive
public import Proofs.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
public import Proofs.Accounting.JointBridge.HonestH200
public import Proofs.Accounting.JointBridge.IntegrityOfIntegrityPeriod
public import Proofs.Accounting.JointBridge.IntegrityOfIntegrityBlock
public import Proofs.Accounting.Pouw
public import Proofs.Accounting.PouwWork
public import Proofs.Accounting.PouwWitness
public import Proofs.Accounting.BandwidthAudit
public import Proofs.Accounting.RateAudit
public import Proofs.Accounting.ReadUnit
public import Proofs.Accounting.ReadUnit.Gates
public import Proofs.Accounting.ReadUnit.Net
public import Proofs.Accounting.ReadUnit.Tie
public import Proofs.Accounting.ReadUnit.Frame
public import Proofs.Accounting.ReadUnit.Sha
public import Proofs.Accounting.ReadUnit.RowSeg
public import Proofs.Accounting.ReadUnit.Leaf
public import Proofs.Accounting.ReadUnit.Select
public import Proofs.Accounting.ReadUnit.Check
public import Proofs.Accounting.ReadUnit.Deployed
public import Proofs.Accounting.ReadUnit.Binding
public import Proofs.Accounting.ReadUnit.Witnesses
public import Proofs.Accounting.BoundReceipt
public import Proofs.Accounting.Capacity
public import Proofs.Accounting.RegisteredCircuit
public import Proofs.Accounting.RegisteredCircuit.Tiles
public import Proofs.Accounting.RegisteredCircuit.Credit
public import Proofs.Accounting.RegisteredCircuit.Slack
public import Proofs.Accounting.RegisteredCircuit.Session
public import Proofs.Accounting.RegisteredCircuit.Requirement
public import Proofs.Accounting.RegisteredCircuit.Scheduled
public import Proofs.Accounting.Remote
public import Proofs.Accounting.PerRunTruncation
public import Proofs.Accounting.PerRunPouw
public import Proofs.Accounting.PerRunJoint
public import Proofs.Accounting.PerRunNecessity
public import Proofs.Accounting.PerRunPriced
public import Proofs.Accounting.PerRunStop
public import Proofs.Accounting.RemoteHeld
public import Proofs.Accounting.RemoteHeld.Meets
public import Proofs.Accounting.RemoteHeld.OverTime
public import Proofs.Accounting.RemoteHeld.Draw
public import Proofs.Accounting.RemoteHeld.DrawMeets
public import Proofs.Accounting.RemoteHeld.Binding
public import Proofs.Accounting.RemoteHeld.Sketch
public import Proofs.Accounting.RemoteHeld.SketchLine
public import Proofs.Accounting.RemoteHeld.SketchP2
public import Proofs.Accounting.RemoteHeld.Line
public import Proofs.Accounting.RemoteHeld.Subprotocol
public import Proofs.Accounting.Requirement
public import Proofs.Accounting.RequirementPouw
public import Proofs.Accounting.Sanity
public import Proofs.Accounting.Stacking
public import Proofs.Accounting.RegisteredCircuit.Witness
public import Proofs.Accounting.RegisteredCircuit.WitnessPrefill
public import Proofs.Accounting.Tag
public import Proofs.Accounting.TS
public import Proofs.Accounting.TS.Stages
public import Proofs.Accounting.TS.Steer
public import Proofs.Accounting.TS.Grind
public import Proofs.Accounting.TS.Fixed
public import Proofs.Accounting.TS.FixedMeets
public import Proofs.Accounting.TS.SharedNoise
public import Proofs.Accounting.TS.J10RChecks
public import Proofs.Accounting.TS.SessionSound
public import Proofs.Accounting.TS.CheckedSound
public import Proofs.Accounting.TS.GroupSound
public import Proofs.Accounting.TS.LanesSound
public import Proofs.Accounting.TS.Session.Lanes.Layout
public import Proofs.Accounting.TS.Session.Lanes.Decode
public import Proofs.Accounting.TS.Session.Lanes.Range
public import Proofs.Accounting.TS.Session.Lanes.Holds
public import Proofs.Accounting.TS.Session.Lanes.PortBits
public import Proofs.Accounting.TS.Session.Lanes.KnownBits
public import Proofs.Accounting.TS.Session.Lanes.KnownOf
public import Proofs.Accounting.TS.Session.Lanes.KnownFields
public import Proofs.Accounting.TS.Session.Lanes.KnownRead
public import Proofs.Accounting.TS.Session.Lanes.Combine
public import Proofs.Accounting.TS.Session.Lanes.Lane
public import Definitions.Pouw.H1Zy.Stream
public import Proofs.Pouw.H1Zy.Tile
public import Proofs.Accounting.TS.HeldSound
public import Proofs.Accounting.TS.Session.Tile.Computes
public import Proofs.Accounting.TS.TileSound
public import Proofs.Accounting.Actual
public import Proofs.Accounting.Responder
public import Proofs.Accounting.Responder.Latency
public import Proofs.Accounting.HMu
public import Proofs.Accounting.CoinFlow
public import Proofs.Accounting.Compose
public import Proofs.Accounting.ComposeMonitoringPeriod
public import Proofs.Accounting.ComposeMonitoringPeriod.Honest
public import Proofs.Accounting.Served
public import Proofs.Accounting.Additive
public import Proofs.Accounting.ServingBlocks
public import Proofs.Accounting.ServingBlocks.Witnesses
public import Proofs.Accounting.Coverage

/-!
# The proofs of resource accounting's guarantees

`Accounting.SecurityProofs.X : Accounting.Guarantees.X` for the guarantees stated under
`Security/Definitions/Accounting/Guarantees/` (`Bandwidth.SecurityProofs.X` for the bandwidth audit's own), the lemmas
behind them, and the proofs of resource accounting's properties, the bridges that read each subprotocol's per-period
guarantee as the joint period's input (`JointBridge/`, such as `Accounting.SecurityProofs.Bridge.SpaceOfPeriodHeld`,
locked in `Security/Properties/lean-audit.json`). `Defs` holds the proof-only vocabulary no guarantee reads,
`Sanity` shows the joint guarantee's hypotheses hold, and `Coverage` proves the inference-only policy's
`Nci.SecurityProofs.CapabilityFloor`. `Security/lean-audit.json` pins each guarantee's theorem, so a
change to the guarantee, or to a definition it reads, changes its record.

## Where the top guarantees are proved

* `Joint`, `CovertResidual`: `Joint.lean`; `JointPerRun`: `PerRunJoint.lean`.
* `JointPeriodLive`, `JointPooledLive`, `JointPooledBlocksLive`: `JointLive.lean`.
* `ComputeOfPouw`: `Pouw.lean`; `Compose.CircuitCredit`: `Compose.lean`; `ComposeMonitoringPeriod.CircuitCreditPeriod`:
  `ComposeMonitoringPeriod.lean`.
* `TS.Sm90SheetHeadline`: `HeadlineDc/Sm90.lean`; `TS.Sm120SheetHeadline`: `Headline/Sm120.lean`.
* `CapacityOfStateBound`: `Capacity.lean`; `Memory.Adds`: `Memory.lean`.
* `BandwidthOfAudit`: `BandwidthAudit.lean`; `Bandwidth.SecurityProofs.AuditRO`: `Bandwidth/AuditRO.lean`.
* `AccelAdditive`: `Additive.lean`.
* `ServingBlocks.ServingBlockBound`: `ServingBlocks.lean`.

## Where to look

* The statements and the vocabulary they read: `Security/Definitions/Accounting.lean`, whose module docstring lists the
  top guarantees with one line each.
* The properties these proofs prove: `Security/Properties/Accounting/`.
* Which of these statements protocol soundness rests on: `Security/README.md`.
-/

@[expose] public section
