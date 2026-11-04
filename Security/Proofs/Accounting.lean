import Proofs.Accounting.Defs
import Proofs.Accounting.Memory
import Proofs.Accounting.MemoryShare
import Proofs.Accounting.Device
import Proofs.Accounting.Gates
import Proofs.Accounting.FillerLuck
import Proofs.Accounting.Headline
import Proofs.Accounting.HeadlineDc
import Proofs.Accounting.HeadlineDc.Meets
import Proofs.Accounting.Gates.Rows
import Proofs.Accounting.Gates.Bitwise
import Proofs.Accounting.Bandwidth
import Proofs.Accounting.ChainWork
import Proofs.Accounting.Prob
import Proofs.Accounting.Joint
import Proofs.Accounting.JointBridge.WorkOfCreditedWords
import Proofs.Accounting.JointBridge.SpaceOfPeriodHeld
import Proofs.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted
import Proofs.Accounting.JointLive
import Proofs.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Proofs.Accounting.JointBridge.HonestH200
import Proofs.Accounting.JointBridge.IntegrityOfIntegrityPeriod
import Proofs.Accounting.JointBridge.IntegrityOfIntegrityBlock
import Proofs.Accounting.Pouw
import Proofs.Accounting.PouwWork
import Proofs.Accounting.BandwidthAudit
import Proofs.Accounting.RateAudit
import Proofs.Accounting.BoundReceipt
import Proofs.Accounting.Capacity
import Proofs.Accounting.DeclaredCircuit
import Proofs.Accounting.DeclaredCircuit.Tiles
import Proofs.Accounting.DeclaredCircuit.Credit
import Proofs.Accounting.DeclaredCircuit.Slack
import Proofs.Accounting.DeclaredCircuit.Session
import Proofs.Accounting.DeclaredCircuit.Requirement
import Proofs.Accounting.DeclaredCircuit.Scheduled
import Proofs.Accounting.Remote
import Proofs.Accounting.PerRunTruncation
import Proofs.Accounting.PerRunPouw
import Proofs.Accounting.PerRunJoint
import Proofs.Accounting.PerRunNecessity
import Proofs.Accounting.PerRunPriced
import Proofs.Accounting.PerRunStop
import Proofs.Accounting.RemoteHeld
import Proofs.Accounting.RemoteHeld.Meets
import Proofs.Accounting.RemoteHeld.OverTime
import Proofs.Accounting.RemoteHeld.Draw
import Proofs.Accounting.RemoteHeld.DrawMeets
import Proofs.Accounting.RemoteHeld.Binding
import Proofs.Accounting.RemoteHeld.Sketch
import Proofs.Accounting.RemoteHeld.SketchLine
import Proofs.Accounting.RemoteHeld.SketchP2
import Proofs.Accounting.RemoteHeld.Line
import Proofs.Accounting.RemoteHeld.Component
import Proofs.Accounting.Requirement
import Proofs.Accounting.RequirementPouw
import Proofs.Accounting.Sanity
import Proofs.Accounting.Stacking
import Proofs.Accounting.DeclaredCircuit.Witness
import Proofs.Accounting.DeclaredCircuit.WitnessPrefill
import Proofs.Accounting.Tag
import Proofs.Accounting.TS
import Proofs.Accounting.TS.Stages
import Proofs.Accounting.TS.Steer
import Proofs.Accounting.TS.Grind
import Proofs.Accounting.TS.Fixed
import Proofs.Accounting.TS.FixedMeets
import Proofs.Accounting.TS.SharedNoise
import Proofs.Accounting.TS.J10RChecks
import Proofs.Accounting.TS.SessionSound
import Proofs.Accounting.TS.CheckedSound
import Proofs.Accounting.TS.GroupSound
import Proofs.Accounting.TS.LanesSound
import Proofs.Accounting.TS.Session.Lanes.Layout
import Proofs.Accounting.TS.Session.Lanes.Decode
import Proofs.Accounting.TS.Session.Lanes.Range
import Proofs.Accounting.TS.Session.Lanes.Holds
import Proofs.Accounting.TS.Session.Lanes.PortBits
import Proofs.Accounting.TS.Session.Lanes.KnownBits
import Proofs.Accounting.TS.Session.Lanes.KnownOf
import Proofs.Accounting.TS.Session.Lanes.KnownFields
import Proofs.Accounting.TS.Session.Lanes.KnownRead
import Proofs.Accounting.TS.Session.Lanes.Combine
import Proofs.Accounting.TS.Session.Lanes.Lane
import Proofs.Pouw.H1Zy.Stream
import Proofs.Accounting.TS.HeldSound
import Proofs.Accounting.TS.Session.Tile.Computes
import Proofs.Accounting.TS.TileSound
import Proofs.Accounting.Actual
import Proofs.Accounting.Responder
import Proofs.Accounting.Responder.Latency
import Proofs.Accounting.HMu
import Proofs.Accounting.CoinFlow
import Proofs.Accounting.Compose
import Proofs.Accounting.ComposePeriod
import Proofs.Accounting.ComposePeriod.Honest
import Proofs.Accounting.Served
import Proofs.Accounting.Additive
import Proofs.Accounting.ServingBlocks
import Proofs.Accounting.ServingBlocks.Witnesses
import Proofs.Accounting.Coverage

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
* `JointPeriodLive`, `JointHorizonLive`, `JointHorizonBlocksLive`: `JointLive.lean`.
* `ComputeOfPouw`: `Pouw.lean`; `Compose.CircuitCredit`: `Compose.lean`; `ComposePeriod.CircuitCreditPeriod`:
  `ComposePeriod.lean`.
* `TS.Sm90SheetHeadline`: `HeadlineDc/Sm90.lean`; `TS.Sm120SheetHeadline`: `Headline/Sm120.lean`.
* `CapacityOfStateBound`: `Capacity.lean`; `Memory.Adds`: `Memory.lean`.
* `BandwidthOfAudit`: `BandwidthAudit.lean`; `Bandwidth.SecurityProofs.AuditRO`: `Bandwidth/AuditRO.lean`.
* `AccelAdditive`: `Additive.lean`; `Actual.Runtime`: `Actual/Runtime.lean`.
* `ServingBlocks.ServingBlockBound`: `ServingBlocks.lean`.

## Where to look

* The statements and the vocabulary they read: `Security/Definitions/Accounting.lean`, whose module docstring lists the
  top guarantees with one line each.
* The properties these proofs prove: `Security/Properties/Accounting/`.
* Which of these statements protocol soundness rests on: `Security/README.md`.
-/
