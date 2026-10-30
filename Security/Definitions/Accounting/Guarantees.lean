import Definitions.Accounting.Guarantees.Actual
import Definitions.Accounting.Guarantees.Actual.Threaded
import Definitions.Accounting.Guarantees.Actual.Dampener
import Definitions.Accounting.Guarantees.Bandwidth
import Definitions.Accounting.Guarantees.Bandwidth.Causal
import Definitions.Accounting.Guarantees.BandwidthAudit
import Definitions.Accounting.Guarantees.BoundReceipt
import Definitions.Accounting.Guarantees.BoundSketch
import Definitions.Accounting.Guarantees.Bridges
import Definitions.Accounting.Guarantees.Capacity
import Definitions.Accounting.Guarantees.CoinFlow
import Definitions.Accounting.Guarantees.Compose
import Definitions.Accounting.Guarantees.ComposePeriod
import Definitions.Accounting.Guarantees.Confidence
import Definitions.Accounting.Guarantees.FillerLuck
import Definitions.Accounting.Guarantees.FpG1
import Definitions.Accounting.Guarantees.HMu
import Definitions.Accounting.Guarantees.DeclaredCircuit
import Definitions.Accounting.Guarantees.Gates
import Definitions.Accounting.Guarantees.GatesBitwise
import Definitions.Accounting.Guarantees.Headline
import Definitions.Accounting.Guarantees.HeadlineDc
import Definitions.Accounting.Guarantees.Joint
import Definitions.Accounting.Guarantees.JointLive
import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfAuditPerRunPeriod
import Definitions.Accounting.Guarantees.JointBridge.CommOfPeriodLeakAccepted
import Definitions.Accounting.Guarantees.Localization
import Definitions.Accounting.Guarantees.Memory
import Definitions.Accounting.Guarantees.PerRun
import Definitions.Accounting.Guarantees.Pouw
import Definitions.Accounting.Guarantees.PouwWork
import Definitions.Accounting.Guarantees.RateAudit
import Definitions.Accounting.Guarantees.RateAuditDeployed
import Definitions.Accounting.Guarantees.RateAuditHorizon
import Definitions.Accounting.Guarantees.RateAuditNamed
import Definitions.Accounting.Guarantees.RateAuditPartial
import Definitions.Accounting.Guarantees.RateAuditRemote
import Definitions.Accounting.Guarantees.RateAuditSketch
import Definitions.Accounting.Guarantees.Responder
import Definitions.Accounting.Guarantees.RemoteHeld
import Definitions.Accounting.Guarantees.RemoteDraw
import Definitions.Accounting.Guarantees.RemoteBinding
import Definitions.Accounting.Guarantees.RemotePass
import Definitions.Accounting.Guarantees.RemoteSketch
import Definitions.Accounting.Guarantees.RemoteSketchLine
import Definitions.Accounting.Guarantees.RemoteSketchP2
import Definitions.Accounting.Guarantees.Requirement
import Definitions.Accounting.Guarantees.RequirementPouw
import Definitions.Accounting.Guarantees.Sketch
import Definitions.Accounting.Guarantees.Stacking
import Definitions.Accounting.Guarantees.Tag
import Definitions.Accounting.Guarantees.TS
import Definitions.Accounting.Guarantees.Sheet
import Definitions.Accounting.Guarantees.TS.Receipt
import Definitions.Accounting.Guarantees.TS.Grind
import Definitions.Accounting.Guarantees.TS.SharedNoise
import Definitions.Accounting.Guarantees.TS.Stages
import Definitions.Accounting.Guarantees.TS.Steer
import Definitions.Accounting.Guarantees.TS.FoldFanIn
import Definitions.Accounting.Guarantees.TS.LowBit
import Definitions.Accounting.Guarantees.TS.Fixed

/-! Resource accounting's guarantee statements: the joint guarantee and its covert residual, the joint guarantee over
a period audited by every protocol, over a horizon of periods and over a horizon of server-time blocks and the
wipes between them, PoUW's and the bandwidth
audit's guarantees as a joint run's compute and bandwidth guarantees, each family's headline `ε` on its exact data,
the filler's per-check luck, a state-bounded game's soundness as its capacity
guarantee, derived state's share of the capacity residual, the rate audit's residency and bandwidth with the rebuild
charged as compute, with partial completion and with a sketch-then-hash tag, the remote rate audit's conservation over
tiers and chained passes, the causal horizon bound on its samples and its conservation with named words, the one
composed guarantee for the remote bandwidth audit as deployed and its deployed instance, the remote
bandwidth audit's conservation and node-level guarantees on the actual-circuit model, the actual circuit's capacity
theorem, runtime corollary, memory cap over threads with its rate, dark gates and the compute dampener's windows,
localization, composition and fit,
the obligations B1, B2 and B4, the tag and sketch guarantees, Lemma B with (H-μ⁺) derived from fresh coins and Lemma B′
for a persistent pattern, the view independence the Build's coin rules give it, co-location and the link relay for
receipts bound to the matmul, what a relay of the deployed bound receipt's sketch must exchange, PoUB's per-pass lemma
(the bandwidth audit's guarantee), their per-run forms, PoUW's work lower bound on an isolated node, PoUW's credit on an
arbitrary declared circuit compiled to its templates, a declared circuit's live and read bits from the blocks the audit
credits, PoUW's credits and help as its placed execution's gates, the grinding premium under hashed seeds and the
deployed tile's moment, stacking's carrier lemma with the deployed remote audit for devices that read from carriers,
also read against a declared circuit's requirement, and the composed guarantee for a declared circuit that passes the
stacking property: its credited live and read bits from an accepted stacked audit. -/
