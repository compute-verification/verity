module

public import Definitions.Accounting.Guarantees.Actual
public import Definitions.Accounting.Guarantees.Actual.Threaded
public import Definitions.Accounting.Guarantees.Actual.Dampener
public import Definitions.Accounting.Guarantees.Additive
public import Definitions.Accounting.Guarantees.Bandwidth
public import Definitions.Accounting.Guarantees.Bandwidth.Causal
public import Definitions.Accounting.Guarantees.BandwidthAudit
public import Definitions.Accounting.Guarantees.BoundReceipt
public import Definitions.Accounting.Guarantees.BoundSketch
public import Definitions.Accounting.Guarantees.Bridges
public import Definitions.Accounting.Guarantees.Capacity
public import Definitions.Accounting.Guarantees.CoinFlow
public import Definitions.Accounting.Guarantees.Compose
public import Definitions.Accounting.Guarantees.ComposeMonitoringPeriod
public import Definitions.Accounting.Guarantees.Confidence
public import Definitions.Accounting.Guarantees.FillerLuck
public import Definitions.Accounting.Guarantees.FpG1
public import Definitions.Accounting.Guarantees.HMu
public import Definitions.Accounting.Guarantees.RegisteredCircuit
public import Definitions.Accounting.Guarantees.Gates
public import Definitions.Accounting.Guarantees.Headline
public import Definitions.Accounting.Guarantees.HeadlineDc
public import Definitions.Accounting.Guarantees.Joint
public import Definitions.Accounting.Guarantees.JointLive
public import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfAuditPerRunPeriod
public import Definitions.Accounting.Guarantees.JointBridge.CommOfPeriodLeakAccepted
public import Definitions.Accounting.Guarantees.JointBridge.CommOfPooledLeakAccepted
public import Definitions.Accounting.Guarantees.Localization
public import Definitions.Accounting.Guarantees.Memory
public import Definitions.Accounting.Guarantees.PerRun
public import Definitions.Accounting.Guarantees.PeriodBinding
public import Definitions.Accounting.Guarantees.Pouw
public import Definitions.Accounting.Guarantees.PouwWork
public import Definitions.Accounting.Guarantees.RateAudit
public import Definitions.Accounting.Guarantees.RateAuditDeployed
public import Definitions.Accounting.Guarantees.RateAuditHorizon
public import Definitions.Accounting.Guarantees.RateAuditNamed
public import Definitions.Accounting.Guarantees.RateAuditPartial
public import Definitions.Accounting.Guarantees.RateAuditRemote
public import Definitions.Accounting.Guarantees.RateAuditSketch
public import Definitions.Accounting.Guarantees.Responder
public import Definitions.Accounting.Guarantees.ServingBlocks
public import Definitions.Accounting.Guarantees.ServingBlocks.Witnesses
public import Definitions.Accounting.Guarantees.RemoteHeld
public import Definitions.Accounting.Guarantees.RemoteDraw
public import Definitions.Accounting.Guarantees.RemoteBinding
public import Definitions.Accounting.Guarantees.RemoteSketch
public import Definitions.Accounting.Guarantees.RemoteSketchLine
public import Definitions.Accounting.Guarantees.RemoteSketchP2
public import Definitions.Accounting.Guarantees.Requirement
public import Definitions.Accounting.Guarantees.Sketch
public import Definitions.Accounting.Guarantees.Stacking
public import Definitions.Accounting.Guarantees.Tag
public import Definitions.Accounting.Guarantees.TS
public import Definitions.Accounting.Guarantees.Sheet
public import Definitions.Accounting.Guarantees.TS.Stages
public import Definitions.Accounting.Guarantees.TS.FoldFanIn
public import Definitions.Accounting.Guarantees.TS.LowBit

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
localization, composition and fit, compute additivity's capacity from the machine model 𝓜,
the obligations B1, B2 and B4, the tag and sketch guarantees, Lemma B with (H-μ⁺) derived from fresh coins and Lemma B′
for a persistent pattern, the view independence the Build's coin rules give it, co-location and the link relay for
receipts bound to the matmul, what a relay of the deployed bound receipt's sketch must exchange, the bandwidth audit's
guarantee, their per-run forms, PoUW's work lower bound on an isolated node, PoUW's credit on an arbitrary declared
circuit compiled to its templates, a declared circuit's live and read bits from the blocks the audit credits, PoUW's
credits and help as its placed execution's gates, the grinding premium under hashed seeds and the deployed tile's
moment, stacking's carrier lemma with the deployed remote audit for devices that read from carriers, also read against
a declared circuit's requirement, and the composed guarantee for a declared circuit that passes the stacking property:
its credited live and read bits from an accepted stacked audit. -/

@[expose] public section
