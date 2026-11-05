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

/-! Resource accounting's vocabulary: the device model every resource audit is about, a joint audit's run on it, and
how PoUW's game, the bandwidth audit's game and a state-bounded capacity game realize its compute, bandwidth and
capacity audits, the filler's cell check and the windows that check it, the rate audit's game with its embedding as a
bandwidth audit whose rebuilds are charged as compute,
with side sources that answer what the chip sent, over memory tiers and chained passes, the causal rounds its samples
close, with partial completion, and its sketch-then-hash variant for a device that reads sectors verbatim, with such
side sources too and for a prover whose memory holds any function of the store, the deployed game that composes them
(partial completion, interactive side sources, tiers, ground chained passes and a session of pooled draws), the remote
bandwidth audit's node and run, the actual circuit, its physical assumptions, a challenge's reach and the device's
schedule it induces, the actual circuit's runs, strategies, soundness and the confidence set, a location's window with
the ring's oracle locations, its provers' cost model, the tags and sketches the audits read, the headline instances as
exact data (a device's sheet, a design's TS model and its floor program), a receipt's skipped values
and the coin rounds that give (H-μ⁺), a Program's coin flow under the Build's two coin rules, the generic-mix game of
receipts bound to the matmul on two locations, the deployed bound receipt's sketch as a two-party protocol, what a
run's guarantee per run is, PoUW's verdict, credits and help on one node of the actual circuit, stacking's carriers:
device memory whose post-seed contents are a circuit's values, seen through its inputs and encodes, a declared
circuit's requirement: the values its schedule keeps live and reads, by block, and when the audit's store and plans are
its, a pass's layout as the verifier's plan of a schedule of counted bits (`ScheduleDeterministic`), the run tree of
a prover's wipes that the memory cap's lifetime pool is over, a device's period as runs of the machine model 𝓜
that compute additivity is stated over, and the gate model the lower bounds on a declared
circuit are proved over (`Gates`). -/

@[expose] public section
