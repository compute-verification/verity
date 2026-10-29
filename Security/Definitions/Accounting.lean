import Definitions.Accounting.Actual
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
import Definitions.Accounting.RateAuditPoub
import Definitions.Accounting.RateAuditSketch
import Definitions.Accounting.RateAuditSketchDeployed
import Definitions.Accounting.RateAuditSketchInteractive
import Definitions.Accounting.RateAuditSketchLayout
import Definitions.Accounting.Poub
import Definitions.Accounting.Poub.Deployed
import Definitions.Accounting.Requirement
import Definitions.Accounting.Requirement.Placed
import Definitions.Accounting.Responder
import Definitions.Accounting.Schedule
import Definitions.Accounting.Served
import Definitions.Accounting.Tag
import Definitions.Accounting.TS
import Definitions.Accounting.TS.Fixed
import Definitions.Accounting.Window

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
a prover's wipes that the memory cap's lifetime pool is over, and the gate model the lower bounds on a declared
circuit are proved over (`Gates`). -/
