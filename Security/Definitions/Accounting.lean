import Definitions.Accounting.Actual
import Definitions.Accounting.Actual.Source
import Definitions.Accounting.Bandwidth
import Definitions.Accounting.BandwidthAudit
import Definitions.Accounting.BoundReceipt
import Definitions.Accounting.BoundSketch
import Definitions.Accounting.Bridge
import Definitions.Accounting.Capacity
import Definitions.Accounting.Confidence
import Definitions.Accounting.CostModel
import Definitions.Accounting.Device
import Definitions.Accounting.Joint
import Definitions.Accounting.Localization
import Definitions.Accounting.Memory
import Definitions.Accounting.PerRun
import Definitions.Accounting.Physical
import Definitions.Accounting.Pouw
import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditDeployed
import Definitions.Accounting.RateAuditHorizon
import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditJoint
import Definitions.Accounting.RateAuditPartial
import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditSketch
import Definitions.Accounting.RateAuditSketchInteractive
import Definitions.Accounting.Remote
import Definitions.Accounting.Remote.Deployed
import Definitions.Accounting.Responder
import Definitions.Accounting.Tag
import Definitions.Accounting.Window

/-! Resource accounting's vocabulary: the device model every resource audit is about, a joint audit's run on it, and
how PoUW's game, the bandwidth audit's game and a state-bounded capacity game realize its compute, bandwidth and
capacity audits, the rate audit's game with its embedding as a bandwidth audit whose rebuilds are charged as compute,
with side sources that answer what the chip sent, over memory tiers and chained passes, the causal rounds its samples
samples close, with partial completion, and its sketch-then-hash variant for a device that reads sectors verbatim, with
such side sources too, the deployed game that composes them (partial completion, interactive side sources, tiers,
ground chained passes and a session of pooled draws), the remote bandwidth audit's node and run, the actual circuit,
its physical assumptions, a challenge's reach and the device's schedule it induces, the actual circuit's runs,
strategies, soundness and the confidence set, a location's window with the ring's oracle locations, its provers' cost
model, the tags and sketches the audits read, the generic-mix game of receipts bound to the matmul on two locations,
the deployed bound receipt's sketch as a two-party protocol, and what a run's guarantee per run is. -/
