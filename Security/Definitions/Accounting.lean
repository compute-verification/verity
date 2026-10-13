import Definitions.Accounting.Memory
import Definitions.Accounting.Bandwidth
import Definitions.Accounting.Device
import Definitions.Accounting.Joint
import Definitions.Accounting.PerRun
import Definitions.Accounting.Pouw
import Definitions.Accounting.BandwidthAudit
import Definitions.Accounting.Capacity
import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditJoint
import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditSketch
import Definitions.Accounting.RateAuditSketchInteractive
import Definitions.Accounting.Tag
import Definitions.Accounting.Actual
import Definitions.Accounting.Physical
import Definitions.Accounting.Localization
import Definitions.Accounting.Bridge
import Definitions.Accounting.Confidence
import Definitions.Accounting.CostModel
import Definitions.Accounting.BoundReceipt
import Definitions.Accounting.BoundSketch

/-! Resource accounting's vocabulary: the device model every resource audit is about, a joint audit's run on it, and
how PoUW's game, the bandwidth audit's game and a state-bounded capacity game realize its compute, bandwidth and
capacity audits, the rate audit's game with its embedding as a bandwidth audit whose rebuilds are charged as compute,
with side sources that answer what the chip sent, and its sketch-then-hash variant for a device that reads sectors
verbatim, with such side sources too, the actual circuit, its physical assumptions, a challenge's reach and the
device's schedule it induces, the actual circuit's runs, strategies, soundness and the confidence set, its provers'
cost model, the tags and sketches the audits read, the generic-mix game of receipts bound to the matmul on two
locations, the deployed bound receipt's sketch as a two-party protocol, and what a run's guarantee per run is. -/
