import Definitions.Accounting.Device
import Definitions.Accounting.Joint
import Definitions.Accounting.Pouw
import Definitions.Accounting.BandwidthAudit
import Definitions.Accounting.Capacity
import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditJoint
import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditHorizon
import Definitions.Accounting.RateAuditPartial
import Definitions.Accounting.RateAuditSketch
import Definitions.Accounting.RateAuditSketchInteractive
import Definitions.Accounting.Remote
import Definitions.Accounting.Tag

/-! Resource accounting's vocabulary: the device model every resource audit is about, a joint audit's run on it, and
how PoUW's game, the bandwidth audit's game and a state-bounded capacity game realize its compute, bandwidth and
capacity audits, the rate audit's game with its embedding as a bandwidth audit whose rebuilds are charged as compute,
with side sources that answer what the chip sent, over memory tiers and chained passes, the causal rounds its
samples close, with partial completion, and its sketch-then-hash variant for a device that reads sectors verbatim, with
such side sources too, the remote bandwidth audit's node and run, and the tags and sketches the audits read. -/
