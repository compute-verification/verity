import Definitions.Accounting.Device
import Definitions.Accounting.Joint
import Definitions.Accounting.Pouw
import Definitions.Accounting.BandwidthAudit
import Definitions.Accounting.Capacity
import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditJoint
import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditSketch
import Definitions.Accounting.Tag

/-! Resource accounting's vocabulary: the device model every resource audit is about, a joint audit's run on it, and
how PoUW's game, the bandwidth audit's game and a state-bounded capacity game realize its compute, bandwidth and
capacity audits, the rate audit's game with its embedding as a bandwidth audit whose rebuilds are charged as compute,
with side sources that answer what the chip sent, over memory tiers and chained passes, and its sketch-then-hash
variant for a device that reads sectors verbatim, and the tags and sketches the audits read. -/
