import Definitions.Accounting.Memory
import Definitions.Accounting.Bandwidth
import Definitions.Accounting.Device
import Definitions.Accounting.Joint
import Definitions.Accounting.PerRun
import Definitions.Accounting.Pouw
import Definitions.Accounting.Remote
import Definitions.Accounting.Tag
import Definitions.Accounting.Actual
import Definitions.Accounting.Physical
import Definitions.Accounting.Localization
import Definitions.Accounting.Bridge
import Definitions.Accounting.Confidence
import Definitions.Accounting.Responder
import Definitions.Accounting.Window
import Definitions.Accounting.CostModel

/-! Resource accounting's vocabulary: the device model every resource audit is about, a joint audit's run on it, how
PoUW's game realizes its compute audit, the remote bandwidth audit's node and run, the actual circuit, its physical
assumptions, a challenge's reach and the device's schedule it induces, the actual circuit's runs, strategies,
soundness and the confidence set, a location's
window with the ring's oracle locations, its provers' cost model, the tags and sketches the audits read, the bandwidth
audit's game, and what a run's guarantee per run is. -/
