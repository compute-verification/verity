import Definitions.Accounting.Remote.Node
import Definitions.Accounting.Remote.Held
import Definitions.Accounting.Remote.Outside
import Definitions.Accounting.Remote.Run
import Definitions.Accounting.Remote.Draw
import Definitions.Accounting.Remote.Sketch

/-! The remote bandwidth audit's vocabulary: the node, the audit's parameters and the γ terms, a run and its routes,
the node's memory as a trace (`Remote.Held`) and what devices outside it compute on (`Remote.Outside`), the verifier's
opening draw with its pooled bound (`Remote.Draw`), and the receipts' sketch over a session's seeds and devices
(`Remote.Sketch`). -/
