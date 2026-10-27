import Definitions.Accounting.Remote.Node
import Definitions.Accounting.Remote.Held
import Definitions.Accounting.Remote.Outside
import Definitions.Accounting.Remote.Run
import Definitions.Accounting.Remote.Draw
import Definitions.Accounting.Remote.Fork
import Definitions.Accounting.Remote.Sketch
import Definitions.Accounting.Remote.SketchLine

/-! The remote bandwidth audit's vocabulary: the node, the audit's parameters and the γ terms, a run and its routes,
the node's memory as a trace (`Remote.Held`) and what devices outside it compute on (`Remote.Outside`), the verifier's
opening draw with its pooled bound (`Remote.Draw`), the receipts' fork game (`Remote.Fork`), the receipts' sketch over
a session's seeds and devices (`Remote.Sketch`), and the draw's coins read as a sketch outcome (`Remote.SketchLine`). -/
