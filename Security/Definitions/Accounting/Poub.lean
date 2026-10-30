import Definitions.Accounting.Poub.Node
import Definitions.Accounting.Poub.Held
import Definitions.Accounting.Poub.Outside
import Definitions.Accounting.Poub.Run
import Definitions.Accounting.Poub.Draw
import Definitions.Accounting.Poub.Fork
import Definitions.Accounting.Poub.Sketch
import Definitions.Accounting.Poub.SketchLine

/-! PoUB's vocabulary: the node, the audit's parameters and the γ terms, a run and its routes,
the node's memory as a trace (`Poub.Held`) and what devices outside it compute on (`Poub.Outside`), the verifier's
opening draw with its pooled bound (`Poub.Draw`), the receipts' fork game (`Poub.Fork`), the receipts' sketch over
a session's seeds and devices (`Poub.Sketch`), and the draw's coins read as a sketch outcome (`Poub.SketchLine`). -/
