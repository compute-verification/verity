import Definitions.Pouw.Guarantees.Barrier
import Definitions.Pouw.Guarantees.Deadline
import Definitions.Pouw.Guarantees.Dimension
import Definitions.Pouw.Guarantees.Dither
import Definitions.Pouw.Guarantees.DitherInstance
import Definitions.Pouw.Guarantees.Eps1
import Definitions.Pouw.Guarantees.Game
import Definitions.Pouw.Guarantees.NCP
import Definitions.Pouw.Guarantees.OnsiteFreed
import Definitions.Pouw.Guarantees.PearlC
import Definitions.Pouw.Guarantees.PearlCTileCheck
import Definitions.Pouw.Guarantees.Receipt
import Definitions.Pouw.Guarantees.RelativeSecurity
import Definitions.Pouw.Guarantees.ShaRealize
import Definitions.Pouw.Guarantees.Window

/-!
# PoUW's guarantees (trusted)

Every statement a document outside the package cites as proved, as a `def G : Prop` over `Definitions.Pouw` and
`Definitions.Pouw.Assumptions`, by topic. Its proof is the theorem `Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`; a pin cited only
inside the package stays a lemma under its own name.
-/
