import Pouw.Guarantees.Barrier
import Pouw.Guarantees.Deadline
import Pouw.Guarantees.Dimension
import Pouw.Guarantees.Game
import Pouw.Guarantees.NCP

/-!
# PoUW's guarantees (trusted)

Every statement a document outside the package cites as proved, as a `def G : Prop` over `Pouw.Protocol` and
`Pouw.Assumptions`, by topic. Its proof is the theorem `Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`; a pin cited only
inside the package stays a lemma under its own name.
-/
