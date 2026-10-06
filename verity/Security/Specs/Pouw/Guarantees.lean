import Specs.Pouw.Guarantees.Barrier
import Specs.Pouw.Guarantees.Deadline
import Specs.Pouw.Guarantees.Dimension
import Specs.Pouw.Guarantees.Game
import Specs.Pouw.Guarantees.NCP
import Specs.Pouw.Guarantees.PearlC

/-!
# PoUW's guarantees (trusted)

Every statement a document outside the package cites as proved, as a `def G : Prop` over `Pouw.Protocol` and
`Pouw.Assumptions`, by topic. Its proof is the theorem `Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`; a pin cited only
inside the package stays a lemma under its own name.
-/
