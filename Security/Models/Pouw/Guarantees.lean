import Models.Pouw.Guarantees.Barrier
import Models.Pouw.Guarantees.Deadline
import Models.Pouw.Guarantees.Dimension
import Models.Pouw.Guarantees.Game
import Models.Pouw.Guarantees.NCP
import Models.Pouw.Guarantees.PearlC
import Models.Pouw.Guarantees.PearlCTileCheck

/-!
# PoUW's guarantees (trusted)

Every statement a document outside the package cites as proved, as a `def G : Prop` over `Definitions.Pouw` and
`Models.Pouw.Assumptions`, by topic. Its proof is the theorem `Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`; a pin cited only
inside the package stays a lemma under its own name.
-/
