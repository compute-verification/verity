module

public import Definitions.Pouw.Guarantees.Advantage
public import Definitions.Pouw.Guarantees.Deadline
public import Definitions.Pouw.Guarantees.Dither
public import Definitions.Pouw.Guarantees.DitherInstance
public import Definitions.Pouw.Guarantees.Eps1
public import Definitions.Pouw.Guarantees.Game
public import Definitions.Pouw.Guarantees.NCP
public import Definitions.Pouw.Guarantees.OnsiteFreed
public import Definitions.Pouw.Guarantees.PearlC
public import Definitions.Pouw.Guarantees.PearlCFlockTile
public import Definitions.Pouw.Guarantees.PearlCTileCheck
public import Definitions.Pouw.Guarantees.Receipt
public import Definitions.Pouw.Guarantees.RelativeSecurity
public import Definitions.Pouw.Guarantees.ShaRealize
public import Definitions.Pouw.Guarantees.Window

/-!
# PoUW's guarantees (trusted)

Every statement a document outside the package cites as proved, as a `def G : Prop` over `Definitions.Pouw` and
`Definitions.Pouw.Assumptions`, by topic. Its proof is the theorem `Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`; a pin cited only
inside the package stays a lemma under its own name.
-/

@[expose] public section
