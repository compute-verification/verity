module

public import Definitions.Core.Guarantees.Circuit
public import Definitions.Core.Guarantees.DeviceFigures
public import Definitions.Core.Guarantees.Game
public import Definitions.Core.Guarantees.Challenger
public import Definitions.Core.Guarantees.NaiveStrassen
public import Definitions.Core.Guarantees.PrivatePartition
public import Definitions.Core.Guarantees.TC
public import Definitions.Core.Guarantees.Formalism
public import Definitions.Core.Guarantees.Formalism.Witnesses
public import Definitions.Core.Guarantees.Expand
public import Definitions.Core.Guarantees.Expand.Witnesses
public import Definitions.Core.Guarantees.Rec
public import Definitions.Core.Guarantees.ValueSalts
public import Definitions.Core.Guarantees.MemCheck

/-!
# Core's guarantees

Each guarantee is a `def G : Prop`, stated over `Protocol`'s definitions with the assumptions it rests on as hypotheses
(`Assumptions`). Its proof is the theorem `Verity.SecurityProofs.G : Verity.Guarantees.G`, and a guarantee without one
is open: nothing may cite it until it is proved.
-/

@[expose] public section
