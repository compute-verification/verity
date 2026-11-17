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
public import Definitions.Core.Guarantees.Transport

/-! Core's guarantees, each a `def G : Prop` proved by `Verity.SecurityProofs.G`; one without that proof is open, and
nothing may cite it. -/

@[expose] public section
