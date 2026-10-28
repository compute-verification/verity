import Definitions.Core.Guarantees.Circuit
import Definitions.Core.Guarantees.Game
import Definitions.Core.Guarantees.NaiveStrassen
import Definitions.Core.Guarantees.PrivatePartition
import Definitions.Core.Guarantees.TC
import Definitions.Core.Guarantees.Formalism
import Definitions.Core.Guarantees.Expand

/-!
# Core's guarantees

Each guarantee is a `def G : Prop`, stated over `Protocol`'s definitions with the assumptions it rests on as hypotheses
(`Assumptions`). Its proof is the theorem `Verity.SecurityProofs.G : Verity.Guarantees.G`, and a guarantee without one
is open: nothing may cite it until it is proved.
-/
