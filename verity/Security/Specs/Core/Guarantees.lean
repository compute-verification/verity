import Specs.Core.Guarantees.Circuit
import Specs.Core.Guarantees.Game
import Specs.Core.Guarantees.PrivatePartition
import Specs.Core.Guarantees.TC

/-!
# Core's guarantees

Each guarantee is a `def G : Prop`, stated over `Protocol`'s definitions with the assumptions it rests on as hypotheses
(`Assumptions`). Its proof is the theorem `Verity.SecurityProofs.G : Verity.Guarantees.G`, and a guarantee without one
is open: nothing may cite it until it is proved.
-/
