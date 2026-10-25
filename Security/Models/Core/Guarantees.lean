import Models.Core.Guarantees.Circuit
import Models.Core.Guarantees.Game
import Models.Core.Guarantees.PrivatePartition
import Models.Core.Guarantees.TC

/-!
# Core's guarantees

Each guarantee is a `def G : Prop`, stated over `Protocol`'s definitions with the assumptions it rests on as hypotheses
(`Assumptions`). Its proof is the theorem `Verity.SecurityProofs.G : Verity.Guarantees.G`, and a guarantee without one
is open: nothing may cite it until it is proved.
-/
