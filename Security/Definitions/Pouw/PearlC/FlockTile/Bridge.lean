import Definitions.Core.Game
import Definitions.Flock.Game

/-!
# C-Flock's games as core's games

C-Flock's soundness is stated over its own public-coin games (`FlockSoundness.Game`), and the hidden audit plays core's
(`Verity.Game`). The two inductives have the same constructors and the same `Strategy`, `prob` and `value`, but are
distinct types. `toVerity` carries a C-Flock game to core's node by node; `stratToVerity` and `stratOfVerity`
carry a prover's strategies both ways. That they are inverse to each other, and that every event has the same
probability under a strategy and its image, are lemmas (`Proofs.Pouw.PearlC.FlockTile.Bridge`).
-/

namespace Pouw.PearlC.FlockTile

variable {α : Type}

/-- C-Flock's game as core's: the same tree, the same messages and the same coins. -/
def toVerity : FlockSoundness.Game α → Verity.Game α
  | .ret a => .ret a
  | .send M k => .send M fun m => toVerity (k m)
  | @FlockSoundness.Game.coin _ C iF iN k => @Verity.Game.coin _ C iF iN fun c => toVerity (k c)

/-- A strategy for C-Flock's game is one for core's: the same message at every node. -/
def stratToVerity : (g : FlockSoundness.Game α) → FlockSoundness.Game.Strategy g →
    Verity.Game.Strategy (toVerity g)
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, stratToVerity (k s.1) s.2⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, s => fun c => stratToVerity (k c) (s c)

/-- A strategy for core's image of C-Flock's game is one for C-Flock's, the inverse of `stratToVerity`. -/
def stratOfVerity : (g : FlockSoundness.Game α) → Verity.Game.Strategy (toVerity g) →
    FlockSoundness.Game.Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, stratOfVerity (k s.1) s.2⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, s => fun c => stratOfVerity (k c) (s c)

end Pouw.PearlC.FlockTile
