import Definitions.Core.Game
import Proofs.Flock.Soundness.Game.Basic

/-!
# C-Flock's games as core's games

C-Flock's soundness is stated over its own public-coin games (`FlockSoundness.Game`), and the hidden audit plays core's
(`Verity.Game`). The two inductives have the same constructors and the same `Strategy`, `prob` and `value`, but are
distinct types. `toVerity` carries a C-Flock game to core's node by node; `stratToVerity` and `stratOfVerity`
carry a prover's strategies both ways and are inverse to each other; and every event has the same probability under a
strategy and its image (`prob_toVerity`, `prob_ofVerity`), and the same value for the optimal prover (`value_toVerity`).
So a bound C-Flock proves on `FlockSoundness.Game.prob` is the same bound on core's game, for the same prover.
-/

namespace Pouw.PearlC.FlockTile

open scoped ENNReal

variable {α β : Type}

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

theorem stratOfVerity_toVerity :
    ∀ (g : FlockSoundness.Game α) (s : FlockSoundness.Game.Strategy g),
      stratOfVerity g (stratToVerity g s) = s
  | .ret _, _ => rfl
  | .send _ k, ⟨m, s'⟩ => by
      show (⟨m, stratOfVerity (k m) (stratToVerity (k m) s')⟩ :
        FlockSoundness.Game.Strategy (FlockSoundness.Game.send _ k)) = ⟨m, s'⟩
      rw [stratOfVerity_toVerity (k m) s']
  | @FlockSoundness.Game.coin _ _ _ _ k, s => funext fun c => stratOfVerity_toVerity (k c) (s c)

theorem stratToVerity_ofVerity :
    ∀ (g : FlockSoundness.Game α) (s : Verity.Game.Strategy (toVerity g)),
      stratToVerity g (stratOfVerity g s) = s
  | .ret _, _ => rfl
  | .send _ k, ⟨m, s'⟩ => by
      show (⟨m, stratToVerity (k m) (stratOfVerity (k m) s')⟩ :
        Verity.Game.Strategy (toVerity (FlockSoundness.Game.send _ k))) = ⟨m, s'⟩
      rw [stratToVerity_ofVerity (k m) s']
  | @FlockSoundness.Game.coin _ _ _ _ k, s => funext fun c => stratToVerity_ofVerity (k c) (s c)

/-- **The bridge keeps every event's probability**: core's `prob` of `p` on the image of C-Flock's game, under the
image of a strategy, is C-Flock's `prob` of `p` under the strategy. -/
theorem prob_toVerity (p : α → Prop) :
    ∀ (g : FlockSoundness.Game α) (s : FlockSoundness.Game.Strategy g),
      Verity.Game.prob p (toVerity g) (stratToVerity g s) = FlockSoundness.Game.prob p g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_toVerity p (k s.1) s.2
  | @FlockSoundness.Game.coin _ C _ _ k, s => by
      show ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * Verity.Game.prob p (toVerity (k c)) (stratToVerity (k c) (s c)) =
        ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * FlockSoundness.Game.prob p (k c) (s c)
      exact Finset.sum_congr rfl fun c _ => by rw [prob_toVerity p (k c) (s c)]

/-- `prob_toVerity` from core's side: C-Flock's `prob` under the strategy a core strategy is, is core's. -/
theorem prob_ofVerity (p : α → Prop) (g : FlockSoundness.Game α) (s : Verity.Game.Strategy (toVerity g)) :
    FlockSoundness.Game.prob p g (stratOfVerity g s) = Verity.Game.prob p (toVerity g) s := by
  rw [← prob_toVerity p g (stratOfVerity g s), stratToVerity_ofVerity]

/-- The bridge keeps the optimal prover's value. -/
theorem value_toVerity (p : α → Prop) :
    ∀ g : FlockSoundness.Game α, Verity.Game.value p (toVerity g) = FlockSoundness.Game.value p g
  | .ret _ => rfl
  | .send M k => by
      show (⨆ m : M, Verity.Game.value p (toVerity (k m))) = ⨆ m : M, FlockSoundness.Game.value p (k m)
      exact iSup_congr fun m => value_toVerity p (k m)
  | @FlockSoundness.Game.coin _ C _ _ k => by
      show ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * Verity.Game.value p (toVerity (k c)) =
        ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * FlockSoundness.Game.value p (k c)
      exact Finset.sum_congr rfl fun c _ => by rw [value_toVerity p (k c)]

/-- Core's soundness of the image is C-Flock's bound on the optimal prover. -/
theorem sound_toVerity (g : FlockSoundness.Game α) (accept E : α → Prop) (δ : ℝ≥0∞) :
    Verity.Sound (toVerity g) accept E δ ↔ FlockSoundness.Game.value (fun a => accept a ∧ ¬ E a) g ≤ δ := by
  rw [Verity.Sound, value_toVerity]

theorem toVerity_bind (f : α → FlockSoundness.Game β) :
    ∀ g : FlockSoundness.Game α, toVerity (g.bind f) = (toVerity g).bind fun a => toVerity (f a)
  | .ret _ => rfl
  | .send M k => congrArg (Verity.Game.send M) (funext fun m => toVerity_bind f (k m))
  | @FlockSoundness.Game.coin _ C iF iN k =>
      congrArg (@Verity.Game.coin _ C iF iN) (funext fun c => toVerity_bind f (k c))

/-- The bridge commutes with relabelling the output. -/
theorem toVerity_map (f : α → β) (g : FlockSoundness.Game α) :
    toVerity (g.map f) = (toVerity g).map f :=
  toVerity_bind (fun a => FlockSoundness.Game.ret (f a)) g

/-- Relabelling core's strategies there and back is the identity (`Strategy.toMap_ofMap`'s other half). -/
theorem ofMap_toMap (f : α → β) :
    ∀ (g : Verity.Game α) (s : Verity.Game.Strategy g),
      Verity.Game.Strategy.ofMap f g (Verity.Game.Strategy.toMap f g s) = s
  | .ret _, _ => rfl
  | .send _ k, ⟨m, s'⟩ => by
      show (⟨m, Verity.Game.Strategy.ofMap f (k m) (Verity.Game.Strategy.toMap f (k m) s')⟩ :
        Verity.Game.Strategy (Verity.Game.send _ k)) = ⟨m, s'⟩
      rw [ofMap_toMap f (k m) s']
  | @Verity.Game.coin _ _ _ _ k, s => funext fun c => ofMap_toMap f (k c) (s c)

end Pouw.PearlC.FlockTile
