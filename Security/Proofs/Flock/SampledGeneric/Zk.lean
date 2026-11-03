module

public import Proofs.Flock.SampledGeneric.Statements

/-!
# The sampled audit's zero knowledge, generic in the proof system (proved)

Two hybrids: replace the session's view on each commitment by its simulator's (`close_coin`, at `ε_zk` because the
proof system is zero knowledge at every commitment the prover's coins give), then replace the commitment to the trace by
one to the fixed trace `X₀` (`Hider.Hiding`, at `ε_hide`, against the post-processing that samples and simulates).
`Close.trans` adds the two.

* `prob_map_strat`: relabelling the output with the strategy that plays the same moves relabels the event;
* `Close.trans`, `close_coin`;
* `Flock.SecurityProofs.GenericSampledZK`.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

theorem prob_map_strat {α β : Type} (p : β → Prop) (f : α → β) :
    ∀ (g : Game α) (s : Strategy g), prob p (g.map f) (Strategy.map f g s) = prob (fun a => p (f a)) g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_map_strat p f (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      show avg (fun c => prob p ((k c).map f) (Strategy.map f (k c) (s c))) = avg fun c => prob _ (k c) (s c)
      exact congrArg avg (funext fun c => prob_map_strat p f (k c) (s c))

theorem Close.trans {V : Type} {a b c : Exp V} {ε₁ ε₂ : ℝ≥0∞} (h₁ : Close a b ε₁) (h₂ : Close b c ε₂) :
    Close a c (ε₁ + ε₂) := fun E =>
  ⟨(h₁ E).1.trans ((add_le_add (h₂ E).1 le_rfl).trans_eq (by rw [add_assoc, add_comm ε₂ ε₁])),
    (h₂ E).2.trans ((add_le_add (h₁ E).2 le_rfl).trans_eq (add_assoc _ _ _))⟩

/-- **Closeness under a uniform coin**: if each coin's two experiments are `ε`-close, so are the experiments that draw
the coin first and relabel each view by the coin. -/
theorem close_coin {Rnd V W : Type} [Fintype Rnd] [Nonempty Rnd] (a b : Rnd → Exp V) (f : Rnd → V → W)
    {ε : ℝ≥0∞} (h : ∀ r, Close (a r) (b r) ε) :
    Close ⟨.coin Rnd fun r => (a r).g.map (f r), fun r => Strategy.map _ _ (a r).s⟩
      ⟨.coin Rnd fun r => (b r).g.map (f r), fun r => Strategy.map _ _ (b r).s⟩ ε := by
  have key : ∀ c : Rnd → Exp V, ∀ E : W → Prop,
      Exp.pr ⟨.coin Rnd fun r => (c r).g.map (f r), fun r => Strategy.map _ _ (c r).s⟩ E =
        avg fun r => (c r).pr fun v => E (f r v) := fun c E =>
    congrArg avg (funext fun r => prob_map_strat E (f r) (c r).g (c r).s)
  intro E
  rw [key a, key b]
  constructor
  · rw [← avg_const (C := Rnd) ε, ← avg_add]
    exact avg_mono fun r => (h r fun v => E (f r v)).1
  · rw [← avg_const (C := Rnd) ε, ← avg_add]
    exact avg_mono fun r => (h r fun v => E (f r v)).2

end FlockSoundness.SampledGeneric

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit's zero knowledge, generic in the proof system** (`Flock.Guarantees.GenericSampledZK`): the real
view is `ε_zk`-close to the hybrid that simulates each session on the real commitment (`close_coin`), which is
`ε_hide`-close to the simulator's, a commitment to `X₀` post-processed the same way (`Hider.Hiding`). -/
theorem Flock.SecurityProofs.GenericSampledZK : Flock.Guarantees.GenericSampledZK := by
  intro n Tr Rnd Com Wit VS View _ _ H Z Rel εzk εh samp wit hZ hH X hX X₀ v
  let F : Com → Exp (Com × Finset (Fin n) × View) := fun c =>
    ⟨(Z.sim (c, samp v c) v).g.map fun w => (c, samp v c, w), Strategy.map _ _ (Z.sim (c, samp v c) v).s⟩
  have h1 : Close (auditView H Z samp wit X v) ((H.exp X).andThen F) εzk :=
    close_coin (fun r => Z.real (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v)
      (fun r => Z.sim (H.commit X r, samp v (H.commit X r)) v) (fun r w => (H.commit X r, samp v (H.commit X r), w))
      fun r => hZ _ _ (hX r _) v
  have h2 : Close ((H.exp X).andThen F) (auditSim H Z samp X₀ v) εh := hH _ X X₀ F
  exact h1.trans h2
