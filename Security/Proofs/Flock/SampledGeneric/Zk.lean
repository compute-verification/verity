module

public import Proofs.Flock.SampledGeneric.Statements

/-!
# The sampled audit's zero knowledge, generic in the proof system (proved)

Two hybrids: replace the session's view on each commitment by its simulator's (`close_coin`, at `ε_zk` because the
proof system is zero knowledge at every commitment the prover's coins give), then replace the commitment to the trace by
one to the fixed trace `X₀` (`Hider.Hiding`, at `ε_hide`, against the post-processing that samples and simulates).
`Close.trans` adds the two.

* `prob_map_strat`: relabelling the output with the strategy that plays the same moves relabels the event;
* `pr_coin_map`, `Close.trans`, `close_coin`, `le_coin`;
* `Flock.SecurityProofs.GenericSampledZK`, and `Flock.SecurityProofs.GenericSampledZKMul` for a proof system zero
  knowledge with a factor (`Zk.ZKMul`, the ZK server's form), by the same two hybrids.
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

/-- An event's probability in the experiment that draws a uniform coin, runs the coin's experiment and relabels its view
by the coin is the average over the coin. -/
theorem pr_coin_map {Rnd V W : Type} [Fintype Rnd] [Nonempty Rnd] (a : Rnd → Exp V) (f : Rnd → V → W)
    (E : W → Prop) :
    Exp.pr ⟨.coin Rnd fun r => (a r).g.map (f r), fun r => Strategy.map _ _ (a r).s⟩ E =
      avg fun r => (a r).pr fun v => E (f r v) :=
  congrArg avg (funext fun r => prob_map_strat E (f r) (a r).g (a r).s)

/-- **Closeness under a uniform coin**: if each coin's two experiments are `ε`-close, so are the experiments that draw
the coin first and relabel each view by the coin. -/
theorem close_coin {Rnd V W : Type} [Fintype Rnd] [Nonempty Rnd] (a b : Rnd → Exp V) (f : Rnd → V → W)
    {ε : ℝ≥0∞} (h : ∀ r, Close (a r) (b r) ε) :
    Close ⟨.coin Rnd fun r => (a r).g.map (f r), fun r => Strategy.map _ _ (a r).s⟩
      ⟨.coin Rnd fun r => (b r).g.map (f r), fun r => Strategy.map _ _ (b r).s⟩ ε := by
  intro E
  rw [pr_coin_map a f E, pr_coin_map b f E]
  constructor
  · rw [← avg_const (C := Rnd) ε, ← avg_add]
    exact avg_mono fun r => (h r fun v => E (f r v)).1
  · rw [← avg_const (C := Rnd) ε, ← avg_add]
    exact avg_mono fun r => (h r fun v => E (f r v)).2

/-- **A bound with a factor under a uniform coin**: if each coin's first experiment is at most `c` times the second plus
`ε r` on the relabelled event, the experiments that draw the coin first are within `c` times plus the average of `ε`. -/
theorem le_coin {Rnd V W : Type} [Fintype Rnd] [Nonempty Rnd] (a b : Rnd → Exp V) (f : Rnd → V → W) (c : ℝ≥0∞)
    (ε : Rnd → ℝ≥0∞) (E : W → Prop)
    (h : ∀ r, (a r).pr (fun v => E (f r v)) ≤ c * (b r).pr (fun v => E (f r v)) + ε r) :
    Exp.pr ⟨.coin Rnd fun r => (a r).g.map (f r), fun r => Strategy.map _ _ (a r).s⟩ E ≤
      c * Exp.pr ⟨.coin Rnd fun r => (b r).g.map (f r), fun r => Strategy.map _ _ (b r).s⟩ E + avg ε := by
  rw [pr_coin_map a f E, pr_coin_map b f E]
  have hm : avg (fun r => c * (b r).pr fun v => E (f r v)) = c * avg fun r => (b r).pr fun v => E (f r v) := by
    rw [mul_comm c, ← avg_mul_const]
    exact congrArg avg (funext fun r => mul_comm _ _)
  rw [← hm, ← avg_add]
  exact avg_mono h

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

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit's zero knowledge with a factor, generic in the proof system**
(`Flock.Guarantees.GenericSampledZKMul`): the real view is at most `c` times the hybrid that simulates each session on
the real commitment plus the average session error (`le_coin`), and the hybrid is within `ε_hide` of the simulator's
(`Hider.Hiding`), which the factor multiplies. -/
theorem Flock.SecurityProofs.GenericSampledZKMul : Flock.Guarantees.GenericSampledZKMul := by
  intro n Tr Rnd Com Wit VS View _ _ H Z Rel c εzk εh samp wit hZ hH X hX X₀ v E
  let F : Com → Exp (Com × Finset (Fin n) × View) := fun cm =>
    ⟨(Z.sim (cm, samp v cm) v).g.map fun w => (cm, samp v cm, w), Strategy.map _ _ (Z.sim (cm, samp v cm) v).s⟩
  have h1 : (auditView H Z samp wit X v).pr E ≤ c * ((H.exp X).andThen F).pr E +
      avg fun r : Rnd => εzk (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v :=
    le_coin (fun r => Z.real (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v)
      (fun r => Z.sim (H.commit X r, samp v (H.commit X r)) v) (fun r w => (H.commit X r, samp v (H.commit X r), w)) c
      (fun r => εzk (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v) E
      fun r => hZ _ _ (hX r _) v _
  have h2 : ((H.exp X).andThen F).pr E ≤ (auditSim H Z samp X₀ v).pr E + εh := (hH _ X X₀ F E).1
  refine h1.trans ?_
  rw [← mul_add]
  gcongr
