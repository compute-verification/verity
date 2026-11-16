module

public import Proofs.Flock.SampledGeneric.Statements

/-!
# The sampled audit's zero knowledge, generic in the proof system (proved)

Two hybrids. First, on each commitment the prover's coins give, replace the session's honest view by its simulator's
(`Close.coin`, at `ε_zk`, because the proof system is zero knowledge at every such statement): this is `auditSim` at the
real trace `X`. Then replace the commitment to `X` by one to the fixed trace `X₀`: `auditSim` at a trace is the
commitment to it post-processed by the verifier's sample and the simulator (`auditSim_eq_andThen`), so hiding (at
`ε_hide`) moves it to `auditSim` at `X₀`. `Close.trans` adds the two.

* `prob_map_strat`: relabelling the output with the strategy that plays the same moves relabels the event;
* `Close.trans`, `Close.symm`, `Close.mono`: closeness is a pseudometric's ball, and grows with its radius;
* `Close.coin`: closeness under a uniform coin flipped first;
* `auditSimAfter`, `auditSim_eq_andThen`: the simulator as a post-processing of the commitment;
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

namespace Close

variable {V : Type} {a b c : Exp V}

theorem trans {ε₁ ε₂ : ℝ≥0∞} (h₁ : Close a b ε₁) (h₂ : Close b c ε₂) : Close a c (ε₁ + ε₂) := fun E =>
  ⟨(h₁ E).1.trans ((add_le_add (h₂ E).1 le_rfl).trans_eq (by rw [add_assoc, add_comm ε₂ ε₁])),
    (h₂ E).2.trans ((add_le_add (h₁ E).2 le_rfl).trans_eq (add_assoc _ _ _))⟩

theorem symm {ε : ℝ≥0∞} (h : Close a b ε) : Close b a ε := fun E => ⟨(h E).2, (h E).1⟩

theorem mono {ε ε' : ℝ≥0∞} (h : Close a b ε) (hε : ε ≤ ε') : Close a b ε' := fun E =>
  ⟨(h E).1.trans (add_le_add le_rfl hε), (h E).2.trans (add_le_add le_rfl hε)⟩

/-- **Closeness under a uniform coin**: if each coin's two experiments are `ε`-close, so are the experiments that sample
the coin first and relabel each view by the coin. -/
theorem coin {Rnd W : Type} [Fintype Rnd] [Nonempty Rnd] (a b : Rnd → Exp V) (f : Rnd → V → W) {ε : ℝ≥0∞}
    (h : ∀ r, Close (a r) (b r) ε) :
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

end Close

variable {n : ℕ} {Tr Rnd Com Wit VS View : Type} [Fintype Rnd] [Nonempty Rnd]

/-- **The simulator after the commitment `c`**: the verifier `v`'s sample on seeing `c`, and the proof system's
simulator on that sample's statement, with the view relabelled to carry `c` and the sample. -/
def auditSimAfter (Z : Zk (Com × Finset (Fin n)) Wit VS View) (samp : VS → Com → Finset (Fin n)) (v : VS) (c : Com) :
    Exp (Com × Finset (Fin n) × View) :=
  ⟨(Z.sim (c, samp v c) v).g.map fun w => (c, samp v c, w), Strategy.map _ _ (Z.sim (c, samp v c) v).s⟩

/-- **The audit's simulator is a post-processing of the commitment**: at any trace `X`, `auditSim` is the commitment to
`X` followed by `auditSimAfter`, which reads only the commitment. -/
theorem auditSim_eq_andThen (H : Hider Tr Rnd Com) (Z : Zk (Com × Finset (Fin n)) Wit VS View)
    (samp : VS → Com → Finset (Fin n)) (X : Tr) (v : VS) :
    auditSim H Z samp X v = (H.exp X).andThen (auditSimAfter Z samp v) := rfl

end FlockSoundness.SampledGeneric

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit's zero knowledge, generic in the proof system** (`Flock.Guarantees.GenericSampledZK`): the real
view is `ε_zk`-close to `auditSim` at the real trace, which simulates each session on the real commitment
(`Close.coin`), and that is `ε_hide`-close to `auditSim` at `X₀`, since both are a commitment post-processed by
`auditSimAfter` (`auditSim_eq_andThen`, `Hider.Hiding`). -/
theorem Flock.SecurityProofs.GenericSampledZK : Flock.Guarantees.GenericSampledZK := by
  intro n Tr Rnd Com Wit VS View _ _ H Z Rel εzk εh samp wit hZ hH X hX X₀ v
  have h1 : Close (auditView H Z samp wit X v) (auditSim H Z samp X v) εzk :=
    Close.coin (fun r => Z.real (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v)
      (fun r => Z.sim (H.commit X r, samp v (H.commit X r)) v) (fun r w => (H.commit X r, samp v (H.commit X r), w))
      fun r => hZ _ _ (hX r _) v
  have h2 : Close (auditSim H Z samp X v) (auditSim H Z samp X₀ v) εh := by
    rw [auditSim_eq_andThen, auditSim_eq_andThen]
    exact hH _ X X₀ (auditSimAfter Z samp v)
  exact h1.trans h2
