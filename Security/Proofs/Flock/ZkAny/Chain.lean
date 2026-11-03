module

public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.Soundness.Audit.Law

/-!
# Zero knowledge at any values across sessions: the chain rule

`OutcomeWithin.chain` composes two sessions, the second adaptive. Its worker, release and outcome may read all of the
first session's randomness, since the worker saw it. Its simulator reads only the first session's public view: its
outcome and its release. Each session's bound is the per-outcome one (`OutcomeWithin`), at every value of what came
before.

The pair is again `OutcomeWithin`, over the product of the randomness spaces and of the outcome sets, at the joint
outcome's errors added. So the rule iterates: session 1 can itself be the first `k` sessions.
`OutcomeWithin.le_card` then bounds the stream by `|St₁ × St₂|` times the uniform-outcome simulator, plus the errors
summed over joint outcomes: `|St₂| ∑ ε₁ + |St₁| ∑ ε₂`.

The proof conditions on the first session's randomness, swaps the second session for its simulator, and exchanges the
order of averaging (`FlockSoundness.Audit.Law.avg_comm`). It then swaps the first session for its simulator, at each
value of the second simulator's coins.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **The chain rule across sessions.** If the first session is per-outcome bounded, and the second is, at every value
of the first session's randomness, with a simulator that reads only the first session's outcome and release, then the
pair is per-outcome bounded at the joint outcome, with the errors added. -/
theorem OutcomeWithin.chain {Ω₁ Ω₂ Ωs₁ Ωs₂ St₁ St₂ V₁ V₂ : Type} [Fintype Ω₁] [Fintype Ω₂] [Fintype Ωs₁]
    [Fintype Ωs₂] [Nonempty Ω₁] [Nonempty Ωs₂] {out₁ : Ω₁ → St₁} {rel₁ : Ω₁ → V₁} {S₁ : St₁ → Ωs₁ → V₁}
    {ε₁ : St₁ → ℝ≥0∞} {out₂ : Ω₁ → Ω₂ → St₂} {rel₂ : Ω₁ → Ω₂ → V₂} {S₂ : St₁ → V₁ → St₂ → Ωs₂ → V₂}
    {ε₂ : St₂ → ℝ≥0∞} (h₁ : OutcomeWithin out₁ rel₁ S₁ ε₁)
    (h₂ : ∀ ω₁, OutcomeWithin (out₂ ω₁) (rel₂ ω₁) (S₂ (out₁ ω₁) (rel₁ ω₁)) ε₂) :
    OutcomeWithin (fun p : Ω₁ × Ω₂ => (out₁ p.1, out₂ p.1 p.2)) (fun p => (rel₁ p.1, rel₂ p.1 p.2))
      (fun s (q : Ωs₁ × Ωs₂) => (S₁ s.1 q.1, S₂ s.1 (S₁ s.1 q.1) s.2 q.2)) (fun s => ε₁ s.1 + ε₂ s.2) := by
  rintro ⟨s₁, s₂⟩ E
  have step₂ : ∀ ω₁, prCoin (fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂)) ≤
      prCoin (fun σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ := by
    intro ω₁
    by_cases h : out₁ ω₁ = s₁
    · calc prCoin (fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂))
          ≤ prCoin (fun ω₂ => out₂ ω₁ ω₂ = s₂ ∧ (fun v => E (rel₁ ω₁, v)) (rel₂ ω₁ ω₂)) :=
            prCoin_mono fun ω₂ hω => ⟨(Prod.mk.inj hω.1).2, hω.2⟩
        _ ≤ prCoin (fun σ₂ => (fun v => E (rel₁ ω₁, v)) (S₂ (out₁ ω₁) (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ :=
            h₂ ω₁ s₂ fun v => E (rel₁ ω₁, v)
        _ = prCoin (fun σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ := by
            simp only [h, true_and]
    · calc prCoin (fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂))
          ≤ prCoin (fun _ : Ω₂ => False) := prCoin_mono fun ω₂ hω => h (Prod.mk.inj hω.1).1
        _ = 0 := by simp [prCoin, avg]
        _ ≤ _ := zero_le
  have swap : ∀ {C D : Type} [Fintype C] [Fintype D] (P : C → D → Prop),
      (avg fun c => prCoin fun d => P c d) = avg fun d => prCoin fun c => P c d := fun P => by
    unfold prCoin
    exact FlockSoundness.Audit.Law.avg_comm _
  calc prCoin (fun p : Ω₁ × Ω₂ => (out₁ p.1, out₂ p.1 p.2) = (s₁, s₂) ∧ E (rel₁ p.1, rel₂ p.1 p.2))
      = avg fun ω₁ => prCoin fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂) :=
        prCoin_prod_split (fun ω₁ ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂))
    _ ≤ avg fun ω₁ => prCoin (fun σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ :=
        avg_mono step₂
    _ = (avg fun σ₂ => prCoin fun ω₁ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ := by
        rw [avg_add, avg_const, swap fun ω₁ σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)]
    _ ≤ (avg fun σ₂ => prCoin (fun σ₁ => E (S₁ s₁ σ₁, S₂ s₁ (S₁ s₁ σ₁) s₂ σ₂)) + ε₁ s₁) + ε₂ s₂ :=
        add_le_add (avg_mono fun σ₂ => h₁ s₁ fun v => E (v, S₂ s₁ v s₂ σ₂)) le_rfl
    _ = prCoin (fun q : Ωs₁ × Ωs₂ => E (S₁ s₁ q.1, S₂ s₁ (S₁ s₁ q.1) s₂ q.2)) + (ε₁ s₁ + ε₂ s₂) := by
        rw [avg_add, avg_const, add_assoc,
          prCoin_prod_split (fun σ₁ σ₂ => E (S₁ s₁ σ₁, S₂ s₁ (S₁ s₁ σ₁) s₂ σ₂)),
          swap fun σ₁ σ₂ => E (S₁ s₁ σ₁, S₂ s₁ (S₁ s₁ σ₁) s₂ σ₂)]

end FlockSoundness.ZK
