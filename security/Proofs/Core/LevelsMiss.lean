module

public import Proofs.Core.Levels
public import Proofs.Core.Draw

/-!
# The sampling step of SPOT's guarantee, for a bad set fixed before the coins

The sampling step of SPOT's soundness at a miss `δ`, with no argument system in it: a set `T` of blocks that
does not depend on the coins escapes the sample with probability exactly `missProb T` (`prCoin_misses`), so a run that
is accepted only where the sample misses its bad set, or where a break occurs, is accepted with a bad set whose miss is
at most `δ` with probability at most `δ` plus the break's (`prCoin_accept_le`). The bad set may depend on the prover's
coins, drawn before the sample and independently of it.

At a design drawn in two stages (`SamplingDesign.bind`), a bad set chosen at each outer block after the outer draw,
by any rule, escapes the inner samples with probability the average over the outer coins of the product of the inner
misses (`prCoin_bind_late`). That committing late does not help is `Proofs.Core.LevelsLate`.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

namespace SamplingDesign

variable {U V : Type}

/-- **A set fixed before the coins escapes the sample with probability its miss.** -/
theorem prCoin_misses (D : SamplingDesign U) (T : Finset U) :
    prCoin (fun ω => Disjoint (D.draw ω) T) = D.missProb T := rfl

/-- **The sampling step of soundness at a miss `δ`.** The prover's coins `p`, drawn before the sample and
independently of it, fix its bad set `T p`. Where every accepted run's sample misses `T p` or a break occurs, the run
is accepted with a bad set whose miss is at most `δ` with probability at most `δ` plus the break's. -/
theorem prCoin_accept_le {P : Type} [Fintype P] [Nonempty P] (D : SamplingDesign U) (T : P → Finset U)
    (acc brk : P → D.Ω → Prop) (h : ∀ p ω, acc p ω → Disjoint (D.draw ω) (T p) ∨ brk p ω) (δ : ℝ≥0∞) :
    prCoin (fun x : P × D.Ω => acc x.1 x.2 ∧ D.missProb (T x.1) ≤ δ) ≤
      δ + prCoin (fun x : P × D.Ω => brk x.1 x.2) := by
  classical
  have hmiss : prCoin (fun x : P × D.Ω => Disjoint (D.draw x.2) (T x.1) ∧ D.missProb (T x.1) ≤ δ) ≤ δ := by
    have hp : ∀ p : P, prCoin (fun ω : D.Ω => Disjoint (D.draw ω) (T p) ∧ D.missProb (T p) ≤ δ) ≤ δ := by
      intro p
      by_cases hδ : D.missProb (T p) ≤ δ
      · exact (prCoin_mono fun ω hω => hω.1).trans hδ
      · exact (prCoin_mono (q := fun _ => False) fun ω hω => hδ hω.2).trans (prCoin_false.trans_le zero_le)
    calc prCoin (fun x : P × D.Ω => Disjoint (D.draw x.2) (T x.1) ∧ D.missProb (T x.1) ≤ δ)
        = avg fun p : P => prCoin fun ω : D.Ω => Disjoint (D.draw ω) (T p) ∧ D.missProb (T p) ≤ δ := by
          unfold prCoin
          exact Tape.avg_prod _
      _ ≤ avg fun _ : P => δ := avg_mono hp
      _ = δ := avg_const δ
  calc prCoin (fun x : P × D.Ω => acc x.1 x.2 ∧ D.missProb (T x.1) ≤ δ)
      ≤ prCoin (fun x : P × D.Ω =>
          (Disjoint (D.draw x.2) (T x.1) ∧ D.missProb (T x.1) ≤ δ) ∨ brk x.1 x.2) :=
        prCoin_mono fun x hx => (h x.1 x.2 hx.1).imp (fun hd => ⟨hd, hx.2⟩) id
    _ ≤ _ := (prCoin_or_le _ _).trans (add_le_add hmiss le_rfl)

variable [Fintype U] [DecidableEq U]

/-- **A bad set chosen after the outer draw**: at a two-stage design `D.bind F`, with the bad set `B ω₁ u` at each
outer unit `u` chosen after the outer coins `ω₁` by any rule, every inner sample misses its bad set with probability
`E_{ω₁}[∏_{u ∈ D.draw ω₁} missProb (F u) (B ω₁ u)]`, since each inner sample is drawn on coins of its own after the
choice. -/
theorem prCoin_bind_late (D : SamplingDesign U) (F : U → SamplingDesign V) (B : D.Ω → U → Finset V) :
    prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) =>
        ∀ u ∈ D.draw ω.1, Disjoint ((F u).draw (ω.2 u)) (B ω.1 u)) =
      avg fun ω₁ => ∏ u ∈ D.draw ω₁, (F u).missProb (B ω₁ u) := by
  calc prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) =>
          ∀ u ∈ D.draw ω.1, Disjoint ((F u).draw (ω.2 u)) (B ω.1 u))
      = avg fun ω₁ => prCoin fun ω₂ : (u : U) → (F u).Ω =>
          ∀ u ∈ D.draw ω₁, Disjoint ((F u).draw (ω₂ u)) (B ω₁ u) := by
        unfold prCoin
        exact Tape.avg_prod _
    _ = _ := congrArg avg (funext fun ω₁ =>
        prCoin_pi_forall (D.draw ω₁) fun u (c : (F u).Ω) => Disjoint ((F u).draw c) (B ω₁ u))

end SamplingDesign

end Verity
