import Proofs.Flock.EndToEnd
import Proofs.Flock.EndToEndHidden

/-!
# A window of member sessions: the union bound over their `EndToEnd` bounds

`Flock.SecurityProofs.window_union`: in a window of member sessions `m ∈ ms`, the probability that every member is
accepted while some member `m` has at least `K₀ m` of its units wrong is at most
$$\sum_{m} \left(\frac{\binom{n_m-K_{0,m}}{kd_m}}{\binom{n_m}{kd_m}} + \varepsilon_{ks,m} + \delta_{link,m}\right).$$
It is the union bound over the members' events (`window_prob_exists_le`), with each member's bound taken as given:
`hE2E m` is `Flock.SecurityProofs.EndToEnd` (or `EndToEndHidden`) instantiated at member `m`'s own `n`, `kd`, `K₀`,
session and live prover, and `p m` is the probability that bound is about, in member `m`'s own game.

The window is one game `g` with one prover strategy `s`. No window game is defined in Lean yet, so the theorem doesn't
derive member `m`'s marginal from `g`: `hmarg m` says that in the window, member `m`'s event (accepted and at least
`K₀ m` units wrong) has probability at most `p m`. That is what composing the members into a window has to establish,
for instance by a member strategy that runs the rest of the window around member `m`'s session. It is a hypothesis
about how the window is built, not an assumption about the world, and the corollary adds no named assumption.

Each member's `kd m` is fixed. When the members' draws are shares of one window draw, the share sizes are random. Apply
the corollary to the window conditioned on the share sizes, one `kd m` per member, or to a window whose share sizes are
fixed. This is the union bound of the members' bounds, not the tight hypergeometric bound of the window's draw.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game
open scoped ENNReal

/-- **The union bound over a finite family of events, for a fixed strategy.** -/
theorem window_prob_exists_le {α ι : Type} (ms : Finset ι) (p : ι → α → Prop) (g : Game α) (s : Strategy g) :
    prob (fun a => ∃ m ∈ ms, p m a) g s ≤ ∑ m ∈ ms, prob (p m) g s := by
  classical
  induction ms using Finset.induction_on with
  | empty =>
      rw [Finset.sum_empty]
      exact (prob_mono (fun a h => by simp at h) g s).trans (prob_false g s).le
  | @insert m ms hm ih =>
      rw [Finset.sum_insert hm]
      refine (prob_mono (q := fun a => p m a ∨ ∃ x ∈ ms, p x a) (fun a ⟨x, hx, h⟩ => ?_) g s).trans
        ((prob_or_le _ _ g s).trans (add_le_add le_rfl ih))
      rcases Finset.mem_insert.1 hx with rfl | hx
      · exact .inl h
      · exact .inr ⟨x, hx, h⟩

/-- **A window of member sessions, end to end.** In a window `g` run against the prover strategy `s`, for members
`m ∈ ms` with acceptance `acc m` and `wrong m` units wrong in the window's outcome: if member `m`'s event in the
window has probability at most `p m` (`hmarg`), and `p m` is at most member `m`'s `EndToEnd` bound at its own
`n m`, `kd m`, `K₀ m`, `ε_ks` and `δ_link` (`hE2E`: `Flock.SecurityProofs.EndToEnd` or `EndToEndHidden` at member `m`),
then
$$\Pr[\forall m,\ \text{member } m \text{ accepted} \wedge \exists m,\ K_{0,m} \le \text{wrong}_m] \le
  \sum_m \left(\frac{\binom{n_m-K_{0,m}}{kd_m}}{\binom{n_m}{kd_m}} + \varepsilon_{ks,m} + \delta_{link,m}\right).$$
Each `kd m` is fixed: for shares of one window draw, apply it conditioned on the share sizes. -/
theorem window_union {α ι : Type} (g : Game α) (s : Strategy g) (ms : Finset ι)
    (acc : ι → α → Prop) (wrong : ι → α → ℕ) (n kd K₀ : ι → ℕ) (p εks δlink : ι → ℝ≥0∞)
    (hmarg : ∀ m ∈ ms, prob (fun a => acc m a ∧ K₀ m ≤ wrong m a) g s ≤ p m)
    (hE2E : ∀ m ∈ ms,
      p m ≤ ((n m - K₀ m).choose (kd m) : ℝ≥0∞) / ((n m).choose (kd m) : ℝ≥0∞) + εks m + δlink m) :
    prob (fun a => (∀ m ∈ ms, acc m a) ∧ ∃ m ∈ ms, K₀ m ≤ wrong m a) g s ≤
      ∑ m ∈ ms, (((n m - K₀ m).choose (kd m) : ℝ≥0∞) / ((n m).choose (kd m) : ℝ≥0∞) + εks m + δlink m) :=
  (prob_mono (q := fun a => ∃ m ∈ ms, acc m a ∧ K₀ m ≤ wrong m a)
      (fun _ ⟨hA, m, hm, hw⟩ => ⟨m, hm, hA m hm, hw⟩) g s).trans
    ((window_prob_exists_le ms _ g s).trans
      (Finset.sum_le_sum fun m hm => (hmarg m hm).trans (hE2E m hm)))

end Flock.SecurityProofs
