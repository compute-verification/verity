module

public import Proofs.Slowdown.Secure
public import Proofs.Slowdown.Threads

/-!
# The threaded bound, a second training bound

`Training.threaded` is the threaded route as a training bound (`Slowdown.Security.Training`), and `threaded_holds`
proves `Training.Holds` of `Training.threaded`: a window outside `I` is a small subcircuit of the
reference training algorithm, so Theorem 1 per parameter slot (`ThreadedReadsSlots`) bounds its counted
multiplications by `S Λ / b_a`, and the windows outside `I` hold every counted multiplication outside `I`'s. So
`Secure` and `RunSecure` hold at the threaded bound as they do at the fit bound (`fit_holds`), with no other change.
-/

@[expose] public section

open InferenceOnly InferenceOnly.SecurityProofs

namespace Slowdown.Security

open Verity Finset

/-- **The threaded bound**, a training bound at the gate count `Γ` and the read size `S₀`. Its premises:

* a thread uses one weight at most `reuse` times;
* each window outside `I` holds at most `Γ` gates of the reference training algorithm;
* it is `(d, e)`-threaded at `(Γ, S₀)` (`Threading.Threaded`);
* its weights are incompressible at `b_w` bits a parameter slot and `b_a` an activation
  (`Assumptions.WeightsIncompressibleSlots`).

The last two are premises about training. The count is `S Λ / b_a` a window, `Λ = max(e, reuse · S₀ / (4 d b_w))`,
defined where `b_w`, `b_a` and `d` are positive and `S ≤ S₀`. -/
noncomputable abbrev Training.threaded (Γ : ℕ) (S₀ b_w b_a : ℝ) (d e reuse : ℕ) : Training where
  Prem := fun {c} T {_} Q I _ => T.reuse ≤ reuse ∧ (∀ j ∉ I, c.D.size (Q.gates j) ≤ Γ) ∧ T.Threaded Γ S₀ d e ∧
    ∃ σ : T.Slots, Assumptions.WeightsIncompressibleSlots c T σ Γ S₀ b_w b_a
  Valid S := 0 < b_w ∧ 0 < b_a ∧ 0 < d ∧ 0 ≤ S₀ ∧ S ≤ S₀
  bound k S := k * (S * max (e : ℝ) (reuse * S₀ / (4 * d * b_w))) / b_a
  bound_zero S := by simp

end Slowdown.Security

namespace Slowdown.SecurityProofs

open Verity Finset Slowdown.Security

/-- **The threaded bound holds** (`Training.threaded`). -/
theorem threaded_holds (Γ : ℕ) (S₀ b_w b_a : ℝ) (d e reuse : ℕ) :
    (Training.threaded Γ S₀ b_w b_a d e reuse).Holds := by
  classical
  intro c T k Q I start S ⟨hbw, hba, hd, hS0, hS⟩ ⟨hre, hΓ, hT, σ, hW⟩ hread
  have hΛ0 : 0 ≤ T.bound S₀ b_w d e := le_max_of_le_left (Nat.cast_nonneg _)
  have hΛ : T.bound S₀ b_w d e ≤ max (e : ℝ) (reuse * S₀ / (4 * d * b_w)) := by
    unfold Threading.bound
    have hpos : (0 : ℝ) < 4 * d * b_w := by positivity
    exact max_le_max le_rfl
      (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (Nat.cast_le.2 hre) hS0) hpos.le)
  have hblock : ∀ j ∉ I, b_a * ((T.mults (Q.gates j)).card : ℝ) ≤
      S * max (e : ℝ) (reuse * S₀ / (4 * d * b_w)) := fun j hj =>
    (ThreadedReadsSlots c T σ Γ S₀ b_w b_a d e hbw hba hd hT hW _
      ⟨Partition.gates_subset_computed _ j, hΓ j hj, (hread j hj).trans hS⟩).trans
      ((mul_le_mul_of_nonneg_right (hread j hj) hΛ0).trans
        (mul_le_mul_of_nonneg_left hΛ ((Nat.cast_nonneg _).trans (hread j hj))))
  have hsub : (T.M.filter fun g => ∀ j ∈ I, g ∉ Q.gates j) ⊆ Iᶜ.biUnion fun j => T.mults (Q.gates j) := by
    intro g hg
    obtain ⟨hgM, hno⟩ := mem_filter.1 hg
    have hne : Q.unit g ≠ none := fun h =>
      absurd ((Q.input_iff g).1 h) (by simp [(mem_filter.1 (T.M_computed hgM)).2])
    obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 hne
    have hgu : g ∈ Q.gates u := (Partition.mem_gates _).2 hu
    exact mem_biUnion.2 ⟨u, mem_compl.2 fun hI => hno u hI hgu, mem_inter.2 ⟨hgM, hgu⟩⟩
  have hcard : ((T.M.filter fun g => ∀ j ∈ I, g ∉ Q.gates j).card : ℝ) ≤
      ∑ j ∈ Iᶜ, ((T.mults (Q.gates j)).card : ℝ) := by
    exact_mod_cast (card_le_card hsub).trans card_biUnion_le
  have hsum : b_a * ∑ j ∈ Iᶜ, ((T.mults (Q.gates j)).card : ℝ) ≤
      ((k - I.card : ℕ) : ℝ) * (S * max (e : ℝ) (reuse * S₀ / (4 * d * b_w))) := by
    rw [mul_sum]
    refine (sum_le_sum fun j hj => hblock j (mem_compl.1 hj)).trans ?_
    rw [sum_const, card_compl, Fintype.card_fin, nsmul_eq_mul]
  change _ ≤ ((k - I.card : ℕ) : ℝ) * (S * max (e : ℝ) (reuse * S₀ / (4 * d * b_w))) / b_a
  rw [le_div_iff₀ hba]
  linarith [mul_le_mul_of_nonneg_left hcard hba.le]

end Slowdown.SecurityProofs
