import Proofs.Flock.Soundness.Rewinding
import Proofs.Flock.Soundness.Merkle

/-!
# The compiled step: from oracles to Merkle caps, for every prover

The oracle model (`table_sound`) sends each committed table whole; the protocol sends a SHA-512
Merkle cap and opens columns with paths at the end of the proof. For an arbitrary prover `P*`
the table a cap "commits to" is the **plurality table** (`Rewinding.IsPlurality`): per position,
the column `P*` opens there in the most runs. The oracle-model prover `P̃` sends the plurality
table where `P*` sends the cap, and otherwise plays as `P*` does.

`compile_step` (one table, committed before any coin, as `root_B` is): if every accepting run
whose openings all agree with the plurality table is also accepting for `P̃` on the same coins
(`couple`, a fact about the two verifiers), then

  `Pr[P* accepted] ≤ Pr[P̃ accepted] + √(N · Q · Pr[two runs conflict])`.

`compile_step_prefix` is the same bound for a table committed after a prefix of coins (levels
`1..R` of each repetition): the plurality table is per prefix, and averaging `√` over prefixes
is bounded by `√` of the average (Cauchy–Schwarz), which is the probability that two runs
sharing the prefix conflict: a fork at the commitment. `conflict_collision` turns every
conflicting pair of verified openings into a collision of the hash (`Merkle.opening_binding`).
With `Pr[conflict] ≤ Adv_CR^{SHA-512}(B)` for the two-run finder `B`, these are the terms of
the compiled theorem (`README.md` §1.2, derived in `DESIGN.md` §2).
-/

open Finset

namespace FlockSoundness.Compiled

open Rewinding

variable {Ω P V : Type} [Fintype Ω] [DecidableEq Ω] [Fintype P] [DecidableEq V]

/-- Probability of an event over uniform coins. -/
noncomputable def pr (E : Ω → Prop) : ℝ := by
  classical exact (#(univ.filter E) : ℝ) / Fintype.card Ω

/-- Probability of an event over two independent runs. -/
noncomputable def pr₂ (E : Ω × Ω → Prop) : ℝ := by
  classical exact (#(univ.filter E) : ℝ) / (Fintype.card Ω) ^ 2

theorem pr_le_add (E A B : Ω → Prop) (h : ∀ ω, E ω → A ω ∨ B ω) : pr E ≤ pr A + pr B := by
  classical
  simp only [pr]
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have : univ.filter E ⊆ univ.filter A ∪ univ.filter B := by
    intro ω
    simp only [mem_filter, mem_univ, true_and, mem_union]
    exact h ω
  exact_mod_cast (card_le_card this).trans (card_union_le _ _)

omit [DecidableEq Ω] in
theorem pr_bad (op : Ω → P → Option V) (vstar : P → V) :
    pr (fun ω => ω ∈ bad op vstar) = (#(bad op vstar) : ℝ) / Fintype.card Ω := by
  classical
  simp only [pr]
  congr 3
  ext ω
  simp

omit [DecidableEq Ω] in
theorem pr₂_conflict (op : Ω → P → Option V) :
    pr₂ (fun x => ∃ p, ConflictAt op x p) = (#(conflict op) : ℝ) / (Fintype.card Ω) ^ 2 := by
  classical
  simp only [pr₂, conflict]
  congr 3
  ext x
  simp

/-- **Compilation, one table committed before any coin.** -/
theorem compile_step [Nonempty Ω] (op : Ω → P → Option V) (vstar : P → V)
    (hv : IsPlurality op vstar) (Q : ℕ) (hQ : ∀ ω, #(univ.filter fun p => (op ω p).isSome) ≤ Q)
    (acceptC acceptI : Ω → Prop)
    (couple : ∀ ω, acceptC ω → (∀ p v, op ω p = some v → v = vstar p) → acceptI ω) :
    pr acceptC ≤ pr acceptI + Real.sqrt (Fintype.card P * Q * pr₂ fun x => ∃ p, ConflictAt op x p) := by
  have hcover : ∀ ω, acceptC ω → acceptI ω ∨ ω ∈ bad op vstar := by
    intro ω hω
    by_cases hb : ω ∈ bad op vstar
    · exact Or.inr hb
    · refine Or.inl (couple ω hω fun p v hp => ?_)
      simp only [bad, Off, mem_filter, mem_univ, true_and, not_exists, not_and, not_not] at hb
      have := hb p (by simp [hp])
      rw [hp] at this
      exact Option.some_injective _ this
  refine (pr_le_add _ _ _ hcover).trans (add_le_add le_rfl ?_)
  rw [pr_bad, pr₂_conflict]
  exact prBad_le vstar hv Q hQ

/-- Jensen for `√` over a uniform average: `avg √x ≤ √(avg x)`. -/
theorem avg_sqrt_le {ι : Type} [Fintype ι] [Nonempty ι] (x : ι → ℝ) (hx : ∀ i, 0 ≤ x i) :
    (∑ i, Real.sqrt (x i)) / Fintype.card ι ≤ Real.sqrt ((∑ i, x i) / Fintype.card ι) := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hcs : (∑ i, Real.sqrt (x i)) ^ 2 ≤ Fintype.card ι * ∑ i, x i := by
    have := sq_sum_le_card_mul_sum_sq (s := (univ : Finset ι)) (f := fun i => Real.sqrt (x i))
    simpa [Real.sq_sqrt (hx _)] using this
  apply Real.le_sqrt_of_sq_le
  rw [div_pow, sq (Fintype.card ι : ℝ), div_le_div_iff₀ (by positivity) hn]
  nlinarith [hcs]

/-- **Compilation, a table committed after a prefix of coins.** Runs are (prefix, suffix); the
plurality table depends on the prefix. The bad-run probability is at most `√(N · Q · c)`, where
`c` is the probability that two runs sharing a random prefix conflict. -/
theorem compile_step_prefix {Ω₁ : Type} [Fintype Ω₁] [Nonempty Ω₁] [Nonempty Ω]
    (op : Ω₁ → Ω → P → Option V) (vstar : Ω₁ → P → V) (hv : ∀ ω₁, IsPlurality (op ω₁) (vstar ω₁))
    (Q : ℕ) (hQ : ∀ ω₁ ω, #(univ.filter fun p => (op ω₁ ω p).isSome) ≤ Q) :
    (∑ ω₁, pr fun ω => ω ∈ bad (op ω₁) (vstar ω₁)) / Fintype.card Ω₁ ≤
      Real.sqrt (Fintype.card P * Q *
        ((∑ ω₁, pr₂ fun x => ∃ p, ConflictAt (op ω₁) x p) / Fintype.card Ω₁)) := by
  set c : Ω₁ → ℝ := fun ω₁ => pr₂ fun x => ∃ p, ConflictAt (op ω₁) x p
  have hc0 : ∀ ω₁, 0 ≤ c ω₁ := fun ω₁ => by
    simp only [c, pr₂]
    positivity
  calc (∑ ω₁, pr fun ω => ω ∈ bad (op ω₁) (vstar ω₁)) / Fintype.card Ω₁
      ≤ (∑ ω₁, Real.sqrt (Fintype.card P * Q * c ω₁)) / Fintype.card Ω₁ := by
        refine div_le_div_of_nonneg_right (sum_le_sum fun ω₁ _ => ?_) (Nat.cast_nonneg _)
        rw [pr_bad]
        have := prBad_le (vstar ω₁) (hv ω₁) Q (hQ ω₁)
        simpa [c, pr₂_conflict] using this
    _ ≤ Real.sqrt ((∑ ω₁, Fintype.card P * Q * c ω₁) / Fintype.card Ω₁) :=
        avg_sqrt_le _ fun ω₁ => mul_nonneg (by positivity) (hc0 ω₁)
    _ = Real.sqrt (Fintype.card P * Q * ((∑ ω₁, c ω₁) / Fintype.card Ω₁)) := by
        rw [← mul_sum, mul_div_assoc]

/-- Every conflicting pair of verified openings against one cap is a collision of the hash. -/
theorem conflict_collision {Col D : Type} (H : List UInt8 → D) (E : Merkle.Enc Col D)
    (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (sibs sibs' : List D) (hc : c ≠ c')
    (h : Merkle.Verifies H E cap depth pos c sibs) (h' : Merkle.Verifies H E cap depth pos c' sibs') :
    Merkle.Collision H :=
  Merkle.opening_binding H E cap depth pos c c' sibs sibs' hc h h'

end FlockSoundness.Compiled
