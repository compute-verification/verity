import Definitions.Pouw.NCP.Word
import Proofs.Pouw.Copy

/-!
# R_word-NCP: the bound under NzNCP, the floor, and the witnesses
-/

namespace Pouw.NCP.WordProofs

open Finset Pouw.NCP.Word Pouw.Proofs

variable {Ω : Type} (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Word.Adversary)

open Classical in
/-- Unit `u`'s fresh positions. -/
noncomputable def freshSet (u : ℕ) : Finset ℕ := (range (n u)).filter (fun p => A.rule u p = .fresh)

open Classical in
/-- Unit `u`'s positions whose rule is neither fresh nor an identity. -/
noncomputable def openSet (u : ℕ) : Finset ℕ :=
  (range (n u)).filter (fun p => A.rule u p ≠ .fresh ∧ ¬ ∀ ω, X ω u p = (A.rule u p).target (X ω u))

theorem count (u : ℕ) : (freshSet n A u).card + identCount n X A u + (openSet n X A u).card = n u := by
  unfold freshSet identCount openSet
  rw [card_filter, card_filter, card_filter, ← sum_add_distrib, ← sum_add_distrib]
  have h1 : ∑ _p ∈ range (n u), (1 : ℕ) = n u := by simp
  refine Eq.trans ?_ h1
  refine sum_congr rfl fun p _ => ?_
  unfold Identity
  split_ifs <;> first | rfl | (exfalso; tauto)

theorem fresh_le (u : ℕ) : 16 * (freshSet n A u).card ≤ unitCost n A u := by
  have h1 : (freshSet n A u).card • 16 ≤ ∑ p ∈ freshSet n A u, (A.rule u p).cost :=
    card_nsmul_le_sum _ _ _ fun p hp => by
      unfold freshSet at hp
      rw [(mem_filter.1 hp).2]
      exact le_rfl
  have h2 : ∑ p ∈ freshSet n A u, (A.rule u p).cost ≤ unitCost n A u :=
    sum_le_sum_of_subset (filter_subset _ _)
  rw [smul_eq_mul] at h1
  omega

open Classical in
theorem mem_openSet {u p : ℕ} : p ∈ openSet n X A u ↔
    p ∈ range (n u) ∧ (A.rule u p ≠ .fresh ∧ ¬ ∀ ω, X ω u p = (A.rule u p).target (X ω u)) := by
  unfold openSet
  exact mem_filter

theorem openSet_sub (u : ℕ) : openSet n X A u ⊆ range (n u) := fun _ hp => ((mem_openSet n X A).1 hp).1

theorem unit_le [Fintype Ω] (hV : Valid n A) (hNz : NzNCP n X) (u : ℕ) :
    pr (fun ω => Word.Correct n X A ω u) ≤ (1 / 2 : ℝ) ^ (openSet n X A u).card := by
  refine (pr_mono fun ω hc p hp => ?_).trans (hNz u _ (openSet_sub n X A u) (A.rule u) fun p hp => ?_)
  · obtain ⟨hp1, hp2⟩ := (mem_openSet n X A).1 hp
    rcases hc p (mem_range.1 hp1) with h | h
    · exact absurd h hp2.1
    · exact h
  · obtain ⟨hp1, hp2⟩ := (mem_openSet n X A).1 hp
    exact ⟨hp2.1, hV u p (mem_range.1 hp1), hp2.2⟩

/-- **R_word-NCP's bound.** -/
theorem wordBound : WordBound := by
  intro Ω _ n X N A γ₀ hγ0 hγ1 hV hNz
  classical
  have h1γ : 0 < 1 - γ₀ := by linarith
  let few := (range N).filter (fun u => (unitCost n A u : ℝ) < (1 - γ₀) * (16 * n u))
  have hsub : ∀ ω, (cost n N A : ℝ) / (1 - γ₀) < 16 * ∑ u ∈ Word.correctSet n X N A ω, (n u : ℝ) →
      ∃ u ∈ few, Word.Correct n X A ω u := by
    intro ω h
    by_contra hne
    push Not at hne
    refine absurd h (not_lt.2 ?_)
    have hK : ∀ u ∈ Word.correctSet n X N A ω, 16 * (n u : ℝ) ≤ (unitCost n A u : ℝ) / (1 - γ₀) := by
      intro u hu
      have hu' := mem_filter.1 hu
      have : u ∉ few := fun hf => hne u hf hu'.2
      have hle : (1 - γ₀) * (16 * n u) ≤ unitCost n A u := by
        by_contra hlt
        exact this (mem_filter.2 ⟨hu'.1, not_le.1 hlt⟩)
      rw [le_div_iff₀ h1γ]
      linarith
    calc 16 * ∑ u ∈ Word.correctSet n X N A ω, (n u : ℝ)
        = ∑ u ∈ Word.correctSet n X N A ω, 16 * (n u : ℝ) := by rw [mul_sum]
      _ ≤ ∑ u ∈ Word.correctSet n X N A ω, (unitCost n A u : ℝ) / (1 - γ₀) := sum_le_sum hK
      _ ≤ ∑ u ∈ range N, (unitCost n A u : ℝ) / (1 - γ₀) :=
          sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun _ _ _ => by positivity
      _ = (cost n N A : ℝ) / (1 - γ₀) := by
          unfold cost
          push_cast
          simp only [div_eq_mul_inv]
          rw [← sum_mul]
  calc _ ≤ pr (fun ω => ∃ u ∈ few, Word.Correct n X A ω u) := pr_mono hsub
    _ ≤ ∑ u ∈ few, pr (fun ω => Word.Correct n X A ω u) := pr_exists_le_sum _ _
    _ ≤ ∑ u ∈ few, (2 : ℝ) ^ (-(γ₀ * n u - identCount n X A u)) := by
        refine sum_le_sum fun u hu => (unit_le n X A hV hNz u).trans (RCopy.half_pow_le_rpow _ _ ?_)
        have hlt := (mem_filter.1 hu).2
        have hc : ((freshSet n A u).card : ℝ) + identCount n X A u + (openSet n X A u).card = n u := by
          exact_mod_cast count n X A u
        have hf : 16 * ((freshSet n A u).card : ℝ) ≤ unitCost n A u := by exact_mod_cast fresh_le n A u
        linarith
    _ ≤ ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * n u - identCount n X A u)) :=
        sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun _ _ _ => by positivity

/-- **The floor.** -/
theorem wordFloor : WordFloor := by
  intro Ω _ n X N A γ₀ w c hγ0 hγ1 hV hNz hn hF
  refine (wordBound Ω n X N A γ₀ hγ0 hγ1 hV hNz).trans ?_
  calc ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * n u - identCount n X A u))
      ≤ ∑ _u ∈ range N, (2 : ℝ) ^ (-(c * (γ₀ * w - 1))) := sum_le_sum fun u _ => by
        apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
        have h1 : (identCount n X A u : ℝ) ≤ c := by exact_mod_cast hF u
        rw [hn u]
        push_cast
        linarith
    _ = N * (2 : ℝ) ^ (-(c * (γ₀ * w - 1))) := by rw [sum_const, card_range, nsmul_eq_mul]

theorem target_congr (r : Word.Rule) (p : ℕ) (h : r.Reads p) (x y : ℕ → ℤ) (hxy : ∀ q < p, x q = y q) :
    r.target x = r.target y := by
  cases r with
  | fresh => rfl
  | copy s a => simp only [Rule.target, hxy a h]
  | sum s a t b => simp only [Rule.target, hxy a h.1, hxy b h.2]
  | const c => rfl

theorem bits_eq {M : ℕ} (ω : Fin M → Bool) (p : ℕ) : bits ω p = (Pouw.Copy.bits ω p : ℤ) := by
  unfold bits Pouw.Copy.bits
  split <;> simp

theorem bits_flip_ne {M : ℕ} (a : Fin M) (ω : Fin M → Bool) (q : ℕ) (hq : q ≠ a) :
    bits (RCopy.flip a ω) q = bits ω q := by
  rw [bits_eq, bits_eq, RCopy.bits_flip_ne a ω q hq]

theorem bits_flip_self {M : ℕ} (a : Fin M) (ω : Fin M → Bool) : bits (RCopy.flip a ω) a ≠ bits ω a := by
  rw [bits_eq, bits_eq]
  exact_mod_cast RCopy.bits_flip_self a ω

/-- Independent uniform bits satisfy NzNCP: flipping the largest constrained bit leaves every target unchanged (they
read earlier positions), so it maps the event for `P` into the event for `P` without that bit, and out of the event
for `P`. -/
theorem bits_nz (M : ℕ) (P : Finset ℕ) :
    P ⊆ range M → ∀ r : ℕ → Word.Rule, (∀ p ∈ P, (r p).Reads p) →
      pr (fun ω : Fin M → Bool => ∀ p ∈ P, bits ω p = (r p).target (bits ω)) ≤ (1 / 2 : ℝ) ^ P.card := by
  classical
  refine Finset.induction_on_max P ?_ ?_
  · intro _ _ _
    simpa using pr_le_one _
  · intro a s hmax ih hP r hr
    have ha : a ∉ s := fun h => lt_irrefl a (hmax a h)
    have haM : a < M := mem_range.1 (hP (mem_insert_self a s))
    let σ := RCopy.flip (⟨a, haM⟩ : Fin M)
    let E := fun ω : Fin M → Bool => ∀ p ∈ insert a s, bits ω p = (r p).target (bits ω)
    let E' := fun ω : Fin M → Bool => ∀ p ∈ s, bits ω p = (r p).target (bits ω)
    have htgt : ∀ ω, ∀ p ∈ insert a s, (r p).target (bits (σ ω)) = (r p).target (bits ω) := fun ω p hp =>
      target_congr (r p) p (hr p hp) _ _ fun q hq => bits_flip_ne _ ω q fun e => by
        have : p ≤ a := by
          rcases mem_insert.1 hp with rfl | hp
          · exact le_rfl
          · exact (hmax p hp).le
        simp only at e
        omega
    have hIH : pr E' ≤ (1 / 2 : ℝ) ^ s.card :=
      ih (fun p hp => hP (mem_insert_of_mem hp)) r fun p hp => hr p (mem_insert_of_mem hp)
    have hcover : ∀ ω, E ω ∨ E (σ ω) → E' ω := by
      rintro ω (h | h) p hp
      · exact h p (mem_insert_of_mem hp)
      · have hpa : p ≠ a := fun e => ha (e ▸ hp)
        rw [← bits_flip_ne (⟨a, haM⟩ : Fin M) ω p hpa, h p (mem_insert_of_mem hp)]
        exact htgt ω p (mem_insert_of_mem hp)
    have hdisj : ∀ ω, ¬ (E ω ∧ E (σ ω)) := by
      rintro ω ⟨h, h'⟩
      apply bits_flip_self (⟨a, haM⟩ : Fin M) ω
      show bits (σ ω) a = bits ω a
      rw [h a (mem_insert_self a s), h' a (mem_insert_self a s)]
      exact htgt ω a (mem_insert_self a s)
    have h2 : 2 * pr E ≤ pr E' := by
      have := pr_mono (P := fun ω => E ω ∨ E (σ ω)) hcover
      rw [pr_or_disjoint _ _ hdisj, pr_equiv σ E] at this
      linarith
    rw [card_insert_of_notMem ha, pow_succ]
    linarith

/-- **NzNCP is satisfiable.** -/
theorem wordNzWitness : WordNzWitness := fun M _ P hP r hr =>
  bits_nz M P hP r fun p hp => (hr p hp).2.1

/-- **FinalOnly is satisfiable.** -/
theorem wordFinalWitness : WordFinalWitness := by
  intro M A hV u
  classical
  show identCount _ _ _ _ ≤ 0
  unfold identCount
  refine (card_eq_zero.2 (filter_eq_empty_iff.2 fun p hp ⟨_, hall0⟩ => ?_)).le
  have hall : ∀ ω : Fin M → Bool, bits ω p = (A.rule u p).target (bits ω) := hall0
  have hpM : p < M := mem_range.1 hp
  let a : Fin M := ⟨p, hpM⟩
  let ω₀ : Fin M → Bool := fun _ => false
  apply bits_flip_self a ω₀
  show bits (RCopy.flip a ω₀) p = bits ω₀ p
  rw [hall (RCopy.flip a ω₀), hall ω₀]
  exact target_congr _ p (hV u p hpM) _ _ fun q hq => bits_flip_ne a ω₀ q (by simp only [a]; omega)

/-- **Zero noise.** -/
theorem wordZeroNoise : WordZeroNoise := by
  intro Ω _ _ n x N γ₀ hγ1 ⟨u0, hu0, hn0⟩
  classical
  have hc : cost n N (constAdv x) = 0 := by simp [cost, unitCost, constAdv, Rule.cost]
  refine ⟨?_, hc, ?_, ?_⟩
  · intro u P _ r hr
    rcases P.eq_empty_or_nonempty with rfl | ⟨p, hp⟩
    · simpa using pr_le_one _
    · rw [pr_eq_zero fun ω h => (hr p hp).2.2 fun _ => h p hp]
      positivity
  · intro u
    unfold identCount
    rw [filter_true_of_mem fun p _ => ⟨fun h => (by simp [constAdv] at h), fun _ => rfl⟩, card_range]
  · apply pr_eq_one
    intro ω
    have hs : Word.correctSet n (fun _ => x) N (constAdv x) ω = range N :=
      filter_true_of_mem fun u _ p _ => Or.inr rfl
    rw [hc, hs]
    have : 0 < ∑ u ∈ range N, (n u : ℝ) :=
      sum_pos' (fun _ _ => by positivity) ⟨u0, mem_range.2 hu0, by exact_mod_cast hn0⟩
    simp only [Nat.cast_zero, zero_div]
    linarith

end Pouw.NCP.WordProofs
