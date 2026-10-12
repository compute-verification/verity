import Models.Accounting.Guarantees.Localization
import Definitions.Accounting.Localization
import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Confidence

/-!
# Localization, proved

`Localization`, over strategies, is function composition. What the rest reads from outside before `u ≤ t₁`
(`crossing_split`) is what it read before `t₀` (part of what it can tell at `t₀`), what `X` sent it during `[t₀, u)`,
and what it heard from the leaves in neither during `W`, cut at `u`. With its inputs fixed, causality's global form
(`global_of_admissible`) makes what the rest does before `u` a function of what `X` sent. The table picks a run for
each key, and the count is the image's. `memory_of_memoryOver`: memory over continuations bounds each run's state.

`localization_wires` is the per-run lemma (not the model's `e = 0`, which reads the strategy): under causality a gate
downstream of `C` runs at or after `C`'s ticks (`le_t_of_downstream`), so one that `A[Y, W]` reads runs during `W`; and
one not in `C` reads a gate downstream of `C`, so it is computed (`downstream_pred`). Placed in `Y` it would be in
`A[Y, W]`, and placed at the verifier's `Reaches` refuses it, so it is in `A[X, W]`. When no wire from `X` to `Y`
carries a gate downstream of `C`, a gate of `A[Y, W]` downstream of `C` reads one, which is then in `A[Y, W]` and
earlier (the gates are in topological order), so none is.
-/

namespace Accounting.SecurityProofs

open Finset

namespace Actual

variable {L : Type*} (A : Placed L)

theorem mem_downstream {C : Finset (Fin A.N)} {g : Fin A.N} :
    g ∈ A.downstream C ↔ ∃ c ∈ C, Relation.ReflTransGen (fun a g => a ∈ A.reads g) c g := by
  simp [Placed.downstream]

/-- A gate downstream of `C` that is not in `C` is computed and reads a gate downstream of `C`. -/
theorem downstream_pred {C : Finset (Fin A.N)} {g : Fin A.N} (hg : g ∈ A.downstream C) (hC : g ∉ C) :
    A.isInput g = false ∧ ∃ a ∈ A.reads g, a ∈ A.downstream C := by
  obtain ⟨c, hc, h⟩ := (mem_downstream A).1 hg
  rcases h.cases_tail with rfl | ⟨a, ha, hag⟩
  · exact absurd hc hC
  refine ⟨?_, a, hag, (mem_downstream A).2 ⟨c, hc, ha⟩⟩
  cases hi : A.isInput g
  · rfl
  · simp [A.input_reads g hi] at hag

/-- Under causality a gate downstream of `C` runs no earlier than a tick `t₀` at most each of `C`'s. -/
theorem le_t_of_downstream {ℓ : L → L → ℕ} (hc : A.Causal ℓ) {C : Finset (Fin A.N)} {t₀ : ℕ}
    (h₀ : ∀ c ∈ C, t₀ ≤ A.t c) {g : Fin A.N} (hg : g ∈ A.downstream C) : t₀ ≤ A.t g := by
  obtain ⟨c, hcC, h⟩ := (mem_downstream A).1 hg
  clear hg
  induction h with
  | refl => exact h₀ c hcC
  | tail _ hab ih => exact ih.trans ((Nat.le_add_right _ _).trans (hc _ _ hab))

/-- **Localization on one circuit's wires.** Under causality, for a set `C` of gates, a location `X` it reaches the
prover through during `W = [t₀, t₁)` (`t₀` at most each tick of `C`) and the rest `Y` of the prover: every gate
downstream of `C` that `A[Y, W]` reads from outside itself is in `A[X, W]`; and when no gate of `C` is placed in `Y` and
no wire from `X` to `Y` during `W` carries a gate downstream of `C`, no gate of `A[Y, W]` is downstream of `C`, nor is
any gate it reads from outside itself, nor any gate placed in `Y` that `A[X, W]` reads. -/
theorem localization_wires {ℓ : L → L → ℕ} (hc : A.Causal ℓ) {C : Finset (Fin A.N)} {X Y : Finset L} {t₀ t₁ : ℕ}
    (h₀ : ∀ c ∈ C, t₀ ≤ A.t c) (hR : A.Reaches C X Y (Set.Ico t₀ t₁)) :
    (∀ a ∈ A.into (A.region Y (Set.Ico t₀ t₁)), a ∈ A.downstream C → a ∈ A.region X (Set.Ico t₀ t₁)) ∧
    ((∀ c ∈ C, A.x c ∉ Y) → A.SendsIndependent C X Y (Set.Ico t₀ t₁) →
      Disjoint (A.region Y (Set.Ico t₀ t₁)) (A.downstream C) ∧
      Disjoint (A.into (A.region Y (Set.Ico t₀ t₁))) (A.downstream C) ∧
      ∀ g ∈ A.region X (Set.Ico t₀ t₁), ∀ a ∈ A.reads g, A.x a ∈ Y → a ∉ A.downstream C) := by
  have egress : ∀ a ∈ A.into (A.region Y (Set.Ico t₀ t₁)), a ∈ A.downstream C → a ∈ A.region X (Set.Ico t₀ t₁) := by
    intro a ha had
    obtain ⟨haY, g, hg, hag⟩ := (mem_into A).1 ha
    obtain ⟨haC, hver⟩ := hR g hg a hag
    have hai := (downstream_pred A had haC).1
    have hat : A.t a ∈ Set.Ico t₀ t₁ :=
      ⟨le_t_of_downstream A hc h₀ had,
        ((Nat.le_add_right _ _).trans (hc g a hag)).trans_lt ((mem_region A).1 hg).2.2.2⟩
    by_cases hx : A.x a ∈ X
    · exact (mem_region A).2 ⟨hai, hx, hat⟩
    by_cases hy : A.x a ∈ Y
    · exact absurd ((mem_region A).2 ⟨hai, hy, hat⟩) haY
    · exact absurd had (hver hx hy)
  refine ⟨egress, fun hCY hS => ?_⟩
  have region : ∀ g ∈ A.region Y (Set.Ico t₀ t₁), g ∉ A.downstream C := by
    intro g
    induction g using WellFoundedLT.induction with
    | _ g ih =>
      intro hg hgd
      obtain ⟨a, hag, had⟩ := (downstream_pred A hgd fun hgC => hCY g hgC ((mem_region A).1 hg).2.1).2
      by_cases ha : a ∈ A.region Y (Set.Ico t₀ t₁)
      · exact ih a (A.reads_lt g a hag) ha had
      · exact hS g hg a hag ((mem_region A).1 (egress a ((mem_into A).2 ⟨ha, g, hg, hag⟩) had)).2.1 had
  refine ⟨disjoint_left.2 region, disjoint_left.2 fun a ha had => ?_, fun g hg a hag hy had => ?_⟩
  · obtain ⟨-, g, hg, hag⟩ := (mem_into A).1 ha
    exact hS g hg a hag ((mem_region A).1 (egress a ha had)).2.1 had
  · have hai := (downstream_pred A had fun haC => hCY a haC hy).1
    refine region a ((mem_region A).2 ⟨hai, hy, le_t_of_downstream A hc h₀ had, ?_⟩) had
    exact ((Nat.le_add_right _ _).trans (hc g a hag)).trans_lt ((mem_region A).1 hg).2.2.2

/-- What `Y` reads from outside before `u ≤ t₁`, for `X` apart from `Y`: what it read before `t₀` (cut at `u`), what it
read from `X` during `[t₀, u)`, and what it read from the leaves in neither during `[t₀, t₁)` (cut at `u`). -/
theorem crossing_split {L : Type*} (R : Actual.Run L) {X Y : Finset L} (h : Disjoint X Y) {t₀ u t₁ : ℕ}
    (hu : u ≤ t₁) : R.crossing (↑Y)ᶜ ↑Y (Set.Iio u) = R.crossing (↑Y)ᶜ ↑Y (Set.Iio t₀) ∩ {p | p.1 < u} ∪
      (R.crossing ↑X ↑Y (Set.Ico t₀ u) ∪ R.crossing (↑X ∪ ↑Y)ᶜ ↑Y (Set.Ico t₀ t₁) ∩ {p | p.1 < u}) := by
  ext p
  simp only [Actual.Run.crossing, Set.mem_union, Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_Iio, Set.mem_Ico]
  constructor
  · rintro ⟨g, a, hr, ha, hg, ht, rfl⟩
    by_cases h0 : R.A.t g < t₀
    · exact Or.inl ⟨⟨g, a, hr, ha, hg, h0, rfl⟩, ht⟩
    by_cases hx : R.A.x a ∈ (↑X : Set L)
    · exact Or.inr (Or.inl ⟨g, a, hr, hx, hg, ⟨not_lt.1 h0, ht⟩, rfl⟩)
    · exact Or.inr (Or.inr ⟨⟨g, a, hr, fun hm => hm.elim hx ha, hg, ⟨not_lt.1 h0, ht.trans_le hu⟩, rfl⟩, ht⟩)
  · rintro (⟨⟨g, a, hr, ha, hg, h0, rfl⟩, ht⟩ | ⟨g, a, hr, ha, hg, ht, rfl⟩ | ⟨⟨g, a, hr, ha, hg, ht, rfl⟩, hlt⟩)
    · exact ⟨g, a, hr, ha, hg, ht, rfl⟩
    · exact ⟨g, a, hr, fun hy => Finset.disjoint_left.1 h ha hy, hg, ht.2, rfl⟩
    · exact ⟨g, a, hr, fun hy => ha (Or.inr hy), hg, hlt, rfl⟩

/-- Memory over continuations bounds each run's state: the per-run state is in the state over continuations. -/
theorem memory_of_memoryOver {Ω L : Type*} {S : Ω → Actual.Run L} {𝒳 : Set (Finset L)} {M : Finset L → ℕ}
    (h : MemoryOver S 𝒳 M) (r : Ω) : (S r).A.Memory 𝒳 M := by
  classical
  intro X hX t
  refine le_trans ?_ (h r X hX t)
  unfold Placed.state Verity.Circuit.width
  refine Finset.sum_le_sum_of_subset fun a ha => ?_
  obtain ⟨hai, hx⟩ := Finset.mem_filter.1 ha
  simp only [stateOver, Finset.mem_filter, Finset.mem_univ, true_and]
  exact ⟨hx, ((mem_into _).1 hai).1, r, rfl, a, hai, rfl⟩

end Actual

theorem Localization : Guarantees.Localization := by
  intro L Ω S X Y t₀ t₁ hXY hA hh hv
  have hin : ∀ r r', (S r).inputs ↑Y = (S r').inputs ↑Y := fun r r' => congrArg Prod.fst (hh r r')
  have hpre : ∀ r r', (S r).crossing (↑Y)ᶜ ↑Y (Set.Iio t₀) = (S r').crossing (↑Y)ᶜ ↑Y (Set.Iio t₀) :=
    fun r r' => congrArg (fun p => p.2.2) (hh r r')
  have key : ∀ u ≤ t₁, ∀ r r', (S r).crossing ↑X ↑Y (Set.Ico t₀ u) = (S r').crossing ↑X ↑Y (Set.Ico t₀ u) →
      (S r).acts ↑Y (Set.Iio u) = (S r').acts ↑Y (Set.Iio u) := by
    intro u hu r r' hx
    refine Actual.global_of_admissible hA Y hin u r r' ?_
    rw [Actual.crossing_split (S r) hXY hu, Actual.crossing_split (S r') hXY hu, hpre r r', hx, hv r r']
  have table : ∀ u ≤ t₁, ∃ T : Set (ℕ × L × ℕ × L × ℕ × ℕ) → Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ),
      ∀ r, (S r).acts ↑Y (Set.Iio u) = T ((S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := by
    intro u hu
    have hf : Function.FactorsThrough (fun r => (S r).acts ↑Y (Set.Iio u))
        (fun r => (S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := fun r r' h => key u hu r r' h
    exact ⟨Function.extend _ _ fun _ => (∅, ∅), fun r => (hf.extend_apply _ r).symm⟩
  refine ⟨table, fun e he => ?_⟩
  obtain ⟨T, hT⟩ := table t₁ le_rfl
  refine le_trans (Set.encard_le_encard ?_) ((Set.encard_image_le T _).trans he)
  rintro _ ⟨r, rfl⟩
  exact ⟨_, ⟨r, rfl⟩, (hT r).symm⟩

end Accounting.SecurityProofs
