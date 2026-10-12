import Models.Accounting.Guarantees.Localization
import Definitions.Accounting.Localization
import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split

/-!
# Localization, proved

`Localization`, over strategies, is function composition. Outside `Y` is `X` and the leaves in neither
(`compl_eq_union`), so what the rest reads from outside during `[t₀, u)` is what `X` sent it then and what the rest
heard from elsewhere, the window's restriction to `u` (`crossing_union`, `crossing_window`); with the state fixed,
causality from `t₀` makes what the rest does then a function of what `X` sent. The table picks a run for each key, and
the count is the image's. `memory_of_memoryOver`: memory over continuations bounds each run's state.

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

theorem crossing_union {L : Type*} (R : Actual.Run L) (P Q Y : Set L) (I : Set ℕ) :
    R.crossing (P ∪ Q) Y I = R.crossing P Y I ∪ R.crossing Q Y I := by
  ext p
  simp only [Actual.Run.crossing, Set.mem_union, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨g, a, h, hP | hQ, hY, hI, rfl⟩
    exacts [Or.inl ⟨g, a, h, hP, hY, hI, rfl⟩, Or.inr ⟨g, a, h, hQ, hY, hI, rfl⟩]
  · rintro (⟨g, a, h, hP, hY, hI, rfl⟩ | ⟨g, a, h, hQ, hY, hI, rfl⟩)
    exacts [⟨g, a, h, Or.inl hP, hY, hI, rfl⟩, ⟨g, a, h, Or.inr hQ, hY, hI, rfl⟩]

theorem crossing_window {L : Type*} (R : Actual.Run L) (P Y : Set L) {t₀ u t₁ : ℕ} (hu : u ≤ t₁) :
    R.crossing P Y (Set.Ico t₀ u) = R.crossing P Y (Set.Ico t₀ t₁) ∩ {p | p.1 < u} := by
  ext p
  simp only [Actual.Run.crossing, Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_Ico]
  constructor
  · rintro ⟨g, a, h, hP, hY, hI, rfl⟩
    exact ⟨⟨g, a, h, hP, hY, ⟨hI.1, hI.2.trans_le hu⟩, rfl⟩, hI.2⟩
  · rintro ⟨⟨g, a, h, hP, hY, hI, rfl⟩, hlt⟩
    exact ⟨g, a, h, hP, hY, ⟨hI.1, hlt⟩, rfl⟩

theorem compl_eq_union {L : Type*} {X Y : Finset L} (h : Disjoint X Y) : (↑Y : Set L)ᶜ = ↑X ∪ (↑X ∪ ↑Y)ᶜ := by
  ext l
  by_cases hx : l ∈ X
  · simp [hx, Finset.disjoint_left.1 h hx]
  · simp [hx]

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
  intro L Ω S X Y t₀ t₁ hXY hc hs hv
  have key : ∀ u ≤ t₁, ∀ r r', (S r).crossing ↑X ↑Y (Set.Ico t₀ u) = (S r').crossing ↑X ↑Y (Set.Ico t₀ u) →
      (S r).acts ↑Y (Set.Ico t₀ u) = (S r').acts ↑Y (Set.Ico t₀ u) := by
    intro u hu r r' hx
    refine hc Y t₀ u r r' (hs r r') ?_
    rw [Actual.compl_eq_union hXY, Actual.crossing_union, Actual.crossing_union, hx,
      Actual.crossing_window (S r) ((↑X ∪ ↑Y)ᶜ) _ hu,
      Actual.crossing_window (S r') ((↑X ∪ ↑Y)ᶜ) _ hu, hv r r']
  have table : ∀ u ≤ t₁, ∃ T : Set (ℕ × L × ℕ × L × ℕ × ℕ) → Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ),
      ∀ r, (S r).acts ↑Y (Set.Ico t₀ u) = T ((S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := by
    intro u hu
    have hf : Function.FactorsThrough (fun r => (S r).acts ↑Y (Set.Ico t₀ u))
        (fun r => (S r).crossing ↑X ↑Y (Set.Ico t₀ u)) := fun r r' h => key u hu r r' h
    exact ⟨Function.extend _ _ fun _ => (∅, ∅), fun r => (hf.extend_apply _ r).symm⟩
  refine ⟨table, fun e he => ?_⟩
  obtain ⟨T, hT⟩ := table t₁ le_rfl
  refine le_trans (Set.encard_le_encard ?_) ((Set.encard_image_le T _).trans he)
  rintro _ ⟨r, rfl⟩
  exact ⟨_, ⟨r, rfl⟩, (hT r).symm⟩

end Accounting.SecurityProofs
