import Models.Accounting.Guarantees.Localization
import Proofs.Accounting.Actual.Split

/-!
# Localization, proved (`e = 0`)

Under causality a gate downstream of `C` runs at or after `C`'s ticks (`le_t_of_downstream`), so one that `A[Y, W]` reads
runs during `W`; and one not in `C` reads a gate downstream of `C`, so it is computed (`downstream_pred`). Placed in `Y`
it would be in `A[Y, W]`, and placed at the verifier's `Reaches` refuses it, so it is in `A[X, W]`. When `X` sends `Y`
nothing of `C`, a gate of `A[Y, W]` downstream of `C` reads one, which is then in `A[Y, W]` and earlier (the gates are in
topological order), so none is.
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

end Actual

open Actual in
theorem Localization : Guarantees.Localization := by
  intro L A ℓ C X Y t₀ t₁ hc h₀ hR
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

end Accounting.SecurityProofs
