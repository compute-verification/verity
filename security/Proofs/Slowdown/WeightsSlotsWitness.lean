module

public import Proofs.Slowdown.FitSlotsWitness
public import Definitions.Slowdown.Assumptions.Threads
import Mathlib.Data.NNRat.Floor
import Proofs.Slowdown.Threads

/-! A subcircuit across an optimizer step, where `WeightsIncompressibleSlots` holds and `WeightsIncompressible`
fails. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs.WeightsSlotsWitness

open Verity Finset Slowdown.SecurityProofs.FitSlotsWitness

theorem span :
    (threading.weightsUsed ({4, 6, 8} : Finset (Fin 10))).card = 2 ∧
      (slots.slotsUsed ({4, 6, 8} : Finset (Fin 10))).card = 1 := by
  decide +kernel

theorem weightsIncompressibleSlots (Γ : ℕ) (S : ℝ) :
    Assumptions.WeightsIncompressibleSlots decl threading slots Γ S 4 1 := by
  have h : ∀ R ∈ decl.D.computed.powerset,
      4 * (slots.slotsUsed R).card + 1 * ∑ t, (threading.threadReads R t).card ≤ freshInput decl R := by
    decide +kernel
  intro R hR
  exact_mod_cast h R (mem_powerset.2 hR.1)

theorem not_weightsIncompressible (Γ : ℕ) (S : ℝ) (hΓ : 3 ≤ Γ) (hS : 6 ≤ S) :
    ¬ Assumptions.WeightsIncompressible decl threading Γ S 4 1 := by
  intro h
  have hsub : ({4, 6, 8} : Finset (Fin 10)) ⊆ decl.D.computed := by decide +kernel
  have hsize : decl.D.size ({4, 6, 8} : Finset (Fin 10)) = 3 := by decide +kernel
  have hr : freshInput decl ({4, 6, 8} : Finset (Fin 10)) = 6 := by decide +kernel
  have hrS : (freshInput decl ({4, 6, 8} : Finset (Fin 10)) : ℝ) ≤ S := by
    rw [hr]
    exact_mod_cast hS
  have h' := h ({4, 6, 8} : Finset (Fin 10)) ⟨hsub, (le_of_eq hsize).trans hΓ, hrS⟩
  have hn : 4 * (threading.weightsUsed ({4, 6, 8} : Finset (Fin 10))).card +
      1 * ∑ t, (threading.threadReads ({4, 6, 8} : Finset (Fin 10)) t).card ≤
        freshInput decl ({4, 6, 8} : Finset (Fin 10)) := by
    exact_mod_cast h'
  have hnot : ¬ (4 * (threading.weightsUsed ({4, 6, 8} : Finset (Fin 10))).card +
      1 * ∑ t, (threading.threadReads ({4, 6, 8} : Finset (Fin 10)) t).card ≤
        freshInput decl ({4, 6, 8} : Finset (Fin 10))) := by
    decide +kernel
  exact hnot hn

theorem threaded (Γ : ℕ) (S : ℝ) : threading.Threaded Γ S 1 0 :=
  threaded_one_zero threading Γ S (by decide +kernel) (by decide +kernel)

theorem reads (Γ : ℕ) (S : ℝ) (R : Finset (Fin 10)) (hR : Small decl Γ S R) :
    (1 : ℝ) * (threading.mults R).card ≤ freshInput decl R * threading.bound S 4 1 0 :=
  ThreadedReadsSlots decl threading slots Γ S 4 1 1 0 (by norm_num) (by norm_num) one_pos (threaded Γ S)
    (weightsIncompressibleSlots Γ S) R hR

end Slowdown.SecurityProofs.WeightsSlotsWitness
