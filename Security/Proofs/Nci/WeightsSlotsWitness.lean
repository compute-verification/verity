import Proofs.Nci.FitSlotsWitness
import Proofs.Nci.Threads

/-!
# A subcircuit across an optimizer step: `WeightsIncompressibleSlots` holds where `WeightsIncompressible` fails

`FitSlotsWitness`'s ten-gate circuit of two optimizer steps on two four-bit parameters, with its slot map (gates 0 and 6
are parameter 0's versions, 1 and 7 parameter 1's). The subcircuit `{4, 6, 8}` spans the update of parameter 0: it
multiplies the step-1 version 0 by activation 2, computes the step-2 version 6 from 0 and that product, and multiplies 6
by activation 3. It reads 0, 2 and 3, 6 bits, and uses two weight gates of one slot.

At `b_w = 4`, `b_a = 1`, `WeightsIncompressible` charges it `2 · 4 + 2 = 10 > 6` and fails at every `Γ ≥ 3`, `S ≥ 6`
(`not_weightsIncompressible`), while `WeightsIncompressibleSlots` charges it `4 + 2 = 6` and holds for every `Γ` and `S`
over all 64 subsets of the computed gates (`weightsIncompressibleSlots`). The circuit is `(1, 0)`-threaded (`threaded`),
so `ThreadedReadsSlots` applies here (`reads`).
-/

namespace Nci.SecurityProofs.WeightsSlotsWitness

open Verity Finset Nci.SecurityProofs.FitSlotsWitness

/-- `{4, 6, 8}` uses two weight gates, both versions of slot 0. -/
theorem span :
    (threading.weightsUsed ({4, 6, 8} : Finset (Fin 10))).card = 2 ∧
      (slots.slotsUsed ({4, 6, 8} : Finset (Fin 10))).card = 1 := by
  decide +kernel

theorem weightsIncompressibleSlots (Γ : ℕ) (S : ℝ) :
    Assumptions.WeightsIncompressibleSlots decl threading slots Γ S 4 1 := by
  have h : ∀ R ∈ decl.D.computed.powerset,
      4 * (slots.slotsUsed R).card + 1 * ∑ t, (threading.threadReads R t).card ≤ novelReads decl R := by
    decide +kernel
  intro R hR
  exact_mod_cast h R (mem_powerset.2 hR.1)

theorem not_weightsIncompressible (Γ : ℕ) (S : ℝ) (hΓ : 3 ≤ Γ) (hS : 6 ≤ S) :
    ¬ Assumptions.WeightsIncompressible decl threading Γ S 4 1 := by
  intro h
  have hsub : ({4, 6, 8} : Finset (Fin 10)) ⊆ decl.D.computed := by decide +kernel
  have hsize : decl.D.size ({4, 6, 8} : Finset (Fin 10)) = 3 := by decide +kernel
  have hr : novelReads decl ({4, 6, 8} : Finset (Fin 10)) = 6 := by decide +kernel
  have hrS : (novelReads decl ({4, 6, 8} : Finset (Fin 10)) : ℝ) ≤ S := by
    rw [hr]
    exact_mod_cast hS
  have h' := h ({4, 6, 8} : Finset (Fin 10)) ⟨hsub, (le_of_eq hsize).trans hΓ, hrS⟩
  have hn : 4 * (threading.weightsUsed ({4, 6, 8} : Finset (Fin 10))).card +
      1 * ∑ t, (threading.threadReads ({4, 6, 8} : Finset (Fin 10)) t).card ≤
        novelReads decl ({4, 6, 8} : Finset (Fin 10)) := by
    exact_mod_cast h'
  have hnot : ¬ (4 * (threading.weightsUsed ({4, 6, 8} : Finset (Fin 10))).card +
      1 * ∑ t, (threading.threadReads ({4, 6, 8} : Finset (Fin 10)) t).card ≤
        novelReads decl ({4, 6, 8} : Finset (Fin 10))) := by
    decide +kernel
  exact hnot hn

/-- The activations are inputs, so no thread value is computed, and none is baseline. -/
theorem threaded (Γ : ℕ) (S : ℝ) : threading.Threaded Γ S 1 0 :=
  threaded_one_zero threading Γ S (by decide +kernel) (by decide +kernel)

/-- `ThreadedReadsSlots` at the witness. -/
theorem reads (Γ : ℕ) (S : ℝ) (R : Finset (Fin 10)) (hR : Small decl Γ S R) :
    (1 : ℝ) * (threading.mults R).card ≤ novelReads decl R * threading.bound S 4 1 0 :=
  ThreadedReadsSlots decl threading slots Γ S 4 1 1 0 (by norm_num) (by norm_num) one_pos (threaded Γ S)
    (weightsIncompressibleSlots Γ S) R hR

end Nci.SecurityProofs.WeightsSlotsWitness
